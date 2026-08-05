"""Failure-isolated stable transcript classification stage."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import logging
from math import isfinite
import re
from time import monotonic
from typing import Protocol

from app.calls.models import CallState
from app.classification.postprocessing import (
    ClassificationPostProcessingMetadata,
    apply_classification_contrast_guards,
    canonicalize_classification_result,
    merge_classification_views,
)
from app.events.labels import ClassificationViewSource
from app.events.models import (
    CoachingAction,
    ClassificationResultEvent,
    TranscriptEvent,
    TranscriptKind,
)


class RuntimeClassifierProtocol(Protocol):
    def classify(
        self,
        *,
        tenant_id: str,
        call_id: str,
        text: str,
        transcript_event_id: str | None = None,
        revision: int | None = None,
        sequence_number: int | None = None,
    ) -> ClassificationResultEvent: ...


class ClassificationProcessingStatus(str, Enum):
    CLASSIFIED = "classified"
    PROVISIONAL_CLASSIFIED = "provisional_classified"
    PARTIAL_SKIPPED = "partial_skipped"
    UNCHANGED_SKIPPED = "unchanged_skipped"
    DUPLICATE_REVISION_SKIPPED = "duplicate_revision_skipped"
    EMPTY_SKIPPED = "empty_skipped"
    DISABLED = "disabled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SafeClassificationError:
    error_type: str
    code: str = "classification_failed"


@dataclass(frozen=True, slots=True)
class StableClassificationOutcome:
    status: ClassificationProcessingStatus
    transcript_revision: int | None
    source_sequence: int | None
    classification_event: ClassificationResultEvent | None = None
    error: SafeClassificationError | None = None
    postprocessing: ClassificationPostProcessingMetadata = (
        ClassificationPostProcessingMetadata()
    )
    context_sentence_count: int = 0
    preceding_sentence_count: int = 0
    delta_word_count: int = 0
    delta_inference_ran: bool = False
    context_inference_ran: bool = False
    delta_inference_time_ms: float | None = None
    context_inference_time_ms: float | None = None
    delta_labels: tuple[str, ...] = ()
    context_labels: tuple[str, ...] = ()
    label_view_sources: tuple[tuple[str, ClassificationViewSource], ...] = ()
    provisional: bool = False


_DEFAULT_PROVISIONAL_LABELS = frozenset(
    {
        "cancellation_request",
        "churn_risk",
        "complaint",
        "price_objection",
        "product_information",
        "renewal_interest",
        "technical_issue",
    }
)
_DEFAULT_CRITICAL_PROVISIONAL_LABELS = frozenset(
    {"cancellation_request", "churn_risk", "complaint"}
)


@dataclass(frozen=True, slots=True)
class ProvisionalClassificationPolicy:
    enabled: bool = False
    minimum_words: int = 3
    minimum_growth_words: int = 1
    minimum_interval_seconds: float = 1.0
    threshold_increment: float = 0.15
    minimum_threshold: float = 0.85
    critical_threshold_increment: float = 0.10
    critical_minimum_threshold: float = 0.90
    allowed_labels: frozenset[str] = _DEFAULT_PROVISIONAL_LABELS
    critical_labels: frozenset[str] = _DEFAULT_CRITICAL_PROVISIONAL_LABELS

    def __post_init__(self) -> None:
        if self.minimum_words < 1 or self.minimum_words > 32:
            raise ValueError("minimum_words must be between 1 and 32")
        if self.minimum_growth_words < 1 or self.minimum_growth_words > 32:
            raise ValueError("minimum_growth_words must be between 1 and 32")
        if not 0 <= self.minimum_interval_seconds <= 60:
            raise ValueError("minimum_interval_seconds must be between 0 and 60")
        for value in (
            self.threshold_increment,
            self.minimum_threshold,
            self.critical_threshold_increment,
            self.critical_minimum_threshold,
        ):
            if not 0 <= value <= 1:
                raise ValueError("provisional thresholds must be between 0 and 1")
        if not self.allowed_labels:
            raise ValueError("allowed_labels cannot be empty")
        if not self.critical_labels.issubset(self.allowed_labels):
            raise ValueError("critical_labels must be included in allowed_labels")


class StableTranscriptClassificationStage:
    def __init__(
        self,
        classifier: RuntimeClassifierProtocol | None,
        *,
        logger: logging.Logger | None = None,
        maximum_preceding_sentences: int = 2,
        provisional_policy: ProvisionalClassificationPolicy | None = None,
        monotonic_clock: Callable[[], float] = monotonic,
    ) -> None:
        if maximum_preceding_sentences < 0 or maximum_preceding_sentences > 2:
            raise ValueError("maximum_preceding_sentences must be between 0 and 2")
        self._classifier = classifier
        self._logger = logger or logging.getLogger(__name__)
        self._maximum_preceding_sentences = maximum_preceding_sentences
        self._provisional_policy = (
            provisional_policy or ProvisionalClassificationPolicy()
        )
        self._monotonic_clock = monotonic_clock
        self._processed_partial_chunk_keys: list[int] = []
        self._last_partial_text = ""
        self._last_partial_word_count = 0
        self._last_partial_wall_evaluation_seconds: float | None = None
        self._last_partial_media_evaluation_seconds: float | None = None
        self._last_partial_media_progress_seconds: float | None = None

    def configure_provisional_policy(
        self,
        policy: ProvisionalClassificationPolicy,
        *,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self._provisional_policy = policy
        if monotonic_clock is not None:
            self._monotonic_clock = monotonic_clock
        self.reset_provisional_source()

    def reset_provisional_source(self) -> None:
        """Reset bounded PARTIAL cadence state for one new uploaded source."""
        self._processed_partial_chunk_keys.clear()
        self._last_partial_text = ""
        self._last_partial_word_count = 0
        self._last_partial_wall_evaluation_seconds = None
        self._last_partial_media_evaluation_seconds = None
        self._last_partial_media_progress_seconds = None

    def process(
        self,
        event: TranscriptEvent,
        *,
        cumulative_stable_transcript: str,
        stable_changed: bool,
        call_state: CallState,
        stable_delta: str | None = None,
        preceding_stable_transcript: str = "",
        allow_provisional: bool = True,
        media_progress_seconds: float | None = None,
    ) -> StableClassificationOutcome:
        if event.kind is TranscriptKind.PARTIAL:
            return self._process_partial(
                event,
                preceding_stable_transcript=preceding_stable_transcript,
                allow_provisional=allow_provisional,
                media_progress_seconds=media_progress_seconds,
            )
        if not stable_changed:
            return self._outcome(
                ClassificationProcessingStatus.UNCHANGED_SKIPPED, event
            )
        if not cumulative_stable_transcript.strip():
            return self._outcome(ClassificationProcessingStatus.EMPTY_SKIPPED, event)
        if (
            call_state.classification_transcript_revision is not None
            and event.revision <= call_state.classification_transcript_revision
        ):
            return self._outcome(
                ClassificationProcessingStatus.DUPLICATE_REVISION_SKIPPED,
                event,
            )
        if self._classifier is None:
            return self._outcome(ClassificationProcessingStatus.DISABLED, event)

        delta = (stable_delta if stable_delta is not None else event.text).strip()
        if not delta:
            return self._outcome(ClassificationProcessingStatus.EMPTY_SKIPPED, event)
        classification_text, preceding_count, sentence_count = _bounded_context(
            preceding_stable_transcript,
            delta,
            maximum_preceding_sentences=self._maximum_preceding_sentences,
        )
        call_state.mark_classification_attempt(
            event.revision, event.source_chunk_sequence
        )
        try:
            raw_delta_result = self._classifier.classify(
                tenant_id=event.tenant_id,
                call_id=event.call_id,
                text=delta,
                transcript_event_id=event.event_id,
                revision=event.revision,
                sequence_number=event.source_chunk_sequence,
            )
            raw_context_result = self._classifier.classify(
                tenant_id=event.tenant_id,
                call_id=event.call_id,
                text=classification_text,
                transcript_event_id=event.event_id,
                revision=event.revision,
                sequence_number=event.source_chunk_sequence,
            )
            delta_result = canonicalize_classification_result(raw_delta_result)
            context_result = canonicalize_classification_result(raw_context_result)
            merged_result, label_view_sources = merge_classification_views(
                delta_result,
                context_result,
            )
            result, postprocessing = apply_classification_contrast_guards(
                classification_text,
                merged_result,
            )
            active_names = {label.name for label in result.labels}
            label_view_sources = {
                label: source
                for label, source in label_view_sources.items()
                if label in active_names
            }
            call_state.apply_classification(
                result,
                transcript_revision=event.revision,
                source_sequence=event.source_chunk_sequence,
                context_sentence_count=sentence_count,
                preceding_sentence_count=preceding_count,
                delta_word_count=len(delta.split()),
                delta_inference_ran=True,
                context_inference_ran=True,
                delta_inference_time_ms=delta_result.processing_time_ms,
                context_inference_time_ms=context_result.processing_time_ms,
                delta_labels=tuple(label.name for label in delta_result.labels),
                context_labels=tuple(label.name for label in context_result.labels),
                label_view_sources=label_view_sources,
            )
            self._logger.info(
                "stable transcript classification completed",
                extra={
                    "tenant_id": event.tenant_id,
                    "call_id": event.call_id,
                    "transcript_revision": event.revision,
                    "source_sequence": event.source_chunk_sequence,
                    "model_id": result.model_id,
                    "threshold_profile_id": result.threshold_profile_id,
                    "labels": [label.name for label in result.labels],
                    "inference_time_ms": result.processing_time_ms,
                    "delta_inference_ran": True,
                    "context_inference_ran": True,
                    "delta_inference_time_ms": delta_result.processing_time_ms,
                    "context_inference_time_ms": context_result.processing_time_ms,
                    "delta_labels": [label.name for label in delta_result.labels],
                    "context_labels": [label.name for label in context_result.labels],
                    "context_sentence_count": sentence_count,
                    "preceding_sentence_count": preceding_count,
                    "delta_word_count": len(delta.split()),
                },
            )
            return StableClassificationOutcome(
                status=ClassificationProcessingStatus.CLASSIFIED,
                transcript_revision=event.revision,
                source_sequence=event.source_chunk_sequence,
                classification_event=result,
                postprocessing=postprocessing,
                context_sentence_count=sentence_count,
                preceding_sentence_count=preceding_count,
                delta_word_count=len(delta.split()),
                delta_inference_ran=True,
                context_inference_ran=True,
                delta_inference_time_ms=delta_result.processing_time_ms,
                context_inference_time_ms=context_result.processing_time_ms,
                delta_labels=tuple(label.name for label in delta_result.labels),
                context_labels=tuple(label.name for label in context_result.labels),
                label_view_sources=tuple(label_view_sources.items()),
            )
        except Exception as error:
            safe_error = SafeClassificationError(error_type=type(error).__name__)
            self._logger.error(
                "stable transcript classification failed",
                extra={
                    "tenant_id": event.tenant_id,
                    "call_id": event.call_id,
                    "transcript_revision": event.revision,
                    "source_sequence": event.source_chunk_sequence,
                    "error_type": safe_error.error_type,
                    "error_code": safe_error.code,
                },
            )
            return StableClassificationOutcome(
                status=ClassificationProcessingStatus.FAILED,
                transcript_revision=event.revision,
                source_sequence=event.source_chunk_sequence,
                error=safe_error,
                context_sentence_count=sentence_count,
                preceding_sentence_count=preceding_count,
                delta_word_count=len(delta.split()),
            )

    def _process_partial(
        self,
        event: TranscriptEvent,
        *,
        preceding_stable_transcript: str,
        allow_provisional: bool,
        media_progress_seconds: float | None,
    ) -> StableClassificationOutcome:
        policy = self._provisional_policy
        if not policy.enabled or not allow_provisional or self._classifier is None:
            return self._outcome(ClassificationProcessingStatus.PARTIAL_SKIPPED, event)
        chunk_key = (
            event.source_chunk_sequence
            if event.source_chunk_sequence is not None
            else event.revision
        )
        if chunk_key in self._processed_partial_chunk_keys:
            return self._outcome(ClassificationProcessingStatus.PARTIAL_SKIPPED, event)
        normalized = " ".join(event.text.casefold().split())
        words = normalized.split()
        if not normalized or len(words) < policy.minimum_words:
            return self._outcome(ClassificationProcessingStatus.PARTIAL_SKIPPED, event)
        if normalized == self._last_partial_text:
            return self._outcome(ClassificationProcessingStatus.PARTIAL_SKIPPED, event)
        if (
            self._last_partial_text
            and normalized.startswith(self._last_partial_text)
            and len(words) - self._last_partial_word_count < policy.minimum_growth_words
        ):
            return self._outcome(ClassificationProcessingStatus.PARTIAL_SKIPPED, event)
        cadence_seconds = self._accepted_cadence_seconds(media_progress_seconds)
        if cadence_seconds is None:
            return self._outcome(ClassificationProcessingStatus.PARTIAL_SKIPPED, event)
        self._processed_partial_chunk_keys.append(chunk_key)
        del self._processed_partial_chunk_keys[:-64]
        self._last_partial_text = normalized
        self._last_partial_word_count = len(words)
        classification_text, preceding_count, sentence_count = _bounded_context(
            preceding_stable_transcript,
            event.text,
            maximum_preceding_sentences=self._maximum_preceding_sentences,
        )
        try:
            raw_delta_result = self._classifier.classify(
                tenant_id=event.tenant_id,
                call_id=event.call_id,
                text=event.text,
                transcript_event_id=event.event_id,
                revision=event.revision,
                sequence_number=event.source_chunk_sequence,
            )
            raw_context_result = self._classifier.classify(
                tenant_id=event.tenant_id,
                call_id=event.call_id,
                text=classification_text,
                transcript_event_id=event.event_id,
                revision=event.revision,
                sequence_number=event.source_chunk_sequence,
            )
            delta_result = canonicalize_classification_result(raw_delta_result)
            context_result = canonicalize_classification_result(raw_context_result)
            merged_result, label_view_sources = merge_classification_views(
                delta_result,
                context_result,
            )
            result, postprocessing = apply_classification_contrast_guards(
                classification_text,
                merged_result,
            )
            admitted_labels = []
            for label in result.labels:
                if label.name not in policy.allowed_labels:
                    continue
                committed_threshold = result.thresholds.get(label.name)
                probability = result.probabilities.get(label.name, label.score)
                if committed_threshold is None:
                    continue
                critical = label.name in policy.critical_labels
                increment = (
                    policy.critical_threshold_increment
                    if critical
                    else policy.threshold_increment
                )
                floor = (
                    policy.critical_minimum_threshold
                    if critical
                    else policy.minimum_threshold
                )
                if probability >= max(committed_threshold + increment, floor):
                    admitted_labels.append(label)
            filtered = result.model_copy(
                update={
                    "labels": admitted_labels,
                    "action": (
                        result.action if admitted_labels else CoachingAction.NO_ACTION
                    ),
                    "provisional": True,
                }
            )
            active_names = {label.name for label in admitted_labels}
            return StableClassificationOutcome(
                status=ClassificationProcessingStatus.PROVISIONAL_CLASSIFIED,
                transcript_revision=event.revision,
                source_sequence=event.source_chunk_sequence,
                classification_event=filtered,
                postprocessing=postprocessing,
                context_sentence_count=sentence_count,
                preceding_sentence_count=preceding_count,
                delta_word_count=len(words),
                delta_inference_ran=True,
                context_inference_ran=True,
                delta_inference_time_ms=delta_result.processing_time_ms,
                context_inference_time_ms=context_result.processing_time_ms,
                delta_labels=tuple(label.name for label in delta_result.labels),
                context_labels=tuple(label.name for label in context_result.labels),
                label_view_sources=tuple(
                    (label, source)
                    for label, source in label_view_sources.items()
                    if label in active_names
                ),
                provisional=True,
            )
        except Exception as error:
            safe_error = SafeClassificationError(error_type=type(error).__name__)
            return StableClassificationOutcome(
                status=ClassificationProcessingStatus.FAILED,
                transcript_revision=event.revision,
                source_sequence=event.source_chunk_sequence,
                error=safe_error,
                delta_word_count=len(words),
                provisional=True,
            )

    def _accepted_cadence_seconds(
        self,
        media_progress_seconds: float | None,
    ) -> float | None:
        interval = self._provisional_policy.minimum_interval_seconds
        if media_progress_seconds is None:
            now = self._monotonic_clock()
            if now < 0:
                raise ValueError("monotonic clock cannot be negative")
            previous = self._last_partial_wall_evaluation_seconds
            if previous is not None and now - previous < interval:
                return None
            self._last_partial_wall_evaluation_seconds = now
            return now

        if not isfinite(media_progress_seconds) or media_progress_seconds < 0:
            return None
        previous_progress = self._last_partial_media_progress_seconds
        if (
            previous_progress is not None
            and media_progress_seconds <= previous_progress
        ):
            return None
        self._last_partial_media_progress_seconds = media_progress_seconds
        previous_evaluation = self._last_partial_media_evaluation_seconds
        if (
            previous_evaluation is not None
            and media_progress_seconds - previous_evaluation < interval
        ):
            return None
        self._last_partial_media_evaluation_seconds = media_progress_seconds
        return media_progress_seconds

    @staticmethod
    def _outcome(
        status: ClassificationProcessingStatus, event: TranscriptEvent
    ) -> StableClassificationOutcome:
        return StableClassificationOutcome(
            status=status,
            transcript_revision=event.revision,
            source_sequence=event.source_chunk_sequence,
        )


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _bounded_context(
    preceding_transcript: str,
    delta: str,
    *,
    maximum_preceding_sentences: int,
) -> tuple[str, int, int]:
    preceding_sentences = _sentences(preceding_transcript)
    selected_preceding = preceding_sentences[-maximum_preceding_sentences:]
    if maximum_preceding_sentences == 0:
        selected_preceding = []
    delta_sentences = _sentences(delta)
    parts = [*selected_preceding, *delta_sentences]
    return " ".join(parts), len(selected_preceding), len(parts)


def _sentences(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(cleaned) if part.strip()]
