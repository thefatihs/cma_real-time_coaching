"""Failure-isolated stable transcript classification stage."""

from dataclasses import dataclass
from enum import Enum
import logging
import re
from typing import Protocol

from app.calls.models import CallState
from app.classification.postprocessing import (
    ClassificationPostProcessingMetadata,
    apply_classification_contrast_guards,
)
from app.events.models import (
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


class StableTranscriptClassificationStage:
    def __init__(
        self,
        classifier: RuntimeClassifierProtocol | None,
        *,
        logger: logging.Logger | None = None,
        maximum_preceding_sentences: int = 2,
    ) -> None:
        if maximum_preceding_sentences < 0 or maximum_preceding_sentences > 2:
            raise ValueError("maximum_preceding_sentences must be between 0 and 2")
        self._classifier = classifier
        self._logger = logger or logging.getLogger(__name__)
        self._maximum_preceding_sentences = maximum_preceding_sentences

    def process(
        self,
        event: TranscriptEvent,
        *,
        cumulative_stable_transcript: str,
        stable_changed: bool,
        call_state: CallState,
        stable_delta: str | None = None,
        preceding_stable_transcript: str = "",
    ) -> StableClassificationOutcome:
        if event.kind is TranscriptKind.PARTIAL:
            return self._outcome(ClassificationProcessingStatus.PARTIAL_SKIPPED, event)
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
            raw_result = self._classifier.classify(
                tenant_id=event.tenant_id,
                call_id=event.call_id,
                text=classification_text,
                transcript_event_id=event.event_id,
                revision=event.revision,
                sequence_number=event.source_chunk_sequence,
            )
            result, postprocessing = apply_classification_contrast_guards(
                classification_text,
                raw_result,
            )
            call_state.apply_classification(
                result,
                transcript_revision=event.revision,
                source_sequence=event.source_chunk_sequence,
                context_sentence_count=sentence_count,
                preceding_sentence_count=preceding_count,
                delta_word_count=len(delta.split()),
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
