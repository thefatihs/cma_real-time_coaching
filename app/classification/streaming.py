"""Failure-isolated stable transcript classification stage."""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Protocol

from app.calls.models import CallState
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


class StableTranscriptClassificationStage:
    def __init__(
        self,
        classifier: RuntimeClassifierProtocol | None,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._classifier = classifier
        self._logger = logger or logging.getLogger(__name__)

    def process(
        self,
        event: TranscriptEvent,
        *,
        cumulative_stable_transcript: str,
        stable_changed: bool,
        call_state: CallState,
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

        call_state.mark_classification_attempt(
            event.revision, event.source_chunk_sequence
        )
        try:
            result = self._classifier.classify(
                tenant_id=event.tenant_id,
                call_id=event.call_id,
                text=cumulative_stable_transcript,
                transcript_event_id=event.event_id,
                revision=event.revision,
                sequence_number=event.source_chunk_sequence,
            )
            call_state.apply_classification(
                result,
                transcript_revision=event.revision,
                source_sequence=event.source_chunk_sequence,
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
                },
            )
            return StableClassificationOutcome(
                status=ClassificationProcessingStatus.CLASSIFIED,
                transcript_revision=event.revision,
                source_sequence=event.source_chunk_sequence,
                classification_event=result,
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
