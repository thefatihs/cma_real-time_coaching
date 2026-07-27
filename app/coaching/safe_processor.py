"""Fail-closed adapter for synchronous coaching processors."""

import logging

from app.coaching.coordinator import (
    CoachingCoordinator,
    CoachingProcessingStatus,
    CoachingStateSnapshot,
    StableCoachingOutcome,
)
from app.events.models import (
    ClassificationResultEvent,
    CoachingSuggestionEvent,
    TranscriptEvent,
)
from app.events.validation import ensure_same_call, ensure_same_tenant

_DELEGATE_EXCEPTION = "delegate_exception"
_INVALID_RESULT_TYPE = "invalid_result_type"
_SCOPE_MISMATCH = "scope_mismatch"
_RESULT_VALIDATION_FAILED = "result_validation_failed"


class SafeCoachingProcessorAdapter:
    def __init__(
        self,
        coordinator: CoachingCoordinator,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._call_state = coordinator.call_state
        self._logger = logger or logging.getLogger(__name__)

    def process_safely(
        self,
        event: TranscriptEvent,
        current_seconds: float,
        *,
        classification_event: ClassificationResultEvent | None = None,
        active_labels: tuple[str, ...] | None = None,
    ) -> StableCoachingOutcome:
        snapshot = self._coordinator.snapshot_coaching_state()
        reason = self._validate_input_scope(event, classification_event)
        if reason is not None:
            return self._fallback(snapshot, event.revision, reason)

        try:
            outcome = self._coordinator.process_safely(
                event,
                current_seconds,
                classification_event=classification_event,
                active_labels=active_labels,
            )
        except Exception:
            return self._fallback(snapshot, event.revision, _DELEGATE_EXCEPTION)

        reason = self._validate_outcome(outcome, event)
        if reason is not None:
            return self._fallback(snapshot, event.revision, reason)
        if outcome.status is CoachingProcessingStatus.FAILED:
            return self._fallback(snapshot, event.revision, _DELEGATE_EXCEPTION)
        return outcome

    def _validate_input_scope(
        self,
        event: TranscriptEvent,
        classification_event: ClassificationResultEvent | None,
    ) -> str | None:
        try:
            ensure_same_tenant(self._call_state, event)
            ensure_same_call(self._call_state, event)
            if event.revision != self._call_state.transcript_revision:
                return _SCOPE_MISMATCH
            if classification_event is not None:
                ensure_same_tenant(event, classification_event)
                ensure_same_call(event, classification_event)
                if classification_event.transcript_event_id != event.event_id:
                    return _SCOPE_MISMATCH
        except (TypeError, ValueError):
            return _SCOPE_MISMATCH
        return None

    def _validate_outcome(
        self,
        outcome: object,
        event: TranscriptEvent,
    ) -> str | None:
        if not isinstance(outcome, StableCoachingOutcome):
            return _INVALID_RESULT_TYPE
        if outcome.transcript_revision != event.revision:
            return _RESULT_VALIDATION_FAILED
        result = outcome.result
        if outcome.status is CoachingProcessingStatus.PROCESSED and result is None:
            return _RESULT_VALIDATION_FAILED
        if result is None:
            return None
        if result.transcript_revision != event.revision:
            return _RESULT_VALIDATION_FAILED
        if result.classification_event is not None and not self._event_scope_matches(
            result.classification_event,
            event,
        ):
            return _SCOPE_MISMATCH
        suggestions = (
            *result.displayed_suggestions,
            *result.suppressed_suggestions,
        )
        if any(not self._suggestion_scope_matches(item, event) for item in suggestions):
            return _SCOPE_MISMATCH
        return None

    def _event_scope_matches(
        self,
        classification: ClassificationResultEvent,
        event: TranscriptEvent,
    ) -> bool:
        return (
            classification.tenant_id == self._call_state.tenant_id
            and classification.call_id == self._call_state.call_id
            and classification.tenant_id == event.tenant_id
            and classification.call_id == event.call_id
            and classification.transcript_event_id == event.event_id
        )

    def _suggestion_scope_matches(
        self,
        suggestion: CoachingSuggestionEvent,
        event: TranscriptEvent,
    ) -> bool:
        return (
            suggestion.tenant_id == self._call_state.tenant_id
            and suggestion.call_id == self._call_state.call_id
            and suggestion.tenant_id == event.tenant_id
            and suggestion.call_id == event.call_id
            and suggestion.source_transcript_event_id == event.event_id
        )

    def _fallback(
        self,
        snapshot: CoachingStateSnapshot,
        revision: int,
        reason: str,
    ) -> StableCoachingOutcome:
        self._coordinator.restore_coaching_state(snapshot)
        self._logger.error(
            "safe coaching processor rejected delegate result",
            extra={"reason_code": reason},
        )
        return StableCoachingOutcome(
            status=CoachingProcessingStatus.FAILED,
            transcript_revision=revision,
            error_type="SafeCoachingProcessorFailure",
            error_code=reason,
        )
