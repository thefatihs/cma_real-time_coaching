"""Coordinate deterministic coaching evaluation with per-call display state."""

from dataclasses import dataclass
from enum import Enum
import logging

from app.calls.models import CallState
from app.coaching.rule_engine import RuleBasedCoachingEngine
from app.events.models import (
    ClassificationResultEvent,
    CoachingSuggestionEvent,
    TranscriptEvent,
    TranscriptKind,
)
from app.events.validation import ensure_same_call, ensure_same_tenant
from app.tenancy.models import TenantConfig


@dataclass(frozen=True, slots=True)
class CoachingCoordinatorResult:
    classification_event: ClassificationResultEvent | None
    displayed_suggestions: tuple[CoachingSuggestionEvent, ...]
    suppressed_suggestions: tuple[CoachingSuggestionEvent, ...]
    matched_rule_ids: tuple[str, ...]
    suppression_reasons: tuple[str, ...]
    transcript_revision: int | None = None


class CoachingProcessingStatus(str, Enum):
    PROCESSED = "processed"
    PARTIAL_SKIPPED = "partial_skipped"
    DUPLICATE_REVISION_SKIPPED = "duplicate_revision_skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StableCoachingOutcome:
    status: CoachingProcessingStatus
    transcript_revision: int
    result: CoachingCoordinatorResult | None = None
    error_type: str | None = None
    error_code: str | None = None


SuggestionFingerprint = tuple[str, str, str, tuple[str, ...]]


class CoachingCoordinator:
    def __init__(
        self,
        tenant_config: TenantConfig,
        call_state: CallState,
        rule_engine: RuleBasedCoachingEngine,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        tenant_id = ensure_same_tenant(
            tenant_config.context,
            call_state,
            rule_engine,
        )
        if tenant_id != tenant_config.context.tenant_id:
            raise ValueError("Coaching coordinator tenant validation failed")
        self._tenant_config = tenant_config
        self._call_state = call_state
        self._rule_engine = rule_engine
        self._displayed_fingerprints: set[SuggestionFingerprint] = set()
        self._processed_revisions: set[int] = set()
        self._logger = logger or logging.getLogger(__name__)

    def process(
        self,
        event: TranscriptEvent,
        current_seconds: float,
        *,
        classification_event: ClassificationResultEvent | None = None,
        active_labels: tuple[str, ...] | None = None,
    ) -> CoachingCoordinatorResult:
        if current_seconds < 0:
            raise ValueError("current_seconds cannot be negative")
        ensure_same_tenant(self._tenant_config.context, self._call_state, event)
        ensure_same_call(self._call_state, event)
        if event.kind is TranscriptKind.PARTIAL:
            return CoachingCoordinatorResult(None, (), (), (), (), event.revision)
        if event.revision in self._processed_revisions:
            return CoachingCoordinatorResult(
                classification_event,
                (),
                (),
                (),
                ("duplicate_revision",),
                event.revision,
            )
        self._processed_revisions.add(event.revision)
        if classification_event is not None:
            ensure_same_tenant(event, classification_event)
            ensure_same_call(event, classification_event)

        labels_for_evaluation = active_labels or ()
        evaluation = self._rule_engine.evaluate(event, labels_for_evaluation)
        classification = classification_event or evaluation.classification_event
        labels = (
            list(labels_for_evaluation)
            if active_labels is not None
            else (
                [label.name for label in classification.labels]
                if classification is not None
                else []
            )
        )
        self._call_state.update_active_labels(labels)

        displayed: list[CoachingSuggestionEvent] = []
        suppressed: list[CoachingSuggestionEvent] = []
        reasons: list[str] = []
        cooldown_available = self._call_state.can_trigger_coaching(
            current_seconds,
            self._tenant_config.coaching.cooldown_seconds,
        )
        maximum = self._tenant_config.coaching.max_active_suggestions

        for suggestion in evaluation.suggestion_events:
            fingerprint = _suggestion_fingerprint(suggestion)
            if fingerprint in self._displayed_fingerprints:
                suppressed.append(suggestion)
                reasons.append("duplicate")
            elif not cooldown_available:
                suppressed.append(suggestion)
                reasons.append("cooldown")
            elif len(displayed) >= maximum:
                suppressed.append(suggestion)
                reasons.append("max_active_suggestions")
            else:
                displayed.append(suggestion)
                self._displayed_fingerprints.add(fingerprint)
                self._call_state.mark_suggestion_shown(suggestion.suggestion_id)
                self._call_state.apply_coaching_suggestion(
                    suggestion,
                    transcript_revision=event.revision,
                    model_id=(
                        classification_event.model_id
                        if classification_event is not None
                        else None
                    ),
                    threshold_profile_id=(
                        classification_event.threshold_profile_id
                        if classification_event is not None
                        else None
                    ),
                )

        if displayed:
            self._call_state.mark_coaching_triggered(current_seconds)

        return CoachingCoordinatorResult(
            classification_event=classification,
            displayed_suggestions=tuple(displayed),
            suppressed_suggestions=tuple(suppressed),
            matched_rule_ids=evaluation.matched_rule_ids,
            suppression_reasons=tuple(reasons),
            transcript_revision=event.revision,
        )

    def process_safely(
        self,
        event: TranscriptEvent,
        current_seconds: float,
        *,
        classification_event: ClassificationResultEvent | None = None,
        active_labels: tuple[str, ...] | None = None,
    ) -> StableCoachingOutcome:
        if event.kind is TranscriptKind.PARTIAL:
            return StableCoachingOutcome(
                CoachingProcessingStatus.PARTIAL_SKIPPED, event.revision
            )
        if event.revision in self._processed_revisions:
            return StableCoachingOutcome(
                CoachingProcessingStatus.DUPLICATE_REVISION_SKIPPED, event.revision
            )
        try:
            result = self.process(
                event,
                current_seconds,
                classification_event=classification_event,
                active_labels=active_labels,
            )
            return StableCoachingOutcome(
                CoachingProcessingStatus.PROCESSED,
                event.revision,
                result=result,
            )
        except Exception as error:
            self._processed_revisions.add(event.revision)
            self._logger.error(
                "stable transcript coaching failed",
                extra={
                    "tenant_id": event.tenant_id,
                    "call_id": event.call_id,
                    "transcript_revision": event.revision,
                    "error_type": type(error).__name__,
                    "error_code": "coaching_failed",
                },
            )
            return StableCoachingOutcome(
                CoachingProcessingStatus.FAILED,
                event.revision,
                error_type=type(error).__name__,
                error_code="coaching_failed",
            )

    def clear(self) -> None:
        self._displayed_fingerprints.clear()
        self._processed_revisions.clear()


def _suggestion_fingerprint(
    suggestion: CoachingSuggestionEvent,
) -> SuggestionFingerprint:
    return (
        suggestion.action.value,
        suggestion.title,
        suggestion.suggestion,
        tuple(suggestion.evidence_ids),
    )
