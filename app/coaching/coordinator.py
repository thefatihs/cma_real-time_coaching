"""Coordinate deterministic coaching evaluation with per-call display state."""

from dataclasses import dataclass

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


SuggestionFingerprint = tuple[str, str, str, tuple[str, ...]]


class CoachingCoordinator:
    def __init__(
        self,
        tenant_config: TenantConfig,
        call_state: CallState,
        rule_engine: RuleBasedCoachingEngine,
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

    def process(
        self, event: TranscriptEvent, current_seconds: float
    ) -> CoachingCoordinatorResult:
        if current_seconds < 0:
            raise ValueError("current_seconds cannot be negative")
        ensure_same_tenant(self._tenant_config.context, self._call_state, event)
        ensure_same_call(self._call_state, event)
        if event.kind is TranscriptKind.PARTIAL:
            return CoachingCoordinatorResult(None, (), (), (), ())

        evaluation = self._rule_engine.evaluate(event)
        classification = evaluation.classification_event
        labels = (
            [label.name for label in classification.labels]
            if classification is not None
            else []
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

        if displayed:
            self._call_state.mark_coaching_triggered(current_seconds)

        return CoachingCoordinatorResult(
            classification_event=classification,
            displayed_suggestions=tuple(displayed),
            suppressed_suggestions=tuple(suppressed),
            matched_rule_ids=evaluation.matched_rule_ids,
            suppression_reasons=tuple(reasons),
        )

    def clear(self) -> None:
        self._displayed_fingerprints.clear()


def _suggestion_fingerprint(
    suggestion: CoachingSuggestionEvent,
) -> SuggestionFingerprint:
    return (
        suggestion.action.value,
        suggestion.title,
        suggestion.suggestion,
        tuple(suggestion.evidence_ids),
    )
