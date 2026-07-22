"""Streamlit-independent state and formatting for the live dashboard demo."""

from dataclasses import dataclass, field
from datetime import datetime
from itertools import count

from app.calls.models import CallState
from app.coaching.coordinator import CoachingCoordinator, CoachingCoordinatorResult
from app.coaching.rule_engine import RuleBasedCoachingEngine
from app.events.models import (
    CoachingAction,
    CoachingSuggestionEvent,
    SuggestionPriority,
    TranscriptEvent,
    TranscriptKind,
)
from live_dashboard.demo_data import DemoScenario, TenantDemo


PRIORITY_RANK = {
    SuggestionPriority.CRITICAL: 4,
    SuggestionPriority.HIGH: 3,
    SuggestionPriority.MEDIUM: 2,
    SuggestionPriority.LOW: 1,
}
CRITICAL_LABEL_MARKERS = ("kritik", "risk", "eskalasyon", "aktarim", "aktarımı")
SUPPRESSION_LABELS = {
    "duplicate": "yinelenen öneri",
    "cooldown": "bekleme süresi",
    "max_active_suggestions": "aktif öneri sınırı",
}


@dataclass(frozen=True, slots=True)
class TranscriptViewModel:
    stable_text: str
    partial_text: str
    latest_event_type: str
    partial_is_changeable: bool = True


@dataclass(frozen=True, slots=True)
class LabelViewModel:
    name: str
    score_percent: str
    critical: bool


@dataclass(frozen=True, slots=True)
class SuggestionCardViewModel:
    priority: SuggestionPriority
    priority_text: str
    title: str
    suggestion: str
    action: str
    timestamp: str


@dataclass(frozen=True, slots=True)
class TimelineItem:
    timestamp: datetime
    event_type: str
    detail: str


@dataclass(slots=True)
class DashboardRuntime:
    tenant: TenantDemo
    scenario: DemoScenario
    call_id: str
    call_state: CallState
    coordinator: CoachingCoordinator
    next_event_index: int = 0
    latest_event: TranscriptEvent | None = None
    latest_action: CoachingAction = CoachingAction.NO_ACTION
    latest_labels: tuple[LabelViewModel, ...] = ()
    suggestions: list[SuggestionCardViewModel] = field(default_factory=list)
    timeline: list[TimelineItem] = field(default_factory=list)
    suppression_reasons: list[str] = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        return 0.0 if self.latest_event is None else self.latest_event.end_seconds

    @property
    def complete(self) -> bool:
        return self.next_event_index >= len(self.scenario.events)


def create_runtime(
    tenant: TenantDemo, scenario: DemoScenario, call_id: str
) -> DashboardRuntime:
    """Create a clean call state, engine, and coordinator."""
    cleaned_call_id = call_id.strip() or "demo-call"
    state = CallState(
        tenant_id=tenant.config.context.tenant_id, call_id=cleaned_call_id
    )
    ids = count(1)
    engine = RuleBasedCoachingEngine(
        tenant.config,
        tenant.rules,
        event_id_factory=lambda: (
            f"{tenant.config.context.tenant_id}-suggestion-{next(ids)}"
        ),
    )
    return DashboardRuntime(
        tenant,
        scenario,
        cleaned_call_id,
        state,
        CoachingCoordinator(tenant.config, state, engine),
    )


def reset_runtime(runtime: DashboardRuntime) -> DashboardRuntime:
    return create_runtime(runtime.tenant, runtime.scenario, runtime.call_id)


def advance_runtime(runtime: DashboardRuntime) -> TranscriptEvent | None:
    """Advance exactly one event; no sleeping, looping, audio, or external access."""
    if runtime.complete:
        return None
    source = runtime.scenario.events[runtime.next_event_index]
    event = source.model_copy(update={"call_id": runtime.call_id})
    runtime.next_event_index += 1
    runtime.latest_event = event
    runtime.call_state.apply_transcript(event)
    runtime.timeline.append(
        TimelineItem(event.created_at_utc, "Transkript", event.kind.value)
    )
    result = runtime.coordinator.process(event, event.end_seconds)
    _apply_coaching_result(runtime, result, event)
    runtime.timeline.sort(key=lambda item: item.timestamp)
    return event


def transcript_view(runtime: DashboardRuntime) -> TranscriptViewModel:
    return TranscriptViewModel(
        runtime.call_state.stable_transcript,
        runtime.call_state.partial_transcript,
        runtime.latest_event.kind.value if runtime.latest_event else "BEKLİYOR",
    )


def suggestion_card(event: CoachingSuggestionEvent) -> SuggestionCardViewModel:
    return SuggestionCardViewModel(
        event.priority,
        event.priority.value,
        event.title,
        event.suggestion,
        event.action.value,
        event.created_at_utc.strftime("%H:%M:%S"),
    )


def ordered_suggestions(
    cards: list[SuggestionCardViewModel],
) -> list[SuggestionCardViewModel]:
    return sorted(cards, key=lambda card: PRIORITY_RANK[card.priority], reverse=True)


def suppression_reason_display(reason: str) -> str:
    return SUPPRESSION_LABELS.get(reason, reason)


def _apply_coaching_result(
    runtime: DashboardRuntime, result: CoachingCoordinatorResult, event: TranscriptEvent
) -> None:
    classification = result.classification_event
    if classification is not None:
        runtime.latest_action = classification.action
        runtime.latest_labels = tuple(
            LabelViewModel(
                label.name,
                f"%{label.score * 100:.0f}",
                any(
                    marker in label.name.casefold() for marker in CRITICAL_LABEL_MARKERS
                ),
            )
            for label in classification.labels
        )
        runtime.timeline.append(
            TimelineItem(
                classification.created_at_utc,
                "Sınıflandırma",
                ", ".join(label.name for label in classification.labels),
            )
        )
    elif event.kind is not TranscriptKind.PARTIAL:
        runtime.latest_action = CoachingAction.NO_ACTION
        runtime.latest_labels = ()
    for item in result.displayed_suggestions:
        runtime.suggestions.append(suggestion_card(item))
        runtime.timeline.append(
            TimelineItem(item.created_at_utc, "Öneri gösterildi", item.title)
        )
    for item, reason in zip(
        result.suppressed_suggestions, result.suppression_reasons, strict=True
    ):
        display = suppression_reason_display(reason)
        runtime.suppression_reasons.append(display)
        runtime.timeline.append(
            TimelineItem(
                item.created_at_utc, "Öneri bastırıldı", f"{item.title}: {display}"
            )
        )
