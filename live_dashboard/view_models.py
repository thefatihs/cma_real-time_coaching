"""Streamlit-independent state and formatting for the live dashboard demo."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count
from pathlib import Path
from time import perf_counter
from typing import Protocol

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
from app.streaming.pipeline import (
    StreamingASRPlan,
    StreamingASRResult,
    StreamingASRStep,
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
ACTION_LABELS = {
    CoachingAction.NO_ACTION: "Aksiyon yok",
    CoachingAction.TEMPLATE_ACTION: "Hazır öneri",
    CoachingAction.RAG_ACTION: "Bilgi arama",
    CoachingAction.ESCALATE: "Yetkiliye aktar",
}


class LocalPipelineProtocol(Protocol):
    def run(
        self,
        audio_path: Path,
        call_id: str,
        *,
        step_callback: Callable[[StreamingASRStep], None] | None = None,
        plan_callback: Callable[[StreamingASRPlan], None] | None = None,
    ) -> StreamingASRResult: ...


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


@dataclass(frozen=True, slots=True)
class StatusCardViewModel:
    label: str
    value: str


@dataclass(frozen=True, slots=True)
class LatencyViewModel:
    chunk_duration_ms: float
    asr_ms: float
    rule_ms: float
    coaching_ms: float
    total_ms: float
    rule_is_approximate: bool = True
    coaching_is_approximate: bool = True


@dataclass(frozen=True, slots=True)
class ProgressViewModel:
    stage: str
    completed_chunks: int
    total_chunks: int
    percentage: float
    time_range: str
    elapsed: str
    average_asr: str
    eta: str
    failed_chunk: int | None


@dataclass(slots=True)
class DashboardRuntime:
    tenant: TenantDemo
    scenario: DemoScenario | None
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
    latency: LatencyViewModel | None = None

    @property
    def elapsed_seconds(self) -> float:
        return 0.0 if self.latest_event is None else self.latest_event.end_seconds

    @property
    def complete(self) -> bool:
        return self.scenario is not None and self.next_event_index >= len(
            self.scenario.events
        )


@dataclass(slots=True)
class LocalExecutionState:
    runtime: DashboardRuntime
    status: str = "idle"
    start_requested: bool = False
    pipeline_calls: int = 0
    current_chunk: int = 0
    total_chunks: int = 0
    error_message: str | None = None
    asr_window_ms: list[float] = field(default_factory=list)
    processing_seconds: float | None = None
    audio_duration_seconds: float | None = None
    stage: str = "Başlatılmadı"
    elapsed_seconds: float = 0.0
    latest_step: StreamingASRStep | None = None
    failed_chunk: int | None = None

    @property
    def real_time_factor(self) -> float | None:
        if not self.processing_seconds or not self.audio_duration_seconds:
            return None
        return self.processing_seconds / self.audio_duration_seconds

    def request_start(self) -> None:
        if self.status == "idle":
            self.start_requested = True


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


def create_local_execution(tenant: TenantDemo, call_id: str) -> LocalExecutionState:
    """Create an idle local execution; constructing it never runs a pipeline."""
    cleaned_call_id = call_id.strip()
    if not cleaned_call_id:
        raise ValueError("call_id cannot be empty")
    state = CallState(
        tenant_id=tenant.config.context.tenant_id, call_id=cleaned_call_id
    )
    ids = count(1)
    engine = RuleBasedCoachingEngine(
        tenant.config,
        tenant.rules,
        event_id_factory=lambda: (
            f"{tenant.config.context.tenant_id}-local-suggestion-{next(ids)}"
        ),
    )
    runtime = DashboardRuntime(
        tenant,
        None,
        cleaned_call_id,
        state,
        CoachingCoordinator(tenant.config, state, engine),
    )
    return LocalExecutionState(runtime)


def reset_local_execution(state: LocalExecutionState) -> LocalExecutionState:
    return create_local_execution(state.runtime.tenant, state.runtime.call_id)


def execute_local_once(
    state: LocalExecutionState,
    pipeline: LocalPipelineProtocol,
    audio_path: Path,
    progress_callback: Callable[[StreamingASRStep], None] | None = None,
    plan_progress_callback: Callable[[StreamingASRPlan], None] | None = None,
    clock: Callable[[], float] = perf_counter,
) -> bool:
    """Run only after an explicit request and at most once for this state."""
    if not state.start_requested or state.status != "idle":
        return False
    state.start_requested = False
    state.status = "running"
    state.stage = "ASR hazırlanıyor"
    state.pipeline_calls += 1

    def on_plan(plan: StreamingASRPlan) -> None:
        if (
            plan.tenant_id != state.runtime.call_state.tenant_id
            or plan.call_id != state.runtime.call_id
        ):
            raise ValueError("Pipeline plan scope does not match dashboard")
        state.total_chunks = plan.total_chunks
        state.audio_duration_seconds = plan.audio_duration_seconds
        state.stage = "ASR işleniyor"
        if plan_progress_callback is not None:
            plan_progress_callback(plan)

    def on_step(step: StreamingASRStep) -> None:
        state.current_chunk = step.sequence_number + 1
        state.latest_step = step
        state.stage = "Koçluk analizi"
        _consume_step(state, step)
        state.elapsed_seconds = max(clock() - started_at, 0.0)
        if progress_callback is not None:
            progress_callback(step)
        if state.current_chunk < state.total_chunks:
            state.stage = "ASR işleniyor"

    started_at = clock()
    try:
        result = pipeline.run(
            audio_path,
            state.runtime.call_id,
            step_callback=on_step,
            plan_callback=on_plan,
        )
        _consume_pipeline_result(state, result)
    except Exception:
        state.status = "error"
        state.stage = "Başarısız"
        state.failed_chunk = min(state.current_chunk + 1, state.total_chunks or 1)
        state.error_message = "Ses işleme sırasında beklenmeyen bir hata oluştu."
        raise
    state.processing_seconds = max(clock() - started_at, 0.0)
    state.elapsed_seconds = state.processing_seconds
    state.audio_duration_seconds = result.audio_duration_seconds
    state.status = "completed"
    state.stage = "Tamamlandı"
    return True


def reset_runtime(runtime: DashboardRuntime) -> DashboardRuntime:
    if runtime.scenario is None:
        raise ValueError("Synthetic scenario is required")
    return create_runtime(runtime.tenant, runtime.scenario, runtime.call_id)


def advance_runtime(runtime: DashboardRuntime) -> TranscriptEvent | None:
    """Advance exactly one event; no sleeping, looping, audio, or external access."""
    if runtime.complete:
        return None
    if runtime.scenario is None:
        raise ValueError("Synthetic scenario is required")
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
        action_display(event.action),
        event.created_at_utc.strftime("%H:%M:%S"),
    )


def ordered_suggestions(
    cards: list[SuggestionCardViewModel],
) -> list[SuggestionCardViewModel]:
    return sorted(cards, key=lambda card: PRIORITY_RANK[card.priority], reverse=True)


def suppression_reason_display(reason: str) -> str:
    return SUPPRESSION_LABELS.get(reason, reason)


def action_display(action: CoachingAction) -> str:
    return ACTION_LABELS[action]


def responsive_rows[T](
    items: tuple[T, ...], maximum_columns: int = 3
) -> tuple[tuple[T, ...], ...]:
    """Split complete view-model values into compact responsive rows."""
    if maximum_columns <= 0:
        raise ValueError("maximum_columns must be positive")
    return tuple(
        items[index : index + maximum_columns]
        for index in range(0, len(items), maximum_columns)
    )


def status_cards(
    runtime: DashboardRuntime, pipeline_status: str
) -> tuple[StatusCardViewModel, ...]:
    return (
        StatusCardViewModel("Tenant", runtime.tenant.config.context.tenant_name),
        StatusCardViewModel("Çağrı kimliği", runtime.call_id),
        StatusCardViewModel(
            "Çağrı durumu", "Tamamlandı" if runtime.complete else "Hazır"
        ),
        StatusCardViewModel("Geçen süre", _format_elapsed(runtime.elapsed_seconds)),
        StatusCardViewModel("İşlem hattı", pipeline_status),
    )


def progress_view(state: LocalExecutionState) -> ProgressViewModel:
    total = max(state.total_chunks, 0)
    completed = min(max(state.current_chunk, 0), total) if total else 0
    percentage = 0.0 if total == 0 else min(completed / total * 100, 100.0)
    if state.latest_step is None:
        time_range = "Henüz başlanmadı"
    else:
        time_range = (
            f"Ses {state.latest_step.chunk_start_seconds:.2f}–"
            f"{state.latest_step.chunk_end_seconds:.2f} sn · "
            f"Pencere {state.latest_step.window_start_seconds:.2f}–"
            f"{state.latest_step.window_end_seconds:.2f} sn"
        )
    recent_asr = state.asr_window_ms[-5:]
    average_seconds = sum(recent_asr) / len(recent_asr) / 1000 if recent_asr else None
    remaining_chunks = max(total - completed, 0)
    eta_seconds = (
        average_seconds * remaining_chunks if average_seconds is not None else None
    )
    return ProgressViewModel(
        stage=state.stage,
        completed_chunks=completed,
        total_chunks=total,
        percentage=percentage,
        time_range=time_range,
        elapsed=_human_duration(state.elapsed_seconds),
        average_asr=(
            "—" if average_seconds is None else f"{average_seconds * 1000:.0f} ms"
        ),
        eta=(
            "Tahmin hazırlanıyor"
            if eta_seconds is None
            else f"Tahmini kalan süre: {_human_duration(eta_seconds)}"
        ),
        failed_chunk=state.failed_chunk,
    )


def _consume_pipeline_result(
    state: LocalExecutionState, result: StreamingASRResult
) -> None:
    runtime = state.runtime
    if (
        result.tenant_id != runtime.call_state.tenant_id
        or result.call_id != runtime.call_id
    ):
        raise ValueError(
            "Pipeline result tenant_id or call_id does not match dashboard"
        )
    if result.final_event is not None:
        _apply_transcript_event(runtime, result.final_event)
    state.total_chunks = result.total_chunks
    state.current_chunk = result.total_chunks


def _consume_step(state: LocalExecutionState, step: StreamingASRStep) -> None:
    for event in step.transcript_events:
        _apply_transcript_event(state.runtime, event)
    rule_ms = 1.0
    coaching_ms = 1.0
    state.runtime.latency = LatencyViewModel(
        chunk_duration_ms=(step.chunk_end_seconds - step.chunk_start_seconds) * 1000,
        asr_ms=step.transcription_time_seconds * 1000,
        rule_ms=rule_ms,
        coaching_ms=coaching_ms,
        total_ms=step.transcription_time_seconds * 1000 + rule_ms + coaching_ms,
    )
    state.asr_window_ms.append(step.transcription_time_seconds * 1000)


def _apply_transcript_event(runtime: DashboardRuntime, event: TranscriptEvent) -> None:
    runtime.latest_event = event
    runtime.call_state.apply_transcript(event)
    runtime.timeline.append(
        TimelineItem(event.created_at_utc, "Transkript", event.kind.value)
    )
    result = runtime.coordinator.process(event, event.end_seconds)
    _apply_coaching_result(runtime, result, event)
    runtime.timeline.sort(key=lambda item: item.timestamp)


def _format_elapsed(seconds: float) -> str:
    minutes, remaining = divmod(int(seconds), 60)
    return f"{minutes:02d}:{remaining:02d}"


def _human_duration(seconds: float) -> str:
    safe_seconds = max(int(seconds) if seconds == seconds else 0, 0)
    minutes, remaining = divmod(safe_seconds, 60)
    if minutes:
        return f"{minutes} dk {remaining} sn"
    return f"{remaining} sn"


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
