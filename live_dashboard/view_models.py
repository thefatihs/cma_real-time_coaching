"""Streamlit-independent state and formatting for the live dashboard demo."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from itertools import count
import logging
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from app.calls.models import CallRevisionLabelDiagnostic, CallState
from app.events.labels import canonical_label, canonical_labels
from app.classification.streaming import (
    ClassificationProcessingStatus,
    StableClassificationOutcome,
)
from app.coaching.coordinator import (
    CoachingCoordinator,
    CoachingCoordinatorResult,
    CoachingProcessingStatus,
    SafeSuggestionDecision,
    StableCoachingOutcome,
)
from app.coaching.rule_engine import RuleBasedCoachingEngine
from app.diarization.composition import (
    DiarizationCompositionOutcome,
    DiarizationCompositionStatus,
)
from app.diarization.models import SpeakerRole
from app.events.models import (
    CoachingAction,
    CoachingSuggestionEvent,
    CoachingSuggestionSource,
    SuggestionPriority,
    TranscriptEvent,
    TranscriptKind,
)
from app.streaming.pipeline import (
    StreamingASRPipeline,
    StreamingASRPlan,
    StreamingASRResult,
    StreamingASRStep,
)
from live_dashboard.demo_data import DemoScenario, TenantDemo
from live_dashboard.uploaded_audio import SafeUploadMetadata


PRIORITY_RANK = {
    SuggestionPriority.CRITICAL: 4,
    SuggestionPriority.HIGH: 3,
    SuggestionPriority.MEDIUM: 2,
    SuggestionPriority.LOW: 1,
}
CRITICAL_LABEL_MARKERS = ("kritik", "risk", "eskalasyon", "aktarim", "aktarımı")
SUPPRESSION_LABELS = {
    "duplicate_same_revision": "yinelenen öneri",
    "cooldown_previously_displayed": "bekleme süresi",
    "rejected_by_capacity": "aktif öneri sınırı",
}
ACTION_LABELS = {
    CoachingAction.NO_ACTION: "Aksiyon yok",
    CoachingAction.TEMPLATE_ACTION: "Hazır öneri",
    CoachingAction.RAG_ACTION: "Bilgi arama",
    CoachingAction.ESCALATE: "Yetkiliye aktar",
}
_MAX_LOCAL_ASR_WINDOWS = 64
_MAX_LOCAL_TIMELINE_ITEMS = 128
_MAX_LOCAL_SUGGESTION_HISTORY = 64
_MAX_LOCAL_DECISIONS = 128
_MAX_LOCAL_DEDUPLICATION_KEYS = 256
INTENT_LABELS = {
    "product_information": "Ürün Bilgisi",
    "price_objection": "Fiyat İtirazı",
    "cancellation_request": "İptal Talebi",
    "technical_issue": "Teknik Sorun",
    "complaint": "Şikâyet",
    "renewal_interest": "Yenileme İlgisi",
    "churn_risk": "Müşteri Kaybı Riski",
    "no_action": "Aksiyon Gerekmiyor",
    "urun_bilgisi": "Ürün bilgisi",
    "paket_sorusu": "Ürün bilgisi",
    "fiyat_itirazi": "Fiyat itirazı",
    "butce_endisesi": "Fiyat itirazı",
    "kritik_eskalasyon": "Kritik risk",
    "yonetici_aktarimi": "Kritik risk",
}
SOURCE_LABELS = {
    CoachingSuggestionSource.RULE: "Kural",
    CoachingSuggestionSource.CLASSIFICATION: "Sınıflandırma",
    CoachingSuggestionSource.BOTH: "Kural + sınıflandırma",
    CoachingSuggestionSource.LLM: "LLM",
}
PRIORITY_SYMBOLS = {
    SuggestionPriority.LOW: "○",
    SuggestionPriority.MEDIUM: "●",
    SuggestionPriority.HIGH: "▲",
    SuggestionPriority.CRITICAL: "◆",
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


class DashboardExecutionStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class DashboardExecutionMode(str, Enum):
    FAST_ANALYSIS = "FAST_ANALYSIS"
    REALTIME_SIMULATION = "REALTIME_SIMULATION"


class DashboardExecutionStage(str, Enum):
    STARTING = "STARTING"
    FILE_PREPARING = "FILE_PREPARING"
    ENGINE_RUNNING = "ENGINE_RUNNING"
    CHUNK_PROCESSING = "CHUNK_PROCESSING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


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
    suggestion_id: str
    priority: SuggestionPriority
    priority_text: str
    title: str
    suggestion: str
    action: str
    timestamp: str
    related_label: str | None
    evidence_ids: tuple[str, ...]
    priority_symbol: str
    source: str
    transcript_revision: int | None
    is_new: bool


@dataclass(frozen=True, slots=True)
class IntentChipViewModel:
    text: str
    score: str
    is_risk: bool
    symbol: str


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
class SpeakerCardViewModel:
    slot: str
    role: str
    aligned_word_count: int
    confidence_bucket: str
    decision_reason: str


@dataclass(frozen=True, slots=True)
class SpeakerDashboardViewModel:
    speakers: tuple[SpeakerCardViewModel, ...]
    speaker_count: int
    turn_count: int
    projected_customer_word_count: int
    unknown_exclusion_count: int


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


@dataclass(frozen=True, slots=True)
class RepresentativeTabViewModel:
    status: tuple[StatusCardViewModel, ...]
    progress: ProgressViewModel
    transcript: TranscriptViewModel
    intent_chips: tuple[IntentChipViewModel, ...]
    detected_intent_chips: tuple[IntentChipViewModel, ...]
    suppressed_count: int
    empty_suggestion_message: str
    safe_messages: tuple[str, ...]
    active_suggestions: tuple[SuggestionCardViewModel, ...] = ()
    suggestion_history: tuple[SuggestionCardViewModel, ...] = ()
    speaker_dashboard: SpeakerDashboardViewModel | None = None

    @property
    def suggestions(self) -> tuple[SuggestionCardViewModel, ...]:
        """Compatibility alias for callers that still read active cards."""
        return self.active_suggestions


@dataclass(frozen=True, slots=True)
class TechnicalTabViewModel:
    progress: ProgressViewModel
    latency: LatencyViewModel | None
    asr_chart: tuple[tuple[int, float], ...]
    pipeline_statuses: tuple[tuple[str, str], ...]
    total_processing: str
    rtf: str
    last_asr: str
    warning: str | None
    error: str | None
    classification_metadata: tuple[tuple[str, str], ...] = ()
    probabilities: tuple[tuple[str, float], ...] = ()
    coaching_metadata: tuple[tuple[str, str], ...] = ()
    failure_details: tuple[tuple[str, str], ...] = ()
    current_labels: tuple[str, ...] = ()
    detected_labels: tuple[str, ...] = ()
    revision_label_timeline: tuple[CallRevisionLabelDiagnostic, ...] = ()
    suggestion_decisions: tuple[SafeSuggestionDecision, ...] = ()


@dataclass(frozen=True, slots=True)
class CallResultTabViewModel:
    completed: bool
    waiting_message: str
    final_transcript: str
    metrics: tuple[StatusCardViewModel, ...]
    detected_labels: tuple[IntentChipViewModel, ...]
    suggestion_timeline: tuple[TimelineItem, ...]
    suppressed_count: int
    model_name: str
    language: str
    audio_metadata: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class DashboardTabsViewModel:
    representative: RepresentativeTabViewModel
    technical: TechnicalTabViewModel
    result: CallResultTabViewModel


@dataclass(frozen=True, slots=True)
class DashboardExecutionSnapshot:
    """Latest bounded, immutable presentation state for one local call."""

    tenant_id: str
    call_id: str
    revision: int
    lifecycle_status: DashboardExecutionStatus
    execution_mode: DashboardExecutionMode
    execution_stage: DashboardExecutionStage
    processed_chunks: int
    total_chunks: int
    processed_audio_seconds: float | None
    total_audio_seconds: float | None
    transcript_revision: int
    transcript: TranscriptViewModel
    intent_risk: tuple[IntentChipViewModel, ...]
    active_coaching: tuple[SuggestionCardViewModel, ...]
    speaker_state: SpeakerDashboardViewModel | None
    latency: LatencyViewModel | None
    failure_reason: str | None
    tabs: DashboardTabsViewModel


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
    suggestion_history: list[SuggestionCardViewModel] = field(default_factory=list)
    timeline: list[TimelineItem] = field(default_factory=list)
    suppression_reasons: list[str] = field(default_factory=list)
    latency: LatencyViewModel | None = None
    detected_label_names: list[str] = field(default_factory=list)
    classification_probabilities: dict[str, float] = field(default_factory=dict)
    classification_failure: bool = False
    coaching_failure: bool = False
    setfit_enabled: bool = False
    coaching_enabled: bool = False
    rule_engine_enabled: bool = False
    service_status_message: str | None = None
    consumed_suggestion_ids: set[str] = field(default_factory=set)
    consumed_classification_event_ids: set[str] = field(default_factory=set)
    consumed_coaching_revisions: set[int] = field(default_factory=set)
    suggestion_decisions: list[SafeSuggestionDecision] = field(default_factory=list)

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
    safe_failure: "SafeRuntimeFailure | None" = None

    @property
    def real_time_factor(self) -> float | None:
        if not self.processing_seconds or not self.audio_duration_seconds:
            return None
        return self.processing_seconds / self.audio_duration_seconds

    def request_start(self) -> None:
        if self.status == "idle":
            self.start_requested = True

    @property
    def start_enabled(self) -> bool:
        return self.status == "idle"

    @property
    def stop_enabled(self) -> bool:
        return self.status == "running"


@dataclass(frozen=True, slots=True)
class SafeRuntimeFailure:
    stage: str
    error_code: str
    chunk_sequence: int | None
    transcript_revision: int | None
    asr_enabled: bool
    classification_enabled: bool
    coaching_enabled: bool
    component: str

    def log_metadata(self) -> dict[str, object]:
        return {
            "failure_stage": self.stage,
            "error_code": self.error_code,
            "chunk_sequence": self.chunk_sequence,
            "transcript_revision": self.transcript_revision,
            "asr_enabled": self.asr_enabled,
            "classification_enabled": self.classification_enabled,
            "coaching_enabled": self.coaching_enabled,
            "component": self.component,
        }


@dataclass(slots=True)
class UploadedAudioSession:
    execution: LocalExecutionState | None = None
    selected_file_identity: str | None = None
    initialized_run_file_identity: str | None = None
    tenant_id: str | None = None
    base_call_id: str | None = None
    run_sequence: int = 0
    uploader_generation: int = 0

    def select(
        self,
        *,
        identity: str,
        tenant: TenantDemo,
        base_call_id: str,
    ) -> tuple[LocalExecutionState, bool]:
        self.selected_file_identity = identity
        current_execution = self.execution
        if (
            current_execution is not None
            and identity == self.initialized_run_file_identity
            and tenant.config.context.tenant_id == self.tenant_id
            and base_call_id == self.base_call_id
        ):
            return current_execution, False
        self.run_sequence += 1
        self.initialized_run_file_identity = identity
        self.tenant_id = tenant.config.context.tenant_id
        self.base_call_id = base_call_id
        self.execution = create_local_execution(
            tenant,
            f"{base_call_id}-upload-{self.run_sequence}",
        )
        return self.execution, True

    def reset(self) -> None:
        self.execution = None
        self.selected_file_identity = None
        self.initialized_run_file_identity = None
        self.tenant_id = None
        self.base_call_id = None
        self.uploader_generation += 1


class _SafeLoggedPipelineError(RuntimeError):
    pass


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
    runtime = DashboardRuntime(
        tenant,
        scenario,
        cleaned_call_id,
        state,
        CoachingCoordinator(tenant.config, state, engine),
    )
    runtime.coaching_enabled = True
    runtime.rule_engine_enabled = True
    return runtime


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
    logger: logging.Logger | None = None,
    retain_pipeline_history: bool = True,
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
        if retain_pipeline_history:
            result = pipeline.run(
                audio_path,
                state.runtime.call_id,
                step_callback=on_step,
                plan_callback=on_plan,
            )
        else:
            result = cast(StreamingASRPipeline, pipeline).run(
                audio_path,
                state.runtime.call_id,
                step_callback=on_step,
                plan_callback=on_plan,
                retain_history=False,
            )
        _consume_pipeline_result(state, result)
    except Exception as error:
        state.status = "error"
        state.stage = "Başarısız"
        state.failed_chunk = min(state.current_chunk + 1, state.total_chunks or 1)
        state.error_message = "Ses işleme sırasında beklenmeyen bir hata oluştu."
        state.safe_failure = SafeRuntimeFailure(
            stage=(
                "finalization"
                if state.total_chunks and state.current_chunk >= state.total_chunks
                else "chunk_processing"
            ),
            error_code=type(error).__name__,
            chunk_sequence=(
                min(state.current_chunk, state.total_chunks - 1)
                if state.total_chunks
                else None
            ),
            transcript_revision=state.runtime.call_state.transcript_revision,
            asr_enabled=True,
            classification_enabled=state.runtime.setfit_enabled,
            coaching_enabled=state.runtime.coaching_enabled,
            component="streaming_asr",
        )
        safe_logger = logger or logging.getLogger(__name__)
        try:
            raise _SafeLoggedPipelineError(state.safe_failure.error_code) from None
        except _SafeLoggedPipelineError:
            safe_logger.exception(
                "uploaded audio pipeline failed",
                extra=state.safe_failure.log_metadata(),
            )
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


def suggestion_card(
    event: CoachingSuggestionEvent,
    *,
    transcript_revision: int | None = None,
) -> SuggestionCardViewModel:
    return SuggestionCardViewModel(
        event.suggestion_id,
        event.priority,
        event.priority.value,
        event.title,
        event.suggestion,
        action_display(event.action),
        event.created_at_utc.strftime("%H:%M:%S"),
        (
            intent_label(canonical)
            if event.label_id
            and (canonical := canonical_label(event.label_id)) is not None
            else None
        ),
        (),
        PRIORITY_SYMBOLS[event.priority],
        SOURCE_LABELS[event.source],
        transcript_revision,
        True,
    )


def ordered_suggestions(
    cards: list[SuggestionCardViewModel],
) -> list[SuggestionCardViewModel]:
    return sorted(
        cards,
        key=lambda card: (
            PRIORITY_RANK[card.priority],
            card.transcript_revision if card.transcript_revision is not None else -1,
        ),
        reverse=True,
    )


def suppression_reason_display(reason: str) -> str:
    return SUPPRESSION_LABELS.get(reason, reason)


def action_display(action: CoachingAction) -> str:
    return ACTION_LABELS[action]


def intent_label(label: str) -> str:
    return INTENT_LABELS.get(label, label.replace("_", " ").title())


def intent_chips(labels: tuple[LabelViewModel, ...]) -> tuple[IntentChipViewModel, ...]:
    return tuple(
        IntentChipViewModel(
            intent_label(canonical),
            label.score_percent,
            label.critical or canonical in {"cancellation_request", "churn_risk"},
            (
                "⚠"
                if label.critical or canonical in {"cancellation_request", "churn_risk"}
                else "●"
            ),
        )
        for label in labels
        if (canonical := canonical_label(label.name)) is not None
    )


def apply_feedback(
    feedback: dict[str, str], suggestion_key: str, value: str
) -> dict[str, str]:
    """Return session-only feedback without touching coaching state."""
    if value not in {"Görüldü", "Uygulandı", "Uygun değil"}:
        raise ValueError("Unknown feedback value")
    return {**feedback, suggestion_key: value}


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


def speaker_dashboard_view(
    outcome: DiarizationCompositionOutcome,
    *,
    tenant_id: str,
    call_id: str,
    transcript_revision: int,
) -> SpeakerDashboardViewModel | None:
    """Reduce a valid diarization outcome to bounded, content-free UI aggregates."""
    role_resolution = outcome.role_resolution
    projection = outcome.customer_projection
    expected_scope = (tenant_id, call_id, transcript_revision)
    if (
        outcome.status
        not in {
            DiarizationCompositionStatus.COMPLETED,
            DiarizationCompositionStatus.EMPTY,
        }
        or role_resolution is None
        or projection is None
        or (outcome.tenant_id, outcome.call_id, outcome.transcript_revision)
        != expected_scope
        or (
            role_resolution.tenant_id,
            role_resolution.call_id,
            role_resolution.transcript_revision,
        )
        != expected_scope
        or (
            projection.tenant_id,
            projection.call_id,
            projection.transcript_revision,
        )
        != expected_scope
    ):
        return None

    assignments = role_resolution.assignments
    diagnostics = role_resolution.diagnostics
    if (
        not assignments
        or len(assignments) > 2
        or len(diagnostics) != len(assignments)
        or any(
            assignment.global_speaker_id is None
            or diagnostic.final_role is not assignment.role
            for assignment, diagnostic in zip(assignments, diagnostics, strict=True)
        )
    ):
        return None

    speaker_rows = sorted(
        zip(assignments, diagnostics, strict=True),
        key=lambda row: row[0].global_speaker_id or "",
    )
    speaker_ids = tuple(assignment.global_speaker_id for assignment, _ in speaker_rows)
    if len(set(speaker_ids)) != len(speaker_ids):
        return None
    aligned_counts = dict.fromkeys(speaker_ids, 0)
    for word in outcome.diarized_words:
        if (
            word.tenant_id != tenant_id
            or word.call_id != call_id
            or word.transcript_revision != transcript_revision
        ):
            return None
        word_speakers = word.global_speaker_ids or (
            (word.global_speaker_id,) if word.global_speaker_id is not None else ()
        )
        for speaker_id in word_speakers:
            if speaker_id in aligned_counts:
                aligned_counts[speaker_id] += 1
    if any(
        turn.tenant_id != tenant_id or turn.call_id != call_id
        for turn in outcome.tracked_turns
    ) or any(
        word.tenant_id != tenant_id
        or word.call_id != call_id
        or word.transcript_revision != transcript_revision
        for word in projection.customer_words
    ):
        return None

    speakers = tuple(
        SpeakerCardViewModel(
            slot=f"SPEAKER_{index}",
            role=(
                "Rol belirleniyor"
                if assignment.role is SpeakerRole.UNKNOWN
                else assignment.role.value
            ),
            aligned_word_count=aligned_counts[assignment.global_speaker_id],
            confidence_bucket=diagnostic.confidence_bucket.value.upper(),
            decision_reason=diagnostic.final_decision_reason.value,
        )
        for index, (assignment, diagnostic) in enumerate(speaker_rows, start=1)
    )
    return SpeakerDashboardViewModel(
        speakers=speakers,
        speaker_count=len(speakers),
        turn_count=len(outcome.tracked_turns),
        projected_customer_word_count=len(projection.customer_words),
        unknown_exclusion_count=projection.excluded_unknown_word_count,
    )


def dashboard_tabs(
    runtime: DashboardRuntime,
    local_state: LocalExecutionState | None = None,
    audio_metadata: SafeUploadMetadata | None = None,
    diarization_outcome: DiarizationCompositionOutcome | None = None,
) -> DashboardTabsViewModel:
    """Build presentation-only data for the three product views."""
    local_mode = local_state is not None
    progress = (
        progress_view(local_state)
        if local_state is not None
        else _synthetic_progress(runtime)
    )
    complete = local_state.status == "completed" if local_state else runtime.complete
    call_status = (
        "Tamamlandı"
        if complete
        else (local_state.stage if local_state else "Demo hazır")
    )
    chunk_status = (
        f"{progress.completed_chunks}/{progress.total_chunks}"
        if progress.total_chunks
        else str(progress.completed_chunks)
    )
    completion_status = (
        f"%{progress.percentage:.0f}" if progress.total_chunks else "Hesaplanıyor"
    )
    status = (
        StatusCardViewModel("Tenant", runtime.tenant.config.context.tenant_name),
        StatusCardViewModel("Çağrı", runtime.call_id),
        StatusCardViewModel("Durum", call_status),
        StatusCardViewModel("Geçen süre", progress.elapsed),
        StatusCardViewModel("Parça", chunk_status),
        StatusCardViewModel("Tamamlanma", completion_status),
    )
    current_names = tuple(
        canonical_labels(
            runtime.call_state.active_labels
            or [label.name for label in runtime.latest_labels]
        )
    )
    latest_by_name = {label.name: label for label in runtime.latest_labels}
    chips = intent_chips(
        tuple(
            LabelViewModel(
                name,
                (latest_by_name[name].score_percent if name in latest_by_name else ""),
                name in {"cancellation_request", "complaint", "churn_risk"},
            )
            for name in current_names
        )
    )
    detected_names = canonical_labels(
        [item.label for item in runtime.call_state.detected_labels]
        or runtime.detected_label_names
    )
    result_chips = intent_chips(
        tuple(
            LabelViewModel(
                name,
                "%100",
                any(marker in name.casefold() for marker in CRITICAL_LABEL_MARKERS),
            )
            for name in detected_names
        )
    )
    latency = runtime.latency or _synthetic_latency(runtime)
    asr_values = local_state.asr_window_ms if local_state else []
    total_processing = (
        f"{local_state.processing_seconds:.2f} sn"
        if local_state and local_state.processing_seconds is not None
        else "—"
    )
    rtf = (
        f"{local_state.real_time_factor:.2f}x"
        if local_state and local_state.real_time_factor is not None
        else "—"
    )
    metadata = (
        (
            ("Kaynak", "Yüklenen ses"),
            ("Biçim", audio_metadata.format_name),
            ("Boyut", f"{audio_metadata.size_bytes / 1024:.1f} KB"),
        )
        if audio_metadata is not None
        else ()
    )
    audio_seconds = (
        local_state.audio_duration_seconds
        if local_state and local_state.audio_duration_seconds is not None
        else runtime.elapsed_seconds
    )
    metrics = (
        StatusCardViewModel("Ses süresi", f"{audio_seconds:.2f} sn"),
        StatusCardViewModel("Toplam parça", str(progress.total_chunks)),
        StatusCardViewModel("İşlem süresi", total_processing),
        StatusCardViewModel(
            "Ortalama ASR",
            "—" if not asr_values else f"{sum(asr_values) / len(asr_values):.0f} ms",
        ),
        StatusCardViewModel("RTF", rtf),
    )
    latest_transcript_revision = runtime.call_state.transcript_revision
    representative_suggestions = tuple(
        replace(
            card,
            is_new=(
                card.transcript_revision is not None
                and card.transcript_revision == latest_transcript_revision
            ),
        )
        for card in ordered_suggestions(runtime.suggestions)
    )
    safe_messages = tuple(
        message
        for active, message in (
            (
                runtime.classification_failure,
                "Sınıflandırma şu anda kullanılamıyor; ses akışı devam ediyor.",
            ),
            (
                runtime.coaching_failure,
                "Koçluk önerileri şu anda güncellenemiyor; ses akışı devam ediyor.",
            ),
            (
                runtime.service_status_message is not None,
                runtime.service_status_message or "",
            ),
            (
                local_state is not None and local_state.error_message is not None,
                (
                    "Ses işleme güvenli biçimde tamamlanamadı."
                    if local_state is not None
                    else ""
                ),
            ),
        )
        if active
    )
    classification_metadata = runtime.call_state.classification_metadata()
    coaching_metadata = (
        runtime.call_state.coaching_suggestions[-1]
        if runtime.call_state.coaching_suggestions
        else None
    )
    return DashboardTabsViewModel(
        representative=RepresentativeTabViewModel(
            status=status,
            progress=progress,
            transcript=transcript_view(runtime),
            intent_chips=chips,
            detected_intent_chips=result_chips,
            suppressed_count=len(runtime.suppression_reasons),
            empty_suggestion_message="Şu anda gösterilecek yeni bir koçluk önerisi yok.",
            safe_messages=safe_messages,
            active_suggestions=representative_suggestions,
            suggestion_history=tuple(ordered_suggestions(runtime.suggestion_history)),
            speaker_dashboard=(
                speaker_dashboard_view(
                    diarization_outcome,
                    tenant_id=runtime.call_state.tenant_id,
                    call_id=runtime.call_id,
                    transcript_revision=runtime.call_state.transcript_revision,
                )
                if diarization_outcome is not None
                else None
            ),
        ),
        technical=TechnicalTabViewModel(
            progress=progress,
            latency=latency,
            asr_chart=tuple(enumerate(asr_values, start=1)),
            pipeline_statuses=(
                (
                    "ASR",
                    "failed"
                    if local_state and local_state.status == "error"
                    else "active"
                    if local_mode
                    else "simulated",
                ),
                (
                    "Rule Engine",
                    "active" if runtime.rule_engine_enabled else "disabled",
                ),
                (
                    "SetFit",
                    "failed"
                    if runtime.classification_failure
                    else "active"
                    if runtime.setfit_enabled
                    else "disabled",
                ),
                (
                    "Coaching",
                    "failed"
                    if runtime.coaching_failure
                    else "active"
                    if runtime.coaching_enabled
                    else "disabled",
                ),
                ("RAG", "not implemented"),
                ("LLM", "not implemented"),
            ),
            total_processing=total_processing,
            rtf=rtf,
            last_asr="—" if not asr_values else f"{asr_values[-1]:.0f} ms",
            warning=(
                "CPU çıkarımı gerçek zamandan yavaş olabilir." if local_mode else None
            ),
            error=local_state.error_message if local_state else None,
            classification_metadata=tuple(
                (label, value)
                for label, value in (
                    ("Model", classification_metadata.model_id),
                    (
                        "Threshold profili",
                        classification_metadata.threshold_profile_id,
                    ),
                    (
                        "Transcript revision",
                        (
                            str(classification_metadata.transcript_revision)
                            if classification_metadata.transcript_revision is not None
                            else None
                        ),
                    ),
                    (
                        "Inference",
                        (
                            f"{classification_metadata.inference_time_ms:.2f} ms"
                            if classification_metadata.inference_time_ms is not None
                            else None
                        ),
                    ),
                    (
                        "Bağlam cümlesi",
                        (
                            str(classification_metadata.context_sentence_count)
                            if classification_metadata.context_sentence_count
                            is not None
                            else None
                        ),
                    ),
                    (
                        "Önceki cümle",
                        (
                            str(classification_metadata.preceding_sentence_count)
                            if classification_metadata.preceding_sentence_count
                            is not None
                            else None
                        ),
                    ),
                    (
                        "Yeni delta kelimesi",
                        (
                            str(classification_metadata.delta_word_count)
                            if classification_metadata.delta_word_count is not None
                            else None
                        ),
                    ),
                    (
                        "Delta inference",
                        "ran" if classification_metadata.delta_inference_ran else "—",
                    ),
                    (
                        "Context inference",
                        (
                            "ran"
                            if classification_metadata.context_inference_ran
                            else "—"
                        ),
                    ),
                    (
                        "Delta inference süresi",
                        (
                            f"{classification_metadata.delta_inference_time_ms:.2f} ms"
                            if classification_metadata.delta_inference_time_ms
                            is not None
                            else None
                        ),
                    ),
                    (
                        "Context inference süresi",
                        (
                            f"{classification_metadata.context_inference_time_ms:.2f} ms"
                            if classification_metadata.context_inference_time_ms
                            is not None
                            else None
                        ),
                    ),
                    (
                        "Delta etiketleri",
                        ", ".join(classification_metadata.delta_labels) or "—",
                    ),
                    (
                        "Context etiketleri",
                        ", ".join(classification_metadata.context_labels) or "—",
                    ),
                )
                if value is not None
            ),
            probabilities=(),
            coaching_metadata=(
                ()
                if coaching_metadata is None
                else (
                    ("Öneri ID", coaching_metadata.suggestion_id),
                    ("Kaynak", SOURCE_LABELS[coaching_metadata.source]),
                    ("Transcript revision", str(coaching_metadata.transcript_revision)),
                )
            ),
            failure_details=(
                ()
                if local_state is None or local_state.safe_failure is None
                else (
                    ("Aşama", local_state.safe_failure.stage),
                    ("Hata kodu", local_state.safe_failure.error_code),
                    (
                        "Parça",
                        (
                            str(local_state.safe_failure.chunk_sequence + 1)
                            if local_state.safe_failure.chunk_sequence is not None
                            else "—"
                        ),
                    ),
                    ("Bileşen", local_state.safe_failure.component),
                )
            ),
            current_labels=current_names,
            detected_labels=tuple(detected_names),
            revision_label_timeline=classification_metadata.revision_label_timeline,
            suggestion_decisions=tuple(runtime.suggestion_decisions),
        ),
        result=CallResultTabViewModel(
            completed=complete,
            waiting_message="Görüşme tamamlandığında sonuç özeti burada görünecek.",
            final_transcript=runtime.call_state.stable_transcript if complete else "",
            metrics=metrics,
            detected_labels=result_chips,
            suggestion_timeline=tuple(
                item
                for item in runtime.timeline
                if item.event_type == "Öneri gösterildi"
            ),
            suppressed_count=len(runtime.suppression_reasons),
            model_name=runtime.tenant.config.asr.model_name,
            language=runtime.tenant.config.asr.language,
            audio_metadata=metadata,
        ),
    )


def execution_snapshot(
    state: LocalExecutionState,
    *,
    revision: int,
    lifecycle_status: DashboardExecutionStatus,
    execution_mode: DashboardExecutionMode = DashboardExecutionMode.FAST_ANALYSIS,
    execution_stage: DashboardExecutionStage | None = None,
    audio_metadata: SafeUploadMetadata | None = None,
    failure_reason: str | None = None,
) -> DashboardExecutionSnapshot:
    """Freeze one bounded latest-value projection without retaining audio."""
    if revision < 0:
        raise ValueError("snapshot revision cannot be negative")
    if failure_reason not in {None, "processing_failed"}:
        raise ValueError("snapshot failure reason is invalid")
    resolved_stage = execution_stage or {
        DashboardExecutionStatus.COMPLETED: DashboardExecutionStage.COMPLETED,
        DashboardExecutionStatus.CANCELLED: DashboardExecutionStage.CANCELLED,
        DashboardExecutionStatus.FAILED: DashboardExecutionStage.FAILED,
    }.get(lifecycle_status, DashboardExecutionStage.STARTING)
    tabs = dashboard_tabs(state.runtime, state, audio_metadata)
    representative = tabs.representative
    transcript = replace(
        representative.transcript,
        stable_text=representative.transcript.stable_text[-12_000:],
        partial_text=representative.transcript.partial_text[-4_000:],
    )
    representative = replace(
        representative,
        transcript=transcript,
        intent_chips=representative.intent_chips[-32:],
        detected_intent_chips=representative.detected_intent_chips[-32:],
        active_suggestions=representative.active_suggestions[:8],
        suggestion_history=representative.suggestion_history[-8:],
    )
    technical = replace(
        tabs.technical,
        asr_chart=tabs.technical.asr_chart[-64:],
        revision_label_timeline=tabs.technical.revision_label_timeline[-32:],
        suggestion_decisions=tabs.technical.suggestion_decisions[-32:],
    )
    result = replace(
        tabs.result,
        final_transcript=tabs.result.final_transcript[-12_000:],
        suggestion_timeline=tabs.result.suggestion_timeline[-32:],
    )
    bounded_tabs = replace(
        tabs,
        representative=representative,
        technical=technical,
        result=result,
    )
    return DashboardExecutionSnapshot(
        tenant_id=state.runtime.call_state.tenant_id,
        call_id=state.runtime.call_id,
        revision=revision,
        lifecycle_status=lifecycle_status,
        execution_mode=execution_mode,
        execution_stage=resolved_stage,
        processed_chunks=state.current_chunk,
        total_chunks=state.total_chunks,
        processed_audio_seconds=(
            state.latest_step.chunk_end_seconds
            if state.latest_step is not None
            else None
        ),
        total_audio_seconds=state.audio_duration_seconds,
        transcript_revision=state.runtime.call_state.transcript_revision,
        transcript=transcript,
        intent_risk=representative.intent_chips,
        active_coaching=representative.active_suggestions,
        speaker_state=representative.speaker_dashboard,
        latency=technical.latency,
        failure_reason=failure_reason,
        tabs=bounded_tabs,
    )


def _synthetic_progress(runtime: DashboardRuntime) -> ProgressViewModel:
    total = len(runtime.scenario.events) if runtime.scenario else 0
    completed = min(runtime.next_event_index, total)
    percentage = completed / total * 100 if total else 0.0
    return ProgressViewModel(
        stage="Tamamlandı" if runtime.complete else "Sentetik oynatma",
        completed_chunks=completed,
        total_chunks=total,
        percentage=percentage,
        time_range=(
            "Henüz başlanmadı"
            if runtime.latest_event is None
            else f"Ses {runtime.latest_event.start_seconds:.2f}–{runtime.latest_event.end_seconds:.2f} sn"
        ),
        elapsed=_human_duration(runtime.elapsed_seconds),
        average_asr="Sentetik değer",
        eta="Sentetik oynatma" if completed else "Tahmin hazırlanıyor",
        failed_chunk=None,
    )


def _synthetic_latency(runtime: DashboardRuntime) -> LatencyViewModel:
    tick = max(runtime.next_event_index, 1)
    return LatencyViewModel(
        chunk_duration_ms=38 + tick,
        asr_ms=122 + tick * 3,
        rule_ms=7 + tick,
        coaching_ms=4 + tick,
        total_ms=171 + tick * 5,
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
        _apply_transcript_event(runtime, result.final_event, run_demo_coaching=False)
    for outcome in result.classification_outcomes:
        _consume_classification_outcome(runtime, outcome)
    for outcome in result.coaching_outcomes:
        _consume_coaching_outcome(runtime, outcome)
    runtime.call_state.update_active_labels(
        list(result.classification_metadata.current_revision_labels)
    )
    if result.classification_metadata.labels_detected_during_call:
        runtime.call_state.detected_labels = list(
            result.classification_metadata.labels_detected_during_call
        )
    if result.classification_metadata.revision_label_timeline:
        runtime.call_state.label_revision_timeline = list(
            result.classification_metadata.revision_label_timeline
        )
    state.total_chunks = result.total_chunks
    state.current_chunk = result.total_chunks


def _consume_step(state: LocalExecutionState, step: StreamingASRStep) -> None:
    for event in step.transcript_events:
        _apply_transcript_event(state.runtime, event, run_demo_coaching=False)
    for outcome in step.classification_outcomes:
        _consume_classification_outcome(state.runtime, outcome)
    for outcome in step.coaching_outcomes:
        _consume_coaching_outcome(state.runtime, outcome)
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
    del state.asr_window_ms[:-_MAX_LOCAL_ASR_WINDOWS]
    runtime = state.runtime
    del runtime.timeline[:-_MAX_LOCAL_TIMELINE_ITEMS]
    del runtime.suggestion_history[:-_MAX_LOCAL_SUGGESTION_HISTORY]
    del runtime.suppression_reasons[:-_MAX_LOCAL_DECISIONS]
    del runtime.suggestion_decisions[:-_MAX_LOCAL_DECISIONS]
    if len(runtime.consumed_suggestion_ids) > _MAX_LOCAL_DEDUPLICATION_KEYS:
        runtime.consumed_suggestion_ids = set(
            sorted(runtime.consumed_suggestion_ids)[-_MAX_LOCAL_DEDUPLICATION_KEYS:]
        )
    if len(runtime.consumed_classification_event_ids) > _MAX_LOCAL_DEDUPLICATION_KEYS:
        runtime.consumed_classification_event_ids = set(
            sorted(runtime.consumed_classification_event_ids)[
                -_MAX_LOCAL_DEDUPLICATION_KEYS:
            ]
        )
    if len(runtime.consumed_coaching_revisions) > _MAX_LOCAL_DEDUPLICATION_KEYS:
        runtime.consumed_coaching_revisions = set(
            sorted(runtime.consumed_coaching_revisions)[-_MAX_LOCAL_DEDUPLICATION_KEYS:]
        )


def _apply_transcript_event(
    runtime: DashboardRuntime,
    event: TranscriptEvent,
    *,
    run_demo_coaching: bool = True,
) -> None:
    runtime.latest_event = event
    runtime.call_state.apply_transcript(event)
    runtime.timeline.append(
        TimelineItem(event.created_at_utc, "Transkript", event.kind.value)
    )
    if run_demo_coaching:
        result = runtime.coordinator.process(event, event.end_seconds)
        _apply_coaching_result(runtime, result, event)
    runtime.timeline.sort(key=lambda item: item.timestamp)


def _consume_classification_outcome(
    runtime: DashboardRuntime, outcome: StableClassificationOutcome
) -> None:
    if outcome.status is ClassificationProcessingStatus.FAILED:
        runtime.classification_failure = True
        return
    classification = outcome.classification_event
    if (
        classification is None
        or classification.transcript_event_id
        in runtime.consumed_classification_event_ids
    ):
        return
    runtime.consumed_classification_event_ids.add(classification.transcript_event_id)
    runtime.classification_failure = False
    runtime.latest_action = classification.action
    current_labels = canonical_labels(label.name for label in classification.labels)
    runtime.latest_labels = tuple(
        LabelViewModel(
            label,
            "",
            label in {"cancellation_request", "complaint", "churn_risk"},
        )
        for label in current_labels
    )
    runtime.classification_probabilities = {}
    runtime.call_state.apply_classification(
        classification,
        transcript_revision=outcome.transcript_revision or 0,
        source_sequence=outcome.source_sequence,
        context_sentence_count=outcome.context_sentence_count,
        preceding_sentence_count=outcome.preceding_sentence_count,
        delta_word_count=outcome.delta_word_count,
        delta_inference_ran=outcome.delta_inference_ran,
        context_inference_ran=outcome.context_inference_ran,
        delta_inference_time_ms=outcome.delta_inference_time_ms,
        context_inference_time_ms=outcome.context_inference_time_ms,
        delta_labels=outcome.delta_labels,
        context_labels=outcome.context_labels,
        label_view_sources=dict(outcome.label_view_sources),
    )
    for label in current_labels:
        if label not in runtime.detected_label_names:
            runtime.detected_label_names.append(label)
    runtime.timeline.append(
        TimelineItem(
            classification.created_at_utc,
            "Sınıflandırma",
            ", ".join(current_labels),
        )
    )


def _consume_coaching_outcome(
    runtime: DashboardRuntime, outcome: StableCoachingOutcome
) -> None:
    if outcome.status is CoachingProcessingStatus.FAILED:
        runtime.coaching_failure = True
        return
    if (
        outcome.result is None
        or outcome.transcript_revision in runtime.consumed_coaching_revisions
    ):
        return
    runtime.consumed_coaching_revisions.add(outcome.transcript_revision)
    runtime.coaching_failure = False
    _apply_coaching_result(
        runtime,
        outcome.result,
        runtime.latest_event,
        apply_state_metadata=True,
    )


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
    runtime: DashboardRuntime,
    result: CoachingCoordinatorResult,
    event: TranscriptEvent | None,
    *,
    apply_state_metadata: bool = False,
) -> None:
    classification = result.classification_event
    replaced_ids = set(result.replaced_suggestion_ids)
    if replaced_ids:
        replaced_cards = [
            card for card in runtime.suggestions if card.suggestion_id in replaced_ids
        ]
        runtime.suggestion_history.extend(replaced_cards)
        runtime.suggestions = [
            card
            for card in runtime.suggestions
            if card.suggestion_id not in replaced_ids
        ]
    runtime.suggestion_decisions.extend(result.suggestion_decisions)
    if apply_state_metadata:
        runtime.call_state.update_active_labels(list(result.current_revision_labels))
    if classification is not None and not apply_state_metadata:
        runtime.latest_action = classification.action
        canonical_scores: dict[str, float] = {}
        for item in classification.labels:
            label = canonical_label(item.name)
            if label is not None:
                canonical_scores[label] = max(
                    item.score,
                    canonical_scores.get(label, 0.0),
                )
        current_labels = canonical_labels(label.name for label in classification.labels)
        runtime.latest_labels = tuple(
            LabelViewModel(
                label,
                f"%{canonical_scores[label] * 100:.0f}",
                label in {"cancellation_request", "complaint", "churn_risk"},
            )
            for label in current_labels
        )
        for label in current_labels:
            if label not in runtime.detected_label_names:
                runtime.detected_label_names.append(label)
        runtime.timeline.append(
            TimelineItem(
                classification.created_at_utc,
                "Sınıflandırma",
                ", ".join(current_labels),
            )
        )
    elif (
        classification is None
        and event is not None
        and event.kind is not TranscriptKind.PARTIAL
        and not apply_state_metadata
    ):
        runtime.latest_action = CoachingAction.NO_ACTION
        runtime.latest_labels = ()
    for item in result.displayed_suggestions:
        if item.suggestion_id in runtime.consumed_suggestion_ids:
            continue
        runtime.consumed_suggestion_ids.add(item.suggestion_id)
        runtime.suggestions.append(
            suggestion_card(
                item,
                transcript_revision=result.transcript_revision,
            )
        )
        if apply_state_metadata:
            runtime.call_state.mark_suggestion_shown(item.suggestion_id)
            runtime.call_state.apply_coaching_suggestion(
                item,
                transcript_revision=result.transcript_revision or 0,
                model_id=classification.model_id if classification else None,
                threshold_profile_id=(
                    classification.threshold_profile_id if classification else None
                ),
            )
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
