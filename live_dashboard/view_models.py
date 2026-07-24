"""Streamlit-independent state and formatting for the live dashboard demo."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import count
from pathlib import Path
from time import perf_counter
from typing import Protocol

from app.calls.models import CallState
from app.classification.streaming import (
    ClassificationProcessingStatus,
    StableClassificationOutcome,
)
from app.coaching.coordinator import (
    CoachingCoordinator,
    CoachingCoordinatorResult,
    CoachingProcessingStatus,
    StableCoachingOutcome,
)
from app.coaching.rule_engine import RuleBasedCoachingEngine
from app.events.models import (
    CoachingAction,
    CoachingSuggestionEvent,
    CoachingSuggestionSource,
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
from live_dashboard.uploaded_audio import SafeUploadMetadata


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
    "iptal_riski": "İptal riski",
    "ayrilma_talebi": "İptal riski",
    "kritik_eskalasyon": "Kritik risk",
    "yonetici_aktarimi": "Kritik risk",
}
SOURCE_LABELS = {
    CoachingSuggestionSource.RULE: "Kural",
    CoachingSuggestionSource.CLASSIFICATION: "Sınıflandırma",
    CoachingSuggestionSource.BOTH: "Kural + sınıflandırma",
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
    suggestions: tuple[SuggestionCardViewModel, ...]
    intent_chips: tuple[IntentChipViewModel, ...]
    suppressed_count: int
    empty_suggestion_message: str
    safe_messages: tuple[str, ...]


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
        intent_label(event.label_id) if event.label_id else None,
        tuple(event.evidence_ids),
        PRIORITY_SYMBOLS[event.priority],
        SOURCE_LABELS[event.source],
        transcript_revision,
        True,
    )


def ordered_suggestions(
    cards: list[SuggestionCardViewModel],
) -> list[SuggestionCardViewModel]:
    return sorted(cards, key=lambda card: PRIORITY_RANK[card.priority], reverse=True)


def suppression_reason_display(reason: str) -> str:
    return SUPPRESSION_LABELS.get(reason, reason)


def action_display(action: CoachingAction) -> str:
    return ACTION_LABELS[action]


def intent_label(label: str) -> str:
    return INTENT_LABELS.get(label, label.replace("_", " ").title())


def intent_chips(labels: tuple[LabelViewModel, ...]) -> tuple[IntentChipViewModel, ...]:
    return tuple(
        IntentChipViewModel(
            intent_label(label.name),
            label.score_percent,
            label.critical or "risk" in label.name.casefold(),
            "⚠" if label.critical or "risk" in label.name.casefold() else "●",
        )
        for label in labels
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


def dashboard_tabs(
    runtime: DashboardRuntime,
    local_state: LocalExecutionState | None = None,
    audio_metadata: SafeUploadMetadata | None = None,
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
    status = (
        StatusCardViewModel("Tenant", runtime.tenant.config.context.tenant_name),
        StatusCardViewModel("Çağrı", runtime.call_id),
        StatusCardViewModel("Durum", call_status),
        StatusCardViewModel("Geçen süre", progress.elapsed),
        StatusCardViewModel(
            "Parça", f"{progress.completed_chunks}/{progress.total_chunks}"
        ),
        StatusCardViewModel("Tamamlanma", f"%{progress.percentage:.0f}"),
    )
    chips = intent_chips(runtime.latest_labels)
    result_chips = intent_chips(
        tuple(
            LabelViewModel(
                name,
                "%100",
                any(marker in name.casefold() for marker in CRITICAL_LABEL_MARKERS),
            )
            for name in runtime.detected_label_names
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
            ("Dosya", audio_metadata.filename),
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
            suggestions=representative_suggestions,
            intent_chips=chips,
            suppressed_count=len(runtime.suppression_reasons),
            empty_suggestion_message="Şu anda gösterilecek yeni bir koçluk önerisi yok.",
            safe_messages=safe_messages,
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
                )
                if value is not None
            ),
            probabilities=tuple(sorted(runtime.classification_probabilities.items())),
            coaching_metadata=(
                ()
                if coaching_metadata is None
                else (
                    ("Öneri ID", coaching_metadata.suggestion_id),
                    ("Kaynak", SOURCE_LABELS[coaching_metadata.source]),
                    ("Transcript revision", str(coaching_metadata.transcript_revision)),
                )
            ),
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
    runtime.latest_labels = tuple(
        LabelViewModel(
            label.name,
            "",
            label.name in {"cancellation_request", "complaint", "churn_risk"},
        )
        for label in classification.labels
    )
    runtime.classification_probabilities = dict(classification.probabilities)
    runtime.call_state.apply_classification(
        classification,
        transcript_revision=outcome.transcript_revision or 0,
        source_sequence=outcome.source_sequence,
    )
    for label in classification.labels:
        if label.name not in runtime.detected_label_names:
            runtime.detected_label_names.append(label.name)
    runtime.timeline.append(
        TimelineItem(
            classification.created_at_utc,
            "Sınıflandırma",
            ", ".join(label.name for label in classification.labels),
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
    if classification is not None and not apply_state_metadata:
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
        for label in classification.labels:
            if label.name not in runtime.detected_label_names:
                runtime.detected_label_names.append(label.name)
        runtime.timeline.append(
            TimelineItem(
                classification.created_at_utc,
                "Sınıflandırma",
                ", ".join(label.name for label in classification.labels),
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
