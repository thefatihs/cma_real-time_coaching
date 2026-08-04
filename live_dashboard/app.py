"""Professional synthetic and opt-in local-file coaching dashboard."""

import sys
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import cast

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.asr.faster_whisper_engine import FasterWhisperEngine  # noqa: E402
from app.audio_ingress.local_microphone import (  # noqa: E402
    LocalMicTestCapability,
    LocalMicrophoneASRReadiness,
    LocalMicrophoneDiagnostics,
    LocalMicrophoneIngressSession,
    LocalMicrophoneStatus,
    LocalMicrophoneTerminalReason,
    LocalMicrophoneTranscriptRejectionReason,
    create_local_mic_test_capability,
    local_microphone_test_enabled,
)
from app.classification.runtime import RuntimeSetFitClassifier  # noqa: E402
from app.classification.streaming import ProvisionalClassificationPolicy  # noqa: E402
from app.events.models import (  # noqa: E402
    AudioChunkEvent,
    CoachingSuggestionLifecycle,
    TranscriptKind,
)
from app.streaming.pipeline import (  # noqa: E402
    StreamingASRPipeline,
    StreamingASRPlan,
    StreamingASRResult,
    StreamingASRStep,
)
from app.streaming.window_transcriber import WindowTranscriber  # noqa: E402
from app.tenancy.models import TenantASRConfig  # noqa: E402
from live_dashboard.demo_data import scenario_for, tenant_demos  # noqa: E402
from live_dashboard.runtime_wiring import (  # noqa: E402
    ArtifactAvailability,
    DashboardExecutionIdentity,
    DashboardExecutionResource,
    DashboardExecutionResourceRegistry,
    DashboardServiceSelection,
    build_live_pipeline,
    default_service_selection,
    inspect_default_artifacts,
    wait_for_live_cadence,
)
from live_dashboard.rag_runtime import DashboardRAGRuntimeController  # noqa: E402
from live_dashboard.presentation import (  # noqa: E402
    OperationalState,
    UIScopeIdentity,
    VISIBLE_ACTIVE_SUGGESTIONS,
    VISIBLE_HISTORY_SUGGESTIONS,
    VISIBLE_TECHNICAL_ROWS,
    VISIBLE_TIMELINE_ROWS,
    bounded_items,
    bounded_text_tail,
    call_status_header,
    coaching_feedback_key,
    operational_status,
    rag_runtime_status_text,
    representative_kpis,
    safe_failure_rows,
    scoped_widget_key,
    synchronize_ui_scope,
    ui_scope_identity,
)
from live_dashboard.local_microphone import (  # noqa: E402
    LOCAL_MIC_WARNING_LINES,
    local_microphone_connection_view,
    microphone_webrtc_streamer,
)
from live_dashboard.uploaded_audio import (  # noqa: E402
    SUPPORTED_UPLOAD_SUFFIXES,
    SafeUploadMetadata,
    safe_upload_metadata,
    safe_upload_identity,
    temporary_uploaded_audio,
)
from live_dashboard.view_models import (  # noqa: E402
    DashboardRuntime,
    DashboardExecutionMode,
    DashboardExecutionSnapshot,
    DashboardExecutionStage,
    DashboardExecutionStatus,
    DashboardTabsViewModel,
    LocalExecutionState,
    StatusCardViewModel,
    UploadedAudioSession,
    apply_feedback,
    advance_runtime,
    create_local_execution,
    create_runtime,
    consume_live_result,
    consume_live_step,
    dashboard_tabs,
    execute_local_once,
    execution_snapshot,
    responsive_rows,
)


st.set_page_config(page_title="Canlı Koçluk", page_icon="🎧", layout="wide")
st.html("""
<style>
div[data-testid="stMetric"] {background:#f7f8fa;border:1px solid #e6e8eb;padding:10px;border-radius:10px}
div[data-testid="stMetricValue"] {font-size:1.1rem;white-space:normal;overflow-wrap:anywhere}
div[data-testid="stVerticalBlock"] {gap:.75rem}
@media (max-width: 760px) {
  div[data-testid="stHorizontalBlock"] {flex-wrap:wrap}
  div[data-testid="column"] {min-width:100%}
}
</style>
""")

_EXECUTION_RESOURCE_CAPACITY = 8
_EXECUTION_RESOURCE_SESSION_KEY = "dashboard_execution_resource_key"
_RAG_RUNTIME_STATUS_SESSION_KEY = "dashboard_rag_runtime_status"
_RAG_RUNTIME_STATUS_NOT_AVAILABLE = object()
_LOCAL_MIC_CALL_ACTIVE_SESSION_KEY = "local_mic_call_active"
_LOCAL_MIC_FINISH_PENDING_SESSION_KEY = "local_mic_finish_pending"
_LOCAL_MIC_ASR_MODEL_NAME = "tiny"
_LOCAL_MIC_ASR_COMPUTE_TYPE = "int8"
_LOCAL_MIC_ASR_CPU_THREADS = 4
_LOCAL_MIC_ASR_BEAM_SIZE = 1
_LOCAL_MIC_ASR_ROLLING_WINDOW_SECONDS = 6.0
_LOCAL_MIC_ASR_STABLE_REGION_SECONDS = 2.0
_EXECUTION_STAGE_TEXT = {
    DashboardExecutionStage.PREPARING_MODEL: (
        "Konuşma modeli hazırlanıyor; henüz konuşmayın"
    ),
    DashboardExecutionStage.WARMING_UP: (
        "Konuşma modeli hazırlanıyor; henüz konuşmayın"
    ),
    DashboardExecutionStage.READY_TO_CAPTURE: "Mikrofon hazır; konuşabilirsiniz",
    DashboardExecutionStage.MODEL_PREPARATION_FAILED: (
        "Konuşma modeli hazırlanamadı; mikrofon testi başlatılamadı"
    ),
    DashboardExecutionStage.PERMISSION_PENDING: "Mikrofon izni bekleniyor",
    DashboardExecutionStage.MICROPHONE_READY: "Mikrofon hazır",
    DashboardExecutionStage.LIVE_AUDIO: "Canlı ses işleniyor",
    DashboardExecutionStage.TRANSCRIPT_UPDATING: "Konuşma metni güncelleniyor",
    DashboardExecutionStage.COACHING_UPDATING: "Anlık öneri hazırlanıyor",
    DashboardExecutionStage.MICROPHONE_PAUSING: "Mikrofon duraklatılıyor",
    DashboardExecutionStage.MICROPHONE_PAUSED: (
        "Mikrofon duraklatıldı. Görüşme verileri korunuyor."
    ),
    DashboardExecutionStage.MICROPHONE_RECONNECTING: (
        "Mikrofon bağlantısı yeniden kuruluyor"
    ),
    DashboardExecutionStage.MICROPHONE_CAPTURE_FAILED: (
        "Mikrofon kullanılamıyor. Yeniden başlatabilirsiniz."
    ),
    DashboardExecutionStage.STOP_REQUESTED: "Mikrofon durduruluyor",
    DashboardExecutionStage.MICROPHONE_DISCONNECTED: ("Mikrofon bağlantısı kesildi"),
    DashboardExecutionStage.MICROPHONE_OVERLOADED: ("Ses kuyruğu kapasitesi aşıldı"),
    DashboardExecutionStage.STARTING: "Analiz başlatılıyor",
    DashboardExecutionStage.FILE_PREPARING: "Ses dosyası hazırlanıyor",
    DashboardExecutionStage.ENGINE_RUNNING: "Analiz motoru çalışıyor",
    DashboardExecutionStage.CHUNK_PROCESSING: "Ses parçaları işleniyor",
    DashboardExecutionStage.COMPLETED: "Analiz tamamlandı",
    DashboardExecutionStage.CANCELLED: "Analiz durduruldu",
    DashboardExecutionStage.FAILED: "Analiz başarısız",
}
_EXECUTION_MODE_TEXT = {
    DashboardExecutionMode.FAST_ANALYSIS: "Hızlı analiz",
    DashboardExecutionMode.REALTIME_SIMULATION: "Gerçek zaman simülasyonu",
    DashboardExecutionMode.LOCAL_MIC_TEST: "Tek konuşmacılı mikrofon testi",
}


@st.cache_resource(show_spinner="ASR modeli yükleniyor…")
def _load_asr_model(
    model_name: str,
    language: str,
    beam_size: int,
    vad_filter: bool,
    condition_on_previous_text: bool,
    initial_prompt: str | None,
) -> FasterWhisperEngine:
    return FasterWhisperEngine(
        model_name,
        device="cpu",
        compute_type="int8",
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
    )


def _local_microphone_asr_config(runtime: DashboardRuntime) -> TenantASRConfig:
    """Return an explicit local-only CPU preset without mutating tenant config."""
    source = runtime.tenant.config.asr.model_dump()
    source.update(
        {
            "model_name": _LOCAL_MIC_ASR_MODEL_NAME,
            "beam_size": _LOCAL_MIC_ASR_BEAM_SIZE,
            "vad_filter": False,
            "rolling_window_seconds": _LOCAL_MIC_ASR_ROLLING_WINDOW_SECONDS,
            "chunk_duration_seconds": 2.0,
            "stable_region_seconds": _LOCAL_MIC_ASR_STABLE_REGION_SECONDS,
        }
    )
    return TenantASRConfig.model_validate(source)


def _create_local_microphone_asr_engine(
    config: TenantASRConfig,
) -> FasterWhisperEngine:
    return FasterWhisperEngine(
        config.model_name,
        device="cpu",
        compute_type=_LOCAL_MIC_ASR_COMPUTE_TYPE,
        language=config.language,
        beam_size=config.beam_size,
        cpu_threads=_LOCAL_MIC_ASR_CPU_THREADS,
        vad_filter=config.vad_filter,
        condition_on_previous_text=config.condition_on_previous_text,
        initial_prompt=config.initial_prompt,
    )


@st.cache_resource(show_spinner=False)
def _load_runtime_classifier() -> RuntimeSetFitClassifier:
    return RuntimeSetFitClassifier()


@st.cache_resource(show_spinner=False)
def _artifact_availability() -> ArtifactAvailability:
    return inspect_default_artifacts()


@st.cache_resource(show_spinner=False)
def _execution_resource_registry(
    capacity: int,
) -> DashboardExecutionResourceRegistry:
    return DashboardExecutionResourceRegistry(capacity=capacity)


@st.cache_resource(show_spinner=False)
def _rag_runtime_controller(
    capacity: int,
) -> DashboardRAGRuntimeController:
    return DashboardRAGRuntimeController(
        registry=_execution_resource_registry(capacity),
    )


def _execution_resource(runtime: DashboardRuntime) -> DashboardExecutionResource:
    controller = _rag_runtime_controller(_EXECUTION_RESOURCE_CAPACITY)
    identity = DashboardExecutionIdentity(
        runtime.tenant.config.context.tenant_id,
        runtime.call_id,
    )
    previous_key = st.session_state.get(_EXECUTION_RESOURCE_SESSION_KEY)
    if previous_key is not None and previous_key != identity.opaque_key:
        if not isinstance(previous_key, str):
            raise ValueError("dashboard execution resource key is invalid")
        controller.close_and_remove(previous_key)
    status, resource = controller.activate(
        tenant_config=runtime.tenant.config,
        call_id=identity.call_id,
    )
    st.session_state[_EXECUTION_RESOURCE_SESSION_KEY] = resource.opaque_key
    st.session_state[_RAG_RUNTIME_STATUS_SESSION_KEY] = status
    return resource


def _close_execution_resource() -> None:
    st.session_state.pop(_RAG_RUNTIME_STATUS_SESSION_KEY, None)
    key = st.session_state.pop(_EXECUTION_RESOURCE_SESSION_KEY, None)
    if key is None:
        return
    if not isinstance(key, str):
        raise ValueError("dashboard execution resource key is invalid")
    _rag_runtime_controller(_EXECUTION_RESOURCE_CAPACITY).close_and_remove(key)


def _retained_execution_resource(
    runtime: DashboardRuntime,
) -> DashboardExecutionResource | None:
    key = st.session_state.get(_EXECUTION_RESOURCE_SESSION_KEY)
    if key is None:
        return None
    if not isinstance(key, str):
        raise ValueError("dashboard execution resource key is invalid")
    return _execution_resource_registry(_EXECUTION_RESOURCE_CAPACITY).lookup(
        key,
        tenant_id=runtime.call_state.tenant_id,
        call_id=runtime.call_id,
    )


def _make_pipeline(
    runtime: DashboardRuntime,
    selection: DashboardServiceSelection,
    availability: ArtifactAvailability,
    execution_resource: DashboardExecutionResource,
) -> StreamingASRPipeline:
    config = runtime.tenant.config
    engine = _load_asr_model(
        config.asr.model_name,
        config.asr.language,
        config.asr.beam_size,
        config.asr.vad_filter,
        config.asr.condition_on_previous_text,
        config.asr.initial_prompt,
    )
    return build_live_pipeline(
        runtime,
        WindowTranscriber(engine),
        selection=selection,
        availability=availability,
        classifier_provider=_load_runtime_classifier,
        integration=execution_resource.integration,
        execution_resource=execution_resource,
    )


def _local_microphone_resource(
    runtime: DashboardRuntime,
) -> DashboardExecutionResource:
    identity = DashboardExecutionIdentity(
        runtime.tenant.config.context.tenant_id,
        runtime.call_id,
    )
    previous_key = st.session_state.get(_EXECUTION_RESOURCE_SESSION_KEY)
    if previous_key is not None and previous_key != identity.opaque_key:
        if not isinstance(previous_key, str):
            raise ValueError("dashboard execution resource key is invalid")
        _rag_runtime_controller(_EXECUTION_RESOURCE_CAPACITY).close_and_remove(
            previous_key
        )
    registry = _execution_resource_registry(_EXECUTION_RESOURCE_CAPACITY)
    resource = registry.find(
        tenant_id=identity.tenant_id,
        call_id=identity.call_id,
    )
    if resource is None:
        resource = registry.acquire(
            tenant_id=identity.tenant_id,
            call_id=identity.call_id,
            integration=None,
        )
    st.session_state[_EXECUTION_RESOURCE_SESSION_KEY] = resource.opaque_key
    return resource


def _local_microphone_session(
    runtime: DashboardRuntime,
    resource: DashboardExecutionResource,
) -> LocalMicrophoneIngressSession:
    existing = resource.microphone_session
    if existing is not None:
        return cast(LocalMicrophoneIngressSession, existing)
    session = LocalMicrophoneIngressSession(
        capability=_local_microphone_capability(runtime, resource),
        resource=resource,
    )
    resource.attach_microphone_session(session)
    return session


def _local_microphone_capability(
    runtime: DashboardRuntime,
    resource: DashboardExecutionResource,
) -> LocalMicTestCapability:
    server_address = st.get_option("server.address")
    return create_local_mic_test_capability(
        tenant_id=runtime.call_state.tenant_id,
        call_id=runtime.call_id,
        resource=resource,
        server_address=(server_address if isinstance(server_address, str) else None),
    )


def _local_microphone_transcript_rejection(
    local: LocalExecutionState,
    step: StreamingASRStep,
) -> tuple[int, LocalMicrophoneTranscriptRejectionReason]:
    expected_scope = (
        local.runtime.call_state.tenant_id,
        local.runtime.call_id,
    )
    if (step.tenant_id, step.call_id) != expected_scope:
        return (
            max(len(step.transcript_events), 1),
            LocalMicrophoneTranscriptRejectionReason.STEP_SCOPE_MISMATCH,
        )
    if any(
        (event.tenant_id, event.call_id) != expected_scope
        for event in step.transcript_events
    ):
        return (
            sum(
                (event.tenant_id, event.call_id) != expected_scope
                for event in step.transcript_events
            ),
            LocalMicrophoneTranscriptRejectionReason.EVENT_SCOPE_MISMATCH,
        )
    expected_revision = local.runtime.call_state.transcript_revision
    rejected_revisions = 0
    for event in step.transcript_events:
        if event.revision < expected_revision:
            rejected_revisions += 1
        expected_revision = max(expected_revision, event.revision)
    if rejected_revisions:
        return (
            rejected_revisions,
            LocalMicrophoneTranscriptRejectionReason.REVISION_REGRESSION,
        )
    return (0, LocalMicrophoneTranscriptRejectionReason.NONE)


def _local_microphone_final_rejection(
    local: LocalExecutionState,
    result: StreamingASRResult,
) -> LocalMicrophoneTranscriptRejectionReason:
    expected_scope = (
        local.runtime.call_state.tenant_id,
        local.runtime.call_id,
    )
    if (result.tenant_id, result.call_id) != expected_scope:
        return LocalMicrophoneTranscriptRejectionReason.RESULT_SCOPE_MISMATCH
    final_event = result.final_event
    if final_event is None:
        return LocalMicrophoneTranscriptRejectionReason.NONE
    if (final_event.tenant_id, final_event.call_id) != expected_scope:
        return LocalMicrophoneTranscriptRejectionReason.EVENT_SCOPE_MISMATCH
    if final_event.revision < local.runtime.call_state.transcript_revision:
        return LocalMicrophoneTranscriptRejectionReason.REVISION_REGRESSION
    return LocalMicrophoneTranscriptRejectionReason.NONE


def _start_local_microphone_worker(
    *,
    local: LocalExecutionState,
    resource: DashboardExecutionResource,
    session: LocalMicrophoneIngressSession,
    selection: DashboardServiceSelection,
    availability: ArtifactAvailability,
) -> bool:
    if resource.worker_active or resource.latest_snapshot is not None:
        return False
    local.request_start()
    local.start_requested = False
    local.status = "running"
    local.pipeline_calls += 1
    local.stage = _EXECUTION_STAGE_TEXT[DashboardExecutionStage.PREPARING_MODEL]
    initial = execution_snapshot(
        local,
        revision=0,
        lifecycle_status=DashboardExecutionStatus.RUNNING,
        execution_mode=DashboardExecutionMode.LOCAL_MIC_TEST,
        execution_stage=DashboardExecutionStage.PREPARING_MODEL,
    )

    def run_microphone(
        cancellation: Event,
        publish: Callable[[DashboardExecutionSnapshot], None],
    ) -> DashboardExecutionSnapshot:
        revision = 0
        started_at = perf_counter()
        try:
            asr_config = _local_microphone_asr_config(local.runtime)
            construction_started = perf_counter()
            engine = _create_local_microphone_asr_engine(asr_config)
            session.record_asr_preparation(
                resource=resource,
                engine_construction_seconds=max(
                    perf_counter() - construction_started,
                    0.0,
                ),
            )
            model_loading_seconds = engine.load_model()
            session.record_asr_preparation(
                resource=resource,
                model_loading_seconds=model_loading_seconds,
            )
            session.set_asr_readiness(
                LocalMicrophoneASRReadiness.WARMING_UP,
                resource=resource,
            )
            revision += 1
            local.stage = _EXECUTION_STAGE_TEXT[DashboardExecutionStage.WARMING_UP]
            publish(
                execution_snapshot(
                    local,
                    revision=revision,
                    lifecycle_status=DashboardExecutionStatus.RUNNING,
                    execution_mode=DashboardExecutionMode.LOCAL_MIC_TEST,
                    execution_stage=DashboardExecutionStage.WARMING_UP,
                )
            )
            warmup_seconds = engine.warm_up()
            session.record_asr_preparation(
                resource=resource,
                warmup_seconds=warmup_seconds,
            )
            pipeline = build_live_pipeline(
                local.runtime,
                WindowTranscriber(engine),
                selection=selection,
                availability=availability,
                classifier_provider=_load_runtime_classifier,
                integration=resource.integration,
                execution_resource=resource,
                asr_config=asr_config,
            )
            pipeline.configure_provisional_coaching(
                ProvisionalClassificationPolicy(enabled=True)
            )
            session.set_asr_readiness(
                LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
                resource=resource,
            )
            revision += 1
            local.stage = _EXECUTION_STAGE_TEXT[
                DashboardExecutionStage.READY_TO_CAPTURE
            ]
            publish(
                execution_snapshot(
                    local,
                    revision=revision,
                    lifecycle_status=DashboardExecutionStatus.RUNNING,
                    execution_mode=DashboardExecutionMode.LOCAL_MIC_TEST,
                    execution_stage=DashboardExecutionStage.READY_TO_CAPTURE,
                )
            )

            def show_step(step: StreamingASRStep) -> None:
                nonlocal revision
                if cancellation.is_set():
                    raise RuntimeError("dashboard_execution_cancelled")
                rejected_count, rejection_reason = (
                    _local_microphone_transcript_rejection(local, step)
                )
                partial_count = sum(
                    event.kind is TranscriptKind.PARTIAL
                    for event in step.transcript_events
                )
                stable_count = sum(
                    event.kind in {TranscriptKind.STABLE, TranscriptKind.FINAL}
                    for event in step.transcript_events
                )
                if rejected_count:
                    session.record_transcript_step(
                        resource=resource,
                        asr_result_non_empty=bool(step.raw_window_text.strip()),
                        asr_segment_count=step.asr_segment_count,
                        window_duration_seconds=step.window_duration_seconds,
                        partial_event_count=0,
                        stable_commit_count=0,
                        rejected_event_count=rejected_count,
                        rejection_reason=rejection_reason,
                    )
                    raise ValueError("local_microphone_transcript_event_rejected")
                consume_live_step(
                    local,
                    step,
                    elapsed_seconds=perf_counter() - started_at,
                )
                session.record_transcript_step(
                    resource=resource,
                    asr_result_non_empty=bool(step.raw_window_text.strip()),
                    asr_segment_count=step.asr_segment_count,
                    window_duration_seconds=step.window_duration_seconds,
                    partial_event_count=partial_count,
                    stable_commit_count=stable_count,
                )
                session.record_asr_inference(
                    resource=resource,
                    audio_preparation_seconds=(step.audio_preparation_time_seconds),
                    inference_seconds=step.transcription_time_seconds,
                )
                session.acknowledge_processed_chunk(resource=resource)
                revision += 1
                stage = (
                    DashboardExecutionStage.COACHING_UPDATING
                    if step.coaching_outcomes
                    else DashboardExecutionStage.TRANSCRIPT_UPDATING
                )
                local.stage = _EXECUTION_STAGE_TEXT[stage]
                publish(
                    execution_snapshot(
                        local,
                        revision=revision,
                        lifecycle_status=DashboardExecutionStatus.RUNNING,
                        execution_mode=DashboardExecutionMode.LOCAL_MIC_TEST,
                        execution_stage=stage,
                    )
                )

            streaming_stage_published = False

            def live_chunks() -> Iterable[AudioChunkEvent]:
                nonlocal revision, streaming_stage_published
                for chunk in session.iter_audio_chunks(cancellation=cancellation):
                    if not streaming_stage_published:
                        streaming_stage_published = True
                        revision += 1
                        local.stage = _EXECUTION_STAGE_TEXT[
                            DashboardExecutionStage.LIVE_AUDIO
                        ]
                        publish(
                            execution_snapshot(
                                local,
                                revision=revision,
                                lifecycle_status=(DashboardExecutionStatus.RUNNING),
                                execution_mode=(DashboardExecutionMode.LOCAL_MIC_TEST),
                                execution_stage=(DashboardExecutionStage.LIVE_AUDIO),
                            )
                        )
                    yield chunk

            result = pipeline.run_live(
                live_chunks(),
                local.runtime.call_id,
                capability=session.capability,
                execution_resource=resource,
                cancellation=cancellation,
                step_callback=show_step,
                retain_history=False,
            )
            final_rejection = _local_microphone_final_rejection(local, result)
            if final_rejection is not LocalMicrophoneTranscriptRejectionReason.NONE:
                session.record_final_transcript_event(
                    resource=resource,
                    accepted=False,
                    rejection_reason=final_rejection,
                )
                raise ValueError("local_microphone_transcript_event_rejected")
            consume_live_result(local, result)
            session.record_final_transcript_event(
                resource=resource,
                accepted=result.final_event is not None,
            )
            local.processing_seconds = max(perf_counter() - started_at, 0.0)
            local.audio_duration_seconds = result.audio_duration_seconds
            session_status = session.diagnostics.status
            if cancellation.is_set():
                lifecycle_status = DashboardExecutionStatus.CANCELLED
                execution_stage = DashboardExecutionStage.CANCELLED
                local.status = "cancelled"
                local.stage = "Mikrofon durduruldu"
            elif session_status is LocalMicrophoneStatus.DISCONNECTED:
                lifecycle_status = DashboardExecutionStatus.CANCELLED
                execution_stage = DashboardExecutionStage.MICROPHONE_DISCONNECTED
                local.status = "cancelled"
                local.stage = _EXECUTION_STAGE_TEXT[execution_stage]
            elif session_status is LocalMicrophoneStatus.OVERLOADED:
                lifecycle_status = DashboardExecutionStatus.FAILED
                execution_stage = DashboardExecutionStage.MICROPHONE_OVERLOADED
                local.status = "error"
                local.stage = _EXECUTION_STAGE_TEXT[execution_stage]
            elif session_status is LocalMicrophoneStatus.FAILED:
                lifecycle_status = DashboardExecutionStatus.FAILED
                execution_stage = DashboardExecutionStage.FAILED
                local.status = "error"
                local.stage = "Mikrofon testi başarısız"
            else:
                lifecycle_status = DashboardExecutionStatus.COMPLETED
                execution_stage = DashboardExecutionStage.COMPLETED
                local.status = "completed"
                local.stage = "Mikrofon durduruldu"
            revision += 1
            return execution_snapshot(
                local,
                revision=revision,
                lifecycle_status=lifecycle_status,
                execution_mode=DashboardExecutionMode.LOCAL_MIC_TEST,
                execution_stage=execution_stage,
            )
        except Exception:
            cancelled = cancellation.is_set()
            preparation_failed = session.diagnostics.asr_readiness in {
                LocalMicrophoneASRReadiness.PREPARING_MODEL,
                LocalMicrophoneASRReadiness.WARMING_UP,
            }
            if cancelled:
                session.close(LocalMicrophoneTerminalReason.RESOURCE_CLOSED)
            else:
                session.fail()
            local.status = "cancelled" if cancelled else "error"
            local.stage = (
                "Mikrofon durduruldu"
                if cancelled
                else _EXECUTION_STAGE_TEXT[
                    (
                        DashboardExecutionStage.MODEL_PREPARATION_FAILED
                        if preparation_failed
                        else DashboardExecutionStage.FAILED
                    )
                ]
            )
            revision += 1
            return execution_snapshot(
                local,
                revision=revision,
                lifecycle_status=(
                    DashboardExecutionStatus.CANCELLED
                    if cancelled
                    else DashboardExecutionStatus.FAILED
                ),
                execution_mode=DashboardExecutionMode.LOCAL_MIC_TEST,
                execution_stage=(
                    DashboardExecutionStage.CANCELLED
                    if cancelled
                    else (
                        DashboardExecutionStage.MODEL_PREPARATION_FAILED
                        if preparation_failed
                        else DashboardExecutionStage.FAILED
                    )
                ),
                failure_reason=None if cancelled else "processing_failed",
            )

    return resource.start_worker(initial, run_microphone)


def _request_local_microphone_finish(
    session: LocalMicrophoneIngressSession,
    resource: DashboardExecutionResource,
) -> bool:
    if st.session_state.get(_LOCAL_MIC_FINISH_PENDING_SESSION_KEY, False):
        return False
    requested = session.finish_call(resource=resource)
    if requested or session.diagnostics.status in {
        LocalMicrophoneStatus.STOP_REQUESTED,
        LocalMicrophoneStatus.COMPLETED,
    }:
        st.session_state[_LOCAL_MIC_FINISH_PENDING_SESSION_KEY] = True
        st.session_state.local_mic_desired_playing = False
        return True
    return False


def _request_local_microphone_reset(
    session: LocalMicrophoneIngressSession,
    resource: DashboardExecutionResource,
) -> bool:
    if st.session_state.get("local_mic_reset_pending", False):
        return False
    session.close(LocalMicrophoneTerminalReason.RESET)
    resource.cancel()
    st.session_state.local_mic_desired_playing = False
    st.session_state[_LOCAL_MIC_CALL_ACTIVE_SESSION_KEY] = False
    st.session_state.pop(_LOCAL_MIC_FINISH_PENDING_SESSION_KEY, None)
    st.session_state.local_mic_reset_pending = True
    return True


def _local_microphone_audio_diagnostic_cards(
    diagnostics: LocalMicrophoneDiagnostics,
) -> tuple[StatusCardViewModel, ...]:
    input_format = "Henüz bilinmiyor"
    if (
        diagnostics.input_sample_rate_hz is not None
        and diagnostics.input_channel_count is not None
    ):
        input_format = (
            f"{diagnostics.input_sample_rate_hz} Hz · "
            f"{diagnostics.input_channel_count} kanal"
        )
    return (
        StatusCardViewModel(
            "Mikrofon karesi",
            str(diagnostics.callback_frame_count),
        ),
        StatusCardViewModel("Girdi biçimi", input_format),
        StatusCardViewModel("Girdi örneği", str(diagnostics.input_sample_count)),
        StatusCardViewModel(
            "Dönüşüm öncesi RMS",
            f"{diagnostics.pre_resample_rms:.6f}",
        ),
        StatusCardViewModel(
            "Dönüşüm öncesi tepe",
            f"{diagnostics.pre_resample_peak:.6f}",
        ),
        StatusCardViewModel(
            "Dönüşüm sonrası RMS",
            f"{diagnostics.post_resample_rms:.6f}",
        ),
        StatusCardViewModel(
            "Dönüşüm sonrası tepe",
            f"{diagnostics.post_resample_peak:.6f}",
        ),
        StatusCardViewModel(
            "Sıfır olmayan örnek",
            f"%{diagnostics.post_resample_nonzero_ratio * 100:.2f}",
        ),
        StatusCardViewModel(
            "Kırpılan örnek",
            f"%{diagnostics.post_resample_clipping_ratio * 100:.2f}",
        ),
        StatusCardViewModel(
            "Üretilen PCM",
            f"{diagnostics.produced_pcm_byte_count} bayt",
        ),
        StatusCardViewModel(
            "ASR segmenti",
            str(diagnostics.asr_segment_count),
        ),
        StatusCardViewModel(
            "Son ASR penceresi",
            f"{diagnostics.latest_window_duration_seconds:.2f} sn",
        ),
        StatusCardViewModel(
            "ASR öncesi ret",
            str(diagnostics.rejected_capture_frame_count),
        ),
    )


def _local_microphone_audio_diagnostic_message(
    diagnostics: LocalMicrophoneDiagnostics,
) -> str | None:
    if diagnostics.rejected_capture_frame_count:
        return (
            "Ses ASR öncesinde güvenli biçimde reddedildi: "
            f"{diagnostics.latest_audio_rejection_reason.value}"
        )
    if diagnostics.rejected_transcript_event_count:
        return (
            "Metin olayı ASR sonrasında güvenli biçimde reddedildi: "
            f"{diagnostics.latest_transcript_rejection_reason.value}"
        )
    if diagnostics.callback_frame_count == 0:
        return "Henüz mikrofon karesi alınmadı."
    if diagnostics.post_resample_rms <= 0.0001:
        return "Mikrofon kareleri alındı ancak dönüştürülen ses etkili olarak sessiz."
    if (
        diagnostics.asr_empty_result_count
        and diagnostics.asr_non_empty_result_count == 0
    ):
        return "Sessiz olmayan PCM ASR'ye ulaştı ancak model metin üretmedi."
    if diagnostics.asr_non_empty_result_count:
        return "Sessiz olmayan PCM ASR tarafından metne dönüştürülüyor."
    return None


def _render_local_microphone_controls(
    *,
    local: LocalExecutionState,
    selection: DashboardServiceSelection,
    availability: ArtifactAvailability,
) -> tuple[DashboardExecutionResource, LocalMicrophoneIngressSession] | None:
    """Render the retained WebRTC component in the same fragment as polling."""
    try:
        if st.session_state.get("local_mic_reset_pending", False):
            retained = _retained_execution_resource(local.runtime)
            if retained is not None and retained.worker_active:
                st.info("Sistem sıfırlanıyor")
                microphone_session = retained.microphone_session
                return (
                    (retained, cast(LocalMicrophoneIngressSession, microphone_session))
                    if microphone_session is not None
                    else None
                )
            if retained is not None:
                _close_execution_resource()
            st.session_state.pop("local_signature", None)
            st.session_state.pop("call_id", None)
            st.session_state.pop("local_mic_desired_playing", None)
            st.session_state.pop("local_mic_reset_pending", None)
            st.session_state.pop(_LOCAL_MIC_FINISH_PENDING_SESSION_KEY, None)
            st.session_state[_LOCAL_MIC_CALL_ACTIVE_SESSION_KEY] = False
            st.session_state.suggestion_feedback = {}
            st.rerun()
            return None

        call_active = st.session_state.get(_LOCAL_MIC_CALL_ACTIVE_SESSION_KEY)
        if call_active is None:
            call_active = _retained_execution_resource(local.runtime) is not None
            st.session_state[_LOCAL_MIC_CALL_ACTIVE_SESSION_KEY] = call_active
        if call_active is not True:
            st.info("Yerel mikrofon testi başlatılmayı bekliyor")
            if st.button(
                "Mikrofon testini başlat",
                type="primary",
                use_container_width=True,
            ):
                st.session_state[_LOCAL_MIC_CALL_ACTIVE_SESSION_KEY] = True
                st.session_state.pop(_LOCAL_MIC_FINISH_PENDING_SESSION_KEY, None)
                st.session_state.pop("local_mic_desired_playing", None)
                st.rerun(scope="fragment")
            return None

        resource = _local_microphone_resource(local.runtime)
        session = _local_microphone_session(local.runtime, resource)
        if resource.latest_snapshot is None:
            _start_local_microphone_worker(
                local=local,
                resource=resource,
                session=session,
                selection=selection,
                availability=availability,
            )
        diagnostics = session.diagnostics
        snapshot = resource.latest_snapshot
        terminal_snapshot = snapshot is not None and snapshot.lifecycle_status in {
            DashboardExecutionStatus.COMPLETED,
            DashboardExecutionStatus.CANCELLED,
            DashboardExecutionStatus.FAILED,
        }
        finish_pending = st.session_state.get(
            _LOCAL_MIC_FINISH_PENDING_SESSION_KEY,
            False,
        )
        if (
            finish_pending
            or diagnostics.status
            in {
                LocalMicrophoneStatus.STOP_REQUESTED,
                LocalMicrophoneStatus.COMPLETED,
            }
            or terminal_snapshot
        ):
            st.session_state.local_mic_desired_playing = False
            if terminal_snapshot:
                assert snapshot is not None
                st.session_state.pop(_LOCAL_MIC_FINISH_PENDING_SESSION_KEY, None)
                if snapshot.lifecycle_status is DashboardExecutionStatus.COMPLETED:
                    st.success("Görüşme tamamlandı")
                elif snapshot.lifecycle_status is DashboardExecutionStatus.FAILED:
                    st.error("Görüşme güvenli biçimde tamamlanamadı")
                else:
                    st.warning("Görüşme durduruldu")
            else:
                st.info("Görüşme tamamlanıyor")
            if st.button("Sistemi sıfırla", use_container_width=True):
                if _request_local_microphone_reset(session, resource):
                    st.rerun(scope="fragment")
            return resource, session
        if diagnostics.asr_readiness not in {
            LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
            LocalMicrophoneASRReadiness.STREAMING,
        }:
            if diagnostics.asr_readiness is LocalMicrophoneASRReadiness.FAILED:
                st.error(
                    _EXECUTION_STAGE_TEXT[
                        DashboardExecutionStage.MODEL_PREPARATION_FAILED
                    ]
                )
            else:
                st.info(
                    _EXECUTION_STAGE_TEXT[
                        (
                            DashboardExecutionStage.WARMING_UP
                            if diagnostics.asr_readiness
                            is LocalMicrophoneASRReadiness.WARMING_UP
                            else DashboardExecutionStage.PREPARING_MODEL
                        )
                    ]
                )
            timings = diagnostics.asr_timings
            preparation_cards = [
                StatusCardViewModel(
                    "Yerel mikrofon modeli",
                    f"{_LOCAL_MIC_ASR_MODEL_NAME} · CPU {_LOCAL_MIC_ASR_COMPUTE_TYPE}",
                )
            ]
            if timings.model_loading_seconds is not None:
                preparation_cards.append(
                    StatusCardViewModel(
                        "Model yükleme",
                        f"{timings.model_loading_seconds:.2f} sn",
                    )
                )
            if timings.warmup_seconds is not None:
                preparation_cards.append(
                    StatusCardViewModel(
                        "Model ısınma",
                        f"{timings.warmup_seconds:.2f} sn",
                    )
                )
            _metric_rows(tuple(preparation_cards))
            if st.button(
                "Sistemi sıfırla",
                use_container_width=True,
            ):
                if _request_local_microphone_reset(session, resource):
                    st.rerun(scope="fragment")
            return resource, session
        resumable_statuses = {
            LocalMicrophoneStatus.PAUSING,
            LocalMicrophoneStatus.PAUSED,
            LocalMicrophoneStatus.PERMISSION_DENIED,
            LocalMicrophoneStatus.DISCONNECTED,
            LocalMicrophoneStatus.FAILED,
        }
        if diagnostics.status in resumable_statuses:
            connection = local_microphone_connection_view(session)
            if diagnostics.status is LocalMicrophoneStatus.FAILED:
                st.error(connection.status_text)
            elif diagnostics.status in {
                LocalMicrophoneStatus.PERMISSION_DENIED,
                LocalMicrophoneStatus.DISCONNECTED,
            }:
                st.warning(connection.status_text)
            else:
                st.info(connection.status_text)
            _metric_rows(
                (
                    StatusCardViewModel(
                        "Alınan ses parçası",
                        str(connection.received_chunk_count),
                    ),
                    StatusCardViewModel(
                        "İşlenen ses",
                        f"{connection.processed_audio_seconds:.2f} sn",
                    ),
                    StatusCardViewModel(
                        "Yakalama oturumu",
                        str(diagnostics.capture_generation),
                    ),
                    StatusCardViewModel(
                        "Metin üreten ASR",
                        str(diagnostics.asr_non_empty_result_count),
                    ),
                    StatusCardViewModel(
                        "Boş ASR sonucu",
                        str(diagnostics.asr_empty_result_count),
                    ),
                    StatusCardViewModel(
                        "Reddedilen metin olayı",
                        str(diagnostics.rejected_transcript_event_count),
                    ),
                )
            )
            _metric_rows(_local_microphone_audio_diagnostic_cards(diagnostics))
            audio_message = _local_microphone_audio_diagnostic_message(diagnostics)
            if audio_message is not None:
                st.caption(audio_message)
            if diagnostics.rejected_transcript_event_count:
                st.caption(
                    "Son güvenli ret nedeni: "
                    f"{diagnostics.latest_transcript_rejection_reason.value}"
                )
            if st.button(
                "Mikrofonu yeniden başlat",
                disabled=diagnostics.status is LocalMicrophoneStatus.PAUSING,
                use_container_width=True,
            ):
                session.resume_capture(
                    _local_microphone_capability(local.runtime, resource),
                    resource=resource,
                )
                st.session_state.local_mic_desired_playing = True
                st.rerun(scope="fragment")
            if st.button("Görüşmeyi bitir", use_container_width=True):
                if _request_local_microphone_finish(session, resource):
                    st.rerun(scope="fragment")
            if st.button("Sistemi sıfırla", use_container_width=True):
                if _request_local_microphone_reset(session, resource):
                    st.rerun(scope="fragment")
            return resource, session
        desired_playing_state = st.session_state.get("local_mic_desired_playing")
        if desired_playing_state is None and diagnostics.status in {
            LocalMicrophoneStatus.READY,
            LocalMicrophoneStatus.STREAMING,
            LocalMicrophoneStatus.RECONNECTING,
        }:
            desired_playing_state = True
        microphone_webrtc_streamer(
            session=session,
            key=session.component_key,
            desired_playing_state=desired_playing_state,
        )
        connection = local_microphone_connection_view(session)
        st.info(connection.status_text)
        diagnostic_cards: list[StatusCardViewModel] = [
            StatusCardViewModel(
                "Alınan ses parçası",
                str(connection.received_chunk_count),
            ),
            StatusCardViewModel(
                "İşlenen ses",
                f"{connection.processed_audio_seconds:.2f} sn",
            ),
            StatusCardViewModel("Bağlantı", connection.status_text),
            StatusCardViewModel(
                "Metin üreten ASR",
                str(diagnostics.asr_non_empty_result_count),
            ),
            StatusCardViewModel(
                "Boş ASR sonucu",
                str(diagnostics.asr_empty_result_count),
            ),
            StatusCardViewModel(
                "Kısmi metin olayı",
                str(diagnostics.partial_event_count),
            ),
            StatusCardViewModel(
                "Kalıcı metin kaydı",
                str(diagnostics.stable_commit_count),
            ),
            StatusCardViewModel(
                "Reddedilen metin olayı",
                str(diagnostics.rejected_transcript_event_count),
            ),
        ]
        if connection.estimated_latency_seconds is not None:
            diagnostic_cards.append(
                StatusCardViewModel(
                    "Tahmini gecikme",
                    f"{connection.estimated_latency_seconds * 1000:.0f} ms",
                )
            )
        timings = diagnostics.asr_timings
        for label, value in (
            ("Model yükleme", timings.model_loading_seconds),
            ("Model ısınma", timings.warmup_seconds),
            ("İlk ses hazırlama", timings.first_audio_preparation_seconds),
            ("İlk ASR çıkarımı", timings.first_inference_seconds),
        ):
            if value is not None:
                diagnostic_cards.append(StatusCardViewModel(label, f"{value:.2f} sn"))
        _metric_rows(tuple(diagnostic_cards))
        _metric_rows(_local_microphone_audio_diagnostic_cards(diagnostics))
        audio_message = _local_microphone_audio_diagnostic_message(diagnostics)
        if audio_message is not None:
            st.caption(audio_message)
        if diagnostics.rejected_transcript_event_count:
            st.caption(
                "Son güvenli ret nedeni: "
                f"{diagnostics.latest_transcript_rejection_reason.value}"
            )
        if st.button("Mikrofonu duraklat", use_container_width=True):
            session.pause_capture(resource=resource)
            st.session_state.local_mic_desired_playing = False
            st.rerun(scope="fragment")
        if st.button("Görüşmeyi bitir", use_container_width=True):
            if _request_local_microphone_finish(session, resource):
                st.rerun(scope="fragment")
        if st.button("Sistemi sıfırla", use_container_width=True):
            if _request_local_microphone_reset(session, resource):
                st.rerun(scope="fragment")
        return resource, session
    except Exception:
        local.status = "error"
        local.stage = "Mikrofon testi başarısız"
        local.error_message = "Mikrofon testi güvenli biçimde başlatılamadı."
        st.error(local.error_message)
        return None


def _local_microphone_presentation_snapshot(
    snapshot: DashboardExecutionSnapshot,
    status: LocalMicrophoneStatus,
) -> DashboardExecutionSnapshot:
    """Project current terminal intent without mutating the worker mailbox."""
    if snapshot.lifecycle_status is not DashboardExecutionStatus.RUNNING:
        return snapshot
    if status is LocalMicrophoneStatus.STOP_REQUESTED:
        return replace(
            snapshot,
            execution_stage=DashboardExecutionStage.STOP_REQUESTED,
        )
    if status is LocalMicrophoneStatus.PAUSING:
        return replace(
            snapshot,
            execution_stage=DashboardExecutionStage.MICROPHONE_PAUSING,
        )
    if status is LocalMicrophoneStatus.PAUSED:
        return replace(
            snapshot,
            execution_stage=DashboardExecutionStage.MICROPHONE_PAUSED,
        )
    if status is LocalMicrophoneStatus.RECONNECTING:
        return replace(
            snapshot,
            execution_stage=DashboardExecutionStage.MICROPHONE_RECONNECTING,
        )
    if status is LocalMicrophoneStatus.DISCONNECTED:
        return replace(
            snapshot,
            execution_stage=DashboardExecutionStage.MICROPHONE_CAPTURE_FAILED,
        )
    if status is LocalMicrophoneStatus.OVERLOADED:
        return replace(
            snapshot,
            lifecycle_status=DashboardExecutionStatus.FAILED,
            execution_stage=DashboardExecutionStage.MICROPHONE_OVERLOADED,
        )
    if status is LocalMicrophoneStatus.FAILED:
        return replace(
            snapshot,
            execution_stage=DashboardExecutionStage.MICROPHONE_CAPTURE_FAILED,
        )
    if status is LocalMicrophoneStatus.PERMISSION_DENIED:
        return replace(
            snapshot,
            execution_stage=DashboardExecutionStage.MICROPHONE_CAPTURE_FAILED,
        )
    return snapshot


def _synthetic_runtime() -> DashboardRuntime:
    demos = tenant_demos()
    signature = (
        st.session_state.tenant_id,
        st.session_state.scenario_id,
        st.session_state.call_id,
    )
    if st.session_state.get("synthetic_signature") != signature:
        tenant_id, scenario_id, call_id = signature
        st.session_state.synthetic_runtime = create_runtime(
            demos[tenant_id], scenario_for(tenant_id, scenario_id), call_id
        )
        st.session_state.synthetic_signature = signature
        st.session_state.playing = False
    return st.session_state.synthetic_runtime


def _local_state() -> LocalExecutionState:
    signature = (st.session_state.tenant_id, st.session_state.call_id)
    if st.session_state.get("local_signature") != signature:
        st.session_state.local_execution = create_local_execution(
            tenant_demos()[signature[0]], signature[1]
        )
        st.session_state.local_signature = signature
    return st.session_state.local_execution


def _metric_rows(items: tuple[object, ...]) -> None:
    for row in responsive_rows(items, 3):
        columns = st.columns(len(row))
        for column, item in zip(columns, row, strict=True):
            label = str(getattr(item, "label"))
            value = str(getattr(item, "value"))
            column.metric(label, value)


def _render_speaker_dashboard(view: DashboardTabsViewModel) -> None:
    speaker_dashboard = view.representative.speaker_dashboard
    if speaker_dashboard is None:
        return
    st.subheader("Konuşmacı Görünümü")
    _metric_rows(
        (
            StatusCardViewModel(
                "Konuşmacı",
                str(speaker_dashboard.speaker_count),
            ),
            StatusCardViewModel(
                "Konuşma turu",
                str(speaker_dashboard.turn_count),
            ),
            StatusCardViewModel(
                "Müşteri kelimesi",
                str(speaker_dashboard.projected_customer_word_count),
            ),
            StatusCardViewModel(
                "UNKNOWN dışlama",
                str(speaker_dashboard.unknown_exclusion_count),
            ),
        )
    )
    for row in responsive_rows(speaker_dashboard.speakers, 2):
        columns = st.columns(len(row))
        for column, speaker in zip(columns, row, strict=True):
            with column:
                with st.container(border=True):
                    st.caption(speaker.slot)
                    st.write(speaker.role)
                    st.caption(
                        f"Hizalanan kelime: {speaker.aligned_word_count} · "
                        f"Güven: {speaker.confidence_bucket} · "
                        f"Karar: {speaker.decision_reason}"
                    )


def _render_representative(
    runtime: DashboardRuntime,
    view: DashboardTabsViewModel,
    scope: UIScopeIdentity,
) -> None:
    _render_representative_view(
        call_id=runtime.call_id,
        transcript_revision=runtime.call_state.transcript_revision,
        view=view,
        scope=scope,
    )


def _render_representative_view(
    *,
    call_id: str,
    transcript_revision: int,
    view: DashboardTabsViewModel,
    scope: UIScopeIdentity,
) -> None:
    representative = view.representative
    operation = operational_status(view)
    with st.container(border=True):
        header = call_status_header(
            view,
            call_id=call_id,
            transcript_revision=transcript_revision,
        )
        st.caption("CANLI GÖRÜŞME")
        header_columns = st.columns(5)
        for column, (label, value) in zip(
            header_columns,
            (
                ("Durum", header.state),
                ("Çağrı", header.masked_call_id),
                ("İlerleme", header.progress),
                ("Revizyon", header.transcript_revision),
                ("Risk durumu", header.current_risk),
            ),
            strict=True,
        ):
            column.metric(label, value)
        status_message = f"{operation.label}: {operation.detail}"
        if operation.state is OperationalState.FAILED:
            st.error(status_message)
        elif operation.state is OperationalState.DEGRADED:
            st.warning(status_message)
        elif operation.state is OperationalState.COMPLETED:
            st.success(status_message)
        else:
            st.info(status_message)

    kpi_columns = st.columns(4)
    for column, kpi in zip(kpi_columns, representative_kpis(view), strict=True):
        column.metric(kpi.label, kpi.value)

    _render_speaker_dashboard(view)

    left, right = st.columns(
        [1.65, 1],
        gap="large",
        vertical_alignment="top",
    )
    with left:
        transcript = representative.transcript
        st.subheader("Canlı Transkript")
        with st.container(height=420, border=True):
            if transcript.stable_text:
                stable_tail = bounded_text_tail(transcript.stable_text)
                st.caption("KESİNLEŞEN KONUŞMA")
                st.write(stable_tail.visible_text)
                if stable_tail.hidden_character_count:
                    st.caption(
                        f"{stable_tail.hidden_character_count} eski karakter "
                        "görünüm dışında; tam metin oturumda korunuyor."
                    )
            else:
                st.info(
                    "Henüz kesinleşen konuşma yok. Ses işleme başladığında "
                    "transkript burada görünecek."
                )
            st.divider()
            st.caption("SON BÖLÜM · DEĞİŞEBİLİR")
            if transcript.partial_text:
                partial_tail = bounded_text_tail(transcript.partial_text)
                st.write(partial_tail.visible_text)
                if partial_tail.hidden_character_count:
                    st.caption(
                        f"{partial_tail.hidden_character_count} eski karakter "
                        "görünüm dışında."
                    )
            else:
                st.caption("Henüz kısmi metin yok.")
            st.caption(f"Son olay türü: {transcript.latest_event_type}")
    with right:
        st.subheader("Öncelikli Koçluk")
        st.caption("ŞU ANKİ NİYET VE RİSKLER")
        if not representative.intent_chips:
            st.info("Henüz güncel bir niyet veya risk tespit edilmedi.")
        for chip in representative.intent_chips:
            st.write(f"{chip.symbol} {chip.text}")
        stored_feedback = st.session_state.get("suggestion_feedback", {})
        feedback = stored_feedback if isinstance(stored_feedback, dict) else {}
        if not representative.active_suggestions:
            st.info(
                "Şu anda aktif bir koçluk önerisi yok. Yeni bir sinyal "
                "oluştuğunda öneri burada görünecek."
            )
        active = bounded_items(
            representative.active_suggestions,
            limit=VISIBLE_ACTIVE_SUGGESTIONS,
        )
        for card in active.visible_items:
            key = coaching_feedback_key(scope, card)
            with st.container(border=True):
                st.caption(f"{card.priority_symbol} {card.priority_text}")
                st.caption(
                    "Geçici anlık öneri"
                    if card.lifecycle is CoachingSuggestionLifecycle.PROVISIONAL
                    else "Kesinleşmiş öneri"
                )
                st.write(card.title)
                st.write(card.suggestion)
                details = [
                    f"Kaynak: {card.source}",
                    f"Saat: {card.timestamp}",
                    "Durum: Yeni" if card.is_new else "Durum: Daha önce gösterildi",
                ]
                if card.transcript_revision is not None:
                    details.append(f"Revizyon: {card.transcript_revision}")
                if card.related_label:
                    details.append(f"Etiket: {card.related_label}")
                st.caption(" · ".join(details))
                for column, value in zip(
                    st.columns(3),
                    ("Görüldü", "Uygulandı", "Uygun değil"),
                    strict=True,
                ):
                    if column.button(value, key=f"{key}-{value}"):
                        feedback = apply_feedback(feedback, key, value)
                        st.session_state.suggestion_feedback = feedback
                if key in feedback:
                    st.caption(f"Geri bildirim: {feedback[key]}")
        if active.hidden_item_count:
            st.caption(f"{active.hidden_item_count} ek aktif öneri görünüm dışında.")
        history_items = representative.suggestion_history
        if history_items and st.toggle(
            f"Önceki önerileri göster ({len(history_items)})",
            value=False,
            key=scoped_widget_key(scope, "representative_history_visible"),
        ):
            history = bounded_items(
                history_items,
                limit=VISIBLE_HISTORY_SUGGESTIONS,
            )
            for card in history.visible_items:
                st.caption(
                    f"{card.priority_symbol} {card.title} · "
                    f"{card.related_label or '—'} · Revizyon "
                    f"{card.transcript_revision if card.transcript_revision is not None else '—'}"
                )
            if history.hidden_item_count:
                st.caption(f"{history.hidden_item_count} eski öneri görünüm dışında.")
        if representative.detected_intent_chips:
            with st.expander("Görüşmede tespit edilenler", expanded=False):
                detected = bounded_items(
                    representative.detected_intent_chips,
                    limit=VISIBLE_TIMELINE_ROWS,
                    newest=True,
                )
                for chip in detected.visible_items:
                    st.write(f"{chip.symbol} {chip.text}")
                if detected.hidden_item_count:
                    st.caption(
                        f"{detected.hidden_item_count} eski tespit görünüm dışında."
                    )


def _render_technical(
    view: DashboardTabsViewModel,
    scope: UIScopeIdentity,
) -> None:
    operation = operational_status(view)
    st.caption(f"Operasyon durumu: {operation.label}")
    if not st.toggle(
        "Teknik ayrıntıları göster",
        value=False,
        key=scoped_widget_key(scope, "technical_details_visible"),
    ):
        st.info("Teknik izleme ayrıntıları varsayılan olarak kapalıdır.")
        return
    technical = view.technical
    progress = technical.progress
    chunk_progress = (
        f"{progress.completed_chunks}/{progress.total_chunks}"
        if progress.total_chunks
        else f"{progress.completed_chunks} · Toplam bilinmiyor"
    )
    completion = (
        f"%{progress.percentage:.0f}" if progress.total_chunks else "Hesaplanıyor"
    )
    remaining = progress.eta if progress.total_chunks else "Toplam süre bilinmiyor"
    metrics = (
        ("Parça", chunk_progress),
        ("Tamamlanma", completion),
        ("Ses / pencere", progress.time_range),
        ("Geçen süre", progress.elapsed),
        ("Tahmini kalan", remaining),
        ("Ortalama ASR", progress.average_asr),
        ("Son ASR", technical.last_asr),
        ("Toplam işlem", technical.total_processing),
        ("RTF", technical.rtf),
    )
    for row in responsive_rows(metrics, 3):
        columns = st.columns(len(row))
        for column, (label, value) in zip(columns, row, strict=True):
            column.metric(label, value)
    if technical.warning:
        st.warning("Yerel CPU işleme süresi gerçek zaman hızını aşabilir.")
    if technical.error:
        st.error("Ses işleme güvenli biçimde tamamlanamadı.")
    st.subheader("Gecikme Dağılımı")
    latency = technical.latency
    if latency:
        values = (
            ("Ses parçası", latency.chunk_duration_ms),
            ("ASR", latency.asr_ms),
            ("Kural sınıflandırma", latency.rule_ms),
            ("Koçluk kararı", latency.coaching_ms),
            ("Toplam", latency.total_ms),
        )
        for row in responsive_rows(values, 3):
            columns = st.columns(len(row))
            for column, (label, value) in zip(columns, row, strict=True):
                column.metric(label, f"{value:.0f} ms")
    st.subheader("Parça Bazında ASR Süresi")
    if technical.asr_chart:
        chart = bounded_items(
            technical.asr_chart,
            limit=VISIBLE_TECHNICAL_ROWS,
            newest=True,
        )
        st.line_chart({"ASR (ms)": [value for _, value in chart.visible_items]})
        if chart.hidden_item_count:
            st.caption(f"{chart.hidden_item_count} eski ASR ölçümü grafiğin dışında.")
    else:
        st.caption("Grafik için henüz ASR ölçümü yok.")
    st.subheader("Pipeline Durumu")
    translations = {
        "active": "Aktif",
        "simulated": "Simüle",
        "not implemented": "Uygulanmadı",
        "disabled": "Devre dışı",
        "failed": "Başarısız",
    }
    for component, status in technical.pipeline_statuses:
        st.write(f"**{component}:** {translations[status]}")
    if technical.classification_metadata:
        st.subheader("Sınıflandırma")
        metadata_rows = bounded_items(
            technical.classification_metadata,
            limit=VISIBLE_TECHNICAL_ROWS,
        )
        for label, value in metadata_rows.visible_items:
            st.write(f"**{label}:** {value}")
        if metadata_rows.hidden_item_count:
            st.caption(
                f"{metadata_rows.hidden_item_count} teknik satır görünüm dışında."
            )
        st.write(
            "**Şu Anki Etiketler:** " + (", ".join(technical.current_labels) or "—")
        )
        st.write(
            "**Görüşmede Tespit Edilenler:** "
            + (", ".join(technical.detected_labels) or "—")
        )
    if technical.probabilities:
        st.caption("Geçici etiket olasılıkları")
        probability_rows = bounded_items(
            technical.probabilities,
            limit=VISIBLE_TECHNICAL_ROWS,
        )
        for label, value in probability_rows.visible_items:
            st.write(f"{label}: {value:.3f}")
    if technical.revision_label_timeline:
        st.subheader("Revizyon Etiket Tanısı")
        diagnostics = bounded_items(
            technical.revision_label_timeline,
            limit=VISIBLE_TIMELINE_ROWS,
            newest=True,
        )
        for diagnostic in diagnostics.visible_items:
            current = ", ".join(diagnostic.current_labels) or "—"
            newly = ", ".join(diagnostic.newly_accumulated_labels) or "—"
            st.write(
                f"**Revizyon {diagnostic.transcript_revision}:** "
                f"güncel={current} · yeni={newly}"
            )
            evidence_rows = bounded_items(
                diagnostic.evidence,
                limit=VISIBLE_TECHNICAL_ROWS,
            )
            for evidence in evidence_rows.visible_items:
                details = [evidence.source.value]
                if evidence.model_id:
                    details.append(f"model={evidence.model_id}")
                if evidence.threshold_profile_id:
                    details.append(f"profile={evidence.threshold_profile_id}")
                st.caption(f"{evidence.label}: " + " · ".join(details))
        if diagnostics.hidden_item_count:
            st.caption(
                f"{diagnostics.hidden_item_count} eski revizyon görünüm dışında."
            )
    if technical.suggestion_decisions:
        st.subheader("Öneri Kapasite Kararları")
        decisions = bounded_items(
            technical.suggestion_decisions,
            limit=VISIBLE_TIMELINE_ROWS,
            newest=True,
        )
        for decision in decisions.visible_items:
            st.write(
                f"Revizyon {decision.transcript_revision} · "
                f"{decision.label_id or '—'} · {decision.priority.value} · "
                f"{decision.reason} · "
                f"geçmişe taşındı={decision.moved_to_history}"
            )
        if decisions.hidden_item_count:
            st.caption(f"{decisions.hidden_item_count} eski karar görünüm dışında.")
    if technical.coaching_metadata:
        st.subheader("Koçluk")
        for label, value in technical.coaching_metadata:
            st.write(f"**{label}:** {value}")
    failures = safe_failure_rows(view)
    if failures:
        st.subheader("Güvenli Hata Tanısı")
        for label, value in failures:
            st.write(f"**{label}:** {value}")


def _render_result(runtime: DashboardRuntime, view: DashboardTabsViewModel) -> None:
    _render_result_view(runtime.call_id, view)


def _render_result_view(call_id: str, view: DashboardTabsViewModel) -> None:
    result = view.result
    if not result.completed:
        st.info(result.waiting_message)
        return
    st.subheader("Final Kümülatif Transkript")
    final_tail = (
        bounded_text_tail(result.final_transcript) if result.final_transcript else None
    )
    st.write(
        final_tail.visible_text
        if final_tail is not None
        else "Final transkript oluşmadı."
    )
    if final_tail is not None and final_tail.hidden_character_count:
        st.caption(
            f"{final_tail.hidden_character_count} eski karakter görünüm dışında."
        )
    _metric_rows(result.metrics)
    st.write(f"**Model:** {result.model_name} · **Dil:** {result.language}")
    if result.detected_labels:
        st.write(
            "**Tespit edilen etiketler:** "
            + ", ".join(chip.text for chip in result.detected_labels)
        )
    st.write(f"**Bastırılan öneri:** {result.suppressed_count}")
    timeline = bounded_items(
        result.suggestion_timeline,
        limit=VISIBLE_TIMELINE_ROWS,
        newest=True,
    )
    for item in timeline.visible_items:
        st.write(f"`{item.timestamp:%H:%M:%S}` {item.detail}")
    if timeline.hidden_item_count:
        st.caption(
            f"{timeline.hidden_item_count} eski zaman çizelgesi satırı görünüm dışında."
        )
    for label, value in result.audio_metadata:
        st.caption(f"{label}: {value}")
    if result.final_transcript:
        st.download_button(
            "Final transkripti indir",
            result.final_transcript,
            file_name=f"{call_id}-transkript.txt",
            mime="text/plain",
        )


def _render_dashboard(
    runtime: DashboardRuntime,
    local_state: LocalExecutionState | None,
    metadata: SafeUploadMetadata | None,
    scope: UIScopeIdentity,
    rag_runtime_status: object = _RAG_RUNTIME_STATUS_NOT_AVAILABLE,
) -> None:
    view = dashboard_tabs(runtime, local_state, metadata)
    _render_dashboard_view(
        call_id=runtime.call_id,
        transcript_revision=runtime.call_state.transcript_revision,
        view=view,
        scope=scope,
        rag_runtime_status=rag_runtime_status,
    )


def _render_dashboard_view(
    *,
    call_id: str,
    transcript_revision: int,
    view: DashboardTabsViewModel,
    scope: UIScopeIdentity,
    rag_runtime_status: object = _RAG_RUNTIME_STATUS_NOT_AVAILABLE,
    execution: DashboardExecutionSnapshot | None = None,
) -> None:
    if execution is not None:
        _render_execution_status(execution)
    if rag_runtime_status is not _RAG_RUNTIME_STATUS_NOT_AVAILABLE:
        st.caption(rag_runtime_status_text(rag_runtime_status))
    representative, technical, result = st.tabs(
        ("Temsilci Görünümü", "Teknik İzleme", "Görüşme Sonucu")
    )
    with representative:
        _render_representative_view(
            call_id=call_id,
            transcript_revision=transcript_revision,
            view=view,
            scope=scope,
        )
    with technical:
        _render_technical(view, scope)
    with result:
        _render_result_view(call_id, view)


def _render_execution_status(snapshot: DashboardExecutionSnapshot) -> None:
    stage_text = _EXECUTION_STAGE_TEXT[snapshot.execution_stage]
    if (
        snapshot.execution_stage is DashboardExecutionStage.CHUNK_PROCESSING
        and snapshot.total_chunks
    ):
        stage_text = (
            f"Ses parçası {snapshot.processed_chunks} / "
            f"{snapshot.total_chunks} işleniyor"
        )
    mode_text = _EXECUTION_MODE_TEXT[snapshot.execution_mode]
    microphone_mode = snapshot.execution_mode is DashboardExecutionMode.LOCAL_MIC_TEST
    if (
        microphone_mode
        and snapshot.execution_stage is DashboardExecutionStage.MODEL_PREPARATION_FAILED
    ):
        stage_text = _EXECUTION_STAGE_TEXT[
            DashboardExecutionStage.MODEL_PREPARATION_FAILED
        ]
    elif (
        microphone_mode
        and snapshot.execution_stage is DashboardExecutionStage.MICROPHONE_DISCONNECTED
    ):
        stage_text = _EXECUTION_STAGE_TEXT[
            DashboardExecutionStage.MICROPHONE_DISCONNECTED
        ]
    elif (
        microphone_mode
        and snapshot.execution_stage is DashboardExecutionStage.MICROPHONE_OVERLOADED
    ):
        stage_text = _EXECUTION_STAGE_TEXT[
            DashboardExecutionStage.MICROPHONE_OVERLOADED
        ]
    elif microphone_mode and snapshot.lifecycle_status in {
        DashboardExecutionStatus.COMPLETED,
        DashboardExecutionStatus.CANCELLED,
    }:
        stage_text = "Mikrofon durduruldu"
    elif (
        microphone_mode and snapshot.lifecycle_status is DashboardExecutionStatus.FAILED
    ):
        stage_text = "Mikrofon testi başarısız"
    message = f"{stage_text} · {mode_text}"
    if snapshot.lifecycle_status is DashboardExecutionStatus.COMPLETED:
        st.success(message)
    elif snapshot.lifecycle_status is DashboardExecutionStatus.CANCELLED:
        st.warning(message)
    elif snapshot.lifecycle_status is DashboardExecutionStatus.FAILED:
        st.error(message)
    else:
        st.info(message)

    if snapshot.total_chunks:
        percentage = min(
            max(snapshot.processed_chunks / snapshot.total_chunks * 100, 0.0),
            100.0,
        )
        chunk_progress = (
            f"{snapshot.processed_chunks}/{snapshot.total_chunks} · %{percentage:.0f}"
        )
    elif microphone_mode:
        chunk_progress = f"{snapshot.processed_chunks} parça · Canlı oturum"
    else:
        chunk_progress = (
            f"{snapshot.processed_chunks} parça · Toplam henüz hesaplanıyor"
        )
    if snapshot.total_audio_seconds is not None:
        processed_audio = snapshot.processed_audio_seconds or 0.0
        audio_progress = f"{processed_audio:.2f}/{snapshot.total_audio_seconds:.2f} sn"
    elif snapshot.processed_audio_seconds is not None:
        audio_progress = f"{snapshot.processed_audio_seconds:.2f} sn işlendi"
    else:
        audio_progress = "Henüz hesaplanıyor"
    _metric_rows(
        (
            StatusCardViewModel("İşlenen parça", chunk_progress),
            StatusCardViewModel("İşlenen ses", audio_progress),
            StatusCardViewModel("Çalışma modu", mode_text),
            StatusCardViewModel("Anlık revizyon", str(snapshot.revision)),
        )
    )
    if (
        snapshot.lifecycle_status is DashboardExecutionStatus.RUNNING
        and snapshot.processed_chunks
    ):
        st.caption(
            "Konuşma metni güncelleniyor · Intent ve risk analizi yapılıyor · "
            "Koçluk önerileri hazırlanıyor"
        )


demos = tenant_demos()
uploaded = None
audio_path_text = ""
source_mode = "Sentetik demo"
playback_mode = "Hızlı analiz"
uploaded_content: bytes | None = None
artifact_availability = _artifact_availability()
service_selection = DashboardServiceSelection(False, False)
upload_session = st.session_state.setdefault(
    "uploaded_audio_session", UploadedAudioSession()
)
local_state_for_render: LocalExecutionState | None = None
ui_scope: UIScopeIdentity | None = None
configured_server_address = st.get_option("server.address")
local_microphone_gate = local_microphone_test_enabled(
    server_address=(
        configured_server_address
        if isinstance(configured_server_address, str)
        else None
    )
)
with st.sidebar:
    st.header("Kontroller")
    with st.expander("Görüşme", expanded=True):
        tenant_id = st.selectbox(
            "Tenant",
            list(demos),
            format_func=lambda tenant: demos[tenant].config.context.tenant_name,
            key="tenant_id",
        )
        st.text_input("Çağrı kimliği", value="demo-call", key="call_id")
        mode = st.radio("Çalışma modu", ("Sentetik Demo", "Yerel Ses Dosyası"))
    with st.expander("Ses Kaynağı", expanded=True):
        if mode == "Sentetik Demo":
            source_mode = "Sentetik demo"
            ui_scope = ui_scope_identity(
                tenant_id=tenant_id,
                call_id=str(st.session_state.call_id),
                source_mode=source_mode,
            )
            synchronize_ui_scope(st.session_state, ui_scope)
            scenarios = {
                item.scenario_id: item.name for item in demos[tenant_id].scenarios
            }
            st.selectbox(
                "Demo senaryosu",
                list(scenarios),
                format_func=scenarios.__getitem__,
                key="scenario_id",
            )
            st.slider("Oynatma hızı", 0.5, 2.0, 1.0, 0.5, key="playback_speed")
        else:
            local_source_options = ["Dosya yükle", "Yerel yol"]
            if local_microphone_gate:
                local_source_options.append("Tek konuşmacılı mikrofon testi")
            source_mode = st.radio(
                "Ses kaynağı", tuple(local_source_options), horizontal=True
            )
            ui_scope = ui_scope_identity(
                tenant_id=tenant_id,
                call_id=str(st.session_state.call_id),
                source_mode=source_mode,
            )
            if synchronize_ui_scope(st.session_state, ui_scope):
                _close_execution_resource()
                st.session_state.pop("local_mic_desired_playing", None)
                st.session_state.pop("local_mic_reset_pending", None)
                st.session_state.pop(_LOCAL_MIC_CALL_ACTIVE_SESSION_KEY, None)
                st.session_state.pop(_LOCAL_MIC_FINISH_PENDING_SESSION_KEY, None)
            if source_mode == "Dosya yükle":
                uploaded = st.file_uploader(
                    "Ses dosyası",
                    type=[
                        suffix.removeprefix(".") for suffix in SUPPORTED_UPLOAD_SUFFIXES
                    ],
                    key=f"audio_upload_{upload_session.uploader_generation}",
                )
                if uploaded is not None:
                    uploaded_content = uploaded.getvalue()
                    metadata = safe_upload_metadata(uploaded.name, uploaded.size)
                    st.session_state.safe_audio_metadata = metadata
                    _, upload_changed = upload_session.select(
                        identity=safe_upload_identity(uploaded_content),
                        tenant=demos[tenant_id],
                        base_call_id=st.session_state.call_id,
                    )
                    if upload_changed:
                        st.session_state.suggestion_feedback = {}
                    st.success("Dosya hazır; Başlat komutu bekleniyor.")
                    st.caption(
                        f"Yüklenen ses · {metadata.format_name} · "
                        f"{metadata.size_bytes / 1024:.1f} KB"
                    )
            elif source_mode == "Yerel yol":
                audio_path_text = st.text_input(
                    "Yerel ses dosyası yolu",
                    type="password",
                )
            else:
                for warning_line in LOCAL_MIC_WARNING_LINES:
                    st.warning(warning_line)
            if source_mode != "Tek konuşmacılı mikrofon testi":
                playback_mode = st.radio(
                    "Oynatma", ("Hızlı analiz", "Gerçek zaman simülasyonu")
                )
    config = demos[tenant_id].config.asr
    service_defaults = default_service_selection(
        artifact_availability,
        deterministic_rules_available=bool(demos[tenant_id].rules),
    )
    with st.expander("Model Ayarları", expanded=False):
        st.text_input("Model", config.model_name, disabled=True)
        st.text_input("Dil", config.language, disabled=True)
        st.number_input(
            "Parça süresi", value=config.chunk_duration_seconds, disabled=True
        )
        st.number_input(
            "Pencere süresi", value=config.rolling_window_seconds, disabled=True
        )
        st.number_input(
            "Kararlı bölge", value=config.stable_region_seconds, disabled=True
        )
        st.checkbox("VAD", value=config.vad_filter, disabled=True)
        enable_setfit = st.checkbox(
            "SetFit sınıflandırmasını etkinleştir",
            value=service_defaults.enable_setfit,
            key="enable_setfit",
        )
        enable_coaching = st.checkbox(
            "Canlı koçluğu etkinleştir",
            value=service_defaults.enable_coaching,
            key="enable_coaching",
        )
        service_selection = DashboardServiceSelection(
            enable_setfit=enable_setfit,
            enable_coaching=enable_coaching,
        )
        if enable_setfit and not artifact_availability.compatible:
            st.info(artifact_availability.safe_message)
    if mode == "Sentetik Demo":
        start, stop = st.columns(2)
        if start.button("Başlat", use_container_width=True):
            st.session_state.playing = True
        if stop.button("Durdur", use_container_width=True):
            st.session_state.playing = False
        if st.button("Sonraki olay", use_container_width=True):
            advance_runtime(_synthetic_runtime())
        if st.button("Sıfırla", use_container_width=True):
            _close_execution_resource()
            st.session_state.pop("synthetic_signature", None)
            st.session_state.playing = False
            st.session_state.suggestion_feedback = {}
    else:
        local = (
            upload_session.execution
            if source_mode == "Dosya yükle" and upload_session.execution is not None
            else _local_state()
        )
        local_state_for_render = local
        if source_mode != "Tek konuşmacılı mikrofon testi":
            st.warning("CPU çıkarımı gerçek zamandan daha yavaş olabilir.")
        retained_before_start = _retained_execution_resource(local.runtime)
        running_before_start = (
            retained_before_start is not None
            and retained_before_start.latest_snapshot is not None
            and retained_before_start.latest_snapshot.lifecycle_status
            is DashboardExecutionStatus.RUNNING
        )
        if source_mode != "Tek konuşmacılı mikrofon testi" and st.button(
            "Başlat",
            type="primary",
            use_container_width=True,
            disabled=not local.start_enabled or running_before_start,
        ):
            if source_mode == "Dosya yükle" and uploaded is None:
                st.error("Önce bir ses dosyası yükleyin.")
            elif source_mode == "Yerel yol" and not audio_path_text.strip():
                st.error("Yerel ses dosyası yolunu girin.")
            else:
                local.request_start()
                local.stage = _EXECUTION_STAGE_TEXT[DashboardExecutionStage.STARTING]

                try:
                    execution_resource = _execution_resource(local.runtime)
                    safe_metadata = st.session_state.get("safe_audio_metadata")
                    selected_metadata = (
                        safe_metadata
                        if isinstance(safe_metadata, SafeUploadMetadata)
                        else None
                    )
                    execution_mode = (
                        DashboardExecutionMode.REALTIME_SIMULATION
                        if playback_mode == "Gerçek zaman simülasyonu"
                        else DashboardExecutionMode.FAST_ANALYSIS
                    )
                    initial = execution_snapshot(
                        local,
                        revision=0,
                        lifecycle_status=DashboardExecutionStatus.RUNNING,
                        execution_mode=execution_mode,
                        execution_stage=DashboardExecutionStage.STARTING,
                        audio_metadata=selected_metadata,
                    )
                    selected_upload = (
                        (uploaded.name, uploaded_content)
                        if source_mode == "Dosya yükle"
                        and uploaded is not None
                        and uploaded_content is not None
                        else None
                    )
                    selected_path = audio_path_text
                    realtime = (
                        execution_mode is DashboardExecutionMode.REALTIME_SIMULATION
                    )

                    def process_uploaded_audio(
                        cancellation: Event,
                        publish: Callable[[DashboardExecutionSnapshot], None],
                    ) -> DashboardExecutionSnapshot:
                        revision = 0
                        pacing_started_at = perf_counter()

                        def publish_stage(stage: DashboardExecutionStage) -> None:
                            nonlocal revision
                            local.stage = _EXECUTION_STAGE_TEXT[stage]
                            revision += 1
                            publish(
                                execution_snapshot(
                                    local,
                                    revision=revision,
                                    lifecycle_status=DashboardExecutionStatus.RUNNING,
                                    execution_mode=execution_mode,
                                    execution_stage=stage,
                                    audio_metadata=selected_metadata,
                                )
                            )

                        publish_stage(DashboardExecutionStage.FILE_PREPARING)
                        pipeline = _make_pipeline(
                            local.runtime,
                            service_selection,
                            artifact_availability,
                            execution_resource,
                        )
                        pipeline.configure_provisional_coaching(
                            ProvisionalClassificationPolicy(enabled=True)
                        )
                        publish_stage(DashboardExecutionStage.ENGINE_RUNNING)

                        def show_plan(_plan: StreamingASRPlan) -> None:
                            if cancellation.is_set():
                                raise RuntimeError("dashboard_execution_cancelled")
                            publish_stage(DashboardExecutionStage.ENGINE_RUNNING)

                        def show_step(step: StreamingASRStep) -> None:
                            if cancellation.is_set():
                                raise RuntimeError("dashboard_execution_cancelled")
                            publish_stage(DashboardExecutionStage.CHUNK_PROCESSING)
                            if wait_for_live_cadence(
                                step,
                                started_at=pacing_started_at,
                                realtime=realtime,
                                clock=perf_counter,
                                cancellation_wait=cancellation.wait,
                            ):
                                raise RuntimeError("dashboard_execution_cancelled")

                        if selected_upload is not None:
                            upload_name, content = selected_upload
                            with temporary_uploaded_audio(upload_name, content) as path:
                                execute_local_once(
                                    local,
                                    pipeline,
                                    path,
                                    show_step,
                                    show_plan,
                                    retain_pipeline_history=False,
                                )
                        else:
                            path = Path(selected_path).expanduser()
                            if not path.is_file():
                                raise ValueError("Ses dosyası bulunamadı")
                            execute_local_once(
                                local,
                                pipeline,
                                path,
                                show_step,
                                show_plan,
                                retain_pipeline_history=False,
                            )
                        execution_resource.drain_completed(
                            current_seconds=local.audio_duration_seconds or 0.0,
                        )
                        revision += 1
                        return execution_snapshot(
                            local,
                            revision=revision,
                            lifecycle_status=DashboardExecutionStatus.COMPLETED,
                            execution_mode=execution_mode,
                            execution_stage=DashboardExecutionStage.COMPLETED,
                            audio_metadata=selected_metadata,
                        )

                    def run_uploaded_audio(
                        cancellation: Event,
                        publish: Callable[[DashboardExecutionSnapshot], None],
                    ) -> DashboardExecutionSnapshot:
                        try:
                            return process_uploaded_audio(cancellation, publish)
                        except Exception:
                            cancelled = cancellation.is_set()
                            local.status = "cancelled" if cancelled else "error"
                            terminal_stage = (
                                DashboardExecutionStage.CANCELLED
                                if cancelled
                                else DashboardExecutionStage.FAILED
                            )
                            local.stage = _EXECUTION_STAGE_TEXT[terminal_stage]
                            local.error_message = (
                                None
                                if cancelled
                                else "Ses işleme güvenli biçimde tamamlanamadı."
                            )
                            return execution_snapshot(
                                local,
                                revision=(
                                    (
                                        execution_resource.latest_snapshot.revision
                                        if execution_resource.latest_snapshot
                                        is not None
                                        else 0
                                    )
                                    + 1
                                ),
                                lifecycle_status=(
                                    DashboardExecutionStatus.CANCELLED
                                    if cancelled
                                    else DashboardExecutionStatus.FAILED
                                ),
                                execution_mode=execution_mode,
                                execution_stage=terminal_stage,
                                audio_metadata=selected_metadata,
                                failure_reason=(
                                    None if cancelled else "processing_failed"
                                ),
                            )

                    if execution_resource.start_worker(initial, run_uploaded_audio):
                        st.rerun()
                except Exception:
                    local.status = "error"
                    local.stage = "Başarısız"
                    local.error_message = "Ses işleme güvenli biçimde tamamlanamadı."
                    st.error(local.error_message)
        if source_mode != "Tek konuşmacılı mikrofon testi" and st.button(
            "Durdur",
            disabled=not (local.stop_enabled or running_before_start),
            use_container_width=True,
        ):
            retained = _retained_execution_resource(local.runtime)
            if retained is not None:
                retained.cancel()
        if source_mode != "Tek konuşmacılı mikrofon testi" and st.button(
            "Sıfırla", use_container_width=True
        ):
            _close_execution_resource()
            if source_mode == "Dosya yükle":
                upload_session.reset()
            else:
                st.session_state.pop("local_signature", None)
            st.session_state.pop("safe_audio_metadata", None)
            st.session_state.suggestion_feedback = {}

st.title("Canlı Koçluk Paneli")
st.caption("Temsilci desteği ve güvenli teknik izleme")
if ui_scope is None:
    raise RuntimeError("Dashboard UI scope was not initialized")
active_ui_scope = ui_scope
if mode == "Sentetik Demo":
    runtime = _synthetic_runtime()
    interval = (
        1.5 / st.session_state.playback_speed
        if st.session_state.get("playing", False)
        else None
    )

    @st.fragment(run_every=interval)
    def live_area() -> None:
        current = _synthetic_runtime()
        if st.session_state.get("playing", False):
            advance_runtime(current)
            if current.complete:
                st.session_state.playing = False
        _render_dashboard(current, None, None, active_ui_scope)

    live_area()
else:
    if local_state_for_render is None:
        raise RuntimeError("Local dashboard state was not initialized")
    render_local_state = local_state_for_render
    safe_metadata = st.session_state.get("safe_audio_metadata")
    retained_resource = _retained_execution_resource(render_local_state.runtime)
    retained_snapshot = (
        None if retained_resource is None else retained_resource.latest_snapshot
    )
    local_microphone_source = source_mode == "Tek konuşmacılı mikrofon testi"
    polling_interval = (
        0.5
        if local_microphone_source
        or (
            retained_snapshot is not None
            and retained_snapshot.lifecycle_status is DashboardExecutionStatus.RUNNING
        )
        else None
    )

    @st.fragment(run_every=polling_interval)
    def local_live_area() -> None:
        local_microphone_state = None
        if local_microphone_source:
            with st.sidebar:
                local_microphone_state = _render_local_microphone_controls(
                    local=render_local_state,
                    selection=service_selection,
                    availability=artifact_availability,
                )
        resource = _retained_execution_resource(render_local_state.runtime)
        snapshot = None if resource is None else resource.latest_snapshot
        if local_microphone_state is not None and snapshot is not None:
            snapshot = _local_microphone_presentation_snapshot(
                snapshot,
                local_microphone_state[1].diagnostics.status,
            )
        rag_status = st.session_state.get(
            _RAG_RUNTIME_STATUS_SESSION_KEY,
            _RAG_RUNTIME_STATUS_NOT_AVAILABLE,
        )
        if snapshot is None:
            _render_dashboard(
                render_local_state.runtime,
                render_local_state,
                safe_metadata,
                active_ui_scope,
                rag_status,
            )
            return
        _render_dashboard_view(
            call_id=snapshot.call_id,
            transcript_revision=snapshot.transcript_revision,
            view=snapshot.tabs,
            scope=active_ui_scope,
            rag_runtime_status=rag_status,
            execution=snapshot,
        )

    local_live_area()
