"""Professional synthetic and opt-in local-file coaching dashboard."""

import sys
from pathlib import Path
from time import sleep

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.asr.faster_whisper_engine import FasterWhisperEngine  # noqa: E402
from app.classification.runtime import RuntimeSetFitClassifier  # noqa: E402
from app.streaming.pipeline import (  # noqa: E402
    StreamingASRPipeline,
    StreamingASRPlan,
    StreamingASRStep,
)
from app.streaming.window_transcriber import WindowTranscriber  # noqa: E402
from live_dashboard.demo_data import scenario_for, tenant_demos  # noqa: E402
from live_dashboard.runtime_wiring import (  # noqa: E402
    ArtifactAvailability,
    DashboardServiceSelection,
    build_live_pipeline,
    default_service_selection,
    inspect_default_artifacts,
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
    DashboardTabsViewModel,
    LocalExecutionState,
    UploadedAudioSession,
    apply_feedback,
    advance_runtime,
    create_local_execution,
    create_runtime,
    dashboard_tabs,
    execute_local_once,
    progress_view,
    responsive_rows,
)


st.set_page_config(page_title="Canlı Koçluk", page_icon="🎧", layout="wide")
st.html("""
<style>
div[data-testid="stMetric"] {background:#f7f8fa;border:1px solid #e6e8eb;padding:10px;border-radius:10px}
div[data-testid="stMetricValue"] {font-size:1.1rem;white-space:normal;overflow-wrap:anywhere}
.stable {border-left:4px solid #2563eb;padding:.9rem;background:#eff6ff;min-height:90px}
.partial {border-left:4px solid #94a3b8;padding:.75rem;background:#f8fafc;color:#64748b}
.chip {display:inline-block;padding:.28rem .62rem;margin:.15rem;border:1px solid #cbd5e1;border-radius:999px;background:#f8fafc;font-size:.88rem}
</style>
""")


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


@st.cache_resource(show_spinner=False)
def _load_runtime_classifier() -> RuntimeSetFitClassifier:
    return RuntimeSetFitClassifier()


@st.cache_resource(show_spinner=False)
def _artifact_availability() -> ArtifactAvailability:
    return inspect_default_artifacts()


def _make_pipeline(
    runtime: DashboardRuntime,
    selection: DashboardServiceSelection,
    availability: ArtifactAvailability,
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
    )


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


def _render_representative(
    runtime: DashboardRuntime, view: DashboardTabsViewModel
) -> None:
    representative = view.representative
    _metric_rows(representative.status)
    progress = representative.progress
    with st.container(border=True):
        st.markdown(f"**İşlem aşaması:** {progress.stage}")
        st.progress(progress.percentage / 100)
        st.caption(f"{progress.eta} · {progress.time_range}")
    if runtime.latest_event is None:
        st.info("Görüşmeyi başlatın; transkript ve koçluk önerileri burada görünecek.")
    left, right = st.columns([1.2, 1])
    with left:
        transcript = representative.transcript
        st.subheader("Canlı Transkript")
        st.markdown(
            f'<div class="stable"><b>Kesinleşen konuşma</b><br>{transcript.stable_text or "Henüz kesinleşen konuşma yok."}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="partial"><b>Son bölüm — değişebilir</b><br><i>{transcript.partial_text or "Henüz kısmi metin yok."}</i></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Son olay türü: {transcript.latest_event_type}")
    with right:
        st.subheader("Anlık Koçluk")
        st.caption("Şu Anki Etiketler")
        if representative.intent_chips:
            chips = " ".join(
                f'<span class="chip">{chip.symbol} {chip.text}</span>'
                for chip in representative.intent_chips
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("Aktif niyet veya risk bulunmuyor.")
        st.caption("Görüşmede Tespit Edilenler")
        if representative.detected_intent_chips:
            detected_chips = " ".join(
                f'<span class="chip">{chip.symbol} {chip.text}</span>'
                for chip in representative.detected_intent_chips
            )
            st.markdown(detected_chips, unsafe_allow_html=True)
        else:
            st.caption("Görüşmede henüz etiket tespit edilmedi.")
        feedback = st.session_state.setdefault("suggestion_feedback", {})
        for message in representative.safe_messages:
            st.warning(message)
        if not representative.suggestions:
            st.info(representative.empty_suggestion_message)
        for index, card in enumerate(representative.suggestions):
            key = f"{card.timestamp}-{card.title}-{index}"
            with st.container(border=True):
                st.markdown(
                    f"**{card.priority_symbol} {card.priority_text} · {card.title}**"
                )
                st.write(card.suggestion)
                details = [
                    f"Öncelik: {card.priority_text}",
                    f"Aksiyon: {card.action}",
                    f"Kaynak: {card.source}",
                    f"Saat: {card.timestamp}",
                    "Durum: Yeni" if card.is_new else "Durum: Daha önce gösterildi",
                ]
                if card.transcript_revision is not None:
                    details.append(f"Revizyon: {card.transcript_revision}")
                if card.related_label:
                    details.append(f"Etiket: {card.related_label}")
                if card.evidence_ids:
                    details.append("Kanıt: " + ", ".join(card.evidence_ids))
                st.caption(" · ".join(details))
                for column, value in zip(
                    st.columns(3), ("Görüldü", "Uygulandı", "Uygun değil"), strict=True
                ):
                    if column.button(value, key=f"feedback-{key}-{value}"):
                        feedback = apply_feedback(feedback, key, value)
                        st.session_state.suggestion_feedback = feedback
                if key in feedback:
                    st.caption(f"Geri bildirim: {feedback[key]}")
        st.caption(f"Bastırılan öneri sayısı: {representative.suppressed_count}")


def _render_technical(view: DashboardTabsViewModel) -> None:
    technical = view.technical
    progress = technical.progress
    metrics = (
        ("Parça", f"{progress.completed_chunks}/{progress.total_chunks}"),
        ("Tamamlanma", f"%{progress.percentage:.0f}"),
        ("Ses / pencere", progress.time_range),
        ("Geçen süre", progress.elapsed),
        ("Tahmini kalan", progress.eta),
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
        st.warning(technical.warning)
    if technical.error:
        st.error(technical.error)
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
        st.line_chart({"ASR (ms)": [value for _, value in technical.asr_chart]})
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
        for label, value in technical.classification_metadata:
            st.write(f"**{label}:** {value}")
        st.write(
            "**Şu Anki Etiketler:** " + (", ".join(technical.current_labels) or "—")
        )
        st.write(
            "**Görüşmede Tespit Edilenler:** "
            + (", ".join(technical.detected_labels) or "—")
        )
    if technical.probabilities:
        st.caption("Geçici etiket olasılıkları")
        for label, value in technical.probabilities:
            st.write(f"{label}: {value:.3f}")
    if technical.coaching_metadata:
        st.subheader("Koçluk")
        for label, value in technical.coaching_metadata:
            st.write(f"**{label}:** {value}")
    if technical.failure_details:
        st.subheader("Güvenli Hata Tanısı")
        for label, value in technical.failure_details:
            st.write(f"**{label}:** {value}")


def _render_result(runtime: DashboardRuntime, view: DashboardTabsViewModel) -> None:
    result = view.result
    if not result.completed:
        st.info(result.waiting_message)
        return
    st.subheader("Final Kümülatif Transkript")
    st.write(result.final_transcript or "Final transkript oluşmadı.")
    _metric_rows(result.metrics)
    st.write(f"**Model:** {result.model_name} · **Dil:** {result.language}")
    if result.detected_labels:
        st.write(
            "**Tespit edilen etiketler:** "
            + ", ".join(chip.text for chip in result.detected_labels)
        )
    st.write(f"**Bastırılan öneri:** {result.suppressed_count}")
    for item in result.suggestion_timeline:
        st.write(f"`{item.timestamp:%H:%M:%S}` {item.detail}")
    for label, value in result.audio_metadata:
        st.caption(f"{label}: {value}")
    if result.final_transcript:
        st.download_button(
            "Final transkripti indir",
            result.final_transcript,
            file_name=f"{runtime.call_id}-transkript.txt",
            mime="text/plain",
        )


def _render_dashboard(
    runtime: DashboardRuntime,
    local_state: LocalExecutionState | None,
    metadata: SafeUploadMetadata | None,
) -> None:
    view = dashboard_tabs(runtime, local_state, metadata)
    representative, technical, result = st.tabs(
        ("Temsilci Görünümü", "Teknik İzleme", "Görüşme Sonucu")
    )
    with representative:
        _render_representative(runtime, view)
    with technical:
        _render_technical(view)
    with result:
        _render_result(runtime, view)


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
            source_mode = st.radio(
                "Ses kaynağı", ("Dosya yükle", "Yerel yol"), horizontal=True
            )
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
                        f"{metadata.filename} · {metadata.format_name} · "
                        f"{metadata.size_bytes / 1024:.1f} KB"
                    )
            else:
                audio_path_text = st.text_input("Yerel ses dosyası yolu")
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
        st.warning("CPU çıkarımı gerçek zamandan daha yavaş olabilir.")
        if st.button(
            "Başlat",
            type="primary",
            use_container_width=True,
            disabled=not local.start_enabled,
        ):
            if source_mode == "Dosya yükle" and uploaded is None:
                st.error("Önce bir ses dosyası yükleyin.")
            elif source_mode == "Yerel yol" and not audio_path_text.strip():
                st.error("Yerel ses dosyası yolunu girin.")
            else:
                local.request_start()
                local.stage = "Model yükleniyor"
                live_progress = st.progress(0, text="Aşama: Model yükleniyor")

                def show_plan(plan: StreamingASRPlan) -> None:
                    live_progress.progress(
                        0,
                        text=(
                            f"ASR işleniyor · 0/{plan.total_chunks} parça · %0 · "
                            "Tahmin hazırlanıyor"
                        ),
                    )

                def show_step(step: StreamingASRStep) -> None:
                    progress = progress_view(local)
                    live_progress.progress(
                        progress.percentage / 100,
                        text=(
                            f"{progress.stage} · {progress.completed_chunks}/"
                            f"{progress.total_chunks} · %{progress.percentage:.0f} · "
                            f"{progress.eta}"
                        ),
                    )
                    if playback_mode == "Gerçek zaman simülasyonu":
                        duration = step.chunk_end_seconds - step.chunk_start_seconds
                        sleep(max(duration - step.transcription_time_seconds, 0.0))

                try:
                    pipeline = _make_pipeline(
                        local.runtime,
                        service_selection,
                        artifact_availability,
                    )
                    if (
                        source_mode == "Dosya yükle"
                        and uploaded is not None
                        and uploaded_content is not None
                    ):
                        with temporary_uploaded_audio(
                            uploaded.name, uploaded_content
                        ) as path:
                            execute_local_once(
                                local, pipeline, path, show_step, show_plan
                            )
                    else:
                        path = Path(audio_path_text).expanduser()
                        if not path.is_file():
                            raise ValueError("Ses dosyası bulunamadı")
                        execute_local_once(local, pipeline, path, show_step, show_plan)
                    live_progress.progress(1.0, text="Tamamlandı · %100")
                except Exception:
                    local.status = "error"
                    local.stage = "Başarısız"
                    local.error_message = "Ses işleme güvenli biçimde tamamlanamadı."
                    st.error(local.error_message)
        st.button("Durdur", disabled=not local.stop_enabled, use_container_width=True)
        if st.button("Sıfırla", use_container_width=True):
            if source_mode == "Dosya yükle":
                upload_session.reset()
            else:
                st.session_state.pop("local_signature", None)
            st.session_state.pop("safe_audio_metadata", None)
            st.session_state.suggestion_feedback = {}

st.title("Canlı Koçluk Paneli")
st.caption("Temsilci desteği ve güvenli teknik izleme")
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
        _render_dashboard(current, None, None)

    live_area()
else:
    if local_state_for_render is None:
        raise RuntimeError("Local dashboard state was not initialized")
    safe_metadata = st.session_state.get("safe_audio_metadata")
    _render_dashboard(
        local_state_for_render.runtime,
        local_state_for_render,
        safe_metadata,
    )
