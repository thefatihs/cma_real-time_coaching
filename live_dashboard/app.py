"""Synthetic and opt-in local-file Streamlit coaching dashboard."""

import sys
from pathlib import Path
from time import sleep

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.asr.faster_whisper_engine import FasterWhisperEngine  # noqa: E402
from app.streaming.pipeline import (  # noqa: E402
    StreamingASRPipeline,
    StreamingASRPlan,
    StreamingASRStep,
)
from app.streaming.window_transcriber import WindowTranscriber  # noqa: E402
from live_dashboard.demo_data import scenario_for, tenant_demos  # noqa: E402
from live_dashboard.view_models import (  # noqa: E402
    DashboardRuntime,
    LocalExecutionState,
    action_display,
    advance_runtime,
    create_local_execution,
    create_runtime,
    execute_local_once,
    ordered_suggestions,
    progress_view,
    responsive_rows,
    status_cards,
    transcript_view,
)
from live_dashboard.uploaded_audio import (  # noqa: E402
    SUPPORTED_UPLOAD_SUFFIXES,
    safe_upload_metadata,
    temporary_uploaded_audio,
)


st.set_page_config(
    page_title="Canlı Koçluk — Güvenli Demo", page_icon="🎧", layout="wide"
)
st.html("""
<style>
div[data-testid="stMetric"] {background:#f7f8fa;border:1px solid #e6e8eb;padding:10px;border-radius:10px}
div[data-testid="stMetricValue"] {font-size:1.12rem;white-space:normal;overflow-wrap:anywhere}
.partial {border-left:4px solid #eab308;padding:.7rem;background:#fffbeb}
.stable {border-left:4px solid #2563eb;padding:.9rem;background:#eff6ff;min-height:90px}
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


def _make_pipeline(runtime: DashboardRuntime) -> StreamingASRPipeline:
    config = runtime.tenant.config
    engine = _load_asr_model(
        config.asr.model_name,
        config.asr.language,
        config.asr.beam_size,
        config.asr.vad_filter,
        config.asr.condition_on_previous_text,
        config.asr.initial_prompt,
    )
    transcriber = WindowTranscriber(engine)
    return StreamingASRPipeline(config.context, config.asr, transcriber)


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


def _render_status(runtime: DashboardRuntime, pipeline_status: str) -> None:
    for row in responsive_rows(status_cards(runtime, pipeline_status), 3):
        columns = st.columns(len(row))
        for column, card in zip(columns, row, strict=True):
            column.metric(card.label, card.value)


def _render_dashboard(
    runtime: DashboardRuntime, pipeline_status: str, *, local_mode: bool
) -> None:
    transcript = transcript_view(runtime)
    _render_status(runtime, pipeline_status)
    if runtime.latest_event is None:
        instruction = (
            "Ses yolu ve kimlikleri kontrol edip **İşlemeyi başlat** düğmesine basın."
            if local_mode
            else "Senaryoyu seçin; ardından **Başlat** veya **Sonraki olay** düğmesini kullanın."
        )
        st.info(f"Henüz işlem başlamadı. {instruction}")

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Canlı Transkript")
        st.markdown(
            f'<div class="stable"><b>Kararlı metin</b><br>{transcript.stable_text or "Henüz kararlı metin yok."}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="partial"><b>Kısmi metin — değişebilir</b><br>{transcript.partial_text or "—"}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Son transkript olayı: {transcript.latest_event_type}")
        st.subheader("Olay Zaman Çizelgesi")
        for item in runtime.timeline:
            st.write(
                f"`{item.timestamp:%H:%M:%S}` **{item.event_type}** — {item.detail}"
            )
    with right:
        st.subheader("Aktif Niyetler ve Riskler")
        if not runtime.latest_labels:
            st.caption("Aktif sınıflandırma yok.")
        for label in runtime.latest_labels:
            (st.error if label.critical else st.info)(
                f"{label.name} · {label.score_percent}"
            )
        st.write(f"Seçilen aksiyon: `{action_display(runtime.latest_action)}`")
        st.subheader("Koçluk Önerileri")
        for card in ordered_suggestions(runtime.suggestions):
            with st.container(border=True):
                message = f"{card.priority_text} · {card.title}"
                (st.error if card.priority.value in {"CRITICAL", "HIGH"} else st.info)(
                    message
                )
                st.write(card.suggestion)
                st.caption(f"Aksiyon: {card.action} · Saat: {card.timestamp}")
        st.metric("Bastırılan öneri", len(runtime.suppression_reasons))
        if runtime.suppression_reasons:
            st.caption("Nedenler: " + ", ".join(runtime.suppression_reasons))

    st.subheader("Gecikme")
    if local_mode and runtime.latency is not None:
        latency_items = (
            ("Ses parçası süresi", runtime.latency.chunk_duration_ms, "gerçek süre"),
            ("ASR", runtime.latency.asr_ms, "gerçek işlem süresi"),
            ("Kural sınıflandırma", runtime.latency.rule_ms, "yaklaşık"),
            ("Koçluk kararı", runtime.latency.coaching_ms, "yaklaşık"),
            ("Toplam", runtime.latency.total_ms, "ASR gerçek; diğerleri yaklaşık"),
        )
    else:
        tick = max(runtime.next_event_index, 1)
        latency_items = tuple(
            (label, value, "sentetik demo")
            for label, value in zip(
                (
                    "Ses parçası",
                    "ASR",
                    "Kural sınıflandırma",
                    "Koçluk kararı",
                    "Toplam",
                ),
                (38 + tick, 122 + tick * 3, 7 + tick, 4 + tick, 171 + tick * 5),
                strict=True,
            )
        )
    for row in responsive_rows(latency_items, 3):
        columns = st.columns(len(row))
        for column, (label, value, note) in zip(columns, row, strict=True):
            column.metric(label, f"{value:.0f} ms", help=note)

    st.subheader("Mimari Durum")
    architecture = (
        ("ASR", "Yerel dosya" if local_mode else "Simüle"),
        ("Kural motoru", "Aktif"),
        ("SetFit", "Uygulanmadı"),
        ("RAG", "Uygulanmadı"),
        ("LLM", "Uygulanmadı"),
    )
    for row in responsive_rows(architecture, 3):
        columns = st.columns(len(row))
        for column, (component, status) in zip(columns, row, strict=True):
            column.metric(component, status)
    st.caption(
        "Ham ses baytları gösterilmez veya loglanmaz. Transkript yalnızca oturum belleğinde tutulur."
    )


demos = tenant_demos()
with st.sidebar:
    st.header("Kontroller")
    mode = st.radio("Çalışma modu", ("Sentetik Demo", "Yerel Ses Dosyası"))
    tenant_id = st.selectbox("Tenant kimliği", list(demos), key="tenant_id")
    st.text_input("Çağrı kimliği", value="demo-call", key="call_id")
    if mode == "Sentetik Demo":
        scenarios = {item.scenario_id: item.name for item in demos[tenant_id].scenarios}
        st.selectbox(
            "Demo senaryosu",
            list(scenarios),
            format_func=scenarios.__getitem__,
            key="scenario_id",
        )
        st.slider("Oynatma hızı", 0.5, 2.0, 1.0, 0.5, key="playback_speed")
        start, stop = st.columns(2)
        if start.button("Başlat", use_container_width=True):
            st.session_state.playing = True
        if stop.button("Durdur", use_container_width=True):
            st.session_state.playing = False
        if st.button("Sonraki olay", use_container_width=True):
            advance_runtime(_synthetic_runtime())
        if st.button("Demoyu sıfırla", use_container_width=True):
            st.session_state.pop("synthetic_signature", None)
            st.session_state.playing = False
    else:
        st.subheader("Kendi Sesimle Test")
        source_mode = st.radio(
            "Ses kaynağı", ("Dosya yükle", "Yerel yol"), horizontal=True
        )
        uploaded = None
        audio_path_text = ""
        if source_mode == "Dosya yükle":
            uploaded = st.file_uploader(
                "Ses dosyası",
                type=[suffix.removeprefix(".") for suffix in SUPPORTED_UPLOAD_SUFFIXES],
                key=f"audio_upload_{st.session_state.get('upload_generation', 0)}",
                help="WAV önerilir; PyAV ile desteklenen yaygın ses biçimleri kabul edilir.",
            )
            if uploaded is not None:
                try:
                    metadata = safe_upload_metadata(uploaded.name, uploaded.size)
                    st.success("Dosya yüklendi; yalnızca Başlat sonrası işlenecek.")
                    st.caption(
                        f"Dosya: {metadata.filename} · Biçim: {metadata.format_name} · Boyut: {metadata.size_bytes / 1024:.1f} KB"
                    )
                except ValueError as error:
                    st.error(str(error))
        else:
            audio_path_text = st.text_input("Yerel ses dosyası yolu", key="audio_path")
        playback_mode = st.radio(
            "Oynatma modu", ("Hızlı analiz", "Gerçek zaman simülasyonu")
        )
        st.warning("CPU üzerinde ASR çıkarımı gerçek zamandan çok daha yavaş olabilir.")
        if playback_mode == "Gerçek zaman simülasyonu":
            st.caption(
                "Parça süreleri mümkün olduğunca taklit edilir; CPU çıkarımı ek gecikme oluşturabilir."
            )
        local = _local_state()
        if st.button(
            "İşlemeyi başlat",
            type="primary",
            use_container_width=True,
            disabled=local.status != "idle",
        ):
            if not tenant_id.strip() or not st.session_state.call_id.strip():
                st.error("Tenant ve çağrı kimliği zorunludur.")
            elif source_mode == "Dosya yükle" and uploaded is None:
                st.error("Önce bir ses dosyası yükleyin.")
            elif source_mode == "Yerel yol" and not audio_path_text.strip():
                st.error("Yerel ses dosyası yolunu girin.")
            else:
                local.request_start()
                local.stage = "Model yükleniyor"
                progress = st.progress(0, text="Aşama: Model yükleniyor")

                def show_plan(plan: StreamingASRPlan) -> None:
                    progress.progress(
                        0,
                        text=(
                            f"Aşama: ASR işleniyor · 0/{plan.total_chunks} parça · %0 · "
                            f"Ses süresi: {plan.audio_duration_seconds:.2f} sn · "
                            "Tahmin hazırlanıyor"
                        ),
                    )

                def update_progress(step: StreamingASRStep) -> None:
                    view = progress_view(local)
                    progress.progress(
                        view.percentage / 100,
                        text=(
                            f"Aşama: {view.stage} · "
                            f"{view.completed_chunks}/{view.total_chunks} parça · "
                            f"%{view.percentage:.0f} · Aralık: {view.time_range} · "
                            f"Geçen: {view.elapsed} · Ortalama ASR: {view.average_asr} · "
                            f"{view.eta}"
                        ),
                    )
                    if playback_mode == "Gerçek zaman simülasyonu":
                        chunk_seconds = float(
                            getattr(step, "chunk_end_seconds")
                        ) - float(getattr(step, "chunk_start_seconds"))
                        asr_seconds = float(getattr(step, "transcription_time_seconds"))
                        sleep(max(chunk_seconds - asr_seconds, 0.0))

                try:
                    if source_mode == "Dosya yükle" and uploaded is not None:
                        with temporary_uploaded_audio(
                            uploaded.name, uploaded.getvalue()
                        ) as temporary_path:
                            execute_local_once(
                                local,
                                _make_pipeline(local.runtime),
                                temporary_path,
                                update_progress,
                                show_plan,
                            )
                    else:
                        path = Path(audio_path_text).expanduser()
                        if not path.is_file():
                            raise ValueError("Belirtilen yerel ses dosyası bulunamadı.")
                        execute_local_once(
                            local,
                            _make_pipeline(local.runtime),
                            path,
                            update_progress,
                            show_plan,
                        )
                    completed_view = progress_view(local)
                    progress.progress(
                        100,
                        text=(
                            f"Aşama: Tamamlandı · {completed_view.completed_chunks}/"
                            f"{completed_view.total_chunks} parça · %100 · "
                            f"Geçen: {completed_view.elapsed} · "
                            f"Ortalama ASR: {completed_view.average_asr}"
                        ),
                    )
                except Exception:
                    if local.status != "error":
                        local.status = "error"
                        local.stage = "Başarısız"
                        local.failed_chunk = local.current_chunk + 1
                        local.error_message = (
                            "Ses işleme sırasında beklenmeyen bir hata oluştu."
                        )
                    failed_view = progress_view(local)
                    progress.progress(
                        failed_view.percentage / 100,
                        text=(
                            f"Aşama: Başarısız · {failed_view.completed_chunks}/"
                            f"{failed_view.total_chunks} parça · "
                            f"Başarısız parça: {failed_view.failed_chunk or 'bilinmiyor'}"
                        ),
                    )
                    st.error(
                        local.error_message
                        or "Ses işleme sırasında beklenmeyen bir hata oluştu."
                    )
        st.button(
            "Durdur",
            use_container_width=True,
            disabled=local.status != "running",
            help="Senkron çalışan mevcut pipeline nedeniyle yalnızca parçalar arasında uygulanabilir.",
        )
        if st.button("Yerel işlemi sıfırla", use_container_width=True):
            st.session_state.pop("local_signature", None)
            st.session_state.upload_generation = (
                st.session_state.get("upload_generation", 0) + 1
            )

st.title("Canlı Koçluk Paneli")
st.caption("Tenant-aware temsilci destek görünümü")
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
        _render_dashboard(current, "Koçluk hazır", local_mode=False)

    live_area()
else:
    local = _local_state()
    _render_dashboard(local.runtime, local.stage, local_mode=True)
    local_progress = progress_view(local)
    if local.total_chunks:
        st.progress(
            local_progress.percentage / 100,
            text=(
                f"{local_progress.completed_chunks}/{local_progress.total_chunks} parça · "
                f"%{local_progress.percentage:.0f} · Aralık: {local_progress.time_range} · "
                f"Geçen: {local_progress.elapsed} · Ortalama ASR: "
                f"{local_progress.average_asr} · {local_progress.eta}"
            ),
        )
    if local.total_chunks:
        st.caption(f"İşlenen parça: {local.current_chunk} / {local.total_chunks}")
    if local.asr_window_ms:
        st.subheader("Pencere Bazında ASR Gecikmesi")
        st.dataframe(
            {
                "Pencere": list(range(1, len(local.asr_window_ms) + 1)),
                "ASR (ms, gerçek)": local.asr_window_ms,
            },
            hide_index=True,
            use_container_width=True,
        )
    if local.processing_seconds is not None:
        rtf = local.real_time_factor
        average_asr_ms = (
            sum(local.asr_window_ms) / len(local.asr_window_ms)
            if local.asr_window_ms
            else 0.0
        )
        summary = (
            ("Toplam parça", str(local.total_chunks)),
            ("Tamamlanan parça", str(local.current_chunk)),
            ("Ses süresi", f"{(local.audio_duration_seconds or 0):.2f} sn"),
            ("Toplam işlem süresi", f"{local.processing_seconds:.2f} sn"),
            ("Ortalama ASR", f"{average_asr_ms:.0f} ms"),
            ("RTF", "—" if rtf is None else f"{rtf:.2f}x"),
        )
        for row in responsive_rows(summary, 3):
            columns = st.columns(len(row))
            for column, (label, value) in zip(columns, row, strict=True):
                column.metric(label, value)
        st.subheader("Final Kümülatif Transkript")
        final_text = local.runtime.call_state.stable_transcript
        st.write(final_text or "Final kararlı transkript oluşmadı.")
        if st.checkbox("Final transkripti dışa aktarmayı etkinleştir"):
            st.download_button(
                "Transkripti indir",
                data=final_text,
                file_name=f"{local.runtime.call_id}-transkript.txt",
                mime="text/plain",
            )
