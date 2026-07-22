"""First synthetic Streamlit live coaching dashboard prototype."""

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_dashboard.demo_data import scenario_for, tenant_demos  # noqa: E402
from live_dashboard.view_models import (  # noqa: E402
    DashboardRuntime,
    advance_runtime,
    create_runtime,
    ordered_suggestions,
    transcript_view,
)


st.set_page_config(
    page_title="Canlı Koçluk — Sentetik Demo", page_icon="🎧", layout="wide"
)
st.html("""
<style>
div[data-testid="stMetric"] {background:#f7f8fa;border:1px solid #e6e8eb;padding:10px;border-radius:10px}
.demo-note {color:#6b7280;font-size:.85rem}.partial {border-left:4px solid #eab308;padding:.7rem;background:#fffbeb}
.stable {border-left:4px solid #2563eb;padding:.9rem;background:#eff6ff;min-height:90px}
</style>
""")


def _runtime() -> DashboardRuntime:
    demos = tenant_demos()
    tenant_id = st.session_state.tenant_id
    signature = (tenant_id, st.session_state.scenario_id, st.session_state.call_id)
    if st.session_state.get("runtime_signature") != signature:
        st.session_state.runtime = create_runtime(
            demos[tenant_id],
            scenario_for(tenant_id, st.session_state.scenario_id),
            st.session_state.call_id,
        )
        st.session_state.runtime_signature = signature
        st.session_state.playing = False
    return st.session_state.runtime


def _format_elapsed(seconds: float) -> str:
    minutes, remaining = divmod(int(seconds), 60)
    return f"{minutes:02d}:{remaining:02d}"


def _render(runtime: DashboardRuntime) -> None:
    transcript = transcript_view(runtime)
    status_columns = st.columns(5)
    status_columns[0].metric("Tenant", runtime.tenant.config.context.tenant_name)
    status_columns[1].metric("Çağrı", runtime.call_id)
    status_columns[2].metric(
        "Çağrı durumu", "Tamamlandı" if runtime.complete else "Canlı demo"
    )
    status_columns[3].metric("Geçen süre", _format_elapsed(runtime.elapsed_seconds))
    status_columns[4].metric(
        "Pipeline", "Koçluk hazır" if runtime.latest_labels else "ASR simülasyonu"
    )

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
        if not runtime.timeline:
            st.info("Demo başlatıldığında sentetik olaylar burada görünecek.")
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
        st.write(f"Seçilen CoachingAction: `{runtime.latest_action.value}`")
        st.subheader("Koçluk Önerileri")
        for card in ordered_suggestions(runtime.suggestions):
            with st.container(border=True):
                if card.priority.value in {"CRITICAL", "HIGH"}:
                    st.error(f"{card.priority_text} · {card.title}")
                else:
                    st.info(f"{card.priority_text} · {card.title}")
                st.write(card.suggestion)
                st.caption(f"Aksiyon: {card.action} · Saat: {card.timestamp}")
        st.metric("Bastırılan öneri", len(runtime.suppression_reasons))
        if runtime.suppression_reasons:
            st.caption("Nedenler: " + ", ".join(runtime.suppression_reasons))

    st.subheader("Gecikme — sentetik demo değerleri")
    tick = max(runtime.next_event_index, 1)
    values = (38 + tick, 122 + tick * 3, 7 + tick, 4 + tick, 171 + tick * 5)
    for column, (label, value) in zip(
        st.columns(5),
        zip(
            ("Ses parçası", "ASR", "Kural sınıflandırma", "Koçluk kararı", "Toplam"),
            values,
            strict=True,
        ),
        strict=True,
    ):
        column.metric(label, f"{value} ms", help="Yalnızca sentetik demo değeri")

    st.subheader("Mimari Durum")
    architecture = st.columns(5)
    for column, (component, status) in zip(
        architecture,
        (
            ("ASR", "Simüle"),
            ("Kural motoru", "Aktif"),
            ("SetFit", "Uygulanmadı"),
            ("RAG", "Uygulanmadı"),
            ("LLM", "Uygulanmadı"),
        ),
        strict=True,
    ):
        column.metric(component, status)
    st.caption(
        "Bu prototip yalnızca sentetik Türkçe metin ve sentetik zamanlama kullanır; ses baytı, gerçek müşteri verisi veya harici servis içermez."
    )


demos = tenant_demos()
with st.sidebar:
    st.header("Demo Kontrolleri")
    tenant_id = st.selectbox("Tenant", list(demos), key="tenant_id")
    st.text_input("Çağrı ID", value="demo-call", key="call_id")
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
        advance_runtime(_runtime())
    if st.button("Demoyu sıfırla", use_container_width=True):
        st.session_state.pop("runtime_signature", None)
        st.session_state.playing = False

runtime = _runtime()
interval = (
    1.5 / st.session_state.playback_speed
    if st.session_state.get("playing", False)
    else None
)


@st.fragment(run_every=interval)
def live_area() -> None:
    current = _runtime()
    if st.session_state.get("playing", False):
        advance_runtime(current)
        if current.complete:
            st.session_state.playing = False
    _render(current)


st.title("Canlı Koçluk Paneli")
st.caption("Güvenli, tenant-aware, sentetik temsilci destek prototipi")
live_area()
