"""Pure presentation helpers for the live coaching dashboard."""

from dataclasses import dataclass

from live_dashboard.view_models import DashboardTabsViewModel


@dataclass(frozen=True, slots=True)
class CallStatusHeader:
    state: str
    masked_call_id: str
    progress: str
    transcript_revision: str
    current_risk: str


@dataclass(frozen=True, slots=True)
class RepresentativeKPI:
    label: str
    value: str


def mask_call_identifier(call_id: str, *, visible_suffix: int = 4) -> str:
    """Mask a call identifier while retaining a short operational suffix."""
    cleaned = call_id.strip()
    if not cleaned:
        return "••••"
    suffix_length = min(max(visible_suffix, 0), len(cleaned))
    suffix = cleaned[-suffix_length:] if suffix_length else ""
    return f"••••{suffix}"


def call_status_header(
    view: DashboardTabsViewModel,
    *,
    call_id: str,
    transcript_revision: int,
) -> CallStatusHeader:
    representative = view.representative
    status_values = {item.label: item.value for item in representative.status}
    progress = representative.progress
    progress_text = (
        f"{progress.completed_chunks}/{progress.total_chunks} parça"
        if progress.total_chunks
        else progress.elapsed
    )
    risks = tuple(chip.text for chip in representative.intent_chips if chip.is_risk)
    if risks:
        current_risk = " · ".join(risks)
    elif representative.intent_chips:
        current_risk = "Aktif risk yok"
    else:
        current_risk = "Henüz risk veya niyet yok"
    return CallStatusHeader(
        state=status_values.get("Durum", progress.stage),
        masked_call_id=mask_call_identifier(call_id),
        progress=progress_text,
        transcript_revision=str(transcript_revision),
        current_risk=current_risk,
    )


def representative_kpis(
    view: DashboardTabsViewModel,
) -> tuple[RepresentativeKPI, ...]:
    representative = view.representative
    return (
        RepresentativeKPI(
            "Kesin transkript",
            "Hazır" if representative.transcript.stable_text else "Bekleniyor",
        ),
        RepresentativeKPI(
            "Aktif niyet / risk",
            str(len(representative.intent_chips)),
        ),
        RepresentativeKPI(
            "Aktif koçluk",
            str(len(representative.active_suggestions)),
        ),
        RepresentativeKPI("ASR sağlığı", _asr_health(view)),
    )


def _asr_health(view: DashboardTabsViewModel) -> str:
    technical = view.technical
    if technical.last_asr != "—":
        return technical.last_asr
    asr_status = next(
        (
            status
            for component, status in technical.pipeline_statuses
            if component == "ASR"
        ),
        "disabled",
    )
    return {
        "failed": "Kullanılamıyor",
        "disabled": "Kullanılamıyor",
        "simulated": "Simülasyon",
        "active": "Ölçüm bekleniyor",
    }.get(asr_status, "Bekleniyor")
