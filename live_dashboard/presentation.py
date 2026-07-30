"""Pure presentation helpers for the live coaching dashboard."""

from collections.abc import MutableMapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any, Generic, TypeVar

from live_dashboard.rag_runtime import DashboardRAGRuntimeStatus
from live_dashboard.view_models import DashboardTabsViewModel, SuggestionCardViewModel


T = TypeVar("T")

VISIBLE_TRANSCRIPT_CHARACTERS = 6_000
VISIBLE_TRANSCRIPT_LINES = 80
VISIBLE_ACTIVE_SUGGESTIONS = 3
VISIBLE_HISTORY_SUGGESTIONS = 5
VISIBLE_TIMELINE_ROWS = 12
VISIBLE_TECHNICAL_ROWS = 12
_SCOPE_STATE_KEY = "_live_dashboard_ui_scope"
_SCOPED_WIDGET_PREFIX = "_live_dashboard_widget_"


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


@dataclass(frozen=True, slots=True)
class UIScopeIdentity:
    key: str


@dataclass(frozen=True, slots=True)
class BoundedText:
    visible_text: str
    hidden_character_count: int


@dataclass(frozen=True, slots=True)
class BoundedItems(Generic[T]):
    visible_items: tuple[T, ...]
    hidden_item_count: int


class OperationalState(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class OperationalStatus:
    state: OperationalState
    label: str
    detail: str


def rag_runtime_status_text(status: object) -> str:
    """Map the immutable runtime state to fixed, secret-safe visible text."""
    if status is DashboardRAGRuntimeStatus.READY:
        return "RAG hazır"
    if status is DashboardRAGRuntimeStatus.DISABLED:
        return "RAG devre dışı"
    return "RAG geçici olarak kullanılamıyor; temel görüşme analizi devam ediyor"


def ui_scope_identity(
    *,
    tenant_id: str,
    call_id: str,
    source_mode: str,
) -> UIScopeIdentity:
    """Return a non-reversible identity containing only trusted UI scope."""
    values = tuple(value.strip() for value in (tenant_id, call_id, source_mode))
    if any(not value for value in values):
        raise ValueError("UI scope values cannot be empty")
    digest = sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:20]
    return UIScopeIdentity(digest)


def synchronize_ui_scope(
    session_state: MutableMapping[Any, Any],
    scope: UIScopeIdentity,
) -> bool:
    """Reset presentation-only state when trusted UI scope changes."""
    if session_state.get(_SCOPE_STATE_KEY) == scope.key:
        return False
    for key in tuple(session_state):
        if isinstance(key, str) and key.startswith(_SCOPED_WIDGET_PREFIX):
            session_state.pop(key, None)
    session_state[_SCOPE_STATE_KEY] = scope.key
    session_state["suggestion_feedback"] = {}
    session_state["playing"] = False
    session_state.pop("safe_audio_metadata", None)
    return True


def scoped_widget_key(scope: UIScopeIdentity, purpose: str) -> str:
    cleaned = purpose.strip()
    if not cleaned:
        raise ValueError("widget purpose cannot be empty")
    purpose_digest = sha256(cleaned.encode("utf-8")).hexdigest()[:12]
    return f"{_SCOPED_WIDGET_PREFIX}{scope.key}_{purpose_digest}"


def coaching_feedback_key(
    scope: UIScopeIdentity,
    card: SuggestionCardViewModel,
) -> str:
    """Build a stable card key independent of display order and content."""
    revision = (
        "none" if card.transcript_revision is None else str(card.transcript_revision)
    )
    material = "\x1f".join(
        (
            scope.key,
            card.suggestion_id,
            revision,
            card.source,
            card.timestamp,
        )
    )
    return f"feedback-{sha256(material.encode('utf-8')).hexdigest()[:24]}"


def bounded_text_tail(
    text: str,
    *,
    maximum_characters: int = VISIBLE_TRANSCRIPT_CHARACTERS,
    maximum_lines: int = VISIBLE_TRANSCRIPT_LINES,
) -> BoundedText:
    if maximum_characters <= 0 or maximum_lines <= 0:
        raise ValueError("text bounds must be positive")
    lines = text.splitlines(keepends=True)
    line_bounded = "".join(lines[-maximum_lines:])
    visible = line_bounded[-maximum_characters:]
    return BoundedText(
        visible_text=visible,
        hidden_character_count=max(len(text) - len(visible), 0),
    )


def bounded_items(
    items: tuple[T, ...],
    *,
    limit: int,
    newest: bool = False,
) -> BoundedItems[T]:
    if limit <= 0:
        raise ValueError("item limit must be positive")
    visible = items[-limit:] if newest else items[:limit]
    return BoundedItems(
        visible_items=visible,
        hidden_item_count=max(len(items) - len(visible), 0),
    )


def operational_status(view: DashboardTabsViewModel) -> OperationalStatus:
    technical = view.technical
    representative = view.representative
    asr_status = next(
        (
            status
            for component, status in technical.pipeline_statuses
            if component == "ASR"
        ),
        "disabled",
    )
    if technical.error is not None or asr_status == "failed":
        return OperationalStatus(
            OperationalState.FAILED,
            "İşlem başarısız",
            "Ses işleme güvenli biçimde tamamlanamadı.",
        )
    if representative.safe_messages or any(
        status == "failed"
        for component, status in technical.pipeline_statuses
        if component != "ASR"
    ):
        return OperationalStatus(
            OperationalState.DEGRADED,
            "Sınırlı hizmet",
            "Bazı yardımcı hizmetler kullanılamıyor; mevcut işlem devam ediyor.",
        )
    if view.result.completed:
        return OperationalStatus(
            OperationalState.COMPLETED,
            "Tamamlandı",
            "Görüşme işleme adımları tamamlandı.",
        )
    waiting_states = {
        "başlatılmadı",
        "demo hazır",
        "hazır",
        "bekliyor",
    }
    state_value = next(
        (item.value for item in representative.status if item.label == "Durum"),
        representative.progress.stage,
    )
    if (
        representative.progress.completed_chunks == 0
        and state_value.casefold() in waiting_states
    ):
        return OperationalStatus(
            OperationalState.WAITING,
            "Başlatılmayı bekliyor",
            "Henüz işlenen bir ses parçası yok.",
        )
    return OperationalStatus(
        OperationalState.RUNNING,
        "İşleniyor",
        "Görüşme verileri işleniyor.",
    )


def safe_failure_rows(
    view: DashboardTabsViewModel,
) -> tuple[tuple[str, str], ...]:
    """Map internal failure metadata to fixed, non-sensitive categories."""
    technical = view.technical
    if technical.error is None and not technical.failure_details:
        return ()
    details = dict(technical.failure_details)
    component = details.get("Bileşen", "")
    category = {
        "ASR": "asr_processing_failed",
        "SetFit": "classification_unavailable",
        "Classification": "classification_unavailable",
        "Coaching": "coaching_unavailable",
    }.get(component, "processing_failed")
    return (
        ("Kategori", category),
        ("Durum", "İşlem güvenli biçimde tamamlanamadı."),
    )


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
