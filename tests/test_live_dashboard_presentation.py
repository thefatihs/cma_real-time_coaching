from dataclasses import replace

from app.events.models import SuggestionPriority
from live_dashboard.demo_data import tenant_demos
from live_dashboard.presentation import (
    call_status_header,
    mask_call_identifier,
    representative_kpis,
)
from live_dashboard.view_models import (
    create_local_execution,
    dashboard_tabs,
    IntentChipViewModel,
    SuggestionCardViewModel,
)


def _tabs():
    state = create_local_execution(tenant_demos()["tenant_alpha"], "private-call-9876")
    return state, dashboard_tabs(state.runtime, state)


def test_call_identifier_is_masked_deterministically() -> None:
    assert mask_call_identifier("private-call-9876") == "••••9876"
    assert mask_call_identifier("abc") == "••••abc"
    assert mask_call_identifier("  ") == "••••"
    assert mask_call_identifier("private-call-9876") == mask_call_identifier(
        "private-call-9876"
    )
    assert "private-call" not in mask_call_identifier("private-call-9876")


def test_header_uses_existing_scope_progress_revision_and_risk() -> None:
    state, tabs = _tabs()
    representative = replace(
        tabs.representative,
        intent_chips=(IntentChipViewModel("İptal Talebi", "", True, "▲"),),
    )

    header = call_status_header(
        replace(tabs, representative=representative),
        call_id=state.runtime.call_id,
        transcript_revision=state.runtime.call_state.transcript_revision,
    )

    assert header.masked_call_id == "••••9876"
    assert header.state == state.stage
    assert header.progress == tabs.representative.progress.elapsed
    assert header.transcript_revision == str(
        state.runtime.call_state.transcript_revision
    )
    assert header.current_risk == "İptal Talebi"


def test_kpis_use_real_view_model_fields_without_percentages() -> None:
    _, tabs = _tabs()
    card = SuggestionCardViewModel(
        suggestion_id="safe-card",
        priority=SuggestionPriority.HIGH,
        priority_text="HIGH",
        title="Güvenli öneri",
        suggestion="Görüşmeyi dikkatle sürdürün.",
        action="Bilgilendir",
        timestamp="12:00:00",
        related_label="İptal Talebi",
        evidence_ids=(),
        priority_symbol="▲",
        source="Kural",
        transcript_revision=1,
        is_new=True,
    )
    representative = replace(
        tabs.representative,
        transcript=replace(
            tabs.representative.transcript,
            stable_text="Sentetik kesin metin.",
        ),
        intent_chips=(
            IntentChipViewModel("İptal Talebi", "", True, "▲"),
            IntentChipViewModel("Teknik Sorun", "", False, "●"),
        ),
        active_suggestions=(card,),
    )
    technical = replace(tabs.technical, last_asr="84 ms")

    kpis = representative_kpis(
        replace(tabs, representative=representative, technical=technical)
    )

    assert [(item.label, item.value) for item in kpis] == [
        ("Kesin transkript", "Hazır"),
        ("Aktif niyet / risk", "2"),
        ("Aktif koçluk", "1"),
        ("ASR sağlığı", "84 ms"),
    ]
    assert all("%" not in item.value for item in kpis)


def test_empty_kpis_and_header_are_professional_and_repeatable() -> None:
    state, tabs = _tabs()

    first_header = call_status_header(
        tabs,
        call_id=state.runtime.call_id,
        transcript_revision=state.runtime.call_state.transcript_revision,
    )
    second_header = call_status_header(
        tabs,
        call_id=state.runtime.call_id,
        transcript_revision=state.runtime.call_state.transcript_revision,
    )

    assert first_header == second_header
    assert first_header.current_risk == "Henüz risk veya niyet yok"
    assert [(item.label, item.value) for item in representative_kpis(tabs)] == [
        ("Kesin transkript", "Bekleniyor"),
        ("Aktif niyet / risk", "0"),
        ("Aktif koçluk", "0"),
        ("ASR sağlığı", "Ölçüm bekleniyor"),
    ]
