from dataclasses import replace

from app.events.models import SuggestionPriority
from live_dashboard.demo_data import tenant_demos
from live_dashboard.presentation import (
    OperationalState,
    bounded_items,
    bounded_text_tail,
    call_status_header,
    coaching_feedback_key,
    mask_call_identifier,
    operational_status,
    representative_kpis,
    safe_failure_rows,
    scoped_widget_key,
    synchronize_ui_scope,
    ui_scope_identity,
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


def _card(*, revision: int, suggestion_id: str = "shared-id"):
    return SuggestionCardViewModel(
        suggestion_id=suggestion_id,
        priority=SuggestionPriority.HIGH,
        priority_text="HIGH",
        title="Sentetik güvenli başlık",
        suggestion="Sentetik güvenli öneri.",
        action="Bilgilendir",
        timestamp="12:00:00",
        related_label="İptal Talebi",
        evidence_ids=(),
        priority_symbol="▲",
        source="Kural",
        transcript_revision=revision,
        is_new=True,
    )


def test_ui_scope_is_deterministic_and_contains_no_private_values() -> None:
    first = ui_scope_identity(
        tenant_id="tenant_alpha",
        call_id="private-call-001",
        source_mode="uploaded",
    )
    second = ui_scope_identity(
        tenant_id="tenant_alpha",
        call_id="private-call-001",
        source_mode="uploaded",
    )
    other_call = ui_scope_identity(
        tenant_id="tenant_alpha",
        call_id="private-call-002",
        source_mode="uploaded",
    )

    assert first == second
    assert first != other_call
    assert "tenant_alpha" not in first.key
    assert "private-call" not in first.key


def test_feedback_keys_survive_reordering_and_separate_scope_and_revision() -> None:
    scope = ui_scope_identity(
        tenant_id="tenant_alpha",
        call_id="call-001",
        source_mode="synthetic",
    )
    other_scope = ui_scope_identity(
        tenant_id="tenant_beta",
        call_id="call-001",
        source_mode="synthetic",
    )
    revision_one = _card(revision=1)
    revision_two = _card(revision=2)

    before = {
        (card.suggestion_id, card.transcript_revision): coaching_feedback_key(
            scope, card
        )
        for card in (revision_one, revision_two)
    }
    after = {
        (card.suggestion_id, card.transcript_revision): coaching_feedback_key(
            scope, card
        )
        for card in (revision_two, revision_one)
    }

    assert coaching_feedback_key(scope, revision_one) == coaching_feedback_key(
        scope, revision_one
    )
    assert (
        len(
            {
                coaching_feedback_key(scope, revision_one),
                coaching_feedback_key(scope, revision_two),
                coaching_feedback_key(other_scope, revision_one),
            }
        )
        == 3
    )
    assert set(before.values()) == set(after.values())


def test_scope_switch_clears_only_presentation_state() -> None:
    first = ui_scope_identity(
        tenant_id="tenant_alpha",
        call_id="call-001",
        source_mode="uploaded",
    )
    second = ui_scope_identity(
        tenant_id="tenant_beta",
        call_id="call-002",
        source_mode="uploaded",
    )
    trusted_runtime = object()
    session: dict[str, object] = {
        "trusted_runtime": trusted_runtime,
        "uploaded_audio_session": object(),
    }
    assert synchronize_ui_scope(session, first)
    session["suggestion_feedback"] = {"old": "Görüldü"}
    session["safe_audio_metadata"] = object()
    session["playing"] = True
    old_widget_key = scoped_widget_key(first, "history")
    session[old_widget_key] = True

    assert synchronize_ui_scope(session, second)

    assert session["trusted_runtime"] is trusted_runtime
    assert "uploaded_audio_session" in session
    assert session["suggestion_feedback"] == {}
    assert session["playing"] is False
    assert "safe_audio_metadata" not in session
    assert old_widget_key not in session
    assert not synchronize_ui_scope(session, second)


def test_bounded_projections_preserve_source_values() -> None:
    text = "\n".join(f"line-{index}" for index in range(100))
    text_snapshot = text
    bounded_text = bounded_text_tail(
        text,
        maximum_characters=60,
        maximum_lines=5,
    )
    items = tuple(range(20))
    items_snapshot = items

    first = bounded_items(items, limit=4, newest=True)
    second = bounded_items(items, limit=4, newest=True)

    assert text == text_snapshot
    assert bounded_text.hidden_character_count > 0
    assert bounded_text.visible_text.endswith("line-99")
    assert items == items_snapshot
    assert first == second
    assert first.visible_items == (16, 17, 18, 19)
    assert first.hidden_item_count == 16


def test_operational_and_failure_statuses_are_fixed_and_safe() -> None:
    _, tabs = _tabs()
    assert operational_status(tabs).state is OperationalState.WAITING
    failed = replace(
        tabs,
        technical=replace(
            tabs.technical,
            error="PRIVATE C:/secret/cache exception",
            failure_details=(
                ("Hata kodu", "PRIVATE_EXCEPTION"),
                ("Bileşen", "ASR"),
            ),
        ),
    )

    status = operational_status(failed)
    rows = safe_failure_rows(failed)
    rendered = repr((status, rows))

    assert status.state is OperationalState.FAILED
    assert rows == (
        ("Kategori", "asr_processing_failed"),
        ("Durum", "İşlem güvenli biçimde tamamlanamadı."),
    )
    assert "PRIVATE" not in rendered
    assert "secret" not in rendered
