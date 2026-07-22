from live_dashboard.demo_data import scenario_for, tenant_demos
from live_dashboard.view_models import (
    advance_runtime,
    create_runtime,
    ordered_suggestions,
    reset_runtime,
    suppression_reason_display,
    transcript_view,
)


def runtime(tenant_id: str = "tenant_alpha", scenario_id: str = "cancel"):
    tenant = tenant_demos()[tenant_id]
    return create_runtime(tenant, scenario_for(tenant_id, scenario_id), "test-call")


def test_partial_updates_transcript_without_coaching_card() -> None:
    subject = runtime()
    advance_runtime(subject)
    view = transcript_view(subject)
    assert view.partial_text == "Aboneliğimi"
    assert view.stable_text == ""
    assert view.partial_is_changeable
    assert subject.suggestions == []
    assert subject.latest_labels == ()


def test_stable_event_creates_formatted_coaching_card() -> None:
    subject = runtime()
    advance_runtime(subject)
    advance_runtime(subject)
    card = subject.suggestions[0]
    assert (card.priority_text, card.title, card.action) == (
        "HIGH",
        "İptal talebini doğrulayın",
        "TEMPLATE_ACTION",
    )
    assert card.suggestion
    assert card.timestamp.count(":") == 2
    assert transcript_view(subject).latest_event_type == "STABLE"


def test_priority_ordering_places_critical_first() -> None:
    normal = runtime("tenant_alpha", "price")
    critical = runtime("tenant_alpha", "critical")
    for subject in (normal, critical):
        while advance_runtime(subject) is not None:
            pass
    cards = ordered_suggestions([*normal.suggestions, *critical.suggestions])
    assert [card.priority_text for card in cards] == ["CRITICAL", "HIGH"]


def test_timeline_is_chronological_and_has_required_event_types() -> None:
    subject = runtime()
    while advance_runtime(subject) is not None:
        pass
    assert subject.timeline == sorted(subject.timeline, key=lambda item: item.timestamp)
    assert {item.event_type for item in subject.timeline} >= {
        "Transkript",
        "Sınıflandırma",
        "Öneri gösterildi",
        "Öneri bastırıldı",
    }


def test_duplicate_suppression_reason_is_displayed() -> None:
    subject = runtime()
    while advance_runtime(subject) is not None:
        pass
    assert subject.suppression_reasons == ["yinelenen öneri"]
    assert suppression_reason_display("cooldown") == "bekleme süresi"


def test_tenant_labels_rules_and_state_are_isolated() -> None:
    alpha = runtime("tenant_alpha", "price")
    beta = runtime("tenant_beta", "price")
    for subject in (alpha, beta):
        advance_runtime(subject)
        advance_runtime(subject)
    assert alpha.call_state.tenant_id != beta.call_state.tenant_id
    assert alpha.latest_labels[0].name == "fiyat_itirazi"
    assert beta.latest_labels[0].name == "butce_endisesi"
    assert alpha.suggestions[0].title != beta.suggestions[0].title


def test_negated_cancellation_does_not_trigger() -> None:
    subject = runtime("tenant_alpha", "negated")
    while advance_runtime(subject) is not None:
        pass
    assert subject.suggestions == []
    assert subject.latest_labels == ()


def test_reset_creates_clean_call_state_and_coordinator() -> None:
    subject = runtime()
    advance_runtime(subject)
    cleaned = reset_runtime(subject)
    assert cleaned is not subject
    assert (
        cleaned.call_state.stable_transcript
        == cleaned.call_state.partial_transcript
        == ""
    )
    assert cleaned.next_event_index == 0
    assert cleaned.suggestions == cleaned.timeline == []


def test_view_models_contain_no_audio_or_private_data() -> None:
    subject = runtime("tenant_beta", "product")
    while advance_runtime(subject) is not None:
        pass
    rendered = repr(subject)
    assert "audio_bytes" not in rendered
    assert "CallMetricPrivate" not in rendered
    assert "müşteri adı" not in rendered.casefold()
