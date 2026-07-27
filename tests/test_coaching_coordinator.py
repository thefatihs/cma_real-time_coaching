from datetime import UTC, datetime
from itertools import count

import pytest

from app.calls.models import CallState
from app.coaching.coordinator import (
    CoachingCoordinator,
    ExternalSuggestionAdmissionReason,
    ExternalSuggestionAdmissionStatus,
)
from app.coaching.rule_engine import CoachingRule, RuleBasedCoachingEngine
from app.events.models import (
    ClassificationLabel,
    ClassificationResultEvent,
    CoachingAction,
    CoachingSuggestionSource,
    SuggestionPriority,
    TranscriptEvent,
    TranscriptKind,
)
from app.tenancy.models import (
    TenantASRConfig,
    TenantClassificationConfig,
    TenantCoachingConfig,
    TenantConfig,
    TenantContext,
    TenantRAGConfig,
)


NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def config(
    *, cooldown: float = 10.0, maximum: int = 2, tenant_id: str = "tenant_alpha"
) -> TenantConfig:
    return TenantConfig(
        context=TenantContext(tenant_id=tenant_id, tenant_name="Synthetic"),
        asr=TenantASRConfig(),
        classification=TenantClassificationConfig(
            model_id="rules-v1",
            labels=[
                "iade",
                "acil",
                "bilgi",
                "product_information",
                "complaint",
                "churn_risk",
                "iptal_riski",
            ],
        ),
        rag=TenantRAGConfig(enabled=False),
        coaching=TenantCoachingConfig(
            cooldown_seconds=cooldown,
            max_active_suggestions=maximum,
            allowed_actions=[action.value for action in CoachingAction],
        ),
    )


def rule(
    rule_id: str = "rule_1",
    *,
    label: str = "iade",
    phrase: str = "iade",
    action: CoachingAction = CoachingAction.TEMPLATE_ACTION,
    title: str = "İade desteği",
) -> CoachingRule:
    return CoachingRule(
        rule_id=rule_id,
        label=label,
        include_any=(phrase,),
        action=action,
        priority=SuggestionPriority.HIGH,
        title=title,
        suggestion=f"{title} adımlarını uygulayın.",
        evidence_ids=(f"{rule_id}_evidence",),
    )


def event(
    kind: TranscriptKind = TranscriptKind.STABLE, **changes: object
) -> TranscriptEvent:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "event_id": "transcript_1",
        "kind": kind,
        "text": "İade ve acil bilgi talebi.",
        "start_seconds": 0.0,
        "end_seconds": 2.0,
        "revision": 1,
        "created_at_utc": NOW,
    }
    values.update(changes)
    return TranscriptEvent.model_validate(values)


def coordinator(
    *rules: CoachingRule,
    tenant: TenantConfig | None = None,
    state: CallState | None = None,
) -> tuple[CoachingCoordinator, CallState]:
    tenant = tenant or config()
    state = state or CallState(tenant_id=tenant.context.tenant_id, call_id="call_001")
    identifiers = count(1)
    engine = RuleBasedCoachingEngine(
        tenant,
        tuple(rules),
        event_id_factory=lambda: f"suggestion_{next(identifiers)}",
        utc_datetime_factory=lambda: NOW,
    )
    return CoachingCoordinator(tenant, state, engine), state


def classification(*labels: str) -> ClassificationResultEvent:
    return ClassificationResultEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        transcript_event_id="transcript_1",
        labels=[ClassificationLabel(name=label, score=0.8) for label in labels],
        action=CoachingAction.TEMPLATE_ACTION,
        model_id="common_turkish_setfit_v2",
        threshold_profile_id="common_turkish_setfit_v2:calibrated:v1",
        created_at_utc=NOW,
    )


def external_candidate(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "transcript_revision": 1,
        "label_id": "cancellation_request",
        "action": CoachingAction.TEMPLATE_ACTION,
        "title": "Synthetic retention guidance",
        "suggestion": "Apply the approved synthetic retention steps.",
        "priority": SuggestionPriority.HIGH,
        "source": CoachingSuggestionSource.LLM,
    }
    values.update(changes)
    return values


def test_partial_is_ignored() -> None:
    subject, state = coordinator(rule())
    result = subject.process(event(TranscriptKind.PARTIAL), 1.0)
    assert result.classification_event is None
    assert result.displayed_suggestions == result.suppressed_suggestions == ()
    assert state.active_labels == []


@pytest.mark.parametrize("kind", [TranscriptKind.STABLE, TranscriptKind.FINAL])
def test_stable_and_final_suggestions_are_displayed(kind: TranscriptKind) -> None:
    subject, state = coordinator(rule())
    result = subject.process(event(kind), 1.0)
    assert len(result.displayed_suggestions) == 1
    assert result.suppressed_suggestions == ()
    assert state.shown_suggestion_ids == [result.displayed_suggestions[0].suggestion_id]
    assert state.last_coaching_trigger_seconds == 1.0


def test_tenant_and_call_mismatches_are_rejected() -> None:
    subject, _ = coordinator(rule())
    with pytest.raises(ValueError, match="tenant_id"):
        subject.process(event(tenant_id="tenant_beta"), 1.0)
    with pytest.raises(ValueError, match="call_id"):
        subject.process(event(call_id="call_002"), 1.0)


def test_initialization_rejects_engine_or_state_from_another_tenant() -> None:
    primary = config()
    other = config(tenant_id="tenant_beta")
    other_engine = RuleBasedCoachingEngine(other, (rule(),))
    with pytest.raises(ValueError, match="tenant_id"):
        CoachingCoordinator(
            primary,
            CallState(tenant_id="tenant_alpha", call_id="call_001"),
            other_engine,
        )
    with pytest.raises(ValueError, match="tenant_id"):
        coordinator(
            rule(),
            tenant=primary,
            state=CallState(tenant_id="tenant_beta", call_id="call_001"),
        )


def test_active_labels_are_updated() -> None:
    subject, state = coordinator(
        rule(label="complaint"),
        rule("rule_2", label="churn_risk", phrase="acil"),
    )
    result = subject.process(event(), 1.0)
    assert result.classification_event is not None
    assert state.active_labels == [
        label.name for label in result.classification_event.labels
    ]


def test_duplicate_suggestion_is_suppressed_and_classification_preserved() -> None:
    subject, _ = coordinator(rule(), tenant=config(cooldown=0))
    subject.process(event(), 1.0)
    result = subject.process(event(event_id="transcript_2", revision=2), 2.0)
    assert result.classification_event is not None
    assert result.displayed_suggestions == ()
    assert len(result.suppressed_suggestions) == 1
    assert result.suppression_reasons == ("duplicate_same_revision",)


def test_cooldown_suppression_then_allowed_after_cooldown() -> None:
    subject, state = coordinator(
        rule("first", phrase="iade"),
        rule("second", phrase="acil", title="Acil destek"),
    )
    first = subject.process(event(text="iade"), 1.0)
    assert len(first.displayed_suggestions) == 1
    blocked = subject.process(
        event(event_id="transcript_2", text="acil", revision=2), 5.0
    )
    assert blocked.suppression_reasons == ("cooldown_previously_displayed",)
    allowed = subject.process(
        event(event_id="transcript_3", text="acil", revision=3), 11.0
    )
    assert len(allowed.displayed_suggestions) == 1
    assert state.last_coaching_trigger_seconds == 11.0


def test_maximum_suggestion_count_suppresses_excess() -> None:
    subject, _ = coordinator(
        rule("one", phrase="iade"),
        rule("two", label="acil", phrase="acil"),
        rule("three", label="bilgi", phrase="bilgi"),
        tenant=config(maximum=2),
    )
    result = subject.process(event(), 1.0)
    assert len(result.displayed_suggestions) == 2
    assert len(result.suppressed_suggestions) == 1
    assert result.suppression_reasons == ("rejected_by_capacity",)


def test_rag_classification_passes_without_suggestion() -> None:
    subject, state = coordinator(rule(action=CoachingAction.RAG_ACTION))
    result = subject.process(event(), 1.0)
    assert result.classification_event is not None
    assert result.classification_event.action is CoachingAction.RAG_ACTION
    assert result.displayed_suggestions == result.suppressed_suggestions == ()
    assert state.last_coaching_trigger_seconds is None


def test_escalation_suggestion_is_displayed() -> None:
    subject, _ = coordinator(rule(action=CoachingAction.ESCALATE))
    result = subject.process(event(), 1.0)
    assert result.displayed_suggestions[0].action is CoachingAction.ESCALATE


def test_no_match_returns_empty_result_and_clears_active_labels() -> None:
    subject, state = coordinator(rule())
    state.active_labels = ["iade"]
    result = subject.process(event(text="eşleşme yok"), 1.0)
    assert result.classification_event is None
    assert result.matched_rule_ids == ()
    assert state.active_labels == []


def test_clear_allows_content_again_without_replacing_call_state() -> None:
    subject, state = coordinator(rule(), tenant=config(cooldown=0))
    subject.process(event(), 1.0)
    subject.clear()
    result = subject.process(event(event_id="transcript_2", revision=2), 2.0)
    assert len(result.displayed_suggestions) == 1
    assert state.shown_suggestion_ids == ["suggestion_1", "suggestion_2"]


def test_negative_time_is_rejected() -> None:
    subject, _ = coordinator(rule())
    with pytest.raises(ValueError, match="current_seconds"):
        subject.process(event(), -0.1)


def test_source_event_and_rule_remain_unchanged() -> None:
    source_rule = rule()
    source_event = event()
    before = (source_rule.model_dump(), source_event.model_dump())
    subject, _ = coordinator(source_rule)
    subject.process(source_event, 1.0)
    assert before == (source_rule.model_dump(), source_event.model_dump())


def test_classification_only_and_agreement_preserve_provenance() -> None:
    subject, _ = coordinator(rule())
    classification_only = subject.process(
        event(text="farkli bir ifade"),
        1.0,
        classification_event=classification("iade"),
        active_labels=("iade",),
    )
    assert classification_only.displayed_suggestions[0].source is (
        CoachingSuggestionSource.CLASSIFICATION
    )

    second, _ = coordinator(rule())
    agreement = second.process(
        event(text="iade"),
        1.0,
        classification_event=classification("iade"),
        active_labels=("iade",),
    )
    assert agreement.displayed_suggestions[0].source is CoachingSuggestionSource.BOTH


def test_explicit_rule_survives_missed_classification_and_stores_safe_metadata() -> (
    None
):
    subject, state = coordinator(rule(action=CoachingAction.ESCALATE))
    result = subject.process(
        event(text="iade"),
        1.0,
        classification_event=classification(),
        active_labels=(),
    )
    suggestion = result.displayed_suggestions[0]
    assert suggestion.source is CoachingSuggestionSource.RULE
    assert suggestion.action is CoachingAction.ESCALATE
    assert state.coaching_suggestions[0].suggestion_id == suggestion.suggestion_id
    assert state.coaching_suggestions[0].transcript_revision == 1
    assert state.coaching_suggestions[0].model_id is None
    assert not hasattr(state.coaching_suggestions[0], "text")
    assert not hasattr(state.coaching_suggestions[0], "probabilities")


def test_same_revision_is_processed_only_once() -> None:
    subject, _ = coordinator(rule(), tenant=config(cooldown=0))
    first = subject.process(event(text="iade"), 1.0)
    duplicate = subject.process(event(text="iade"), 2.0)
    assert len(first.displayed_suggestions) == 1
    assert duplicate.displayed_suggestions == ()
    assert duplicate.suppression_reasons == ("duplicate_revision",)


EXACT_CANCELLATION = (
    "Aboneliğimi bugün iptal ettirmek istiyorum. Lütfen iptal işlemini başlatın."
)


def test_cancellation_classification_without_text_rule_creates_suggestion() -> None:
    subject, _ = coordinator()
    result = subject.process(
        event(text="Üyeliğimin sonlandırılmasını değerlendiriyorum."),
        1.0,
        classification_event=classification("cancellation_request"),
        active_labels=("cancellation_request",),
    )
    assert len(result.displayed_suggestions) == 1
    assert result.suppressed_suggestions == ()
    assert result.suppression_reasons == ()
    suggestion = result.displayed_suggestions[0]
    assert suggestion.source is CoachingSuggestionSource.CLASSIFICATION
    assert "iptal nedenini netleştirin" in suggestion.suggestion


def test_exact_cancellation_rule_only_and_both_provenance() -> None:
    rule_subject, rule_state = coordinator()
    rule_only = rule_subject.process(event(text=EXACT_CANCELLATION), 1.0)
    assert len(rule_only.displayed_suggestions) == 1
    assert rule_only.displayed_suggestions[0].source is CoachingSuggestionSource.RULE
    assert [item.label for item in rule_state.detected_labels] == [
        "cancellation_request"
    ]
    assert rule_state.detected_labels[0].source is CoachingSuggestionSource.RULE

    combined_subject, combined_state = coordinator()
    combined = combined_subject.process(
        event(text=EXACT_CANCELLATION),
        1.0,
        classification_event=classification("cancellation_request"),
        active_labels=("cancellation_request",),
    )
    assert len(combined.displayed_suggestions) == 1
    assert combined.displayed_suggestions[0].source is CoachingSuggestionSource.BOTH
    assert combined.suppressed_suggestions == ()
    assert combined_state.detected_labels[0].source is CoachingSuggestionSource.BOTH
    assert combined_state.label_revision_timeline[0].evidence[0].source is (
        CoachingSuggestionSource.BOTH
    )


def test_internal_cancellation_rule_label_is_never_stored() -> None:
    subject, state = coordinator(
        rule(label="iptal_riski", phrase="iptal etmek istiyorum")
    )
    result = subject.process(
        event(text="Aboneliğimi iptal etmek istiyorum."),
        1.0,
        classification_event=classification(),
        active_labels=(),
    )
    assert result.current_revision_labels == ("cancellation_request",)
    assert result.displayed_suggestions[0].label_id == "cancellation_request"
    assert state.active_labels == ["cancellation_request"]
    assert [item.label for item in state.detected_labels] == ["cancellation_request"]
    assert "iptal_riski" not in repr(state)


def test_cancellation_suggestion_is_not_displayed_twice() -> None:
    subject, _ = coordinator(tenant=config(cooldown=0))
    first = subject.process(
        event(text=EXACT_CANCELLATION),
        1.0,
        classification_event=classification("cancellation_request"),
        active_labels=("cancellation_request",),
    )
    duplicate = subject.process(
        event(
            event_id="transcript_2",
            revision=2,
            text=EXACT_CANCELLATION,
        ),
        2.0,
        classification_event=classification("cancellation_request").model_copy(
            update={"transcript_event_id": "transcript_2"}
        ),
        active_labels=("cancellation_request",),
    )
    assert len(first.displayed_suggestions) == 1
    assert first.suppressed_suggestions == ()
    assert duplicate.displayed_suggestions == ()
    assert duplicate.suppression_reasons == ("duplicate_same_revision",)


def test_short_call_suggestion_provenance_and_deduplication_are_unchanged() -> None:
    subject, _ = coordinator(tenant=config(cooldown=0))
    first = subject.process(
        event(text=EXACT_CANCELLATION),
        1.0,
        classification_event=classification("cancellation_request"),
        active_labels=("cancellation_request",),
    )
    repeated = subject.process(
        event(
            event_id="transcript_2",
            revision=2,
            text=EXACT_CANCELLATION,
        ),
        2.0,
        classification_event=classification("cancellation_request").model_copy(
            update={"transcript_event_id": "transcript_2"}
        ),
        active_labels=("cancellation_request",),
    )

    assert first.displayed_suggestions[0].source is CoachingSuggestionSource.BOTH
    assert repeated.displayed_suggestions == ()
    assert repeated.suppression_reasons == ("duplicate_same_revision",)


def test_accumulated_old_label_does_not_trigger_new_coaching() -> None:
    subject, state = coordinator(tenant=config(cooldown=0))
    state.record_detected_labels(
        ["complaint"],
        transcript_revision=1,
        source=CoachingSuggestionSource.CLASSIFICATION,
        model_id="synthetic-classifier",
    )
    result = subject.process(
        event(text="Yeni revizyonda eşleşen bir sinyal yok."),
        2.0,
        classification_event=classification(),
        active_labels=(),
    )
    assert [item.label for item in state.detected_labels] == ["complaint"]
    assert result.displayed_suggestions == ()
    assert result.suppressed_suggestions == ()


def test_long_call_current_critical_suggestions_replace_older_active_cards() -> None:
    subject, state = coordinator(tenant=config(cooldown=20, maximum=2))
    price = subject.process(
        event(text="Bu ücret çok pahalı.", revision=7),
        7.0,
        classification_event=classification("price_objection"),
        active_labels=("price_objection",),
    )
    complaint = subject.process(
        event(event_id="transcript_13", text="Şikayetçiyim.", revision=13),
        13.0,
        classification_event=classification("complaint").model_copy(
            update={"transcript_event_id": "transcript_13"}
        ),
        active_labels=("complaint",),
    )
    final = subject.process(
        event(
            event_id="transcript_15",
            text=EXACT_CANCELLATION,
            revision=15,
        ),
        15.0,
        classification_event=classification(
            "churn_risk", "cancellation_request"
        ).model_copy(update={"transcript_event_id": "transcript_15"}),
        active_labels=("churn_risk", "cancellation_request"),
    )

    assert price.displayed_suggestions[0].label_id == "price_objection"
    assert complaint.displayed_suggestions[0].label_id == "complaint"
    assert {item.label_id for item in state.active_coaching_suggestions} == {
        "churn_risk",
        "cancellation_request",
    }
    assert {item.label_id for item in state.coaching_suggestion_history} == {
        "price_objection",
        "complaint",
    }
    assert len(final.displayed_suggestions) == 2
    assert len(final.replaced_suggestion_ids) == 2
    assert all(
        item.priority is SuggestionPriority.HIGH for item in state.coaching_suggestions
    )
    assert not {
        item.suggestion_id for item in state.active_coaching_suggestions
    }.intersection(item.suggestion_id for item in state.coaching_suggestion_history)
    assert all(
        decision.reason == "replaced_by_newer_priority" and decision.moved_to_history
        for decision in final.suggestion_decisions
        if decision.moved_to_history
    )
    assert (
        next(
            item
            for item in final.displayed_suggestions
            if item.label_id == "cancellation_request"
        ).source
        is CoachingSuggestionSource.BOTH
    )


def test_current_high_suggestion_replaces_older_lower_priority_suggestion() -> None:
    subject, state = coordinator(tenant=config(cooldown=0, maximum=1))
    subject.process(
        event(text="Ürün bilgisi.", revision=1),
        1.0,
        classification_event=classification("product_information"),
        active_labels=("product_information",),
    )
    current = subject.process(
        event(event_id="transcript_2", text=EXACT_CANCELLATION, revision=2),
        2.0,
        classification_event=classification("cancellation_request").model_copy(
            update={"transcript_event_id": "transcript_2"}
        ),
        active_labels=("cancellation_request",),
    )

    assert state.active_coaching_suggestions[0].label_id == "cancellation_request"
    assert state.coaching_suggestion_history[0].label_id == "product_information"
    assert current.replaced_suggestion_ids


def test_equal_priority_capacity_order_is_stable() -> None:
    subject, state = coordinator(tenant=config(cooldown=0, maximum=1))
    result = subject.process(
        event(text="Eşleşmeyen ifade."),
        1.0,
        classification_event=classification("product_information", "renewal_interest"),
        active_labels=("product_information", "renewal_interest"),
    )

    assert [item.label_id for item in result.displayed_suggestions] == [
        "product_information"
    ]
    assert [item.label_id for item in result.suppressed_suggestions] == [
        "renewal_interest"
    ]
    assert result.suppression_reasons == ("rejected_by_capacity",)
    assert state.active_coaching_suggestions[0].label_id == "product_information"


def test_capacity_rejection_does_not_start_candidate_cooldown() -> None:
    subject, state = coordinator(tenant=config(cooldown=60, maximum=1))
    first = subject.process(
        event(text="Eşleşmeyen ifade."),
        1.0,
        classification_event=classification(
            "product_information",
            "renewal_interest",
        ),
        active_labels=("product_information", "renewal_interest"),
    )
    assert [item.label_id for item in first.suppressed_suggestions] == [
        "renewal_interest"
    ]
    assert first.suppression_reasons == ("rejected_by_capacity",)

    retried = subject.process(
        event(event_id="transcript_2", text="Eşleşmeyen yeni ifade.", revision=2),
        2.0,
        classification_event=classification("renewal_interest").model_copy(
            update={"transcript_event_id": "transcript_2"}
        ),
        active_labels=("renewal_interest",),
    )

    assert [item.label_id for item in retried.displayed_suggestions] == [
        "renewal_interest"
    ]
    assert retried.suppression_reasons == ()
    assert state.active_coaching_suggestions[0].label_id == "renewal_interest"
    assert state.last_coaching_trigger_seconds == 2.0


def test_valid_external_llm_candidate_uses_existing_admission_lifecycle() -> None:
    subject, state = coordinator(tenant=config(cooldown=10, maximum=2))
    state.transcript_revision = 1

    result = subject.admit_external_suggestion(
        external_candidate(),
        current_seconds=1.0,
    )

    assert result.status is ExternalSuggestionAdmissionStatus.ADMITTED
    assert result.reason is ExternalSuggestionAdmissionReason.ADMITTED
    assert len(state.active_coaching_suggestions) == 1
    stored = state.active_coaching_suggestions[0]
    assert stored.label_id == "cancellation_request"
    assert stored.source is CoachingSuggestionSource.LLM
    assert state.shown_suggestion_ids == [stored.suggestion_id]
    assert state.last_coaching_trigger_seconds == 1.0


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"tenant_id": "tenant_beta"}, ExternalSuggestionAdmissionReason.INVALID_SCOPE),
        ({"call_id": "call_002"}, ExternalSuggestionAdmissionReason.INVALID_SCOPE),
        (
            {"transcript_revision": 0},
            ExternalSuggestionAdmissionReason.INVALID_REVISION,
        ),
    ],
)
def test_external_candidate_wrong_scope_or_revision_is_rejected_without_mutation(
    changes: dict[str, object],
    reason: ExternalSuggestionAdmissionReason,
) -> None:
    subject, state = coordinator()
    state.transcript_revision = 1
    before = subject.snapshot_coaching_state()

    result = subject.admit_external_suggestion(
        external_candidate(**changes),
        current_seconds=1.0,
    )

    assert result.status is ExternalSuggestionAdmissionStatus.REJECTED
    assert result.reason is reason
    assert subject.snapshot_coaching_state() == before


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"title": " "}, ExternalSuggestionAdmissionReason.INVALID_CANDIDATE),
        ({"suggestion": ""}, ExternalSuggestionAdmissionReason.INVALID_CANDIDATE),
        (
            {"source": CoachingSuggestionSource.RULE},
            ExternalSuggestionAdmissionReason.INVALID_CANDIDATE,
        ),
        ({"label_id": "not_a_label"}, ExternalSuggestionAdmissionReason.UNKNOWN_LABEL),
        ({"label_id": "iptal_riski"}, ExternalSuggestionAdmissionReason.UNKNOWN_LABEL),
        ({"label_id": "no_action"}, ExternalSuggestionAdmissionReason.UNKNOWN_LABEL),
    ],
)
def test_malformed_or_noncanonical_external_candidate_is_rejected_without_mutation(
    changes: dict[str, object],
    reason: ExternalSuggestionAdmissionReason,
) -> None:
    subject, state = coordinator()
    state.transcript_revision = 1
    before = subject.snapshot_coaching_state()

    result = subject.admit_external_suggestion(
        external_candidate(**changes),
        current_seconds=1.0,
    )

    assert result.status is ExternalSuggestionAdmissionStatus.REJECTED
    assert result.reason is reason
    assert subject.snapshot_coaching_state() == before


def test_external_duplicate_and_cooldown_apply_only_to_displayed_candidates() -> None:
    subject, state = coordinator(tenant=config(cooldown=10, maximum=3))
    state.transcript_revision = 1
    admitted = subject.admit_external_suggestion(
        external_candidate(),
        current_seconds=1.0,
    )
    duplicate = subject.admit_external_suggestion(
        external_candidate(),
        current_seconds=2.0,
    )
    cooldown = subject.admit_external_suggestion(
        external_candidate(
            title="Different synthetic guidance",
            suggestion="Use different approved synthetic steps.",
        ),
        current_seconds=2.0,
    )

    assert admitted.status is ExternalSuggestionAdmissionStatus.ADMITTED
    assert duplicate.reason is (
        ExternalSuggestionAdmissionReason.DUPLICATE_PREVIOUSLY_DISPLAYED
    )
    assert cooldown.reason is (
        ExternalSuggestionAdmissionReason.COOLDOWN_PREVIOUSLY_DISPLAYED
    )
    assert len(state.coaching_suggestions) == 1


def test_capacity_rejected_external_candidate_does_not_start_cooldown() -> None:
    subject, state = coordinator(tenant=config(cooldown=60, maximum=1))
    state.transcript_revision = 1
    subject.admit_external_suggestion(
        external_candidate(
            label_id="complaint",
            title="Synthetic complaint guidance",
            suggestion="Apply approved complaint steps.",
            priority=SuggestionPriority.CRITICAL,
        ),
        current_seconds=1.0,
    )
    rejected = subject.admit_external_suggestion(
        external_candidate(),
        current_seconds=1.0,
    )
    state.transcript_revision = 2
    retried = subject.admit_external_suggestion(
        external_candidate(
            transcript_revision=2,
            priority=SuggestionPriority.CRITICAL,
        ),
        current_seconds=2.0,
    )

    assert rejected.reason is ExternalSuggestionAdmissionReason.REJECTED_BY_CAPACITY
    assert retried.status is ExternalSuggestionAdmissionStatus.ADMITTED
    assert state.active_coaching_suggestions[0].label_id == "cancellation_request"


def test_newer_high_external_candidate_replaces_active_card_and_preserves_history() -> (
    None
):
    subject, state = coordinator(tenant=config(cooldown=0, maximum=1))
    state.transcript_revision = 1
    subject.process(
        event(text="Ürün bilgisi.", revision=1),
        1.0,
        classification_event=classification("product_information"),
        active_labels=("product_information",),
    )
    state.transcript_revision = 2

    result = subject.admit_external_suggestion(
        external_candidate(transcript_revision=2),
        current_seconds=2.0,
    )

    assert result.status is ExternalSuggestionAdmissionStatus.ADMITTED
    assert [item.label_id for item in state.active_coaching_suggestions] == [
        "cancellation_request"
    ]
    assert [item.label_id for item in state.coaching_suggestion_history] == [
        "product_information"
    ]
    active_ids = {item.suggestion_id for item in state.active_coaching_suggestions}
    history_ids = {item.suggestion_id for item in state.coaching_suggestion_history}
    assert active_ids.isdisjoint(history_ids)


def test_external_equal_rank_order_is_deterministic() -> None:
    subject, state = coordinator(tenant=config(cooldown=0, maximum=2))
    state.transcript_revision = 1

    subject.admit_external_suggestion(
        external_candidate(
            label_id="complaint",
            title="Synthetic complaint guidance",
            suggestion="Apply approved complaint steps.",
        ),
        current_seconds=1.0,
    )
    subject.admit_external_suggestion(
        external_candidate(),
        current_seconds=1.0,
    )

    assert [item.label_id for item in state.active_coaching_suggestions] == [
        "cancellation_request",
        "complaint",
    ]


def test_external_admission_diagnostics_contain_only_fixed_safe_values() -> None:
    subject, state = coordinator()
    state.transcript_revision = 1
    sensitive_marker = "PRIVATE_SYNTHETIC_MARKER"

    result = subject.admit_external_suggestion(
        external_candidate(suggestion=sensitive_marker),
        current_seconds=1.0,
    )

    assert set(result.__dataclass_fields__) == {"status", "reason"}
    assert sensitive_marker not in repr(result)
