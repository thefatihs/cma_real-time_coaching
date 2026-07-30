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
    CoachingSuggestionEvent,
    CoachingSuggestionLifecycle,
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
    *,
    cooldown: float = 10.0,
    maximum: int = 2,
    tenant_id: str = "tenant_alpha",
    allowed_actions: list[str] | None = None,
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
            allowed_actions=allowed_actions
            or [action.value for action in CoachingAction],
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


def classification(
    *labels: str,
    provisional: bool = False,
) -> ClassificationResultEvent:
    return ClassificationResultEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        transcript_event_id="transcript_1",
        labels=[ClassificationLabel(name=label, score=0.8) for label in labels],
        action=CoachingAction.TEMPLATE_ACTION,
        model_id="common_turkish_setfit_v2",
        threshold_profile_id="common_turkish_setfit_v2:calibrated:v1",
        provisional=provisional,
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


def external_suggestion(
    transcript: TranscriptEvent,
    **changes: object,
) -> CoachingSuggestionEvent:
    values: dict[str, object] = {
        "tenant_id": transcript.tenant_id,
        "call_id": transcript.call_id,
        "suggestion_id": f"external_{transcript.revision}",
        "source_transcript_event_id": transcript.event_id,
        "action": CoachingAction.TEMPLATE_ACTION,
        "priority": SuggestionPriority.HIGH,
        "source": CoachingSuggestionSource.LLM,
        "label_id": "complaint",
        "title": "Synthetic external guidance",
        "suggestion": "Use the synthetic external guidance.",
        "evidence_ids": ["document_1:chunk_1"],
        "created_at_utc": NOW,
    }
    values.update(changes)
    return CoachingSuggestionEvent.model_validate(values)


def apply_current_transcript(state: CallState, transcript: TranscriptEvent) -> None:
    state.apply_transcript(transcript)


def test_partial_is_ignored() -> None:
    subject, state = coordinator(rule())
    result = subject.process(event(TranscriptKind.PARTIAL), 1.0)
    assert result.classification_event is None
    assert result.displayed_suggestions == result.suppressed_suggestions == ()
    assert state.active_labels == []


def test_partial_classification_creates_provisional_card() -> None:
    subject, state = coordinator()
    transcript = event(TranscriptKind.PARTIAL, text="synthetic complaint now")
    result = subject.process(
        transcript,
        1.0,
        classification_event=classification("complaint", provisional=True),
        active_labels=("complaint",),
    )

    assert len(result.displayed_suggestions) == 1
    suggestion = result.displayed_suggestions[0]
    assert suggestion.lifecycle is CoachingSuggestionLifecycle.PROVISIONAL
    assert result.lifecycle is CoachingSuggestionLifecycle.PROVISIONAL
    assert state.classification_transcript_revision is None
    assert state.coaching_transcript_revision is None
    assert (
        state.active_coaching_suggestions[0].lifecycle
        is CoachingSuggestionLifecycle.PROVISIONAL
    )


def test_matching_commit_promotes_same_logical_card_without_duplicate() -> None:
    subject, state = coordinator()
    provisional = subject.process(
        event(TranscriptKind.PARTIAL, text="synthetic complaint now"),
        1.0,
        classification_event=classification("complaint", provisional=True),
        active_labels=("complaint",),
    ).displayed_suggestions[0]
    committed_event = event(
        TranscriptKind.STABLE,
        event_id="stable-2",
        revision=2,
        text="synthetic complaint confirmed",
    )
    committed_classification = classification("complaint").model_copy(
        update={"transcript_event_id": committed_event.event_id}
    )

    confirmed = subject.process(
        committed_event,
        2.0,
        classification_event=committed_classification,
        active_labels=("complaint",),
    )

    assert [item.suggestion_id for item in confirmed.displayed_suggestions] == [
        provisional.suggestion_id
    ]
    assert (
        confirmed.displayed_suggestions[0].lifecycle
        is CoachingSuggestionLifecycle.CONFIRMED
    )
    assert len(state.active_coaching_suggestions) == 1
    assert (
        state.active_coaching_suggestions[0].lifecycle
        is CoachingSuggestionLifecycle.CONFIRMED
    )


def test_unsupported_commit_withdraws_provisional_card() -> None:
    subject, state = coordinator()
    provisional = subject.process(
        event(TranscriptKind.PARTIAL, text="synthetic complaint now"),
        1.0,
        classification_event=classification("complaint", provisional=True),
        active_labels=("complaint",),
    ).displayed_suggestions[0]
    committed_event = event(
        TranscriptKind.STABLE,
        event_id="stable-2",
        revision=2,
        text="neutral committed text",
    )

    committed = subject.process(
        committed_event,
        2.0,
        classification_event=classification().model_copy(
            update={
                "transcript_event_id": committed_event.event_id,
                "action": CoachingAction.NO_ACTION,
            }
        ),
        active_labels=(),
    )

    assert committed.withdrawn_suggestion_ids == (provisional.suggestion_id,)
    assert state.active_coaching_suggestions == []
    assert (
        state.coaching_suggestion_history[-1].lifecycle
        is CoachingSuggestionLifecycle.WITHDRAWN
    )


def test_changed_commit_replaces_provisional_with_confirmed_card() -> None:
    subject, state = coordinator()
    provisional = subject.process(
        event(TranscriptKind.PARTIAL, text="synthetic complaint now"),
        1.0,
        classification_event=classification("complaint", provisional=True),
        active_labels=("complaint",),
    ).displayed_suggestions[0]
    committed_event = event(
        TranscriptKind.STABLE,
        event_id="stable-2",
        revision=2,
        text="synthetic churn risk confirmed",
    )

    committed = subject.process(
        committed_event,
        12.0,
        classification_event=classification("churn_risk").model_copy(
            update={"transcript_event_id": committed_event.event_id}
        ),
        active_labels=("churn_risk",),
    )

    assert provisional.suggestion_id in committed.withdrawn_suggestion_ids
    assert len(committed.displayed_suggestions) == 1
    assert committed.displayed_suggestions[0].label_id == "churn_risk"
    assert (
        committed.displayed_suggestions[0].lifecycle
        is CoachingSuggestionLifecycle.CONFIRMED
    )
    assert len(state.active_coaching_suggestions) == 1


def test_provisional_partial_key_history_is_bounded() -> None:
    subject, _ = coordinator()

    for revision in range(1, 70):
        transcript = event(
            TranscriptKind.PARTIAL,
            event_id=f"partial-{revision}",
            revision=revision,
            source_chunk_sequence=revision,
            text="synthetic complaint now",
        )
        subject.process(
            transcript,
            float(revision),
            classification_event=classification(
                "complaint", provisional=True
            ).model_copy(update={"transcript_event_id": transcript.event_id}),
            active_labels=("complaint",),
        )

    snapshot = subject.snapshot_coaching_state()
    assert len(snapshot.processed_partial_keys) == 64
    assert snapshot.processed_partial_keys == tuple(range(6, 70))
    assert len(snapshot.provisional_suggestions) == 1


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


def test_external_llm_suggestion_is_admitted_with_exact_scope_and_source() -> None:
    subject, state = coordinator(tenant=config(cooldown=0))
    transcript = event()
    apply_current_transcript(state, transcript)
    suggestion = external_suggestion(transcript)

    result = subject.process_external_suggestion(transcript, suggestion, 1.0)

    assert result.displayed_suggestions == (suggestion,)
    assert result.suppressed_suggestions == ()
    assert result.transcript_revision == transcript.revision
    admitted = result.displayed_suggestions[0]
    assert admitted.tenant_id == transcript.tenant_id
    assert admitted.call_id == transcript.call_id
    assert admitted.source_transcript_event_id == transcript.event_id
    assert admitted.source is CoachingSuggestionSource.LLM
    assert state.active_coaching_suggestions[0].source is CoachingSuggestionSource.LLM


@pytest.mark.parametrize(
    ("suggestion_changes", "message"),
    [
        ({"tenant_id": "tenant_beta"}, "tenant_id"),
        ({"call_id": "call_002"}, "call_id"),
        ({"source_transcript_event_id": "transcript_other"}, "source transcript"),
    ],
)
def test_invalid_external_scope_fails_without_state_mutation(
    suggestion_changes: dict[str, object],
    message: str,
) -> None:
    subject, state = coordinator()
    transcript = event()
    apply_current_transcript(state, transcript)
    before = subject.snapshot_coaching_state()

    with pytest.raises(ValueError, match=message):
        subject.process_external_suggestion(
            transcript,
            external_suggestion(transcript, **suggestion_changes),
            1.0,
        )

    assert subject.snapshot_coaching_state() == before


def test_stale_external_revision_fails_without_state_mutation() -> None:
    subject, state = coordinator()
    current = event(event_id="transcript_2", revision=2)
    apply_current_transcript(state, current)
    stale = event()
    before = subject.snapshot_coaching_state()

    with pytest.raises(ValueError, match="revision"):
        subject.process_external_suggestion(
            stale,
            external_suggestion(stale),
            1.0,
        )

    assert subject.snapshot_coaching_state() == before


def test_external_and_rule_revision_tracking_are_independent() -> None:
    rule_first, rule_first_state = coordinator(rule(), tenant=config(cooldown=0))
    transcript = event()
    apply_current_transcript(rule_first_state, transcript)
    rule_result = rule_first.process(transcript, 1.0)
    external_result = rule_first.process_external_suggestion(
        transcript,
        external_suggestion(
            transcript,
            title="Different synthetic external guidance",
        ),
        2.0,
    )

    external_first, external_first_state = coordinator(
        rule(),
        tenant=config(cooldown=0),
    )
    apply_current_transcript(external_first_state, transcript)
    first_external_result = external_first.process_external_suggestion(
        transcript,
        external_suggestion(
            transcript,
            title="Different synthetic external guidance",
        ),
        1.0,
    )
    later_rule_result = external_first.process(transcript, 2.0)

    assert rule_result.displayed_suggestions
    assert external_result.suppression_reasons != ("duplicate_external_revision",)
    assert first_external_result.displayed_suggestions
    assert later_rule_result.suppression_reasons != ("duplicate_revision",)


def test_repeated_external_revision_is_suppressed_deterministically() -> None:
    subject, state = coordinator(tenant=config(cooldown=0))
    transcript = event()
    apply_current_transcript(state, transcript)
    suggestion = external_suggestion(transcript)
    subject.process_external_suggestion(transcript, suggestion, 1.0)

    first_repeat = subject.process_external_suggestion(transcript, suggestion, 2.0)
    second_repeat = subject.process_external_suggestion(transcript, suggestion, 3.0)

    assert first_repeat == second_repeat
    assert first_repeat.displayed_suggestions == ()
    assert first_repeat.suppressed_suggestions == (suggestion,)
    assert first_repeat.suppression_reasons == ("duplicate_external_revision",)


def test_external_disallowed_action_uses_normal_suppression_result() -> None:
    subject, state = coordinator(
        tenant=config(
            allowed_actions=[CoachingAction.TEMPLATE_ACTION.value],
        )
    )
    transcript = event()
    apply_current_transcript(state, transcript)
    suggestion = external_suggestion(
        transcript,
        action=CoachingAction.ESCALATE,
    )
    coordinator_before = subject.snapshot_coaching_state()
    state_before = state.model_copy(deep=True)

    first = subject.process_external_suggestion(transcript, suggestion, 1.0)
    second = subject.process_external_suggestion(transcript, suggestion, 2.0)

    assert first == second
    assert first.displayed_suggestions == ()
    assert first.suppressed_suggestions == (suggestion,)
    assert first.suppression_reasons == ("action_not_allowed",)
    assert first.suggestion_decisions[0].reason == "action_not_allowed"
    assert not first.suggestion_decisions[0].moved_to_history
    assert subject.snapshot_coaching_state() == coordinator_before
    assert subject.snapshot_coaching_state().processed_external_revisions == frozenset()
    assert state == state_before


def test_classification_template_preserves_head_allowed_action_behavior() -> None:
    tenant = config(allowed_actions=[CoachingAction.NO_ACTION.value])
    subject, state = coordinator(tenant=tenant)
    transcript = event(text="Synthetic unmatched text.")

    result = subject.process(
        transcript,
        1.0,
        classification_event=classification("cancellation_request"),
        active_labels=("cancellation_request",),
    )

    assert result.displayed_suggestions
    assert result.displayed_suggestions[0].action.value not in (
        tenant.coaching.allowed_actions
    )
    assert result.suppression_reasons == ()
    assert state.active_coaching_suggestions


def test_configured_rule_with_disallowed_action_keeps_constructor_validation() -> None:
    tenant = config(allowed_actions=[CoachingAction.NO_ACTION.value])

    with pytest.raises(ValueError, match="Rule action is not allowed"):
        coordinator(rule(), tenant=tenant)


def test_external_duplicate_fingerprint_uses_shared_policy() -> None:
    subject, state = coordinator(tenant=config(cooldown=0))
    first = event()
    apply_current_transcript(state, first)
    subject.process_external_suggestion(first, external_suggestion(first), 1.0)
    second = event(event_id="transcript_2", revision=2)
    apply_current_transcript(state, second)

    result = subject.process_external_suggestion(
        second,
        external_suggestion(second),
        2.0,
    )

    assert result.displayed_suggestions == ()
    assert result.suppression_reasons == ("duplicate_same_revision",)


def test_external_cooldown_uses_shared_policy() -> None:
    subject, state = coordinator(tenant=config(cooldown=10))
    first = event()
    apply_current_transcript(state, first)
    subject.process_external_suggestion(first, external_suggestion(first), 1.0)
    second = event(event_id="transcript_2", revision=2)
    apply_current_transcript(state, second)

    result = subject.process_external_suggestion(
        second,
        external_suggestion(
            second,
            title="Updated synthetic external guidance",
        ),
        5.0,
    )

    assert result.displayed_suggestions == ()
    assert result.suppression_reasons == ("cooldown_previously_displayed",)


def test_external_capacity_rejection_and_replacement_use_call_state_policy() -> None:
    rejected_subject, rejected_state = coordinator(tenant=config(cooldown=0, maximum=1))
    first = event()
    apply_current_transcript(rejected_state, first)
    rejected_subject.process_external_suggestion(
        first,
        external_suggestion(first, priority=SuggestionPriority.CRITICAL),
        1.0,
    )
    second = event(event_id="transcript_2", revision=2)
    apply_current_transcript(rejected_state, second)
    rejected = rejected_subject.process_external_suggestion(
        second,
        external_suggestion(
            second,
            priority=SuggestionPriority.LOW,
            title="Lower priority synthetic guidance",
        ),
        2.0,
    )

    replacing_subject, replacing_state = coordinator(
        tenant=config(cooldown=0, maximum=1)
    )
    apply_current_transcript(replacing_state, first)
    replacing_subject.process_external_suggestion(
        first,
        external_suggestion(first, priority=SuggestionPriority.LOW),
        1.0,
    )
    apply_current_transcript(replacing_state, second)
    replacement = external_suggestion(
        second,
        priority=SuggestionPriority.CRITICAL,
        title="Critical synthetic guidance",
    )
    replaced = replacing_subject.process_external_suggestion(
        second,
        replacement,
        2.0,
    )

    assert rejected.suppression_reasons == ("rejected_by_capacity",)
    assert rejected_state.active_coaching_suggestions[0].priority is (
        SuggestionPriority.CRITICAL
    )
    assert replaced.displayed_suggestions == (replacement,)
    assert replaced.replaced_suggestion_ids == ("external_1",)
    assert replacing_state.active_coaching_suggestions[0].suggestion_id == "external_2"
    assert replacing_state.coaching_suggestion_history[0].suggestion_id == "external_1"
    assert [decision.reason for decision in replaced.suggestion_decisions] == [
        "admitted",
        "replaced_by_newer_priority",
    ]


def test_snapshot_restore_includes_external_revision_state() -> None:
    subject, state = coordinator(tenant=config(cooldown=0))
    transcript = event()
    apply_current_transcript(state, transcript)
    before = subject.snapshot_coaching_state()
    suggestion = external_suggestion(transcript)
    subject.process_external_suggestion(transcript, suggestion, 1.0)

    subject.restore_coaching_state(before)
    restored = subject.process_external_suggestion(transcript, suggestion, 1.0)

    assert restored.displayed_suggestions == (suggestion,)
    assert restored.suppression_reasons == ()
