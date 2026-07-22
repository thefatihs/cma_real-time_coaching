from datetime import UTC, datetime
from itertools import count

import pytest

from app.calls.models import CallState
from app.coaching.coordinator import CoachingCoordinator
from app.coaching.rule_engine import CoachingRule, RuleBasedCoachingEngine
from app.events.models import (
    CoachingAction,
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
            model_id="rules-v1", labels=["iade", "acil", "bilgi"]
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
    subject, state = coordinator(rule(), rule("rule_2", label="acil", phrase="acil"))
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
    assert result.suppression_reasons == ("duplicate",)


def test_cooldown_suppression_then_allowed_after_cooldown() -> None:
    subject, state = coordinator(
        rule("first", phrase="iade"),
        rule("second", label="acil", phrase="acil", title="Acil destek"),
    )
    first = subject.process(event(text="iade"), 1.0)
    assert len(first.displayed_suggestions) == 1
    blocked = subject.process(
        event(event_id="transcript_2", text="acil", revision=2), 5.0
    )
    assert blocked.suppression_reasons == ("cooldown",)
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
    assert result.suppression_reasons == ("max_active_suggestions",)


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
