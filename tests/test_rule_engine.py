from datetime import UTC, datetime
from itertools import count

import pytest
from pydantic import ValidationError

from app.coaching.rule_engine import (
    RULE_ONLY_PARTIAL_MODEL_ID,
    CoachingRule,
    RuleBasedCoachingEngine,
)
from app.events.models import (
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


def tenant_config(**changes: object) -> TenantConfig:
    values: dict[str, object] = {
        "context": TenantContext(tenant_id="tenant_alpha", tenant_name="Alpha"),
        "asr": TenantASRConfig(),
        "classification": TenantClassificationConfig(
            model_id="rules-v1", labels=["iade", "şikayet", "acil"]
        ),
        "rag": TenantRAGConfig(enabled=False),
        "coaching": TenantCoachingConfig(
            allowed_actions=[action.value for action in CoachingAction]
        ),
    }
    values.update(changes)
    return TenantConfig.model_validate(values)


def rule(rule_id: str = "rule_1", **changes: object) -> CoachingRule:
    values: dict[str, object] = {
        "rule_id": rule_id,
        "label": "iade",
        "include_any": ("ürün iadesi",),
        "action": CoachingAction.TEMPLATE_ACTION,
        "priority": SuggestionPriority.HIGH,
        "title": "İade desteği",
        "suggestion": "İade adımlarını açıklayın.",
        "evidence_ids": ("policy_1",),
    }
    values.update(changes)
    return CoachingRule.model_validate(values)


def transcript(
    kind: TranscriptKind = TranscriptKind.STABLE, **changes: object
) -> TranscriptEvent:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "event_id": "transcript_1",
        "kind": kind,
        "text": "Ürün iadesi hakkında bilgi istiyorum.",
        "start_seconds": 0.0,
        "end_seconds": 2.0,
        "revision": 1,
        "created_at_utc": NOW,
    }
    values.update(changes)
    return TranscriptEvent.model_validate(values)


def engine(
    *rules: CoachingRule, config: TenantConfig | None = None
) -> RuleBasedCoachingEngine:
    identifiers = count(1)
    return RuleBasedCoachingEngine(
        config or tenant_config(),
        tuple(rules),
        event_id_factory=lambda: f"suggestion_{next(identifiers)}",
        utc_datetime_factory=lambda: NOW,
    )


def test_partial_transcript_is_ignored() -> None:
    result = engine(rule()).evaluate(transcript(TranscriptKind.PARTIAL))
    assert result.classification_event is None
    assert result.suggestion_events == result.matched_rule_ids == ()


def test_rule_only_partial_uses_existing_explicit_evidence_without_probabilities() -> (
    None
):
    event = transcript(TranscriptKind.PARTIAL)

    result = engine(rule()).classify_partial(event)

    assert result is not None
    assert result.provisional
    assert result.model_id == RULE_ONLY_PARTIAL_MODEL_ID
    assert result.threshold_profile_id is None
    assert result.probabilities == result.thresholds == {}
    assert [label.name for label in result.labels] == ["iade"]


def test_rule_only_partial_without_existing_match_returns_none() -> None:
    assert (
        engine(rule()).classify_partial(
            transcript(TranscriptKind.PARTIAL, text="Eşleşmeyen sentetik ifade.")
        )
        is None
    )


@pytest.mark.parametrize("kind", [TranscriptKind.STABLE, TranscriptKind.FINAL])
def test_stable_and_final_events_are_evaluated(kind: TranscriptKind) -> None:
    result = engine(rule()).evaluate(transcript(kind))
    assert result.classification_event is not None
    assert (
        result.classification_event.tenant_id,
        result.classification_event.call_id,
    ) == ("tenant_alpha", "call_001")
    assert result.matched_rule_ids == ("rule_1",)


def test_tenant_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="Mismatched tenant_id"):
        engine(rule()).evaluate(transcript(tenant_id="tenant_beta"))


def test_include_any_include_all_and_exclude_matching() -> None:
    rules = (
        rule("any", include_any=("ürün iadesi", "değişim")),
        rule("all", include_any=(), include_all=("ürün", "bilgi istiyorum")),
        rule("excluded", include_any=("iade",), exclude_any=("bilgi istiyorum",)),
    )
    result = engine(*rules).evaluate(transcript())
    assert result.matched_rule_ids == ("any", "all")


def test_case_whitespace_and_surrounding_punctuation_are_normalized() -> None:
    matching = rule(include_any=("iade talebi",))
    event = transcript(text="  !!!İADE,    TALEBİ???  ")
    assert engine(matching).evaluate(event).matched_rule_ids == ("rule_1",)


def test_disabled_rule_is_ignored() -> None:
    assert engine(rule(enabled=False)).evaluate(transcript()).matched_rule_ids == ()


def test_multiple_labels_and_strongest_action_are_deterministic() -> None:
    result = engine(
        rule("template"),
        rule(
            "rag",
            label="şikayet",
            include_any=("bilgi",),
            action=CoachingAction.RAG_ACTION,
        ),
        rule(
            "escalate",
            label="acil",
            include_any=("istiyorum",),
            action=CoachingAction.ESCALATE,
        ),
    ).evaluate(transcript())
    assert result.classification_event is not None
    assert [label.name for label in result.classification_event.labels] == [
        "acil",
        "iade",
        "şikayet",
    ]
    assert result.classification_event.action is CoachingAction.ESCALATE


@pytest.mark.parametrize(
    ("action", "expected_suggestions"),
    [
        (CoachingAction.TEMPLATE_ACTION, 1),
        (CoachingAction.ESCALATE, 1),
        (CoachingAction.RAG_ACTION, 0),
    ],
)
def test_suggestion_creation_by_action(
    action: CoachingAction, expected_suggestions: int
) -> None:
    configured = rule(action=action)
    result = engine(configured).evaluate(transcript())
    assert result.classification_event is not None
    assert len(result.suggestion_events) == expected_suggestions
    if result.suggestion_events:
        suggestion = result.suggestion_events[0]
        assert (suggestion.action, suggestion.priority, suggestion.title) == (
            action,
            SuggestionPriority.HIGH,
            "İade desteği",
        )
        assert suggestion.evidence_ids == ["policy_1"]


def test_duplicate_label_suggestions_are_deduplicated() -> None:
    result = engine(rule("first"), rule("second", include_any=("bilgi",))).evaluate(
        transcript()
    )
    assert result.matched_rule_ids == ("first", "second")
    assert len(result.suggestion_events) == 1


def test_duplicate_rule_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="rule IDs"):
        engine(rule("same"), rule("same", include_any=("bilgi",)))


def test_unknown_label_and_disallowed_action_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown tenant classification label"):
        engine(rule(label="bilinmeyen"))
    config = tenant_config(
        coaching=TenantCoachingConfig(allowed_actions=[CoachingAction.NO_ACTION.value])
    )
    with pytest.raises(ValueError, match="not allowed"):
        engine(rule(), config=config)


@pytest.mark.parametrize(
    "changes",
    [
        {"rule_id": " "},
        {"include_any": ()},
        {"include_any": ("",)},
        {"include_any": ("iade", "İADE!")},
        {"evidence_ids": ("doc", "doc")},
    ],
)
def test_rule_validation(changes: dict[str, object]) -> None:
    values = rule().model_dump()
    values.update(changes)
    with pytest.raises(ValidationError):
        CoachingRule.model_validate(values)


def test_source_event_and_rules_remain_unchanged() -> None:
    source_rule = rule()
    source_event = transcript()
    before = (source_rule.model_dump(), source_event.model_dump())
    engine(source_rule).evaluate(source_event)
    assert before == (source_rule.model_dump(), source_event.model_dump())


@pytest.mark.parametrize(
    "label",
    [
        "product_information",
        "price_objection",
        "cancellation_request",
        "technical_issue",
        "complaint",
        "renewal_interest",
        "churn_risk",
    ],
)
def test_every_business_classification_label_has_a_coaching_template(
    label: str,
) -> None:
    result = engine().evaluate(
        transcript(text="Metin kuralıyla eşleşmeyen sentetik ifade."),
        (label,),
    )
    assert len(result.suggestion_events) == 1
    assert result.suggestion_events[0].source is (
        CoachingSuggestionSource.CLASSIFICATION
    )


def test_common_turkish_cancellation_forms_match_one_safe_rule_suggestion() -> None:
    exact = (
        "Aboneliğimi bugün iptal ettirmek istiyorum. Lütfen iptal işlemini başlatın."
    )
    result = engine().evaluate(transcript(text=exact))
    assert result.matched_rule_ids == ("general-explicit-cancellation",)
    assert len(result.suggestion_events) == 1
    suggestion = result.suggestion_events[0]
    assert suggestion.source is CoachingSuggestionSource.RULE
    assert "iptal nedenini netleştirin" in suggestion.suggestion


@pytest.mark.parametrize(
    "text",
    [
        "İptal etmek istemiyorum.",
        "Aboneliğimi iptal etmeyeceğim.",
        "İptal talebim yok.",
    ],
)
def test_negated_cancellation_does_not_match_general_rule(text: str) -> None:
    result = engine().evaluate(transcript(text=text))
    assert result.matched_rule_ids == ()
    assert result.suggestion_events == ()
