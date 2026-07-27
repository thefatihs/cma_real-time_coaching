import math

import pytest
from pydantic import ValidationError

from app.events.models import CoachingAction, SuggestionPriority
from app.integration import (
    CoachingSuggestionFactory,
    DeterministicLLMCoachingSuggestionFactory,
    OrchestrationRunner,
    RAGCoachingIntegrationPolicy,
    RAGCoachingProcessorDecorator,
)


def policy(
    **overrides: object,
) -> RAGCoachingIntegrationPolicy:
    values: dict[str, object] = {
        "rag_llm_enabled_labels": ("product_information", "complaint"),
        "title": "Synthetic guidance",
        "action": CoachingAction.TEMPLATE_ACTION,
        "priority": SuggestionPriority.MEDIUM,
        "label_id": "product_information",
        "expires_after_seconds": 30.0,
    }
    values.update(overrides)
    return RAGCoachingIntegrationPolicy.model_validate(values)


def test_valid_policy_preserves_explicit_values_and_enum_types() -> None:
    subject = policy()

    assert subject.rag_llm_enabled_labels == (
        "product_information",
        "complaint",
    )
    assert subject.title == "Synthetic guidance"
    assert subject.action is CoachingAction.TEMPLATE_ACTION
    assert subject.priority is SuggestionPriority.MEDIUM
    assert subject.label_id == "product_information"
    assert subject.expires_after_seconds == 30.0


def test_policy_is_frozen() -> None:
    subject = policy()

    with pytest.raises(ValidationError, match="frozen"):
        subject.title = "Changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    [
        "rag_llm_enabled_labels",
        "title",
        "action",
        "priority",
        "label_id",
        "expires_after_seconds",
    ],
)
def test_all_fields_are_required(field_name: str) -> None:
    assert RAGCoachingIntegrationPolicy.model_fields[field_name].is_required()


def test_text_and_label_order_are_normalized_deterministically() -> None:
    subject = policy(
        rag_llm_enabled_labels=(" second ", "first "),
        title="  Synthetic guidance  ",
        label_id="  product_information  ",
    )

    assert subject.rag_llm_enabled_labels == ("second", "first")
    assert subject.title == "Synthetic guidance"
    assert subject.label_id == "product_information"


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ((), "cannot be empty"),
        (("product_information", " "), "blank labels"),
        (("complaint", "complaint"), "unique"),
        (("complaint", " complaint "), "unique"),
    ],
)
def test_invalid_label_collections_are_rejected(
    labels: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        policy(rag_llm_enabled_labels=labels)


def test_blank_title_is_rejected() -> None:
    with pytest.raises(ValidationError, match="title"):
        policy(title=" ")


def test_blank_supplied_label_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="label_id"):
        policy(label_id=" ")


def test_none_label_id_and_expiry_are_accepted() -> None:
    subject = policy(label_id=None, expires_after_seconds=None)

    assert subject.label_id is None
    assert subject.expires_after_seconds is None


@pytest.mark.parametrize("expiry", [-0.1, math.inf, -math.inf, math.nan])
def test_invalid_expiry_is_rejected(expiry: float) -> None:
    with pytest.raises(ValidationError, match="expires_after_seconds"):
        policy(expires_after_seconds=expiry)


def test_equality_and_serialization_are_deterministic() -> None:
    first = policy()
    second = policy()

    assert first == second
    assert first.model_dump() == second.model_dump()
    assert first.model_dump_json() == second.model_dump_json()
    assert f'"action":"{CoachingAction.TEMPLATE_ACTION.value}"' in (
        first.model_dump_json()
    )
    assert f'"priority":"{SuggestionPriority.MEDIUM.value}"' in (
        first.model_dump_json()
    )


def test_existing_integration_exports_remain_available() -> None:
    assert CoachingSuggestionFactory is not None
    assert DeterministicLLMCoachingSuggestionFactory is not None
    assert OrchestrationRunner is not None
    assert RAGCoachingProcessorDecorator is not None
