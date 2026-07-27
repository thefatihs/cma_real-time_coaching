import pytest
from pydantic import ValidationError

from app.prompting import (
    DeterministicPromptBuilder,
    PromptBuilder,
    PromptBuildRequest,
    PromptBuildResult,
    PromptContextItem,
)


def context(
    chunk_id: str,
    *,
    document_id: str = "guide",
    text: str | None = None,
    score: float = 0.8,
) -> PromptContextItem:
    return PromptContextItem(
        document_id=document_id,
        chunk_id=chunk_id,
        text=text or f"Synthetic context for {chunk_id}.",
        score=score,
    )


def request(**changes: object) -> PromptBuildRequest:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "user_input": "Synthetic user question.",
        "retrieved_context": (),
    }
    values.update(changes)
    return PromptBuildRequest.model_validate(values)


def test_protocol_compatibility_and_scope_preservation() -> None:
    builder: PromptBuilder = DeterministicPromptBuilder()

    result = builder.build(request(tenant_id="tenant_beta", call_id="call_002"))

    assert result.tenant_id == "tenant_beta"
    assert result.call_id == "call_002"


def test_context_order_is_deterministic() -> None:
    result = DeterministicPromptBuilder().build(
        request(
            retrieved_context=(
                context("chunk_b", document_id="guide_b", score=0.9),
                context("chunk_c", document_id="guide_a", score=0.9),
                context("chunk_a", document_id="guide_a", score=0.9),
                context("chunk_high", score=1.0),
            )
        )
    )

    positions = [
        result.user_prompt.index(chunk_id)
        for chunk_id in ("chunk_high", "chunk_a", "chunk_c", "chunk_b")
    ]
    assert positions == sorted(positions)


def test_empty_context_is_explicit_and_safe() -> None:
    result = DeterministicPromptBuilder().build(request())

    assert "Synthetic user question." in result.user_prompt
    assert "No retrieved context." in result.user_prompt


def test_repeated_builds_are_stable() -> None:
    builder = DeterministicPromptBuilder()
    source = request(retrieved_context=(context("chunk_1"),))

    assert builder.build(source) == builder.build(source)


def test_context_text_is_included_without_object_representation() -> None:
    item = context("chunk_1", text="Synthetic approved information.")

    result = DeterministicPromptBuilder().build(request(retrieved_context=(item,)))

    assert "Synthetic approved information." in result.user_prompt
    assert "PromptContextItem(" not in result.user_prompt
    assert repr(item) not in result.user_prompt


@pytest.mark.parametrize(
    ("model", "field"),
    [
        (PromptContextItem, "document_id"),
        (PromptContextItem, "chunk_id"),
        (PromptContextItem, "text"),
        (PromptBuildRequest, "tenant_id"),
        (PromptBuildRequest, "call_id"),
        (PromptBuildRequest, "user_input"),
        (PromptBuildResult, "tenant_id"),
        (PromptBuildResult, "call_id"),
        (PromptBuildResult, "system_prompt"),
        (PromptBuildResult, "user_prompt"),
    ],
)
def test_required_fields_reject_blank_values(
    model: type[PromptContextItem | PromptBuildRequest | PromptBuildResult],
    field: str,
) -> None:
    values_by_model: dict[type[object], dict[str, object]] = {
        PromptContextItem: {
            "document_id": "guide",
            "chunk_id": "chunk_1",
            "text": "Synthetic context.",
            "score": 0.8,
        },
        PromptBuildRequest: {
            "tenant_id": "tenant_alpha",
            "call_id": "call_001",
            "user_input": "Synthetic input.",
        },
        PromptBuildResult: {
            "tenant_id": "tenant_alpha",
            "call_id": "call_001",
            "system_prompt": "Synthetic system prompt.",
            "user_prompt": "Synthetic user prompt.",
        },
    }
    values = values_by_model[model].copy()
    values[field] = " "

    with pytest.raises(ValidationError, match=f"{field} cannot be empty"):
        model.model_validate(values)


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_context_score_must_be_a_probability(score: float) -> None:
    with pytest.raises(ValidationError, match="between 0 and 1"):
        context("chunk_1", score=score)


def test_models_are_immutable() -> None:
    item = context("chunk_1")
    source = request(retrieved_context=(item,))
    result = DeterministicPromptBuilder().build(source)

    with pytest.raises(ValidationError):
        item.text = "Synthetic changed context."
    with pytest.raises(ValidationError):
        source.user_input = "Synthetic changed input."
    with pytest.raises(ValidationError):
        result.user_prompt = "Synthetic changed prompt."
