import pytest
from pydantic import ValidationError

from app.llm import InMemoryLLMGateway, LLMGateway, LLMRequest, LLMResponse


def request(**changes: object) -> LLMRequest:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "input_text": "Synthetic question.",
    }
    values.update(changes)
    return LLMRequest.model_validate(values)


def test_gateway_satisfies_protocol_and_preserves_scope() -> None:
    gateway: LLMGateway = InMemoryLLMGateway()

    response = gateway.generate(request())

    assert response == LLMResponse(
        tenant_id="tenant_alpha",
        call_id="call_001",
        text="Synthetic LLM response.",
    )


def test_output_is_deterministic() -> None:
    gateway = InMemoryLLMGateway()

    first = gateway.generate(request(input_text="Synthetic first input."))
    second = gateway.generate(request(input_text="Synthetic second input."))

    assert first == second


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", ""),
        ("tenant_id", " "),
        ("call_id", ""),
        ("input_text", " "),
    ],
)
def test_request_rejects_invalid_input(field: str, value: str) -> None:
    with pytest.raises(ValidationError, match=f"{field} cannot be empty"):
        request(**{field: value})


@pytest.mark.parametrize("field", ["tenant_id", "call_id", "text"])
def test_response_rejects_invalid_input(field: str) -> None:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "text": "Synthetic response.",
    }
    values[field] = " "

    with pytest.raises(ValidationError, match=f"{field} cannot be empty"):
        LLMResponse.model_validate(values)


def test_models_are_immutable() -> None:
    source_request = request()
    response = InMemoryLLMGateway().generate(source_request)

    with pytest.raises(ValidationError):
        source_request.input_text = "Synthetic changed input."
    with pytest.raises(ValidationError):
        response.text = "Synthetic changed response."
