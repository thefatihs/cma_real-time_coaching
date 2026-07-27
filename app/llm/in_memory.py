"""Deterministic synthetic-only LLM gateway."""

from app.llm.models import LLMRequest, LLMResponse

_SYNTHETIC_RESPONSE = "Synthetic LLM response."


class InMemoryLLMGateway:
    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            tenant_id=request.tenant_id,
            call_id=request.call_id,
            text=_SYNTHETIC_RESPONSE,
        )
