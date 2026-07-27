"""Provider-neutral LLM gateway interface."""

from typing import Protocol

from app.llm.models import LLMRequest, LLMResponse


class LLMGateway(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse: ...
