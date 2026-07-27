"""Provider-neutral prompt builder interface."""

from typing import Protocol

from app.prompting.models import PromptBuildRequest, PromptBuildResult


class PromptBuilder(Protocol):
    def build(self, request: PromptBuildRequest) -> PromptBuildResult: ...
