"""Internal models for deterministic prompt construction."""

from pydantic import BaseModel, ConfigDict, field_validator


class PromptContextItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    chunk_id: str
    text: str
    score: float

    @field_validator("document_id", "chunk_id", "text")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("score must be between 0 and 1")
        return value


class PromptBuildRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    call_id: str
    user_input: str
    retrieved_context: tuple[PromptContextItem, ...] = ()

    @field_validator("tenant_id", "call_id", "user_input")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))


class PromptBuildResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    call_id: str
    system_prompt: str
    user_prompt: str

    @field_validator("tenant_id", "call_id", "system_prompt", "user_prompt")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
