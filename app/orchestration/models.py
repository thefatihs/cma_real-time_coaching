"""Immutable internal orchestration models."""

from pydantic import BaseModel, ConfigDict, field_validator


class OrchestrationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    call_id: str
    knowledge_base_id: str
    user_input: str
    top_k: int

    @field_validator(
        "tenant_id",
        "call_id",
        "knowledge_base_id",
        "user_input",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k must be positive")
        return value


class OrchestrationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    call_id: str
    generated_text: str

    @field_validator("tenant_id", "call_id", "generated_text")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned
