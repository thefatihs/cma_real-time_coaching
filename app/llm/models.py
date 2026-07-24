"""Internal models for provider-neutral LLM generation."""

from pydantic import BaseModel, ConfigDict, field_validator


class LLMRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    call_id: str
    input_text: str

    @field_validator("tenant_id", "call_id", "input_text")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    call_id: str
    text: str

    @field_validator("tenant_id", "call_id", "text")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned
