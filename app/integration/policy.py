"""Provider-neutral policy for RAG-generated coaching suggestions."""

from math import isfinite

from pydantic import BaseModel, ConfigDict, field_validator

from app.events.models import CoachingAction, SuggestionPriority


class RAGCoachingIntegrationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    rag_llm_enabled_labels: tuple[str, ...]
    title: str
    action: CoachingAction
    priority: SuggestionPriority
    label_id: str | None
    expires_after_seconds: float | None

    @field_validator("rag_llm_enabled_labels")
    @classmethod
    def validate_enabled_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if not cleaned:
            raise ValueError("rag_llm_enabled_labels cannot be empty")
        if any(not value for value in cleaned):
            raise ValueError("rag_llm_enabled_labels cannot contain blank labels")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("rag_llm_enabled_labels must be unique")
        return cleaned

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return _required_text(value, "title")

    @field_validator("label_id")
    @classmethod
    def validate_label_id(cls, value: str | None) -> str | None:
        return None if value is None else _required_text(value, "label_id")

    @field_validator("expires_after_seconds")
    @classmethod
    def validate_expiry(cls, value: float | None) -> float | None:
        if value is not None and (not isfinite(value) or value < 0):
            raise ValueError(
                "expires_after_seconds must be finite and cannot be negative"
            )
        return value


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
