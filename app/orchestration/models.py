"""Immutable internal orchestration models."""

from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class OrchestrationCitationReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    chunk_id: str

    @field_validator("document_id", "chunk_id")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned


class OrchestrationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    call_id: str
    transcript_revision: int
    knowledge_base_id: str
    user_input: str
    top_k: int
    minimum_score: float = 0.0

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

    @field_validator("transcript_revision")
    @classmethod
    def validate_transcript_revision(cls, value: int) -> int:
        if value < 0:
            raise ValueError("transcript_revision cannot be negative")
        return value

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k must be positive")
        return value

    @field_validator("minimum_score")
    @classmethod
    def validate_minimum_score(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("minimum_score must be between 0 and 1")
        return value


class OrchestrationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    call_id: str
    transcript_revision: int
    generated_text: str
    citations: tuple[OrchestrationCitationReference, ...]

    @field_validator("tenant_id", "call_id", "generated_text")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned

    @field_validator("transcript_revision")
    @classmethod
    def validate_transcript_revision(cls, value: int) -> int:
        if value < 0:
            raise ValueError("transcript_revision cannot be negative")
        return value

    @model_validator(mode="after")
    def validate_unique_citations(self) -> Self:
        identities = tuple(
            (citation.document_id, citation.chunk_id) for citation in self.citations
        )
        if len(identities) != len(set(identities)):
            raise ValueError("citation identities must be unique")
        return self
