"""Internal models for provider-neutral retrieval."""

from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class RetrievalDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    knowledge_base_id: str
    document_id: str
    chunk_id: str
    text: str
    score: float

    @field_validator(
        "tenant_id",
        "knowledge_base_id",
        "document_id",
        "chunk_id",
        "text",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("score must be between 0 and 1")
        return value


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    knowledge_base_id: str
    documents: tuple[RetrievalDocument, ...] = ()

    @field_validator("tenant_id", "knowledge_base_id")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned

    @model_validator(mode="after")
    def validate_document_scope(self) -> Self:
        if any(document.tenant_id != self.tenant_id for document in self.documents):
            raise ValueError("retrieval document tenant_id does not match result")
        if any(
            document.knowledge_base_id != self.knowledge_base_id
            for document in self.documents
        ):
            raise ValueError(
                "retrieval document knowledge_base_id does not match result"
            )
        return self
