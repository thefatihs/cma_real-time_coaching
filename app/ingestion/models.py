"""Immutable provider-neutral document ingestion inputs."""

from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.vector_store.models import Metadata


class DocumentChunkInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    chunk_id: str
    text: str
    metadata: Metadata = ()

    @field_validator("document_id", "chunk_id", "text")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: Metadata) -> Metadata:
        result: list[tuple[str, str]] = []
        keys: set[str] = set()
        for key, item in value:
            clean_key = _required_text(key, "metadata key")
            clean_item = _required_text(item, f"metadata value for {clean_key}")
            if clean_key in keys:
                raise ValueError("metadata keys must be unique")
            keys.add(clean_key)
            result.append((clean_key, clean_item))
        return tuple(result)


class DocumentIngestionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    knowledge_base_id: str
    chunks: tuple[DocumentChunkInput, ...]

    @field_validator("tenant_id", "knowledge_base_id")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))

    @model_validator(mode="after")
    def validate_chunks(self) -> Self:
        if not self.chunks:
            raise ValueError("chunks cannot be empty")
        identities = tuple((chunk.document_id, chunk.chunk_id) for chunk in self.chunks)
        if len(identities) != len(set(identities)):
            raise ValueError("document chunk identities must be unique")
        return self


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
