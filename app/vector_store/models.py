"""Immutable internal vector-store models."""

from math import isfinite
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

Metadata = tuple[tuple[str, str], ...]


class VectorRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    knowledge_base_id: str
    document_id: str
    chunk_id: str
    text: str
    embedding: tuple[float, ...]
    metadata: Metadata = ()

    @field_validator(
        "tenant_id",
        "knowledge_base_id",
        "document_id",
        "chunk_id",
        "text",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        return _validated_embedding(value, "embedding")

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


class SearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    knowledge_base_id: str
    query_embedding: tuple[float, ...]
    top_k: int
    minimum_score: float = 0.0

    @field_validator("tenant_id", "knowledge_base_id")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))

    @field_validator("query_embedding")
    @classmethod
    def validate_query_embedding(
        cls,
        value: tuple[float, ...],
    ) -> tuple[float, ...]:
        return _validated_embedding(value, "query_embedding")

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k must be positive")
        return value

    @field_validator("minimum_score")
    @classmethod
    def validate_minimum_score(cls, value: float) -> float:
        return _validated_score(value, "minimum_score")


class VectorSearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    record: VectorRecord
    score: float

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        return _validated_score(value, "score")


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    knowledge_base_id: str
    hits: tuple[VectorSearchHit, ...] = ()

    @field_validator("tenant_id", "knowledge_base_id")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))

    @model_validator(mode="after")
    def validate_hit_scope_and_identity(self) -> Self:
        if any(hit.record.tenant_id != self.tenant_id for hit in self.hits):
            raise ValueError("vector record tenant_id does not match search result")
        if any(
            hit.record.knowledge_base_id != self.knowledge_base_id for hit in self.hits
        ):
            raise ValueError(
                "vector record knowledge_base_id does not match search result"
            )
        identities = tuple(
            (hit.record.document_id, hit.record.chunk_id) for hit in self.hits
        )
        if len(identities) != len(set(identities)):
            raise ValueError("vector search hit identities must be unique")
        return self


def _validated_embedding(
    value: tuple[float, ...],
    field_name: str,
) -> tuple[float, ...]:
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    if any(not isfinite(item) for item in value):
        raise ValueError(f"{field_name} must contain only finite values")
    return value


def _validated_score(value: float, field_name: str) -> float:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return value


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
