"""Immutable knowledge-base embedding compatibility contract."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator


class EmbeddingDistanceMetric(str, Enum):
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


class KnowledgeBaseEmbeddingProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    knowledge_base_id: str
    model_id: str
    vector_dimension: int
    normalize_embeddings: bool
    distance_metric: EmbeddingDistanceMetric

    @field_validator("tenant_id", "knowledge_base_id", "model_id")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned

    @field_validator("vector_dimension", mode="before")
    @classmethod
    def validate_vector_dimension(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("vector_dimension must be an integer")
        if value <= 0:
            raise ValueError("vector_dimension must be positive")
        return value

    @field_validator("normalize_embeddings", mode="before")
    @classmethod
    def validate_normalize_embeddings(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("normalize_embeddings must be a boolean")
        return value
