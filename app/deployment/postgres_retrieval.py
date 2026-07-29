"""Explicit PostgreSQL RAG retrieval deployment operation."""

from math import isfinite
from typing import Any

from psycopg import Connection
from pydantic import BaseModel, ConfigDict, field_validator

from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
    PsycopgConnect,
    compose_profile_bound_postgres_rag,
)
from app.embeddings.sentence_transformers import BackendFactory
from app.retrieval.models import RetrievalResult
from app.vector_store.embedding_profile import KnowledgeBaseEmbeddingProfile
from app.vector_store.postgres.readiness import PostgreSQLSchemaReadinessChecker


class PostgreSQLRAGRetrievalRequest(BaseModel):
    """Immutable scoped query for explicit PostgreSQL RAG retrieval."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    knowledge_base_id: str
    query: str
    top_k: int
    minimum_score: float

    @field_validator("tenant_id", "knowledge_base_id", "query")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned

    @field_validator("top_k", mode="before")
    @classmethod
    def validate_top_k(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("top_k must be an integer")
        if value <= 0:
            raise ValueError("top_k must be positive")
        return value

    @field_validator("minimum_score", mode="before")
    @classmethod
    def validate_minimum_score(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("minimum_score must be numeric")
        numeric = float(value)
        if not isfinite(numeric) or not 0 <= numeric <= 1:
            raise ValueError("minimum_score must be finite and between 0 and 1")
        return numeric


def retrieve_profile_bound_postgres_rag(
    *,
    postgres_settings: PostgreSQLVectorStoreSettings,
    knowledge_base_settings: KnowledgeBaseRAGProviderSettings,
    request: PostgreSQLRAGRetrievalRequest,
    psycopg_connect: PsycopgConnect,
    embedding_backend_factory: BackendFactory | None = None,
) -> RetrievalResult:
    """Verify readiness and profile identity before explicitly retrieving."""
    if not isinstance(postgres_settings, PostgreSQLVectorStoreSettings):
        raise ValueError(
            "postgres_settings must be PostgreSQLVectorStoreSettings",
        )
    if not isinstance(knowledge_base_settings, KnowledgeBaseRAGProviderSettings):
        raise ValueError(
            "knowledge_base_settings must be KnowledgeBaseRAGProviderSettings",
        )
    if not isinstance(request, PostgreSQLRAGRetrievalRequest):
        raise ValueError("request must be PostgreSQLRAGRetrievalRequest")
    if not callable(psycopg_connect):
        raise ValueError("psycopg_connect must be callable")
    if embedding_backend_factory is not None and not callable(
        embedding_backend_factory
    ):
        raise ValueError("embedding_backend_factory must be callable")
    if request.tenant_id != knowledge_base_settings.tenant_id:
        raise ValueError("request tenant_id does not match provider scope")
    if request.knowledge_base_id != knowledge_base_settings.knowledge_base_id:
        raise ValueError("request knowledge_base_id does not match provider scope")

    composition = compose_profile_bound_postgres_rag(
        postgres_settings=postgres_settings,
        knowledge_base_settings=knowledge_base_settings,
        psycopg_connect=psycopg_connect,
        embedding_backend_factory=embedding_backend_factory,
    )

    def readiness_connection_factory() -> Connection[Any]:
        return psycopg_connect(
            conninfo=postgres_settings.dsn.get_secret_value(),
            connect_timeout=postgres_settings.connect_timeout_seconds,
            sslmode=postgres_settings.ssl_mode,
            application_name=postgres_settings.application_name,
            autocommit=False,
        )

    readiness_checker = PostgreSQLSchemaReadinessChecker(
        connection_factory=readiness_connection_factory,
    )
    readiness_checker.verify()

    registered = composition.profile_repository.get_profile(
        tenant_id=request.tenant_id,
        knowledge_base_id=request.knowledge_base_id,
    )
    if registered is None:
        raise ValueError("embedding profile is not registered")
    if not isinstance(registered, KnowledgeBaseEmbeddingProfile):
        raise ValueError("profile repository returned an invalid embedding profile")
    if registered != composition.profile:
        raise ValueError("profile repository returned a conflicting embedding profile")

    return composition.retriever.retrieve(
        tenant_id=request.tenant_id,
        knowledge_base_id=request.knowledge_base_id,
        query=request.query,
        top_k=request.top_k,
        minimum_score=request.minimum_score,
    )
