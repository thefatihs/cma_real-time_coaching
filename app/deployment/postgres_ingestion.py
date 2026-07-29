"""Explicit PostgreSQL RAG chunk ingestion deployment operation."""

from typing import Any

from psycopg import Connection

from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
    PsycopgConnect,
    compose_profile_bound_postgres_rag,
)
from app.embeddings.sentence_transformers import BackendFactory
from app.ingestion.models import DocumentIngestionRequest
from app.vector_store.embedding_profile import KnowledgeBaseEmbeddingProfile
from app.vector_store.models import VectorBatchWriteResult
from app.vector_store.postgres.readiness import PostgreSQLSchemaReadinessChecker


def ingest_profile_bound_postgres_rag(
    *,
    postgres_settings: PostgreSQLVectorStoreSettings,
    knowledge_base_settings: KnowledgeBaseRAGProviderSettings,
    request: DocumentIngestionRequest,
    psycopg_connect: PsycopgConnect,
    embedding_backend_factory: BackendFactory | None = None,
) -> VectorBatchWriteResult:
    """Verify readiness and profile identity before explicitly ingesting chunks."""
    if not isinstance(postgres_settings, PostgreSQLVectorStoreSettings):
        raise ValueError(
            "postgres_settings must be PostgreSQLVectorStoreSettings",
        )
    if not isinstance(knowledge_base_settings, KnowledgeBaseRAGProviderSettings):
        raise ValueError(
            "knowledge_base_settings must be KnowledgeBaseRAGProviderSettings",
        )
    if not isinstance(request, DocumentIngestionRequest):
        raise ValueError("request must be DocumentIngestionRequest")
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

    return composition.ingestion_service.ingest(request)
