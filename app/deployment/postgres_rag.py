"""Explicit PostgreSQL RAG readiness and profile provisioning."""

from typing import Any

from psycopg import Connection

from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
    PsycopgConnect,
    compose_profile_bound_postgres_rag,
)
from app.embeddings.sentence_transformers import BackendFactory
from app.vector_store.embedding_profile import KnowledgeBaseEmbeddingProfile
from app.vector_store.postgres.readiness import PostgreSQLSchemaReadinessChecker


def provision_profile_bound_postgres_rag(
    *,
    postgres_settings: PostgreSQLVectorStoreSettings,
    knowledge_base_settings: KnowledgeBaseRAGProviderSettings,
    psycopg_connect: PsycopgConnect,
    embedding_backend_factory: BackendFactory | None = None,
) -> KnowledgeBaseEmbeddingProfile:
    """Verify readiness, then explicitly provision one immutable profile."""
    if not isinstance(postgres_settings, PostgreSQLVectorStoreSettings):
        raise ValueError(
            "postgres_settings must be PostgreSQLVectorStoreSettings",
        )
    if not isinstance(knowledge_base_settings, KnowledgeBaseRAGProviderSettings):
        raise ValueError(
            "knowledge_base_settings must be KnowledgeBaseRAGProviderSettings",
        )
    if not callable(psycopg_connect):
        raise ValueError("psycopg_connect must be callable")
    if embedding_backend_factory is not None and not callable(
        embedding_backend_factory
    ):
        raise ValueError("embedding_backend_factory must be callable")

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
    registered = composition.profile_repository.register_profile(composition.profile)
    if not isinstance(registered, KnowledgeBaseEmbeddingProfile):
        raise ValueError(
            "profile repository returned an invalid embedding profile",
        )
    if registered != composition.profile:
        raise ValueError(
            "profile repository returned a conflicting embedding profile",
        )
    return registered
