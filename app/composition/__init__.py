"""Production-safe dependency composition boundaries."""

from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLRAGComposition,
    PostgreSQLVectorStoreSettings,
    compose_profile_bound_postgres_rag,
)

__all__ = [
    "KnowledgeBaseRAGProviderSettings",
    "PostgreSQLRAGComposition",
    "PostgreSQLVectorStoreSettings",
    "compose_profile_bound_postgres_rag",
]
