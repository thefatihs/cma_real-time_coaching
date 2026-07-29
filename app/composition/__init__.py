"""Production-safe dependency composition boundaries."""

from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLRAGComposition,
    PostgreSQLVectorStoreSettings,
    compose_profile_bound_postgres_rag,
)
from app.composition.postgres_rag_orchestration import (
    LLMGatewayFactory,
    PostgreSQLRAGOrchestrationComposition,
    compose_profile_bound_postgres_rag_orchestration,
)
from app.composition.postgres_rag_runtime import (
    ProfileVerifiedPostgreSQLRAGRunner,
)

__all__ = [
    "KnowledgeBaseRAGProviderSettings",
    "LLMGatewayFactory",
    "ProfileVerifiedPostgreSQLRAGRunner",
    "PostgreSQLRAGOrchestrationComposition",
    "PostgreSQLRAGComposition",
    "PostgreSQLVectorStoreSettings",
    "compose_profile_bound_postgres_rag",
    "compose_profile_bound_postgres_rag_orchestration",
]
