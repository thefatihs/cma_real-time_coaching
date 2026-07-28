"""Provider-neutral embedding-profile persistence boundary."""

from typing import Protocol

from app.vector_store.embedding_profile import KnowledgeBaseEmbeddingProfile


class EmbeddingProfileRepository(Protocol):
    """Register and retrieve immutable profiles by tenant/knowledge-base scope.

    Implementations must treat ``(tenant_id, knowledge_base_id)`` as the
    canonical identity. Registering a missing profile inserts and returns it.
    Registering a completely equal profile is an idempotent no-op that returns
    the canonical stored profile. Any differing model ID, vector dimension,
    normalization policy, or distance metric must fail closed without changing
    the stored profile. Replacement and deletion are not supported.

    Lookups return the exact stored immutable profile or ``None`` when missing.
    Tenant and knowledge-base scopes must remain isolated, required scope text
    must be validated consistently, and malformed stored profiles must fail
    closed rather than being silently repaired.
    """

    def register_profile(
        self,
        profile: KnowledgeBaseEmbeddingProfile,
    ) -> KnowledgeBaseEmbeddingProfile:
        """Insert a missing profile or return the equal canonical stored profile."""
        ...

    def get_profile(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> KnowledgeBaseEmbeddingProfile | None:
        """Return the exact stored profile for a validated scope, if present."""
        ...
