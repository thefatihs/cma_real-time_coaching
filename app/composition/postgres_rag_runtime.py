"""Reusable readiness-verified PostgreSQL RAG orchestration runtime."""

from threading import Lock

from app.composition.postgres_rag_orchestration import (
    PostgreSQLRAGOrchestrationComposition,
)
from app.orchestration.models import OrchestrationRequest, OrchestrationResult
from app.vector_store.embedding_profile import KnowledgeBaseEmbeddingProfile
from app.vector_store.postgres.readiness import PostgreSQLSchemaReadinessChecker


class ProfileVerifiedPostgreSQLRAGRunner:
    """Run one profile-bound orchestration after explicit preparation."""

    def __init__(
        self,
        composition: PostgreSQLRAGOrchestrationComposition,
        readiness_checker: PostgreSQLSchemaReadinessChecker,
    ) -> None:
        if not isinstance(composition, PostgreSQLRAGOrchestrationComposition):
            raise ValueError(
                "composition must be PostgreSQLRAGOrchestrationComposition"
            )
        if not isinstance(readiness_checker, PostgreSQLSchemaReadinessChecker):
            raise ValueError(
                "readiness_checker must be PostgreSQLSchemaReadinessChecker"
            )
        self._composition = composition
        self._readiness_checker = readiness_checker
        self._prepared = False
        self._preparation_lock = Lock()

    def prepare(self) -> None:
        """Verify schema readiness and the exact registered profile once."""
        with self._preparation_lock:
            if self._prepared:
                return
            self._readiness_checker.verify()
            expected_profile = self._composition.postgres_rag.profile
            registered_profile = (
                self._composition.postgres_rag.profile_repository.get_profile(
                    tenant_id=expected_profile.tenant_id,
                    knowledge_base_id=expected_profile.knowledge_base_id,
                )
            )
            if registered_profile is None:
                raise ValueError("embedding profile is not registered")
            if not isinstance(
                registered_profile,
                KnowledgeBaseEmbeddingProfile,
            ):
                raise ValueError(
                    "profile repository returned an invalid embedding profile"
                )
            if registered_profile != expected_profile:
                raise ValueError(
                    "profile repository returned a conflicting embedding profile"
                )
            self._prepared = True

    def run(
        self,
        request: OrchestrationRequest,
    ) -> OrchestrationResult | None:
        """Delegate one validated request after successful preparation."""
        if not isinstance(request, OrchestrationRequest):
            raise ValueError("request must be OrchestrationRequest")
        expected_profile = self._composition.postgres_rag.profile
        if request.tenant_id != expected_profile.tenant_id:
            raise ValueError("request tenant_id does not match the bound profile")
        if request.knowledge_base_id != expected_profile.knowledge_base_id:
            raise ValueError(
                "request knowledge_base_id does not match the bound profile"
            )
        with self._preparation_lock:
            if not self._prepared:
                raise RuntimeError("PostgreSQL RAG runtime is not prepared")
        return self._composition.orchestrator.run(request)
