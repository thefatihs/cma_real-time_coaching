"""SQL-free PostgreSQL embedding-profile repository domain implementation."""

from app.vector_store.embedding_profile import (
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
)
from app.vector_store.postgres.contracts import (
    PostgreSQLVectorTransaction,
    PostgreSQLVectorTransactionRunner,
)


class PostgreSQLEmbeddingProfileRepository:
    """Persist immutable profiles through a runner-owned transaction boundary."""

    def __init__(
        self,
        transaction_runner: PostgreSQLVectorTransactionRunner,
    ) -> None:
        if not callable(getattr(transaction_runner, "run_in_transaction", None)):
            raise ValueError("transaction_runner.run_in_transaction must be callable")
        self._transaction_runner = transaction_runner

    def register_profile(
        self,
        profile: KnowledgeBaseEmbeddingProfile,
    ) -> KnowledgeBaseEmbeddingProfile:
        _validate_profile(profile)

        def register(
            transaction: PostgreSQLVectorTransaction,
        ) -> KnowledgeBaseEmbeddingProfile:
            transaction.acquire_scope_lock(
                tenant_id=profile.tenant_id,
                knowledge_base_id=profile.knowledge_base_id,
            )
            stored = transaction.get_profile(
                tenant_id=profile.tenant_id,
                knowledge_base_id=profile.knowledge_base_id,
                for_update=True,
            )
            if stored is None:
                transaction.insert_profile(profile)
                return profile
            _validate_profile_scope(
                stored,
                tenant_id=profile.tenant_id,
                knowledge_base_id=profile.knowledge_base_id,
            )
            if stored != profile:
                raise ValueError("embedding profile conflicts with stored profile")
            return stored

        return self._transaction_runner.run_in_transaction(register)

    def get_profile(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> KnowledgeBaseEmbeddingProfile | None:
        tenant = _required_text(tenant_id, "tenant_id")
        knowledge_base = _required_text(
            knowledge_base_id,
            "knowledge_base_id",
        )

        def lookup(
            transaction: PostgreSQLVectorTransaction,
        ) -> KnowledgeBaseEmbeddingProfile | None:
            stored = transaction.get_profile(
                tenant_id=tenant,
                knowledge_base_id=knowledge_base,
                for_update=False,
            )
            if stored is None:
                return None
            _validate_profile_scope(
                stored,
                tenant_id=tenant,
                knowledge_base_id=knowledge_base,
            )
            return stored

        return self._transaction_runner.run_in_transaction(lookup)


def _validate_profile_scope(
    value: object,
    *,
    tenant_id: str,
    knowledge_base_id: str,
) -> None:
    profile = _validated_profile(value)
    if profile.tenant_id != tenant_id:
        raise ValueError("stored profile tenant_id does not match requested scope")
    if profile.knowledge_base_id != knowledge_base_id:
        raise ValueError(
            "stored profile knowledge_base_id does not match requested scope"
        )


def _validate_profile(value: object) -> None:
    _validated_profile(value)


def _validated_profile(value: object) -> KnowledgeBaseEmbeddingProfile:
    if not isinstance(value, KnowledgeBaseEmbeddingProfile):
        raise ValueError("profile must be a KnowledgeBaseEmbeddingProfile")
    try:
        tenant_id = value.tenant_id
        knowledge_base_id = value.knowledge_base_id
        model_id = value.model_id
        vector_dimension = value.vector_dimension
        normalize_embeddings = value.normalize_embeddings
        distance_metric = value.distance_metric
    except AttributeError as error:
        raise ValueError("embedding profile is malformed") from error
    _require_canonical_text(tenant_id, "tenant_id")
    _require_canonical_text(knowledge_base_id, "knowledge_base_id")
    _require_canonical_text(model_id, "model_id")
    if type(vector_dimension) is not int or vector_dimension <= 0:
        raise ValueError("vector_dimension must be a positive integer")
    if type(normalize_embeddings) is not bool:
        raise ValueError("normalize_embeddings must be a boolean")
    if not isinstance(distance_metric, EmbeddingDistanceMetric):
        raise ValueError("distance_metric must be an EmbeddingDistanceMetric")
    return value


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def _require_canonical_text(value: object, field_name: str) -> None:
    cleaned = _required_text(value, field_name)
    if value != cleaned:
        raise ValueError(f"{field_name} must be canonical")
