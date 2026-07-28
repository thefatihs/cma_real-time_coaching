from inspect import Parameter, signature
from typing import Any, get_type_hints

import pytest
from pydantic import ValidationError

from app.vector_store import (
    AtomicVectorBatchWriter,
    EmbeddingDistanceMetric,
    EmbeddingProfileRepository,
    InMemoryVectorStore,
    KnowledgeBaseEmbeddingProfile,
    SearchRequest,
    SearchResult,
    VectorBatchWriteRequest,
    VectorBatchWriteResult,
    VectorRecord,
    VectorRecordIdentity,
    VectorSearchHit,
    VectorStore,
)

ScopeKey = tuple[str, str]


class FakeEmbeddingProfileRepository:
    def __init__(self) -> None:
        self._profiles: dict[ScopeKey, KnowledgeBaseEmbeddingProfile] = {}

    def register_profile(
        self,
        profile: KnowledgeBaseEmbeddingProfile,
    ) -> KnowledgeBaseEmbeddingProfile:
        key = (profile.tenant_id, profile.knowledge_base_id)
        existing = self._profiles.get(key)
        if existing is None:
            self._profiles[key] = profile
            return profile
        if existing != profile:
            raise ValueError("embedding profile conflicts with stored profile")
        return existing

    def get_profile(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> KnowledgeBaseEmbeddingProfile | None:
        key = (
            _required_text(tenant_id, "tenant_id"),
            _required_text(knowledge_base_id, "knowledge_base_id"),
        )
        return self._profiles.get(key)


def profile(
    *,
    tenant_id: str = "tenant_alpha",
    knowledge_base_id: str = "kb_support",
    model_id: str = "synthetic-embedding-v1",
    vector_dimension: int = 384,
    normalize_embeddings: bool = True,
    distance_metric: EmbeddingDistanceMetric = EmbeddingDistanceMetric.COSINE,
) -> KnowledgeBaseEmbeddingProfile:
    return KnowledgeBaseEmbeddingProfile(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        model_id=model_id,
        vector_dimension=vector_dimension,
        normalize_embeddings=normalize_embeddings,
        distance_metric=distance_metric,
    )


def consume_repository(
    repository: EmbeddingProfileRepository,
    value: KnowledgeBaseEmbeddingProfile,
) -> tuple[
    KnowledgeBaseEmbeddingProfile,
    KnowledgeBaseEmbeddingProfile | None,
]:
    registered = repository.register_profile(value)
    stored = repository.get_profile(
        tenant_id=value.tenant_id,
        knowledge_base_id=value.knowledge_base_id,
    )
    return registered, stored


def test_fake_is_structurally_compatible_and_consumer_uses_both_methods() -> None:
    repository: EmbeddingProfileRepository = FakeEmbeddingProfileRepository()
    value = profile()

    registered, stored = consume_repository(repository, value)

    assert registered is value
    assert stored is value


def test_missing_scope_returns_none() -> None:
    repository = FakeEmbeddingProfileRepository()

    assert (
        repository.get_profile(
            tenant_id="tenant_alpha",
            knowledge_base_id="kb_missing",
        )
        is None
    )


def test_equal_registration_is_idempotent_and_returns_canonical_profile() -> None:
    repository = FakeEmbeddingProfileRepository()
    original = profile()
    equal_copy = profile()

    first = repository.register_profile(original)
    second = repository.register_profile(equal_copy)
    third = repository.register_profile(equal_copy)

    assert first is original
    assert second is original
    assert third is original
    assert (
        repository.get_profile(
            tenant_id="tenant_alpha",
            knowledge_base_id="kb_support",
        )
        is original
    )


@pytest.mark.parametrize(
    "conflicting",
    [
        profile(model_id="synthetic-embedding-v2"),
        profile(vector_dimension=768),
        profile(normalize_embeddings=False),
        profile(distance_metric=EmbeddingDistanceMetric.DOT_PRODUCT),
    ],
    ids=["model-id", "dimension", "normalization", "metric"],
)
def test_conflicts_fail_closed_without_replacing_profile(
    conflicting: KnowledgeBaseEmbeddingProfile,
) -> None:
    repository = FakeEmbeddingProfileRepository()
    original = profile()
    repository.register_profile(original)

    with pytest.raises(ValueError, match="conflicts"):
        repository.register_profile(conflicting)

    assert (
        repository.get_profile(
            tenant_id="tenant_alpha",
            knowledge_base_id="kb_support",
        )
        is original
    )


def test_tenant_and_knowledge_base_scopes_are_isolated() -> None:
    repository = FakeEmbeddingProfileRepository()
    tenant_alpha = profile()
    tenant_beta = profile(tenant_id="tenant_beta")
    other_knowledge_base = profile(knowledge_base_id="kb_sales")

    repository.register_profile(tenant_alpha)
    repository.register_profile(tenant_beta)
    repository.register_profile(other_knowledge_base)

    assert (
        repository.get_profile(
            tenant_id="tenant_alpha",
            knowledge_base_id="kb_support",
        )
        is tenant_alpha
    )
    assert (
        repository.get_profile(
            tenant_id="tenant_beta",
            knowledge_base_id="kb_support",
        )
        is tenant_beta
    )
    assert (
        repository.get_profile(
            tenant_id="tenant_alpha",
            knowledge_base_id="kb_sales",
        )
        is other_knowledge_base
    )


def test_lookup_scope_is_normalized_and_blank_values_are_rejected() -> None:
    repository = FakeEmbeddingProfileRepository()
    value = profile()
    repository.register_profile(value)

    assert (
        repository.get_profile(
            tenant_id=" tenant_alpha ",
            knowledge_base_id=" kb_support ",
        )
        is value
    )
    with pytest.raises(ValueError, match="tenant_id"):
        repository.get_profile(tenant_id=" ", knowledge_base_id="kb_support")
    with pytest.raises(ValueError, match="knowledge_base_id"):
        repository.get_profile(tenant_id="tenant_alpha", knowledge_base_id=" ")


def test_returned_profile_is_immutable() -> None:
    repository = FakeEmbeddingProfileRepository()
    repository.register_profile(profile())
    stored = repository.get_profile(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
    )

    assert stored is not None
    with pytest.raises(ValidationError):
        stored.model_id = "changed"


def test_repeated_operations_are_deterministic() -> None:
    repository = FakeEmbeddingProfileRepository()
    value = profile()

    results = tuple(repository.register_profile(value) for _ in range(3))
    lookups = tuple(
        repository.get_profile(
            tenant_id="tenant_alpha",
            knowledge_base_id="kb_support",
        )
        for _ in range(3)
    )

    assert results == lookups == (value, value, value)
    assert all(item is value for item in (*results, *lookups))


def test_protocol_method_signatures_and_annotations_are_exact() -> None:
    register = signature(EmbeddingProfileRepository.register_profile)
    lookup = signature(EmbeddingProfileRepository.get_profile)
    register_hints = get_type_hints(EmbeddingProfileRepository.register_profile)
    lookup_hints = get_type_hints(EmbeddingProfileRepository.get_profile)

    assert tuple(register.parameters) == ("self", "profile")
    assert register.parameters["profile"].kind is Parameter.POSITIONAL_OR_KEYWORD
    assert register_hints == {
        "profile": KnowledgeBaseEmbeddingProfile,
        "return": KnowledgeBaseEmbeddingProfile,
    }
    assert tuple(lookup.parameters) == (
        "self",
        "tenant_id",
        "knowledge_base_id",
    )
    assert lookup.parameters["tenant_id"].kind is Parameter.KEYWORD_ONLY
    assert lookup.parameters["knowledge_base_id"].kind is Parameter.KEYWORD_ONLY
    assert lookup_hints == {
        "tenant_id": str,
        "knowledge_base_id": str,
        "return": KnowledgeBaseEmbeddingProfile | None,
    }
    assert not any(
        fragment in repr(hints).lower()
        for hints in (register_hints, lookup_hints)
        for fragment in ("sql", "database", "client", "connection")
    )


def test_new_and_existing_vector_store_exports_remain_available() -> None:
    exports: tuple[Any, ...] = (
        AtomicVectorBatchWriter,
        EmbeddingDistanceMetric,
        EmbeddingProfileRepository,
        InMemoryVectorStore,
        KnowledgeBaseEmbeddingProfile,
        SearchRequest,
        SearchResult,
        VectorBatchWriteRequest,
        VectorBatchWriteResult,
        VectorRecord,
        VectorRecordIdentity,
        VectorSearchHit,
        VectorStore,
    )

    assert all(export is not None for export in exports)


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
