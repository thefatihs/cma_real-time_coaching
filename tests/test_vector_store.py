from math import inf, nan

import pytest
from pydantic import ValidationError

from app.vector_store import (
    InMemoryVectorStore,
    SearchRequest,
    SearchResult,
    VectorRecord,
    VectorStore,
)


def record(
    chunk_id: str = "chunk_1",
    *,
    tenant_id: str = "tenant_alpha",
    knowledge_base_id: str = "kb_support",
    document_id: str = "guide",
    embedding: tuple[float, ...] = (1.0, 0.0),
) -> VectorRecord:
    return VectorRecord(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id=chunk_id,
        embedding=embedding,
        metadata=(("source", "synthetic"),),
    )


def request(**changes: object) -> SearchRequest:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "knowledge_base_id": "kb_support",
        "query_embedding": (1.0, 0.0),
        "top_k": 5,
    }
    values.update(changes)
    return SearchRequest.model_validate(values)


def test_protocol_compatibility_and_upsert() -> None:
    store: VectorStore = InMemoryVectorStore()
    stored = record("chunk_1")

    store.upsert(stored)

    assert store.search(request()).records == (stored,)


def test_upsert_replaces_chunk_within_scope() -> None:
    store = InMemoryVectorStore()
    store.upsert(record("chunk_1", document_id="old"))
    replacement = record("chunk_1", document_id="new")

    store.upsert(replacement)

    assert store.search(request()).records == (replacement,)


def test_search_order_is_deterministic() -> None:
    store = InMemoryVectorStore()
    for item in (
        record("chunk_b", document_id="guide_b", embedding=(0.5, 0.0)),
        record("chunk_c", document_id="guide_a", embedding=(0.5, 0.0)),
        record("chunk_a", document_id="guide_a", embedding=(0.5, 0.0)),
        record("chunk_high", embedding=(1.0, 0.0)),
    ):
        store.upsert(item)

    result = store.search(request())

    assert [item.chunk_id for item in result.records] == [
        "chunk_high",
        "chunk_a",
        "chunk_c",
        "chunk_b",
    ]


def test_tenant_isolation() -> None:
    store = InMemoryVectorStore()
    store.upsert(record("alpha"))
    store.upsert(record("beta", tenant_id="tenant_beta"))

    result = store.search(request())

    assert [item.chunk_id for item in result.records] == ["alpha"]


def test_knowledge_base_isolation() -> None:
    store = InMemoryVectorStore()
    store.upsert(record("support"))
    store.upsert(record("sales", knowledge_base_id="kb_sales"))

    result = store.search(request())

    assert [item.chunk_id for item in result.records] == ["support"]


def test_top_k_limits_results() -> None:
    store = InMemoryVectorStore()
    store.upsert(record("first", embedding=(1.0, 0.0)))
    store.upsert(record("second", embedding=(0.5, 0.0)))

    result = store.search(request(top_k=1))

    assert [item.chunk_id for item in result.records] == ["first"]


def test_repeated_searches_are_identical() -> None:
    store = InMemoryVectorStore()
    store.upsert(record("chunk_1"))
    source_request = request()

    assert store.search(source_request) == store.search(source_request)


@pytest.mark.parametrize("embedding", [(), (nan, 0.0), (inf,), (-inf,)])
def test_invalid_record_embeddings_are_rejected(
    embedding: tuple[float, ...],
) -> None:
    with pytest.raises(ValidationError):
        record("chunk_1", embedding=embedding)


@pytest.mark.parametrize("embedding", [(), (nan,), (inf,)])
def test_invalid_query_embeddings_are_rejected(
    embedding: tuple[float, ...],
) -> None:
    with pytest.raises(ValidationError):
        request(query_embedding=embedding)


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (record, "tenant_id"),
        (record, "knowledge_base_id"),
        (record, "document_id"),
        (record, "chunk_id"),
        (request, "tenant_id"),
        (request, "knowledge_base_id"),
    ],
)
def test_required_ids_are_validated(factory: object, field: str) -> None:
    with pytest.raises(ValidationError, match=f"{field} cannot be empty"):
        factory(**{field: " "})  # type: ignore[operator]


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="top_k must be positive"):
        request(top_k=0)


def test_embedding_dimensions_are_enforced_per_scope() -> None:
    store = InMemoryVectorStore()
    store.upsert(record("chunk_1"))

    with pytest.raises(ValueError, match="dimension"):
        store.upsert(record("chunk_2", embedding=(1.0,)))
    with pytest.raises(ValueError, match="dimension"):
        store.search(request(query_embedding=(1.0,)))


def test_models_are_immutable() -> None:
    stored = record("chunk_1")
    source_request = request()
    result = SearchResult(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        records=(stored,),
    )

    with pytest.raises(ValidationError):
        stored.embedding = (0.0, 1.0)
    with pytest.raises(ValidationError):
        source_request.top_k = 1
    with pytest.raises(ValidationError):
        result.records = ()


def test_empty_search_is_safe() -> None:
    result = InMemoryVectorStore().search(request())

    assert result.records == ()
