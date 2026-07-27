from dataclasses import dataclass, field
from math import inf, nan

import pytest
from pydantic import ValidationError

from app.embeddings import QueryEmbedder
from app.retrieval import RetrievalDocument, Retriever, VectorBackedRetriever
from app.vector_store import (
    SearchRequest,
    SearchResult,
    VectorRecord,
    VectorSearchHit,
    VectorStore,
)


@dataclass
class FakeQueryEmbedder:
    embedding: tuple[float, ...] = (1.0, 0.0)
    calls: list[tuple[str, str, str]] = field(default_factory=list)
    error: Exception | None = None

    def embed_query(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        text: str,
    ) -> tuple[float, ...]:
        self.calls.append((tenant_id, knowledge_base_id, text))
        if self.error is not None:
            raise self.error
        return self.embedding


@dataclass
class FakeVectorStore:
    result: SearchResult
    requests: list[SearchRequest] = field(default_factory=list)
    error: Exception | None = None

    def upsert(self, record: VectorRecord) -> None:
        raise AssertionError("upsert must not be called during retrieval")

    def search(self, request: SearchRequest) -> SearchResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result


def record(
    chunk_id: str = "chunk_1",
    *,
    tenant_id: str = "tenant_alpha",
    knowledge_base_id: str = "kb_support",
    document_id: str = "guide",
    text: str = "Synthetic retrieved context.",
    embedding: tuple[float, ...] = (1.0, 0.0),
    metadata: tuple[tuple[str, str], ...] = (("source", "synthetic"),),
) -> VectorRecord:
    return VectorRecord(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id=chunk_id,
        text=text,
        embedding=embedding,
        metadata=metadata,
    )


def hit(
    chunk_id: str = "chunk_1",
    *,
    score: float = 0.9,
    **record_changes: object,
) -> VectorSearchHit:
    stored = record(chunk_id)
    if record_changes:
        stored = VectorRecord.model_validate({**stored.model_dump(), **record_changes})
    return VectorSearchHit(record=stored, score=score)


def result(*hits: VectorSearchHit) -> SearchResult:
    return SearchResult(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        hits=hits,
    )


def subject(
    search_result: SearchResult,
    *,
    embedding: tuple[float, ...] = (1.0, 0.0),
) -> tuple[VectorBackedRetriever, FakeQueryEmbedder, FakeVectorStore]:
    embedder = FakeQueryEmbedder(embedding=embedding)
    store = FakeVectorStore(search_result)
    retriever: Retriever = VectorBackedRetriever(embedder, store)
    assert isinstance(retriever, VectorBackedRetriever)
    return retriever, embedder, store


def retrieve(
    retriever: Retriever,
    *,
    top_k: int = 3,
    minimum_score: float = 0.0,
) -> tuple[RetrievalDocument, ...]:
    return retriever.retrieve(
        tenant_id=" tenant_alpha ",
        knowledge_base_id=" kb_support ",
        query=" Synthetic question. ",
        top_k=top_k,
        minimum_score=minimum_score,
    ).documents


def test_exact_embedding_and_search_propagation_and_mapping() -> None:
    retriever, embedder, store = subject(result(hit()))

    documents = retrieve(retriever, minimum_score=0.6)

    assert embedder.calls == [("tenant_alpha", "kb_support", "Synthetic question.")]
    assert store.requests == [
        SearchRequest(
            tenant_id="tenant_alpha",
            knowledge_base_id="kb_support",
            query_embedding=(1.0, 0.0),
            top_k=3,
            minimum_score=0.6,
        )
    ]
    assert documents == (
        RetrievalDocument(
            tenant_id="tenant_alpha",
            knowledge_base_id="kb_support",
            document_id="guide",
            chunk_id="chunk_1",
            text="Synthetic retrieved context.",
            score=0.9,
        ),
    )
    assert len(embedder.calls) == len(store.requests) == 1


def test_empty_search_result_is_safe() -> None:
    retriever, _, _ = subject(result())

    assert retrieve(retriever) == ()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("tenant_id", "tenant_beta"),
        ("knowledge_base_id", "kb_other"),
    ],
)
def test_result_scope_mismatch_fails_closed(field_name: str, value: str) -> None:
    search_result = result()
    search_result = search_result.model_copy(update={field_name: value})
    retriever, _, _ = subject(search_result)

    with pytest.raises(ValueError, match=field_name):
        retrieve(retriever)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("tenant_id", "tenant_beta"),
        ("knowledge_base_id", "kb_other"),
    ],
)
def test_hit_scope_mismatch_fails_closed(field_name: str, value: str) -> None:
    mismatched = record().model_copy(update={field_name: value})
    search_result = SearchResult.model_construct(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        hits=(VectorSearchHit(record=mismatched, score=0.9),),
    )
    retriever, _, _ = subject(search_result)

    with pytest.raises(ValueError, match=field_name):
        retrieve(retriever)


@pytest.mark.parametrize("field_name", ["document_id", "chunk_id", "text"])
def test_blank_hit_content_fails_closed(field_name: str) -> None:
    invalid_record = record().model_copy(update={field_name: " "})
    search_result = SearchResult.model_construct(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        hits=(VectorSearchHit(record=invalid_record, score=0.9),),
    )
    retriever, _, _ = subject(search_result)

    with pytest.raises(ValueError, match=field_name):
        retrieve(retriever)


@pytest.mark.parametrize("score", [-0.1, 1.1, nan, inf])
def test_invalid_hit_score_fails_closed(score: float) -> None:
    invalid_hit = VectorSearchHit.model_construct(record=record(), score=score)
    search_result = SearchResult.model_construct(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        hits=(invalid_hit,),
    )
    retriever, _, _ = subject(search_result)

    with pytest.raises(ValueError, match="score"):
        retrieve(retriever)


def test_hit_dimension_mismatch_fails_closed() -> None:
    retriever, _, _ = subject(
        result(hit(embedding=(1.0,))),
        embedding=(1.0, 0.0),
    )

    with pytest.raises(ValueError, match="dimension"):
        retrieve(retriever)


def test_duplicate_identity_defense_fails_before_mapping() -> None:
    duplicate_hits = (hit(score=0.9), hit(score=0.8))
    search_result = SearchResult.model_construct(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        hits=duplicate_hits,
    )
    retriever, _, _ = subject(search_result)

    with pytest.raises(ValueError, match="identities must be unique"):
        retrieve(retriever)


def test_minimum_score_is_reenforced_if_store_ignores_it() -> None:
    retriever, _, _ = subject(
        result(
            hit("high", score=0.8),
            hit("low", score=0.4),
        )
    )

    documents = retrieve(retriever, minimum_score=0.6)

    assert [document.chunk_id for document in documents] == ["high"]


def test_top_k_and_deterministic_tie_ordering_are_enforced() -> None:
    retriever, _, _ = subject(
        result(
            hit("chunk_b", document_id="guide_b"),
            hit("chunk_c", document_id="guide_a"),
            hit("chunk_a", document_id="guide_a"),
        )
    )

    documents = retrieve(retriever, top_k=2)

    assert [document.chunk_id for document in documents] == [
        "chunk_a",
        "chunk_c",
    ]


def test_repeated_retrieval_is_deterministic() -> None:
    retriever, embedder, store = subject(result(hit()))

    first = retrieve(retriever)
    second = retrieve(retriever)

    assert first == second
    assert len(embedder.calls) == len(store.requests) == 2


@pytest.mark.parametrize("embedding", [(), (nan,), (inf,), (-inf,)])
def test_invalid_embedder_vector_is_rejected(
    embedding: tuple[float, ...],
) -> None:
    retriever, _, store = subject(result(), embedding=embedding)

    with pytest.raises(ValidationError, match="query_embedding"):
        retrieve(retriever)

    assert store.requests == []


def test_embedder_and_store_exceptions_propagate() -> None:
    embedder_error = RuntimeError("synthetic embedder failure")
    embedder = FakeQueryEmbedder(error=embedder_error)
    retriever = VectorBackedRetriever(embedder, FakeVectorStore(result()))
    with pytest.raises(RuntimeError, match="embedder failure"):
        retrieve(retriever)

    store_error = RuntimeError("synthetic store failure")
    store = FakeVectorStore(result(), error=store_error)
    retriever = VectorBackedRetriever(FakeQueryEmbedder(), store)
    with pytest.raises(RuntimeError, match="store failure"):
        retrieve(retriever)


def test_retrieval_document_does_not_leak_embedding_or_metadata() -> None:
    retriever, _, _ = subject(result(hit()))

    document = retrieve(retriever)[0]

    assert set(type(document).model_fields) == {
        "tenant_id",
        "knowledge_base_id",
        "document_id",
        "chunk_id",
        "text",
        "score",
    }
    assert not hasattr(document, "embedding")
    assert not hasattr(document, "metadata")


def test_protocols_are_structurally_compatible() -> None:
    embedder: QueryEmbedder = FakeQueryEmbedder()
    store: VectorStore = FakeVectorStore(result())
    retriever: Retriever = VectorBackedRetriever(embedder, store)

    assert retriever is not None
