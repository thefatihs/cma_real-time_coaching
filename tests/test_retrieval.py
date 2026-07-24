import pytest
from pydantic import ValidationError

from app.retrieval import (
    InMemoryRetriever,
    RetrievalDocument,
    RetrievalResult,
    Retriever,
)


def document(
    chunk_id: str,
    *,
    tenant_id: str = "tenant_alpha",
    knowledge_base_id: str = "kb_support",
    document_id: str = "guide",
    score: float = 0.8,
) -> RetrievalDocument:
    return RetrievalDocument(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id=chunk_id,
        text=f"Synthetic support content for {chunk_id}.",
        score=score,
    )


def test_same_tenant_retrieval_and_protocol_compatibility() -> None:
    retriever: Retriever = InMemoryRetriever((document("chunk_1"),))

    result = retriever.retrieve(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        top_k=5,
    )

    assert result == RetrievalResult(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        documents=(document("chunk_1"),),
    )


def test_cross_tenant_documents_are_never_returned() -> None:
    retriever = InMemoryRetriever(
        (
            document("alpha_chunk"),
            document("beta_chunk", tenant_id="tenant_beta", score=1.0),
        )
    )

    result = retriever.retrieve(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        top_k=5,
    )

    assert [item.chunk_id for item in result.documents] == ["alpha_chunk"]


def test_knowledge_base_isolation_is_enforced() -> None:
    retriever = InMemoryRetriever(
        (
            document("support_chunk"),
            document("sales_chunk", knowledge_base_id="kb_sales", score=1.0),
        )
    )

    result = retriever.retrieve(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        top_k=5,
    )

    assert [item.chunk_id for item in result.documents] == ["support_chunk"]


def test_top_k_and_minimum_score_are_applied() -> None:
    retriever = InMemoryRetriever(
        (
            document("high", score=0.9),
            document("medium", score=0.7),
            document("low", score=0.4),
        )
    )

    result = retriever.retrieve(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        top_k=1,
        minimum_score=0.6,
    )

    assert [item.chunk_id for item in result.documents] == ["high"]


def test_ordering_is_deterministic_for_equal_scores() -> None:
    retriever = InMemoryRetriever(
        (
            document("chunk_b", document_id="guide_b"),
            document("chunk_c", document_id="guide_a"),
            document("chunk_a", document_id="guide_a"),
        )
    )

    result = retriever.retrieve(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        top_k=5,
    )

    assert [item.chunk_id for item in result.documents] == [
        "chunk_a",
        "chunk_c",
        "chunk_b",
    ]


def test_empty_result_is_safe() -> None:
    result = InMemoryRetriever().retrieve(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        top_k=5,
    )

    assert result.documents == ()


@pytest.mark.parametrize("tenant_id", ["", "  "])
def test_invalid_tenant_identity_is_rejected(tenant_id: str) -> None:
    with pytest.raises(ValueError, match="tenant_id cannot be empty"):
        InMemoryRetriever().retrieve(
            tenant_id=tenant_id,
            knowledge_base_id="kb_support",
            top_k=5,
        )


def test_missing_tenant_identity_is_rejected_by_interface() -> None:
    with pytest.raises(TypeError):
        InMemoryRetriever().retrieve(  # type: ignore[call-arg]
            knowledge_base_id="kb_support",
            top_k=5,
        )


def test_document_requires_valid_tenant_identity() -> None:
    with pytest.raises(ValidationError, match="tenant_id cannot be empty"):
        document("chunk_1", tenant_id=" ")


def test_duplicate_chunk_within_namespace_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        InMemoryRetriever(
            (
                document("chunk_1", document_id="guide_a"),
                document("chunk_1", document_id="guide_b"),
            )
        )


def test_same_chunk_id_in_different_namespaces_is_allowed() -> None:
    retriever = InMemoryRetriever(
        (
            document("chunk_1"),
            document("chunk_1", tenant_id="tenant_beta"),
            document("chunk_1", knowledge_base_id="kb_sales"),
        )
    )

    result = retriever.retrieve(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        top_k=5,
    )

    assert result.documents == (document("chunk_1"),)


def test_result_rejects_documents_from_another_scope() -> None:
    with pytest.raises(ValidationError, match="tenant_id does not match"):
        RetrievalResult(
            tenant_id="tenant_alpha",
            knowledge_base_id="kb_support",
            documents=(document("chunk_1", tenant_id="tenant_beta"),),
        )
