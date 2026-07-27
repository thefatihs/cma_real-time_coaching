import pytest
from pydantic import ValidationError

from app.vector_store import (
    AtomicVectorBatchWriter,
    InMemoryVectorStore,
    SearchRequest,
    VectorBatchWriteRequest,
    VectorBatchWriteResult,
    VectorRecord,
    VectorRecordIdentity,
    VectorStore,
)


def record(
    document_id: str,
    chunk_id: str,
    *,
    tenant_id: str = "tenant_alpha",
    knowledge_base_id: str = "kb_support",
    text: str | None = None,
    embedding: tuple[float, ...] = (1.0, 0.0),
    metadata: tuple[tuple[str, str], ...] = (("source", "synthetic"),),
) -> VectorRecord:
    return VectorRecord(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id=chunk_id,
        text=text or f"Synthetic content for {document_id}/{chunk_id}.",
        embedding=embedding,
        metadata=metadata,
    )


def batch(
    *records: VectorRecord,
    tenant_id: str = "tenant_alpha",
    knowledge_base_id: str = "kb_support",
) -> VectorBatchWriteRequest:
    return VectorBatchWriteRequest(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        records=records,
    )


def identities(
    *values: tuple[str, str],
) -> tuple[VectorRecordIdentity, ...]:
    return tuple(
        VectorRecordIdentity(document_id=document_id, chunk_id=chunk_id)
        for document_id, chunk_id in values
    )


def stored_records(
    store: InMemoryVectorStore,
    *,
    dimension: int = 2,
) -> tuple[VectorRecord, ...]:
    result = store.search(
        SearchRequest(
            tenant_id="tenant_alpha",
            knowledge_base_id="kb_support",
            query_embedding=(1.0,) + (0.0,) * (dimension - 1),
            top_k=100,
            minimum_score=0.0,
        )
    )
    return tuple(hit.record for hit in result.hits)


def test_models_are_frozen_and_normalize_required_identifiers() -> None:
    identity = VectorRecordIdentity(document_id=" guide ", chunk_id=" chunk_1 ")
    request = batch(record("guide", "chunk_1"))
    result = VectorBatchWriteResult(
        tenant_id=" tenant_alpha ",
        knowledge_base_id=" kb_support ",
        inserted_identities=(identity,),
        unchanged_identities=(),
    )

    assert identity == VectorRecordIdentity(document_id="guide", chunk_id="chunk_1")
    assert result.tenant_id == "tenant_alpha"
    assert result.knowledge_base_id == "kb_support"
    with pytest.raises(ValidationError):
        identity.chunk_id = "changed"
    with pytest.raises(ValidationError):
        request.records = ()
    with pytest.raises(ValidationError):
        result.inserted_identities = ()


def test_protocols_are_structurally_compatible() -> None:
    concrete = InMemoryVectorStore()
    store: VectorStore = concrete
    writer: AtomicVectorBatchWriter = concrete

    assert store is concrete
    assert writer is concrete


def test_empty_batch_is_rejected() -> None:
    with pytest.raises(ValidationError, match="records cannot be empty"):
        batch()


def test_mixed_tenant_is_rejected() -> None:
    with pytest.raises(ValidationError, match="tenant_id"):
        batch(
            record("guide_a", "chunk_1"),
            record("guide_b", "chunk_2", tenant_id="tenant_beta"),
        )


def test_mixed_knowledge_base_is_rejected() -> None:
    with pytest.raises(ValidationError, match="knowledge_base_id"):
        batch(
            record("guide_a", "chunk_1"),
            record("guide_b", "chunk_2", knowledge_base_id="kb_other"),
        )


def test_duplicate_identity_inside_batch_is_rejected() -> None:
    with pytest.raises(ValidationError, match="identities must be unique"):
        batch(
            record("guide", "chunk_1"),
            record("guide", "chunk_1"),
        )


def test_same_chunk_id_in_different_documents_is_valid() -> None:
    store = InMemoryVectorStore()
    first = record("guide_a", "chunk_1")
    second = record("guide_b", "chunk_1")

    result = store.admit_batch(batch(first, second))

    assert result.inserted_identities == identities(
        ("guide_a", "chunk_1"),
        ("guide_b", "chunk_1"),
    )
    assert stored_records(store) == (first, second)


def test_new_records_are_inserted_and_existing_identical_records_are_unchanged() -> (
    None
):
    store = InMemoryVectorStore()
    existing = record("guide_a", "chunk_1")
    added = record("guide_b", "chunk_2")
    store.upsert(existing)

    result = store.admit_batch(batch(added, existing))

    assert result.inserted_identities == identities(("guide_b", "chunk_2"))
    assert result.unchanged_identities == identities(("guide_a", "chunk_1"))
    assert stored_records(store) == (existing, added)


@pytest.mark.parametrize(
    "conflicting",
    [
        record("guide", "chunk_1", text="Synthetic changed text."),
        record("guide", "chunk_1", embedding=(0.5, 0.5)),
        record(
            "guide",
            "chunk_1",
            metadata=(("source", "synthetic_changed"),),
        ),
    ],
    ids=["text", "embedding", "metadata"],
)
def test_changed_existing_record_conflicts(conflicting: VectorRecord) -> None:
    store = InMemoryVectorStore()
    original = record("guide", "chunk_1")
    store.upsert(original)

    with pytest.raises(ValueError, match="conflicts"):
        store.admit_batch(batch(conflicting))

    assert stored_records(store) == (original,)


def test_conflicting_batch_performs_no_partial_insertion() -> None:
    store = InMemoryVectorStore()
    original = record("guide_b", "chunk_2")
    store.upsert(original)
    new_record = record("guide_a", "chunk_1")
    conflict = record("guide_b", "chunk_2", text="Synthetic conflict.")

    with pytest.raises(ValueError, match="conflicts"):
        store.admit_batch(batch(new_record, conflict))

    assert stored_records(store) == (original,)


def test_dimension_mismatch_performs_no_mutation() -> None:
    store = InMemoryVectorStore()
    original = record("guide_a", "chunk_1")
    store.upsert(original)

    with pytest.raises(ValueError, match="dimension"):
        store.admit_batch(
            batch(record("guide_b", "chunk_2", embedding=(1.0, 0.0, 0.0)))
        )

    assert stored_records(store) == (original,)


def test_failed_batch_preserves_records_and_dimension_bookkeeping() -> None:
    store = InMemoryVectorStore()
    original = record("guide_a", "chunk_1")
    store.upsert(original)
    conflict = record("guide_a", "chunk_1", text="Synthetic conflict.")

    with pytest.raises(ValueError, match="conflicts"):
        store.admit_batch(batch(conflict))

    accepted = record("guide_b", "chunk_2", embedding=(0.5, 0.5))
    result = store.admit_batch(batch(accepted))

    assert result.inserted_identities == identities(("guide_b", "chunk_2"))
    assert stored_records(store) == (original, accepted)


def test_mixed_batch_dimensions_are_rejected_before_mutation() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValidationError, match="equal dimensions"):
        store.admit_batch(
            batch(
                record("guide_a", "chunk_1"),
                record("guide_b", "chunk_2", embedding=(1.0, 0.0, 0.0)),
            )
        )

    store.upsert(record("guide_c", "chunk_3", embedding=(1.0, 0.0, 0.0)))
    assert len(stored_records(store, dimension=3)) == 1


def test_result_ordering_is_deterministic() -> None:
    store = InMemoryVectorStore()
    unchanged_b = record("guide_b", "chunk_2")
    unchanged_a = record("guide_a", "chunk_3")
    store.upsert(unchanged_b)
    store.upsert(unchanged_a)

    result = store.admit_batch(
        batch(
            unchanged_b,
            record("guide_c", "chunk_2"),
            unchanged_a,
            record("guide_a", "chunk_1"),
        )
    )

    assert result.inserted_identities == identities(
        ("guide_a", "chunk_1"),
        ("guide_c", "chunk_2"),
    )
    assert result.unchanged_identities == identities(
        ("guide_a", "chunk_3"),
        ("guide_b", "chunk_2"),
    )


def test_repeated_identical_batch_is_deterministic() -> None:
    store = InMemoryVectorStore()
    request = batch(
        record("guide_b", "chunk_2"),
        record("guide_a", "chunk_1"),
    )

    first = store.admit_batch(request)
    second = store.admit_batch(request)
    third = store.admit_batch(request)

    assert first.inserted_identities == identities(
        ("guide_a", "chunk_1"),
        ("guide_b", "chunk_2"),
    )
    assert second == third
    assert second.inserted_identities == ()
    assert second.unchanged_identities == first.inserted_identities


def test_result_excludes_record_content() -> None:
    result = InMemoryVectorStore().admit_batch(batch(record("guide", "chunk_1")))

    assert set(type(result).model_fields) == {
        "tenant_id",
        "knowledge_base_id",
        "inserted_identities",
        "unchanged_identities",
    }
    identity = result.inserted_identities[0]
    assert set(type(identity).model_fields) == {"document_id", "chunk_id"}
    assert not hasattr(identity, "text")
    assert not hasattr(identity, "metadata")
    assert not hasattr(identity, "embedding")
