"""Tests for the SQL-free PostgreSQL vector boundary."""

import dataclasses
import inspect
import json
import math
from collections.abc import Callable
from typing import TypeVar, get_type_hints

import pytest

from app.vector_store import (
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
    VectorRecordIdentity,
)
from app.vector_store.postgres import (
    PostgreSQLCosineSearchRow,
    PostgreSQLStoredVectorRow,
    PostgreSQLVectorTransaction,
    PostgreSQLVectorTransactionRunner,
    canonicalize_float32_embedding,
    cosine_distance_to_relevance,
    cosine_minimum_score_to_maximum_distance,
    decode_ordered_metadata,
    encode_ordered_metadata,
    order_cosine_search_rows,
)

T = TypeVar("T")


def _profile() -> KnowledgeBaseEmbeddingProfile:
    return KnowledgeBaseEmbeddingProfile(
        tenant_id="tenant_synthetic",
        knowledge_base_id="kb_synthetic",
        model_id="model_synthetic",
        vector_dimension=2,
        normalize_embeddings=True,
        distance_metric=EmbeddingDistanceMetric.COSINE,
    )


def _stored_row(
    *,
    document_id: str = "document_a",
    chunk_id: str = "chunk_a",
) -> PostgreSQLStoredVectorRow:
    return PostgreSQLStoredVectorRow(
        tenant_id="tenant_synthetic",
        knowledge_base_id="kb_synthetic",
        document_id=document_id,
        chunk_id=chunk_id,
        text="Synthetic knowledge text.",
        embedding=(0.25, 0.5),
        metadata_json='[["kind","synthetic"]]',
    )


def _search_row(
    document_id: str,
    chunk_id: str,
    cosine_distance: float,
    embedding: tuple[float, ...],
) -> PostgreSQLCosineSearchRow:
    return PostgreSQLCosineSearchRow(
        tenant_id="tenant_synthetic",
        knowledge_base_id="kb_synthetic",
        document_id=document_id,
        chunk_id=chunk_id,
        text="Synthetic search text.",
        embedding=embedding,
        metadata_json="[]",
        cosine_distance=cosine_distance,
    )


class FakeTransaction:
    def __init__(self) -> None:
        self.scope_locks: list[tuple[str, str]] = []
        self.profile: KnowledgeBaseEmbeddingProfile | None = None
        self.profile_reads: list[tuple[str, str, bool]] = []
        self.inserted_profiles: list[KnowledgeBaseEmbeddingProfile] = []
        self.record_reads: list[tuple[str, str, tuple[VectorRecordIdentity, ...]]] = []
        self.records: tuple[PostgreSQLStoredVectorRow, ...] = ()
        self.inserted_rows: list[tuple[PostgreSQLStoredVectorRow, ...]] = []
        self.replaced_rows: list[PostgreSQLStoredVectorRow] = []
        self.search_rows: tuple[PostgreSQLCosineSearchRow, ...] = ()
        self.searches: list[tuple[str, str, tuple[float, ...], int, float]] = []

    def acquire_scope_lock(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> None:
        self.scope_locks.append((tenant_id, knowledge_base_id))

    def get_profile(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        for_update: bool,
    ) -> KnowledgeBaseEmbeddingProfile | None:
        self.profile_reads.append((tenant_id, knowledge_base_id, for_update))
        return self.profile

    def insert_profile(self, profile: KnowledgeBaseEmbeddingProfile) -> None:
        self.inserted_profiles.append(profile)

    def get_records(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        identities: tuple[VectorRecordIdentity, ...],
    ) -> tuple[PostgreSQLStoredVectorRow, ...]:
        self.record_reads.append((tenant_id, knowledge_base_id, identities))
        return self.records

    def insert_records(
        self,
        rows: tuple[PostgreSQLStoredVectorRow, ...],
    ) -> None:
        self.inserted_rows.append(rows)

    def replace_record(self, row: PostgreSQLStoredVectorRow) -> None:
        self.replaced_rows.append(row)

    def search_cosine(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        query_embedding: tuple[float, ...],
        top_k: int,
        maximum_cosine_distance: float,
    ) -> tuple[PostgreSQLCosineSearchRow, ...]:
        self.searches.append(
            (
                tenant_id,
                knowledge_base_id,
                query_embedding,
                top_k,
                maximum_cosine_distance,
            )
        )
        return self.search_rows


class FakeRunner:
    def __init__(self, transaction: FakeTransaction) -> None:
        self.transaction = transaction
        self.callback_calls = 0
        self.commits = 0
        self.rollbacks = 0
        self.releases = 0

    def run_in_transaction(
        self,
        operation: Callable[[PostgreSQLVectorTransaction], T],
    ) -> T:
        try:
            self.callback_calls += 1
            result = operation(self.transaction)
        except Exception:
            self.rollbacks += 1
            raise
        else:
            self.commits += 1
            return result
        finally:
            self.releases += 1


def _accept_transaction(value: PostgreSQLVectorTransaction) -> None:
    del value


def _accept_runner(value: PostgreSQLVectorTransactionRunner) -> None:
    del value


def test_fake_transaction_and_runner_structurally_match_protocols() -> None:
    transaction = FakeTransaction()
    runner = FakeRunner(transaction)

    _accept_transaction(transaction)
    _accept_runner(runner)


def test_runner_success_lifecycle_and_result_identity() -> None:
    runner = FakeRunner(FakeTransaction())
    expected = object()

    result = runner.run_in_transaction(lambda transaction: expected)

    assert result is expected
    assert runner.callback_calls == 1
    assert runner.commits == 1
    assert runner.rollbacks == 0
    assert runner.releases == 1


def test_runner_failure_lifecycle_and_exception_identity() -> None:
    runner = FakeRunner(FakeTransaction())
    expected = RuntimeError("synthetic transaction failure")

    def fail(transaction: PostgreSQLVectorTransaction) -> None:
        del transaction
        raise expected

    with pytest.raises(RuntimeError) as captured:
        runner.run_in_transaction(fail)

    assert captured.value is expected
    assert runner.callback_calls == 1
    assert runner.commits == 0
    assert runner.rollbacks == 1
    assert runner.releases == 1


def test_transaction_records_all_domain_operations() -> None:
    transaction = FakeTransaction()
    profile = _profile()
    identity = VectorRecordIdentity(document_id="document_a", chunk_id="chunk_a")
    stored_row = _stored_row()
    search_row = _search_row("document_a", "chunk_a", 0.25, (0.25, 0.5))
    transaction.profile = profile
    transaction.records = (stored_row,)
    transaction.search_rows = (search_row,)

    transaction.acquire_scope_lock(
        tenant_id="tenant_synthetic",
        knowledge_base_id="kb_synthetic",
    )
    assert (
        transaction.get_profile(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            for_update=True,
        )
        is profile
    )
    transaction.insert_profile(profile)
    assert transaction.get_records(
        tenant_id="tenant_synthetic",
        knowledge_base_id="kb_synthetic",
        identities=(identity,),
    ) == (stored_row,)
    transaction.insert_records((stored_row,))
    transaction.replace_record(stored_row)
    returned_search_rows = transaction.search_cosine(
        tenant_id="tenant_synthetic",
        knowledge_base_id="kb_synthetic",
        query_embedding=(0.25, 0.5),
        top_k=3,
        maximum_cosine_distance=0.5,
    )
    assert returned_search_rows == (search_row,)
    assert returned_search_rows[0].embedding == (0.25, 0.5)

    assert transaction.scope_locks == [("tenant_synthetic", "kb_synthetic")]
    assert transaction.profile_reads == [("tenant_synthetic", "kb_synthetic", True)]
    assert transaction.inserted_profiles == [profile]
    assert transaction.record_reads == [
        ("tenant_synthetic", "kb_synthetic", (identity,))
    ]
    assert transaction.inserted_rows == [(stored_row,)]
    assert transaction.replaced_rows == [stored_row]
    assert transaction.searches == [
        ("tenant_synthetic", "kb_synthetic", (0.25, 0.5), 3, 0.5)
    ]


def test_transaction_protocol_does_not_expose_lifecycle_methods() -> None:
    assert not hasattr(PostgreSQLVectorTransaction, "commit")
    assert not hasattr(PostgreSQLVectorTransaction, "rollback")
    assert not hasattr(PostgreSQLVectorTransaction, "close")


def test_stored_row_type_is_frozen_slotted_and_has_exact_fields() -> None:
    assert tuple(
        field.name for field in dataclasses.fields(PostgreSQLStoredVectorRow)
    ) == (
        "tenant_id",
        "knowledge_base_id",
        "document_id",
        "chunk_id",
        "text",
        "embedding",
        "metadata_json",
    )
    assert "__slots__" in PostgreSQLStoredVectorRow.__dict__


def test_search_row_type_is_frozen_slotted_and_has_exact_fields() -> None:
    assert tuple(
        field.name for field in dataclasses.fields(PostgreSQLCosineSearchRow)
    ) == (
        "tenant_id",
        "knowledge_base_id",
        "document_id",
        "chunk_id",
        "text",
        "embedding",
        "metadata_json",
        "cosine_distance",
    )
    assert "__slots__" in PostgreSQLCosineSearchRow.__dict__


def test_search_row_requires_and_preserves_tuple_embedding() -> None:
    parameters = inspect.signature(PostgreSQLCosineSearchRow).parameters
    embedding = (0.125, 0.75)

    row = _search_row("document_a", "chunk_a", 0.25, embedding)

    assert parameters["embedding"].default is inspect.Parameter.empty
    assert get_type_hints(PostgreSQLCosineSearchRow)["embedding"] == tuple[float, ...]
    assert row.embedding is embedding


def test_search_row_contains_all_vector_record_stored_fields() -> None:
    row_fields = {field.name for field in dataclasses.fields(PostgreSQLCosineSearchRow)}

    assert {
        "tenant_id",
        "knowledge_base_id",
        "document_id",
        "chunk_id",
        "text",
        "embedding",
        "metadata_json",
    } <= row_fields


def test_stored_row_is_immutable() -> None:
    row = _stored_row()

    with pytest.raises(dataclasses.FrozenInstanceError):
        row.text = "Changed synthetic text."  # type: ignore[misc]


def test_search_row_is_immutable() -> None:
    row = _search_row("document_a", "chunk_a", 0.25, (0.25, 0.5))

    with pytest.raises(dataclasses.FrozenInstanceError):
        row.text = "Changed synthetic text."  # type: ignore[misc]


def test_protocol_annotations_reuse_existing_contracts_and_no_client_types() -> None:
    transaction_methods = (
        "acquire_scope_lock",
        "get_profile",
        "insert_profile",
        "get_records",
        "insert_records",
        "replace_record",
        "search_cosine",
    )
    rendered_annotations = " ".join(
        str(get_type_hints(getattr(PostgreSQLVectorTransaction, method)))
        for method in transaction_methods
    )
    runner_annotations = str(
        get_type_hints(PostgreSQLVectorTransactionRunner.run_in_transaction)
    )

    assert "KnowledgeBaseEmbeddingProfile" in rendered_annotations
    assert "VectorRecordIdentity" in rendered_annotations
    for forbidden in ("psycopg", "cursor", "connection", "sqlalchemy"):
        assert forbidden not in rendered_annotations.lower()
        assert forbidden not in runner_annotations.lower()


def test_protocol_exposes_only_approved_domain_methods() -> None:
    methods = {
        name
        for name, value in inspect.getmembers(
            PostgreSQLVectorTransaction,
            inspect.isfunction,
        )
        if not name.startswith("_")
    }

    assert methods == {
        "acquire_scope_lock",
        "get_profile",
        "get_records",
        "insert_profile",
        "insert_records",
        "replace_record",
        "search_cosine",
    }


def test_float32_canonicalization_is_exact_idempotent_and_nonmutating() -> None:
    embedding = [0.1, -0.2]
    original = embedding.copy()

    canonical = canonicalize_float32_embedding(
        embedding,
        expected_dimension=2,
    )

    assert canonical == (
        0.10000000149011612,
        -0.20000000298023224,
    )
    assert canonicalize_float32_embedding(canonical, expected_dimension=2) == canonical
    assert embedding == original


@pytest.mark.parametrize("expected_dimension", [True, False, 0, -1, 1.0, "2"])
def test_float32_rejects_invalid_expected_dimension(
    expected_dimension: object,
) -> None:
    with pytest.raises(ValueError):
        canonicalize_float32_embedding(
            (0.1,),
            expected_dimension=expected_dimension,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "embedding",
    [
        (),
        "0.1",
        b"0.1",
        iter((0.1,)),
        (True,),
        ("0.1",),
        (float("nan"),),
        (float("inf"),),
        (float("-inf"),),
        (1e100,),
    ],
)
def test_float32_rejects_invalid_embedding(embedding: object) -> None:
    with pytest.raises(ValueError):
        canonicalize_float32_embedding(embedding, expected_dimension=1)


def test_float32_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        canonicalize_float32_embedding((0.1, 0.2), expected_dimension=1)


def test_ordered_metadata_round_trip_is_compact_unicode_and_deterministic() -> None:
    metadata = (("kind", "synthetic"), ("language", "Türkçe"))

    encoded = encode_ordered_metadata(metadata)

    assert encoded == '[["kind","synthetic"],["language","Türkçe"]]'
    assert json.loads(encoded) == [
        ["kind", "synthetic"],
        ["language", "Türkçe"],
    ]
    assert decode_ordered_metadata(encoded) == metadata
    assert encode_ordered_metadata(metadata) == encoded


@pytest.mark.parametrize(
    "metadata",
    [
        ((" key", "value"),),
        (("key ", "value"),),
        (("key", " value"),),
        (("key", "value "),),
        (("", "value"),),
        (("key", ""),),
        (("key", "value"), ("key", "other")),
        (("key", 1),),
        ("not-a-pair",),
    ],
)
def test_metadata_encoding_rejects_noncanonical_or_malformed_values(
    metadata: object,
) -> None:
    with pytest.raises(ValueError):
        encode_ordered_metadata(metadata)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "metadata_json",
    [
        None,
        1,
        "{",
        '{"key":"value"}',
        '["not-a-pair"]',
        '[["key"]]',
        '[["key","value","extra"]]',
        '[[1,"value"]]',
        '[["key",1]]',
        '[[" key","value"]]',
        '[["key"," value"]]',
        '[["key","value"],["key","other"]]',
    ],
)
def test_metadata_decoding_rejects_malformed_or_noncanonical_values(
    metadata_json: object,
) -> None:
    with pytest.raises(ValueError):
        decode_ordered_metadata(metadata_json)


@pytest.mark.parametrize(
    "score, expected_distance",
    [(0.0, 2.0), (0.25, 1.5), (0.5, 1.0), (1.0, 0.0)],
)
def test_cosine_score_to_distance_boundaries(
    score: float,
    expected_distance: float,
) -> None:
    assert cosine_minimum_score_to_maximum_distance(score) == expected_distance


@pytest.mark.parametrize(
    "distance, expected_relevance",
    [(0.0, 1.0), (0.5, 0.75), (1.0, 0.5), (2.0, 0.0)],
)
def test_cosine_distance_to_relevance_boundaries(
    distance: float,
    expected_relevance: float,
) -> None:
    assert cosine_distance_to_relevance(distance) == expected_relevance


@pytest.mark.parametrize(
    "value",
    [True, False, "0.5", float("nan"), float("inf"), -0.1, 1.1],
)
def test_cosine_score_rejects_invalid_values_without_clamping(value: object) -> None:
    with pytest.raises(ValueError):
        cosine_minimum_score_to_maximum_distance(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [True, False, "0.5", float("nan"), float("inf"), -0.1, 2.1],
)
def test_cosine_distance_rejects_invalid_values_without_clamping(
    value: object,
) -> None:
    with pytest.raises(ValueError):
        cosine_distance_to_relevance(value)


def test_search_rows_order_by_relevance_then_document_and_chunk() -> None:
    rows = (
        _search_row("document_b", "chunk_a", 0.5, (0.1, 0.2)),
        _search_row("document_a", "chunk_b", 0.5, (0.3, 0.4)),
        _search_row("document_z", "chunk_z", 0.25, (0.5, 0.6)),
        _search_row("document_a", "chunk_a", 0.5, (0.7, 0.8)),
    )
    original_embeddings = tuple(row.embedding for row in rows)

    ordered = order_cosine_search_rows(rows)

    assert tuple((row.document_id, row.chunk_id) for row in ordered) == (
        ("document_z", "chunk_z"),
        ("document_a", "chunk_a"),
        ("document_a", "chunk_b"),
        ("document_b", "chunk_a"),
    )
    assert tuple(row.embedding for row in ordered) == (
        (0.5, 0.6),
        (0.7, 0.8),
        (0.3, 0.4),
        (0.1, 0.2),
    )
    assert tuple(row.embedding for row in rows) == original_embeddings
    assert rows[0].document_id == "document_b"
    assert order_cosine_search_rows(rows) == ordered


def test_malformed_distance_prevents_any_ordered_result() -> None:
    rows = (
        _search_row("document_a", "chunk_a", 0.25, (0.1, 0.2)),
        _search_row("document_b", "chunk_b", math.nan, (0.3, 0.4)),
    )

    with pytest.raises(ValueError):
        order_cosine_search_rows(rows)


def test_postgres_package_exports_are_exact() -> None:
    import app.vector_store.postgres as postgres

    assert postgres.__all__ == [
        "PostgreSQLCosineSearchRow",
        "PostgreSQLStoredVectorRow",
        "PostgreSQLVectorTransaction",
        "PostgreSQLVectorTransactionRunner",
        "canonicalize_float32_embedding",
        "cosine_distance_to_relevance",
        "cosine_minimum_score_to_maximum_distance",
        "decode_ordered_metadata",
        "encode_ordered_metadata",
        "order_cosine_search_rows",
    ]
