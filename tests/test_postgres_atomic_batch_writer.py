"""Tests for profile-bound PostgreSQL atomic vector batch admission."""

from collections.abc import Callable
from dataclasses import replace
from typing import TypeVar, cast

import pytest

from app.vector_store import (
    AtomicVectorBatchWriter,
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
    VectorBatchWriteRequest,
    VectorRecord,
    VectorRecordIdentity,
)
from app.vector_store.postgres.adapter import ProfileBoundPostgreSQLVectorStore
from app.vector_store.postgres.codecs import (
    canonicalize_float32_embedding,
    encode_ordered_metadata,
)
from app.vector_store.postgres.contracts import (
    PostgreSQLCosineSearchRow,
    PostgreSQLStoredVectorRow,
    PostgreSQLVectorTransaction,
)

T = TypeVar("T")
Scope = tuple[str, str]
Identity = tuple[str, str]


def profile(
    *,
    tenant_id: str = "tenant_synthetic",
    knowledge_base_id: str = "kb_synthetic",
    model_id: str = "model_synthetic",
    vector_dimension: int = 2,
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


def record(
    document_id: str = "document_a",
    chunk_id: str = "chunk_a",
    *,
    text: str = "Synthetic knowledge text.",
    embedding: tuple[float, ...] = (0.1, 0.2),
    metadata: tuple[tuple[str, str], ...] = (("kind", "synthetic"),),
    tenant_id: str = "tenant_synthetic",
    knowledge_base_id: str = "kb_synthetic",
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


def request(*records: VectorRecord) -> VectorBatchWriteRequest:
    return VectorBatchWriteRequest(
        tenant_id="tenant_synthetic",
        knowledge_base_id="kb_synthetic",
        records=records,
    )


def stored_row(value: VectorRecord) -> PostgreSQLStoredVectorRow:
    return PostgreSQLStoredVectorRow(
        tenant_id=value.tenant_id,
        knowledge_base_id=value.knowledge_base_id,
        document_id=value.document_id,
        chunk_id=value.chunk_id,
        text=value.text,
        embedding=canonicalize_float32_embedding(
            value.embedding,
            expected_dimension=2,
        ),
        metadata_json=encode_ordered_metadata(value.metadata),
    )


class FakeTransaction:
    def __init__(
        self,
        *,
        stored_profile: KnowledgeBaseEmbeddingProfile | None = None,
        rows: tuple[PostgreSQLStoredVectorRow, ...] = (),
        failure: BaseException | None = None,
    ) -> None:
        self.stored_profile = stored_profile
        self.rows = {self._key(row): row for row in rows}
        self.returned_rows: tuple[PostgreSQLStoredVectorRow, ...] | None = None
        self.failure = failure
        self.calls: list[tuple[object, ...]] = []
        self.inserted: list[tuple[PostgreSQLStoredVectorRow, ...]] = []

    def acquire_scope_lock(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> None:
        self.calls.append(("lock", tenant_id, knowledge_base_id))
        if self.failure is not None:
            raise self.failure

    def get_profile(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        for_update: bool,
    ) -> KnowledgeBaseEmbeddingProfile | None:
        self.calls.append(("profile", tenant_id, knowledge_base_id, for_update))
        return self.stored_profile

    def insert_profile(self, profile: KnowledgeBaseEmbeddingProfile) -> None:
        raise AssertionError(f"unexpected insert_profile: {profile!r}")

    def get_records(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        identities: tuple[VectorRecordIdentity, ...],
    ) -> tuple[PostgreSQLStoredVectorRow, ...]:
        self.calls.append(("records", tenant_id, knowledge_base_id, identities))
        if self.returned_rows is not None:
            return self.returned_rows
        return tuple(
            self.rows[key]
            for identity in identities
            if (key := (identity.document_id, identity.chunk_id)) in self.rows
        )

    def insert_records(
        self,
        rows: tuple[PostgreSQLStoredVectorRow, ...],
    ) -> None:
        self.calls.append(("insert", rows))
        self.inserted.append(rows)
        self.rows.update((self._key(row), row) for row in rows)

    def replace_record(self, row: PostgreSQLStoredVectorRow) -> None:
        raise AssertionError(f"unexpected replace_record: {row!r}")

    def search_cosine(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        query_embedding: tuple[float, ...],
        top_k: int,
        maximum_cosine_distance: float,
    ) -> tuple[PostgreSQLCosineSearchRow, ...]:
        raise AssertionError(
            "unexpected search_cosine: "
            f"{tenant_id}, {knowledge_base_id}, {query_embedding}, "
            f"{top_k}, {maximum_cosine_distance}"
        )

    @staticmethod
    def _key(row: PostgreSQLStoredVectorRow) -> Identity:
        return (row.document_id, row.chunk_id)


class FakeRunner:
    def __init__(self, transaction: FakeTransaction) -> None:
        self.transaction = transaction
        self.callback_count = 0
        self.commit_count = 0
        self.rollback_count = 0
        self.release_count = 0

    def run_in_transaction(
        self,
        operation: Callable[[PostgreSQLVectorTransaction], T],
    ) -> T:
        self.callback_count += 1
        try:
            result = operation(self.transaction)
        except BaseException:
            self.rollback_count += 1
            raise
        else:
            self.commit_count += 1
            return result
        finally:
            self.release_count += 1


class FalseyRunner(FakeRunner):
    def __bool__(self) -> bool:
        return False


def store(
    *,
    expected_profile: KnowledgeBaseEmbeddingProfile | None = None,
    transaction: FakeTransaction | None = None,
    runner: FakeRunner | None = None,
) -> tuple[ProfileBoundPostgreSQLVectorStore, FakeRunner, FakeTransaction]:
    selected_profile = profile() if expected_profile is None else expected_profile
    selected_transaction = (
        FakeTransaction(stored_profile=selected_profile)
        if transaction is None
        else transaction
    )
    selected_runner = FakeRunner(selected_transaction) if runner is None else runner
    return (
        ProfileBoundPostgreSQLVectorStore(
            expected_profile=selected_profile,
            transaction_runner=selected_runner,
        ),
        selected_runner,
        selected_transaction,
    )


def test_structurally_satisfies_atomic_batch_writer() -> None:
    concrete, _, _ = store()

    writer: AtomicVectorBatchWriter = concrete

    assert writer is concrete


def test_constructor_preserves_falsey_callable_runner_without_execution() -> None:
    transaction = FakeTransaction(stored_profile=profile())
    runner = FalseyRunner(transaction)

    concrete, returned_runner, _ = store(transaction=transaction, runner=runner)

    assert concrete is not None
    assert returned_runner is runner
    assert runner.callback_count == 0


@pytest.mark.parametrize("runner", [None, object()])
def test_constructor_rejects_noncallable_runner_without_transaction(
    runner: object,
) -> None:
    with pytest.raises(ValueError, match="run_in_transaction"):
        ProfileBoundPostgreSQLVectorStore(
            expected_profile=profile(),
            transaction_runner=cast(object, runner),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_profile",
    [
        object(),
        KnowledgeBaseEmbeddingProfile.model_construct(),
        KnowledgeBaseEmbeddingProfile.model_construct(
            tenant_id=" tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            model_id="model_synthetic",
            vector_dimension=2,
            normalize_embeddings=True,
            distance_metric=EmbeddingDistanceMetric.COSINE,
        ),
        KnowledgeBaseEmbeddingProfile.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            model_id="model_synthetic",
            vector_dimension=True,
            normalize_embeddings=True,
            distance_metric=EmbeddingDistanceMetric.COSINE,
        ),
        KnowledgeBaseEmbeddingProfile.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            model_id="model_synthetic",
            vector_dimension=2,
            normalize_embeddings=1,
            distance_metric=EmbeddingDistanceMetric.COSINE,
        ),
        KnowledgeBaseEmbeddingProfile.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            model_id="model_synthetic",
            vector_dimension=2,
            normalize_embeddings=True,
            distance_metric="cosine",
        ),
    ],
)
def test_constructor_rejects_malformed_profiles_without_transaction(
    invalid_profile: object,
) -> None:
    runner = FakeRunner(FakeTransaction())

    with pytest.raises(ValueError):
        ProfileBoundPostgreSQLVectorStore(
            expected_profile=cast(
                KnowledgeBaseEmbeddingProfile,
                invalid_profile,
            ),
            transaction_runner=runner,
        )

    assert runner.callback_count == 0


@pytest.mark.parametrize(
    "metric",
    [EmbeddingDistanceMetric.DOT_PRODUCT, EmbeddingDistanceMetric.EUCLIDEAN],
)
def test_constructor_rejects_non_cosine_profile(
    metric: EmbeddingDistanceMetric,
) -> None:
    runner = FakeRunner(FakeTransaction())

    with pytest.raises(ValueError, match="cosine"):
        ProfileBoundPostgreSQLVectorStore(
            expected_profile=profile(distance_metric=metric),
            transaction_runner=runner,
        )

    assert runner.callback_count == 0


def test_missing_records_insert_atomically_in_deterministic_order() -> None:
    concrete, runner, transaction = store()
    later = record("document_b", "chunk_b")
    earlier = record("document_a", "chunk_a", embedding=(0.3, 0.4))
    batch = request(later, earlier)
    original_records = batch.records

    result = concrete.admit_batch(batch)

    assert runner.callback_count == runner.commit_count == runner.release_count == 1
    assert runner.rollback_count == 0
    assert [call[0] for call in transaction.calls] == [
        "lock",
        "profile",
        "records",
        "insert",
    ]
    assert transaction.calls[0] == ("lock", "tenant_synthetic", "kb_synthetic")
    assert transaction.calls[1] == (
        "profile",
        "tenant_synthetic",
        "kb_synthetic",
        True,
    )
    record_read = cast(
        tuple[str, str, str, tuple[VectorRecordIdentity, ...]],
        transaction.calls[2],
    )
    assert record_read[:3] == ("records", "tenant_synthetic", "kb_synthetic")
    assert tuple(
        (identity.document_id, identity.chunk_id) for identity in record_read[3]
    ) == (("document_a", "chunk_a"), ("document_b", "chunk_b"))
    inserted_rows = transaction.inserted[0]
    assert tuple(FakeTransaction._key(row) for row in inserted_rows) == (
        ("document_a", "chunk_a"),
        ("document_b", "chunk_b"),
    )
    assert inserted_rows[1].embedding == canonicalize_float32_embedding(
        later.embedding,
        expected_dimension=2,
    )
    assert inserted_rows[1].metadata_json == '[["kind","synthetic"]]'
    assert tuple(
        (identity.document_id, identity.chunk_id)
        for identity in result.inserted_identities
    ) == (("document_a", "chunk_a"), ("document_b", "chunk_b"))
    assert result.unchanged_identities == ()
    assert batch.records is original_records
    assert batch.records == (later, earlier)


def test_equal_canonical_record_is_unchanged_without_insert() -> None:
    candidate = record()
    transaction = FakeTransaction(
        stored_profile=profile(),
        rows=(stored_row(candidate),),
    )
    concrete, _, _ = store(transaction=transaction)

    result = concrete.admit_batch(request(candidate))

    assert result.inserted_identities == ()
    assert result.unchanged_identities == (
        VectorRecordIdentity(document_id="document_a", chunk_id="chunk_a"),
    )
    assert transaction.inserted == []


def test_canonical_float32_values_define_embedding_equality() -> None:
    candidate = record(embedding=(0.1, 0.2))
    canonical = stored_row(candidate)
    assert canonical.embedding != candidate.embedding
    transaction = FakeTransaction(stored_profile=profile(), rows=(canonical,))
    concrete, _, _ = store(transaction=transaction)

    result = concrete.admit_batch(request(candidate))

    assert len(result.unchanged_identities) == 1
    assert transaction.inserted == []


def test_mixed_batch_inserts_missing_and_preserves_equal_record() -> None:
    existing_b = record("document_b", "chunk_b")
    existing_d = record("document_d", "chunk_d", embedding=(0.5, 0.6))
    added_a = record("document_a", "chunk_a", embedding=(0.3, 0.4))
    added_c = record("document_c", "chunk_c", embedding=(0.7, 0.8))
    transaction = FakeTransaction(
        stored_profile=profile(),
        rows=(stored_row(existing_d), stored_row(existing_b)),
    )
    concrete, _, _ = store(transaction=transaction)
    batch = request(existing_d, added_c, existing_b, added_a)
    original_records = batch.records

    result = concrete.admit_batch(batch)

    record_read = cast(
        tuple[str, str, str, tuple[VectorRecordIdentity, ...]],
        transaction.calls[2],
    )
    assert tuple(
        (identity.document_id, identity.chunk_id) for identity in record_read[3]
    ) == (
        ("document_a", "chunk_a"),
        ("document_b", "chunk_b"),
        ("document_c", "chunk_c"),
        ("document_d", "chunk_d"),
    )
    assert tuple(FakeTransaction._key(row) for row in transaction.inserted[0]) == (
        ("document_a", "chunk_a"),
        ("document_c", "chunk_c"),
    )
    assert tuple(
        (identity.document_id, identity.chunk_id)
        for identity in result.inserted_identities
    ) == (
        ("document_a", "chunk_a"),
        ("document_c", "chunk_c"),
    )
    assert tuple(
        (identity.document_id, identity.chunk_id)
        for identity in result.unchanged_identities
    ) == (
        ("document_b", "chunk_b"),
        ("document_d", "chunk_d"),
    )
    assert batch.records is original_records
    assert batch.records == (existing_d, added_c, existing_b, added_a)


@pytest.mark.parametrize(
    "conflicting",
    [
        replace(stored_row(record()), text="Different synthetic text."),
        replace(stored_row(record()), embedding=(0.25, 0.5)),
        replace(stored_row(record()), metadata_json='[["kind","different"]]'),
    ],
)
def test_record_conflict_rejects_complete_batch_without_writes(
    conflicting: PostgreSQLStoredVectorRow,
) -> None:
    transaction = FakeTransaction(stored_profile=profile(), rows=(conflicting,))
    concrete, runner, _ = store(transaction=transaction)

    with pytest.raises(ValueError, match="conflicts"):
        concrete.admit_batch(
            request(
                record("document_new", "chunk_new"),
                record(),
            )
        )

    assert transaction.inserted == []
    assert runner.commit_count == 0
    assert runner.rollback_count == 1


@pytest.mark.parametrize(
    "stored_profile",
    [
        None,
        profile(tenant_id="tenant_other"),
        profile(knowledge_base_id="kb_other"),
        profile(model_id="model_other"),
        profile(vector_dimension=3),
        profile(normalize_embeddings=False),
        KnowledgeBaseEmbeddingProfile.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            model_id="model_synthetic",
            vector_dimension=2,
            normalize_embeddings=True,
            distance_metric="cosine",
        ),
    ],
)
def test_missing_mismatched_or_malformed_stored_profile_fails_before_record_read(
    stored_profile: KnowledgeBaseEmbeddingProfile | None,
) -> None:
    transaction = FakeTransaction(stored_profile=stored_profile)
    concrete, runner, _ = store(transaction=transaction)

    with pytest.raises(ValueError):
        concrete.admit_batch(request(record()))

    assert [call[0] for call in transaction.calls] == ["lock", "profile"]
    assert transaction.inserted == []
    assert runner.rollback_count == 1


@pytest.mark.parametrize(
    "malformed_request",
    [
        object(),
        VectorBatchWriteRequest.model_construct(),
        VectorBatchWriteRequest.model_construct(
            tenant_id="tenant_other",
            knowledge_base_id="kb_synthetic",
            records=(record(),),
        ),
        VectorBatchWriteRequest.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_other",
            records=(record(),),
        ),
        VectorBatchWriteRequest.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            records=(),
        ),
        VectorBatchWriteRequest.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            records=(
                VectorRecord.model_construct(
                    tenant_id="tenant_synthetic",
                    knowledge_base_id="kb_synthetic",
                    document_id="document_a",
                    chunk_id="chunk_a",
                    text="Synthetic text.",
                    embedding=(True, 0.2),
                    metadata=(),
                ),
            ),
        ),
        VectorBatchWriteRequest.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            records=(
                record(),
                record(),
            ),
        ),
    ],
)
def test_batch_prevalidation_fails_before_transaction(
    malformed_request: object,
) -> None:
    concrete, runner, _ = store()

    with pytest.raises(ValueError):
        concrete.admit_batch(
            cast(VectorBatchWriteRequest, malformed_request),
        )

    assert runner.callback_count == 0


@pytest.mark.parametrize(
    "returned_row",
    [
        replace(stored_row(record()), tenant_id="tenant_other"),
        replace(stored_row(record()), knowledge_base_id="kb_other"),
        replace(stored_row(record()), document_id="document_unexpected"),
        replace(stored_row(record()), text=" Synthetic text."),
        replace(stored_row(record()), embedding=(0.1, 0.2)),
        replace(stored_row(record()), embedding=(True, 0.2)),
        replace(stored_row(record()), metadata_json='{"kind":"synthetic"}'),
    ],
)
def test_malformed_or_unexpected_stored_row_fails_without_writes(
    returned_row: PostgreSQLStoredVectorRow,
) -> None:
    transaction = FakeTransaction(stored_profile=profile())
    transaction.returned_rows = (returned_row,)
    concrete, runner, _ = store(transaction=transaction)

    with pytest.raises(ValueError):
        concrete.admit_batch(request(record()))

    assert transaction.inserted == []
    assert runner.rollback_count == 1


def test_duplicate_stored_row_fails_without_writes() -> None:
    returned = stored_row(record())
    transaction = FakeTransaction(stored_profile=profile())
    transaction.returned_rows = (returned, returned)
    concrete, _, _ = store(transaction=transaction)

    with pytest.raises(ValueError, match="duplicate"):
        concrete.admit_batch(request(record()))

    assert transaction.inserted == []


def test_provider_exception_propagates_with_exact_identity() -> None:
    expected = RuntimeError("synthetic provider failure")
    transaction = FakeTransaction(stored_profile=profile(), failure=expected)
    concrete, runner, _ = store(transaction=transaction)

    with pytest.raises(RuntimeError) as captured:
        concrete.admit_batch(request(record()))

    assert captured.value is expected
    assert runner.rollback_count == runner.release_count == 1
    assert runner.commit_count == 0
    assert transaction.inserted == []


def test_repeated_batch_admission_is_idempotent() -> None:
    concrete, runner, transaction = store()
    batch = request(record())

    first = concrete.admit_batch(batch)
    second = concrete.admit_batch(batch)

    assert len(first.inserted_identities) == 1
    assert first.unchanged_identities == ()
    assert second.inserted_identities == ()
    assert len(second.unchanged_identities) == 1
    assert len(transaction.inserted) == 1
    assert runner.callback_count == runner.commit_count == runner.release_count == 2
    assert runner.rollback_count == 0
