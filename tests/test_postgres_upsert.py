"""Tests for profile-bound PostgreSQL vector upsert."""

from collections.abc import Callable
from typing import TypeVar, cast

import pytest

from app.vector_store import (
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
    VectorBatchWriteRequest,
    VectorRecord,
    VectorRecordIdentity,
)
from app.vector_store.postgres import (
    PostgreSQLCosineSearchRow,
    PostgreSQLStoredVectorRow,
)
from app.vector_store.postgres.adapter import ProfileBoundPostgreSQLVectorStore
from app.vector_store.postgres.contracts import PostgreSQLVectorTransaction

T = TypeVar("T")
Identity = tuple[str, str, str, str]


def profile(
    *,
    tenant_id: str = "tenant_synthetic",
    knowledge_base_id: str = "kb_synthetic",
    model_id: str = "model_synthetic",
    vector_dimension: int = 2,
    normalize_embeddings: bool = True,
) -> KnowledgeBaseEmbeddingProfile:
    return KnowledgeBaseEmbeddingProfile(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        model_id=model_id,
        vector_dimension=vector_dimension,
        normalize_embeddings=normalize_embeddings,
        distance_metric=EmbeddingDistanceMetric.COSINE,
    )


def record(
    *,
    tenant_id: str = "tenant_synthetic",
    knowledge_base_id: str = "kb_synthetic",
    document_id: str = "document_synthetic",
    chunk_id: str = "chunk_synthetic",
    text: str = "Synthetic knowledge text.",
    embedding: tuple[float, ...] = (0.1, 0.2),
    metadata: tuple[tuple[str, str], ...] = (
        ("kind", "synthetic"),
        ("language", "en"),
    ),
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


class FakeTransaction:
    def __init__(
        self,
        *,
        stored_profile: KnowledgeBaseEmbeddingProfile | None,
        replace_failure: BaseException | None = None,
    ) -> None:
        self.stored_profile = stored_profile
        self.replace_failure = replace_failure
        self.calls: list[tuple[object, ...]] = []
        self.rows: dict[Identity, PostgreSQLStoredVectorRow] = {}

    def acquire_scope_lock(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> None:
        self.calls.append(("lock", tenant_id, knowledge_base_id))

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
        return tuple(
            self.rows[key]
            for identity in identities
            if (
                key := (
                    tenant_id,
                    knowledge_base_id,
                    identity.document_id,
                    identity.chunk_id,
                )
            )
            in self.rows
        )

    def insert_records(
        self,
        rows: tuple[PostgreSQLStoredVectorRow, ...],
    ) -> None:
        self.calls.append(("insert", rows))
        self.rows.update((self._identity(row), row) for row in rows)

    def replace_record(self, row: PostgreSQLStoredVectorRow) -> None:
        self.calls.append(("replace", row))
        if self.replace_failure is not None:
            raise self.replace_failure
        self.rows[self._identity(row)] = row

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
    def _identity(row: PostgreSQLStoredVectorRow) -> Identity:
        return (
            row.tenant_id,
            row.knowledge_base_id,
            row.document_id,
            row.chunk_id,
        )


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


def store(
    *,
    expected_profile: KnowledgeBaseEmbeddingProfile | None = None,
    transaction: FakeTransaction | None = None,
) -> tuple[ProfileBoundPostgreSQLVectorStore, FakeRunner, FakeTransaction]:
    selected_profile = profile() if expected_profile is None else expected_profile
    selected_transaction = (
        FakeTransaction(stored_profile=selected_profile)
        if transaction is None
        else transaction
    )
    runner = FakeRunner(selected_transaction)
    return (
        ProfileBoundPostgreSQLVectorStore(
            expected_profile=selected_profile,
            transaction_runner=runner,
        ),
        runner,
        selected_transaction,
    )


def test_upsert_maps_complete_canonical_row_and_uses_exact_transaction_order() -> None:
    concrete, runner, transaction = store()
    candidate = record()

    result = concrete.upsert(candidate)

    assert result is None
    assert runner.callback_count == runner.commit_count == runner.release_count == 1
    assert runner.rollback_count == 0
    assert [call[0] for call in transaction.calls] == [
        "lock",
        "profile",
        "replace",
    ]
    assert transaction.calls[0] == ("lock", "tenant_synthetic", "kb_synthetic")
    assert transaction.calls[1] == (
        "profile",
        "tenant_synthetic",
        "kb_synthetic",
        True,
    )
    replaced = cast(PostgreSQLStoredVectorRow, transaction.calls[2][1])
    assert replaced == PostgreSQLStoredVectorRow(
        tenant_id="tenant_synthetic",
        knowledge_base_id="kb_synthetic",
        document_id="document_synthetic",
        chunk_id="chunk_synthetic",
        text="Synthetic knowledge text.",
        embedding=(0.10000000149011612, 0.20000000298023224),
        metadata_json='[["kind","synthetic"],["language","en"]]',
    )


@pytest.mark.parametrize(
    "invalid",
    [
        object(),
        VectorRecord.model_construct(),
        VectorRecord.model_construct(
            tenant_id="tenant_other",
            knowledge_base_id="kb_synthetic",
            document_id="document_synthetic",
            chunk_id="chunk_synthetic",
            text="Synthetic text.",
            embedding=(0.1, 0.2),
            metadata=(),
        ),
        VectorRecord.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_other",
            document_id="document_synthetic",
            chunk_id="chunk_synthetic",
            text="Synthetic text.",
            embedding=(0.1, 0.2),
            metadata=(),
        ),
        VectorRecord.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            document_id=" document_synthetic",
            chunk_id="chunk_synthetic",
            text="Synthetic text.",
            embedding=(0.1, 0.2),
            metadata=(),
        ),
        VectorRecord.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            document_id="document_synthetic",
            chunk_id="chunk_synthetic",
            text=" Synthetic text.",
            embedding=(0.1, 0.2),
            metadata=(),
        ),
        VectorRecord.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            document_id="document_synthetic",
            chunk_id="chunk_synthetic",
            text="Synthetic text.",
            embedding=(True, 0.2),
            metadata=(),
        ),
        VectorRecord.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            document_id="document_synthetic",
            chunk_id="chunk_synthetic",
            text="Synthetic text.",
            embedding=(0.1,),
            metadata=(),
        ),
        VectorRecord.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            document_id="document_synthetic",
            chunk_id="chunk_synthetic",
            text="Synthetic text.",
            embedding=(float("nan"), 0.2),
            metadata=(),
        ),
        VectorRecord.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            document_id="document_synthetic",
            chunk_id="chunk_synthetic",
            text="Synthetic text.",
            embedding=(0.1, 0.2),
            metadata=((" key", "value"),),
        ),
    ],
    ids=[
        "wrong-type",
        "malformed",
        "tenant",
        "knowledge-base",
        "document",
        "text",
        "boolean",
        "dimension",
        "non-finite",
        "metadata",
    ],
)
def test_invalid_record_fails_before_runner_invocation(invalid: object) -> None:
    concrete, runner, transaction = store()

    with pytest.raises(ValueError):
        concrete.upsert(cast(VectorRecord, invalid))

    assert runner.callback_count == 0
    assert transaction.calls == []


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
def test_profile_failure_prevents_replacement(
    stored_profile: KnowledgeBaseEmbeddingProfile | None,
) -> None:
    transaction = FakeTransaction(stored_profile=stored_profile)
    concrete, runner, _ = store(transaction=transaction)

    with pytest.raises(ValueError):
        concrete.upsert(record())

    assert [call[0] for call in transaction.calls] == ["lock", "profile"]
    assert runner.rollback_count == runner.release_count == 1
    assert runner.commit_count == 0


def test_same_full_identity_replaces_existing_content() -> None:
    concrete, _, transaction = store()
    original = record(text="Original synthetic text.")
    replacement = record(
        text="Replacement synthetic text.",
        embedding=(0.3, 0.4),
        metadata=(("kind", "replacement"),),
    )

    concrete.upsert(original)
    concrete.upsert(replacement)

    stored = transaction.rows[
        (
            "tenant_synthetic",
            "kb_synthetic",
            "document_synthetic",
            "chunk_synthetic",
        )
    ]
    assert stored.text == "Replacement synthetic text."
    assert stored.embedding == (0.30000001192092896, 0.4000000059604645)
    assert stored.metadata_json == '[["kind","replacement"]]'
    assert [call[0] for call in transaction.calls].count("replace") == 2


def test_same_chunk_id_under_different_documents_remains_distinct() -> None:
    concrete, _, transaction = store()

    concrete.upsert(record(document_id="document_a", chunk_id="shared_chunk"))
    concrete.upsert(record(document_id="document_b", chunk_id="shared_chunk"))

    assert set(transaction.rows) == {
        (
            "tenant_synthetic",
            "kb_synthetic",
            "document_a",
            "shared_chunk",
        ),
        (
            "tenant_synthetic",
            "kb_synthetic",
            "document_b",
            "shared_chunk",
        ),
    }


def test_repeated_upsert_is_deterministic() -> None:
    concrete, runner, transaction = store()
    candidate = record()

    concrete.upsert(candidate)
    first = transaction.rows.copy()
    concrete.upsert(candidate)
    second = transaction.rows.copy()

    assert first == second
    assert runner.callback_count == runner.commit_count == runner.release_count == 2
    assert runner.rollback_count == 0


def test_provider_exception_identity_and_runner_rollback_are_preserved() -> None:
    expected = RuntimeError("synthetic provider failure")
    transaction = FakeTransaction(
        stored_profile=profile(),
        replace_failure=expected,
    )
    concrete, runner, _ = store(transaction=transaction)

    with pytest.raises(RuntimeError) as captured:
        concrete.upsert(record())

    assert captured.value is expected
    assert [call[0] for call in transaction.calls] == [
        "lock",
        "profile",
        "replace",
    ]
    assert runner.rollback_count == runner.release_count == 1
    assert runner.commit_count == 0


def test_admit_batch_behavior_remains_available_and_unchanged() -> None:
    concrete, _, transaction = store()
    candidate = record()
    batch = VectorBatchWriteRequest(
        tenant_id=candidate.tenant_id,
        knowledge_base_id=candidate.knowledge_base_id,
        records=(candidate,),
    )

    result = concrete.admit_batch(batch)

    assert result.inserted_identities == (
        VectorRecordIdentity(
            document_id="document_synthetic",
            chunk_id="chunk_synthetic",
        ),
    )
    assert result.unchanged_identities == ()
    assert [call[0] for call in transaction.calls] == [
        "lock",
        "profile",
        "records",
        "insert",
    ]


def test_partial_adapter_has_no_search_and_is_not_exported() -> None:
    import app.vector_store.postgres as postgres

    concrete, _, _ = store()

    assert not hasattr(concrete, "search")
    assert "ProfileBoundPostgreSQLVectorStore" not in postgres.__all__
