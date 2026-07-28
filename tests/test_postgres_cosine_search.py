"""Tests for profile-bound PostgreSQL cosine search."""

from collections.abc import Callable
from typing import TypeVar, cast

import pytest

from app.vector_store import (
    AtomicVectorBatchWriter,
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
    SearchRequest,
    VectorBatchWriteRequest,
    VectorRecord,
    VectorRecordIdentity,
    VectorStore,
)
from app.vector_store.postgres import (
    PostgreSQLCosineSearchRow,
    PostgreSQLStoredVectorRow,
    ProfileBoundPostgreSQLVectorStore,
)
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


def request(
    *,
    tenant_id: str = "tenant_synthetic",
    knowledge_base_id: str = "kb_synthetic",
    query_embedding: tuple[float, ...] = (0.1, 0.2),
    top_k: int = 3,
    minimum_score: float = 0.25,
) -> SearchRequest:
    return SearchRequest(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        query_embedding=query_embedding,
        top_k=top_k,
        minimum_score=minimum_score,
    )


def search_row(
    document_id: str = "document_synthetic",
    chunk_id: str = "chunk_synthetic",
    *,
    tenant_id: str = "tenant_synthetic",
    knowledge_base_id: str = "kb_synthetic",
    text: str = "Synthetic knowledge text.",
    embedding: tuple[float, ...] = (
        0.10000000149011612,
        0.20000000298023224,
    ),
    metadata_json: str = '[["kind","synthetic"],["language","en"]]',
    cosine_distance: float = 0.5,
) -> PostgreSQLCosineSearchRow:
    return PostgreSQLCosineSearchRow(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id=chunk_id,
        text=text,
        embedding=embedding,
        metadata_json=metadata_json,
        cosine_distance=cosine_distance,
    )


class FakeTransaction:
    def __init__(
        self,
        *,
        stored_profile: KnowledgeBaseEmbeddingProfile | None,
        search_rows: object = (),
        search_failure: BaseException | None = None,
    ) -> None:
        self.stored_profile = stored_profile
        self.search_rows = search_rows
        self.search_failure = search_failure
        self.calls: list[tuple[object, ...]] = []
        self.records: dict[Identity, PostgreSQLStoredVectorRow] = {}

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
            self.records[key]
            for identity in identities
            if (
                key := (
                    tenant_id,
                    knowledge_base_id,
                    identity.document_id,
                    identity.chunk_id,
                )
            )
            in self.records
        )

    def insert_records(
        self,
        rows: tuple[PostgreSQLStoredVectorRow, ...],
    ) -> None:
        self.calls.append(("insert", rows))
        self.records.update((self._identity(row), row) for row in rows)

    def replace_record(self, row: PostgreSQLStoredVectorRow) -> None:
        self.calls.append(("replace", row))
        self.records[self._identity(row)] = row

    def search_cosine(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        query_embedding: tuple[float, ...],
        top_k: int,
        maximum_cosine_distance: float,
    ) -> tuple[PostgreSQLCosineSearchRow, ...]:
        self.calls.append(
            (
                "search",
                tenant_id,
                knowledge_base_id,
                query_embedding,
                top_k,
                maximum_cosine_distance,
            )
        )
        if self.search_failure is not None:
            raise self.search_failure
        return cast(tuple[PostgreSQLCosineSearchRow, ...], self.search_rows)

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


def vector_record(
    *,
    text: str = "Synthetic knowledge text.",
) -> VectorRecord:
    return VectorRecord(
        tenant_id="tenant_synthetic",
        knowledge_base_id="kb_synthetic",
        document_id="document_synthetic",
        chunk_id="chunk_synthetic",
        text=text,
        embedding=(0.1, 0.2),
        metadata=(("kind", "synthetic"),),
    )


def test_constructor_remains_side_effect_free_and_protocol_compatible() -> None:
    concrete, runner, transaction = store()

    vector_store: VectorStore = concrete
    batch_writer: AtomicVectorBatchWriter = concrete

    assert vector_store is batch_writer is concrete
    assert runner.callback_count == 0
    assert transaction.calls == []


def test_exact_read_only_transaction_order_and_query_propagation() -> None:
    transaction = FakeTransaction(
        stored_profile=profile(),
        search_rows=(search_row(),),
    )
    concrete, runner, _ = store(transaction=transaction)

    result = concrete.search(request())

    assert result.tenant_id == "tenant_synthetic"
    assert result.knowledge_base_id == "kb_synthetic"
    assert runner.callback_count == runner.commit_count == runner.release_count == 1
    assert runner.rollback_count == 0
    assert transaction.calls == [
        ("profile", "tenant_synthetic", "kb_synthetic", False),
        (
            "search",
            "tenant_synthetic",
            "kb_synthetic",
            (0.10000000149011612, 0.20000000298023224),
            3,
            1.5,
        ),
    ]


@pytest.mark.parametrize(
    "invalid",
    [
        object(),
        SearchRequest.model_construct(),
        SearchRequest.model_construct(
            tenant_id="tenant_other",
            knowledge_base_id="kb_synthetic",
            query_embedding=(0.1, 0.2),
            top_k=1,
            minimum_score=0.0,
        ),
        SearchRequest.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_other",
            query_embedding=(0.1, 0.2),
            top_k=1,
            minimum_score=0.0,
        ),
        SearchRequest.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            query_embedding=(True, 0.2),
            top_k=1,
            minimum_score=0.0,
        ),
        SearchRequest.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            query_embedding=(0.1,),
            top_k=1,
            minimum_score=0.0,
        ),
        SearchRequest.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            query_embedding=(float("nan"), 0.2),
            top_k=1,
            minimum_score=0.0,
        ),
        SearchRequest.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            query_embedding=(0.1, 0.2),
            top_k=True,
            minimum_score=0.0,
        ),
        SearchRequest.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            query_embedding=(0.1, 0.2),
            top_k=0,
            minimum_score=0.0,
        ),
        SearchRequest.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            query_embedding=(0.1, 0.2),
            top_k=1,
            minimum_score=True,
        ),
        SearchRequest.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            query_embedding=(0.1, 0.2),
            top_k=1,
            minimum_score=float("nan"),
        ),
        SearchRequest.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            query_embedding=(0.1, 0.2),
            top_k=1,
            minimum_score=1.1,
        ),
    ],
    ids=[
        "wrong-type",
        "malformed",
        "tenant",
        "knowledge-base",
        "boolean-vector",
        "dimension",
        "non-finite-vector",
        "boolean-top-k",
        "nonpositive-top-k",
        "boolean-score",
        "non-finite-score",
        "out-of-range-score",
    ],
)
def test_invalid_request_fails_before_runner_invocation(invalid: object) -> None:
    concrete, runner, transaction = store()

    with pytest.raises(ValueError):
        concrete.search(cast(SearchRequest, invalid))

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
def test_profile_failure_prevents_search(
    stored_profile: KnowledgeBaseEmbeddingProfile | None,
) -> None:
    transaction = FakeTransaction(stored_profile=stored_profile)
    concrete, runner, _ = store(transaction=transaction)

    with pytest.raises(ValueError):
        concrete.search(request())

    assert transaction.calls == [("profile", "tenant_synthetic", "kb_synthetic", False)]
    assert runner.rollback_count == runner.release_count == 1
    assert runner.commit_count == 0


def test_empty_result_is_safe() -> None:
    concrete, _, _ = store()

    result = concrete.search(request())

    assert result.hits == ()


def test_complete_row_maps_to_provider_neutral_hit() -> None:
    row = search_row()
    transaction = FakeTransaction(stored_profile=profile(), search_rows=(row,))
    concrete, _, _ = store(transaction=transaction)

    result = concrete.search(request())

    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.score == 0.75
    assert hit.record == VectorRecord(
        tenant_id="tenant_synthetic",
        knowledge_base_id="kb_synthetic",
        document_id="document_synthetic",
        chunk_id="chunk_synthetic",
        text="Synthetic knowledge text.",
        embedding=(0.10000000149011612, 0.20000000298023224),
        metadata=(("kind", "synthetic"), ("language", "en")),
    )
    assert not hasattr(hit, "cosine_distance")


@pytest.mark.parametrize(
    "distance, expected_score",
    [(0.0, 1.0), (0.5, 0.75), (1.0, 0.5), (2.0, 0.0)],
)
def test_distance_boundaries_convert_to_relevance(
    distance: float,
    expected_score: float,
) -> None:
    transaction = FakeTransaction(
        stored_profile=profile(),
        search_rows=(search_row(cosine_distance=distance),),
    )
    concrete, _, _ = store(transaction=transaction)

    result = concrete.search(request(minimum_score=0.0))

    assert result.hits[0].score == expected_score


def test_rows_order_by_score_then_document_and_chunk_without_mutation() -> None:
    rows = (
        search_row("document_b", "chunk_a", cosine_distance=0.5),
        search_row("document_a", "chunk_b", cosine_distance=0.5),
        search_row("document_z", "chunk_z", cosine_distance=0.25),
        search_row("document_a", "chunk_a", cosine_distance=0.5),
    )
    original = rows
    transaction = FakeTransaction(stored_profile=profile(), search_rows=rows)
    concrete, _, _ = store(transaction=transaction)

    result = concrete.search(request(top_k=4))

    assert tuple(
        (hit.record.document_id, hit.record.chunk_id) for hit in result.hits
    ) == (
        ("document_z", "chunk_z"),
        ("document_a", "chunk_a"),
        ("document_a", "chunk_b"),
        ("document_b", "chunk_a"),
    )
    assert transaction.search_rows is original
    assert transaction.search_rows == rows


def test_threshold_is_reenforced_after_complete_row_validation() -> None:
    rows = (
        search_row("document_high", "chunk_high", cosine_distance=0.1),
        search_row("document_low", "chunk_low", cosine_distance=1.8),
    )
    transaction = FakeTransaction(stored_profile=profile(), search_rows=rows)
    concrete, _, _ = store(transaction=transaction)

    result = concrete.search(request(top_k=2, minimum_score=0.8))

    assert tuple(hit.record.document_id for hit in result.hits) == ("document_high",)


def test_duplicate_identity_is_rejected_before_threshold_filtering() -> None:
    duplicate = search_row(
        "document_duplicate",
        "chunk_duplicate",
        cosine_distance=1.9,
    )
    transaction = FakeTransaction(
        stored_profile=profile(),
        search_rows=(duplicate, duplicate),
    )
    concrete, runner, _ = store(transaction=transaction)

    with pytest.raises(ValueError, match="identities must be unique"):
        concrete.search(request(top_k=2, minimum_score=0.9))

    assert runner.rollback_count == 1


def test_overflow_is_rejected_before_hit_mapping() -> None:
    rows = (
        search_row("document_a", "chunk_a"),
        search_row("document_b", "chunk_b"),
    )
    transaction = FakeTransaction(stored_profile=profile(), search_rows=rows)
    concrete, _, _ = store(transaction=transaction)

    with pytest.raises(ValueError, match="more than top_k"):
        concrete.search(request(top_k=1))


@pytest.mark.parametrize(
    "returned",
    [
        [search_row()],
        (object(),),
        (search_row(tenant_id="tenant_other"),),
        (search_row(knowledge_base_id="kb_other"),),
        (search_row(document_id=" document_synthetic"),),
        (search_row(chunk_id="chunk_synthetic "),),
        (search_row(text=" Synthetic text."),),
        (search_row(embedding=(0.1, 0.2)),),
        (search_row(embedding=cast(tuple[float, ...], (True, 0.2))),),
        (search_row(embedding=(0.1,)),),
        (search_row(metadata_json='{"kind":"synthetic"}'),),
        (search_row(cosine_distance=float("nan")),),
        (search_row(cosine_distance=-0.1),),
        (search_row(cosine_distance=2.1),),
    ],
    ids=[
        "container",
        "row-type",
        "tenant",
        "knowledge-base",
        "document",
        "chunk",
        "text",
        "noncanonical-embedding",
        "boolean-embedding",
        "dimension",
        "metadata",
        "non-finite-distance",
        "negative-distance",
        "excess-distance",
    ],
)
def test_malformed_backend_result_fails_closed(returned: object) -> None:
    transaction = FakeTransaction(stored_profile=profile(), search_rows=returned)
    concrete, runner, _ = store(transaction=transaction)

    with pytest.raises(ValueError):
        concrete.search(request())

    assert runner.rollback_count == runner.release_count == 1
    assert runner.commit_count == 0


def test_provider_exception_identity_and_runner_lifecycle_are_preserved() -> None:
    expected = RuntimeError("synthetic provider failure")
    transaction = FakeTransaction(
        stored_profile=profile(),
        search_failure=expected,
    )
    concrete, runner, _ = store(transaction=transaction)

    with pytest.raises(RuntimeError) as captured:
        concrete.search(request())

    assert captured.value is expected
    assert runner.callback_count == runner.rollback_count == runner.release_count == 1
    assert runner.commit_count == 0


def test_request_and_rows_remain_immutable_across_repeated_search() -> None:
    row = search_row()
    rows = (row,)
    search_request = request()
    original_dump = search_request.model_dump()
    transaction = FakeTransaction(stored_profile=profile(), search_rows=rows)
    concrete, _, _ = store(transaction=transaction)

    first = concrete.search(search_request)
    second = concrete.search(search_request)

    assert first == second
    assert search_request.model_dump() == original_dump
    assert transaction.search_rows is rows
    assert transaction.search_rows == (row,)


def test_upsert_and_admit_batch_regression_behavior_remains_available() -> None:
    concrete, _, transaction = store()
    existing = vector_record(text="Original synthetic text.")
    replacement = vector_record(text="Replacement synthetic text.")

    concrete.upsert(existing)
    concrete.upsert(replacement)
    batch_result = concrete.admit_batch(
        VectorBatchWriteRequest(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
            records=(replacement,),
        )
    )

    assert batch_result.inserted_identities == ()
    assert batch_result.unchanged_identities == (
        VectorRecordIdentity(
            document_id="document_synthetic",
            chunk_id="chunk_synthetic",
        ),
    )
    assert [call[0] for call in transaction.calls] == [
        "lock",
        "profile",
        "replace",
        "lock",
        "profile",
        "replace",
        "lock",
        "profile",
        "records",
    ]


def test_provider_package_export_behavior_is_exact() -> None:
    import app.vector_store as vector_store
    import app.vector_store.postgres as postgres

    previous_exports = {
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
    }

    assert postgres.ProfileBoundPostgreSQLVectorStore is (
        ProfileBoundPostgreSQLVectorStore
    )
    assert postgres.__all__.count("ProfileBoundPostgreSQLVectorStore") == 1
    assert previous_exports < set(postgres.__all__)
    assert "ProfileBoundPostgreSQLVectorStore" not in vector_store.__all__
    assert not hasattr(vector_store, "ProfileBoundPostgreSQLVectorStore")
