"""Tests for the SQL-free PostgreSQL embedding-profile repository."""

from collections.abc import Callable
from typing import TypeVar, cast, get_type_hints

import pytest

from app.vector_store import (
    EmbeddingDistanceMetric,
    EmbeddingProfileRepository,
    KnowledgeBaseEmbeddingProfile,
    VectorRecordIdentity,
)
from app.vector_store.postgres import (
    PostgreSQLCosineSearchRow,
    PostgreSQLEmbeddingProfileRepository,
    PostgreSQLStoredVectorRow,
    PostgreSQLVectorTransaction,
)

T = TypeVar("T")


def _profile(
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


class FakeTransaction:
    def __init__(self) -> None:
        self.profile: object | None = None
        self.calls: list[tuple[object, ...]] = []
        self.inserted_profiles: list[KnowledgeBaseEmbeddingProfile] = []
        self.session_error: BaseException | None = None

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
        self.calls.append(("get_profile", tenant_id, knowledge_base_id, for_update))
        if self.session_error is not None:
            raise self.session_error
        return cast(KnowledgeBaseEmbeddingProfile | None, self.profile)

    def insert_profile(self, profile: KnowledgeBaseEmbeddingProfile) -> None:
        self.calls.append(("insert_profile", profile))
        self.inserted_profiles.append(profile)
        self.profile = profile

    def get_records(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        identities: tuple[VectorRecordIdentity, ...],
    ) -> tuple[PostgreSQLStoredVectorRow, ...]:
        raise AssertionError("get_records must not be called")

    def insert_records(
        self,
        rows: tuple[PostgreSQLStoredVectorRow, ...],
    ) -> None:
        raise AssertionError("insert_records must not be called")

    def replace_record(self, row: PostgreSQLStoredVectorRow) -> None:
        raise AssertionError("replace_record must not be called")

    def search_cosine(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        query_embedding: tuple[float, ...],
        top_k: int,
        maximum_cosine_distance: float,
    ) -> tuple[PostgreSQLCosineSearchRow, ...]:
        raise AssertionError("search_cosine must not be called")


class FakeRunner:
    def __init__(self, transaction: FakeTransaction) -> None:
        self.transaction = transaction
        self.run_calls = 0
        self.callback_calls = 0
        self.commits = 0
        self.rollbacks = 0
        self.releases = 0
        self.runner_error: BaseException | None = None

    def run_in_transaction(
        self,
        operation: Callable[[PostgreSQLVectorTransaction], T],
    ) -> T:
        self.run_calls += 1
        if self.runner_error is not None:
            raise self.runner_error
        try:
            self.callback_calls += 1
            result = operation(self.transaction)
        except BaseException:
            self.rollbacks += 1
            raise
        else:
            self.commits += 1
            return result
        finally:
            self.releases += 1


def _accept_repository(repository: EmbeddingProfileRepository) -> None:
    del repository


def test_repository_structurally_satisfies_protocol() -> None:
    repository = PostgreSQLEmbeddingProfileRepository(FakeRunner(FakeTransaction()))

    _accept_repository(repository)


def test_constructor_accepts_runner_without_invoking_it() -> None:
    runner = FakeRunner(FakeTransaction())

    repository = PostgreSQLEmbeddingProfileRepository(runner)

    assert repository._transaction_runner is runner
    assert runner.run_calls == 0


@pytest.mark.parametrize(
    "runner",
    [
        object(),
        None,
        type("MissingMethod", (), {})(),
        type("NonCallableMethod", (), {"run_in_transaction": None})(),
    ],
)
def test_constructor_rejects_invalid_runner(runner: object) -> None:
    with pytest.raises(ValueError, match="run_in_transaction"):
        PostgreSQLEmbeddingProfileRepository(runner)  # type: ignore[arg-type]


def test_missing_registration_uses_exact_order_and_object_identity() -> None:
    transaction = FakeTransaction()
    runner = FakeRunner(transaction)
    repository = PostgreSQLEmbeddingProfileRepository(runner)
    value = _profile()

    result = repository.register_profile(value)

    assert result is value
    assert transaction.profile is value
    assert transaction.inserted_profiles == [value]
    assert transaction.calls == [
        ("lock", "tenant_synthetic", "kb_synthetic"),
        ("get_profile", "tenant_synthetic", "kb_synthetic", True),
        ("insert_profile", value),
    ]
    assert runner.run_calls == 1
    assert runner.callback_calls == 1
    assert runner.commits == 1
    assert runner.rollbacks == 0
    assert runner.releases == 1


def test_equal_registration_returns_exact_canonical_stored_object() -> None:
    transaction = FakeTransaction()
    stored = _profile()
    transaction.profile = stored
    runner = FakeRunner(transaction)
    repository = PostgreSQLEmbeddingProfileRepository(runner)
    equal_copy = _profile()

    result = repository.register_profile(equal_copy)

    assert result is stored
    assert transaction.profile is stored
    assert transaction.inserted_profiles == []
    assert transaction.calls == [
        ("lock", "tenant_synthetic", "kb_synthetic"),
        ("get_profile", "tenant_synthetic", "kb_synthetic", True),
    ]
    assert runner.commits == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"model_id": "other_model"},
        {"vector_dimension": 3},
        {"normalize_embeddings": False},
        {"distance_metric": EmbeddingDistanceMetric.DOT_PRODUCT},
    ],
    ids=["model", "dimension", "normalization", "metric"],
)
def test_conflicting_registration_fails_without_insertion_or_state_change(
    changes: dict[str, object],
) -> None:
    transaction = FakeTransaction()
    stored = _profile()
    transaction.profile = stored
    runner = FakeRunner(transaction)
    repository = PostgreSQLEmbeddingProfileRepository(runner)
    conflicting = stored.model_copy(update=changes)

    with pytest.raises(ValueError, match="conflicts"):
        repository.register_profile(conflicting)

    assert transaction.profile is stored
    assert transaction.inserted_profiles == []
    assert transaction.calls == [
        ("lock", "tenant_synthetic", "kb_synthetic"),
        ("get_profile", "tenant_synthetic", "kb_synthetic", True),
    ]
    assert runner.callback_calls == 1
    assert runner.commits == 0
    assert runner.rollbacks == 1
    assert runner.releases == 1


@pytest.mark.parametrize(
    "stored",
    [
        _profile(tenant_id="other_tenant"),
        _profile(knowledge_base_id="other_kb"),
        object(),
        KnowledgeBaseEmbeddingProfile.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
        ),
    ],
    ids=["wrong-tenant", "wrong-kb", "wrong-type", "malformed-profile"],
)
def test_registration_rejects_invalid_stored_profile_without_further_calls(
    stored: object,
) -> None:
    transaction = FakeTransaction()
    transaction.profile = stored
    runner = FakeRunner(transaction)
    repository = PostgreSQLEmbeddingProfileRepository(runner)

    with pytest.raises(ValueError):
        repository.register_profile(_profile())

    assert transaction.inserted_profiles == []
    assert transaction.calls == [
        ("lock", "tenant_synthetic", "kb_synthetic"),
        ("get_profile", "tenant_synthetic", "kb_synthetic", True),
    ]
    assert runner.commits == 0
    assert runner.rollbacks == 1


def test_registration_rejects_non_profile_before_runner() -> None:
    runner = FakeRunner(FakeTransaction())
    repository = PostgreSQLEmbeddingProfileRepository(runner)

    with pytest.raises(ValueError, match="KnowledgeBaseEmbeddingProfile"):
        repository.register_profile(object())  # type: ignore[arg-type]

    assert runner.run_calls == 0


def test_repeated_equal_registration_is_deterministic() -> None:
    transaction = FakeTransaction()
    runner = FakeRunner(transaction)
    repository = PostgreSQLEmbeddingProfileRepository(runner)
    value = _profile()

    results = tuple(repository.register_profile(value) for _ in range(3))

    assert all(result is value for result in results)
    assert transaction.inserted_profiles == [value]
    assert runner.run_calls == 3
    assert runner.callback_calls == 3
    assert runner.commits == 3


def test_lookup_normalizes_scope_and_returns_exact_stored_object() -> None:
    transaction = FakeTransaction()
    stored = _profile()
    transaction.profile = stored
    runner = FakeRunner(transaction)
    repository = PostgreSQLEmbeddingProfileRepository(runner)

    result = repository.get_profile(
        tenant_id=" tenant_synthetic ",
        knowledge_base_id=" kb_synthetic ",
    )

    assert result is stored
    assert transaction.calls == [
        ("get_profile", "tenant_synthetic", "kb_synthetic", False)
    ]
    assert transaction.inserted_profiles == []
    assert runner.run_calls == 1
    assert runner.callback_calls == 1
    assert runner.commits == 1
    assert runner.rollbacks == 0
    assert runner.releases == 1


@pytest.mark.parametrize(
    ("tenant_id", "knowledge_base_id"),
    [
        ("", "kb_synthetic"),
        ("   ", "kb_synthetic"),
        ("tenant_synthetic", ""),
        ("tenant_synthetic", "   "),
        (1, "kb_synthetic"),
        ("tenant_synthetic", 1),
    ],
)
def test_invalid_lookup_scope_is_rejected_before_runner(
    tenant_id: object,
    knowledge_base_id: object,
) -> None:
    runner = FakeRunner(FakeTransaction())
    repository = PostgreSQLEmbeddingProfileRepository(runner)

    with pytest.raises(ValueError):
        repository.get_profile(
            tenant_id=tenant_id,  # type: ignore[arg-type]
            knowledge_base_id=knowledge_base_id,  # type: ignore[arg-type]
        )

    assert runner.run_calls == 0


def test_missing_lookup_returns_none_without_lock_or_mutation() -> None:
    transaction = FakeTransaction()
    runner = FakeRunner(transaction)
    repository = PostgreSQLEmbeddingProfileRepository(runner)

    result = repository.get_profile(
        tenant_id="tenant_synthetic",
        knowledge_base_id="kb_synthetic",
    )

    assert result is None
    assert transaction.calls == [
        ("get_profile", "tenant_synthetic", "kb_synthetic", False)
    ]
    assert transaction.inserted_profiles == []


@pytest.mark.parametrize(
    "stored",
    [
        _profile(tenant_id="other_tenant"),
        _profile(knowledge_base_id="other_kb"),
        object(),
        KnowledgeBaseEmbeddingProfile.model_construct(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
        ),
    ],
    ids=["wrong-tenant", "wrong-kb", "wrong-type", "malformed-profile"],
)
def test_lookup_rejects_invalid_stored_result(stored: object) -> None:
    transaction = FakeTransaction()
    transaction.profile = stored
    runner = FakeRunner(transaction)
    repository = PostgreSQLEmbeddingProfileRepository(runner)

    with pytest.raises(ValueError):
        repository.get_profile(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
        )

    assert transaction.calls == [
        ("get_profile", "tenant_synthetic", "kb_synthetic", False)
    ]
    assert transaction.inserted_profiles == []
    assert runner.commits == 0
    assert runner.rollbacks == 1
    assert runner.releases == 1


def test_repeated_lookup_returns_same_stored_object() -> None:
    transaction = FakeTransaction()
    stored = _profile()
    transaction.profile = stored
    runner = FakeRunner(transaction)
    repository = PostgreSQLEmbeddingProfileRepository(runner)

    results = tuple(
        repository.get_profile(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
        )
        for _ in range(3)
    )

    assert all(result is stored for result in results)
    assert runner.run_calls == 3
    assert runner.callback_calls == 3
    assert runner.commits == 3


def test_runner_provider_exception_identity_propagates() -> None:
    runner = FakeRunner(FakeTransaction())
    error = RuntimeError("synthetic runner failure")
    runner.runner_error = error
    repository = PostgreSQLEmbeddingProfileRepository(runner)

    with pytest.raises(RuntimeError) as captured:
        repository.get_profile(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
        )

    assert captured.value is error
    assert runner.callback_calls == 0


def test_session_provider_exception_identity_propagates_and_rolls_back() -> None:
    transaction = FakeTransaction()
    error = RuntimeError("synthetic session failure")
    transaction.session_error = error
    runner = FakeRunner(transaction)
    repository = PostgreSQLEmbeddingProfileRepository(runner)

    with pytest.raises(RuntimeError) as captured:
        repository.get_profile(
            tenant_id="tenant_synthetic",
            knowledge_base_id="kb_synthetic",
        )

    assert captured.value is error
    assert runner.callback_calls == 1
    assert runner.commits == 0
    assert runner.rollbacks == 1
    assert runner.releases == 1


def test_new_export_preserves_existing_postgres_exports() -> None:
    import app.vector_store.postgres as postgres

    assert postgres.PostgreSQLEmbeddingProfileRepository is (
        PostgreSQLEmbeddingProfileRepository
    )
    assert set(postgres.__all__) == {
        "PostgreSQLCosineSearchRow",
        "PostgreSQLStoredVectorRow",
        "PostgreSQLVectorTransaction",
        "PostgreSQLVectorTransactionRunner",
        "ProfileBoundPostgreSQLVectorStore",
        "canonicalize_float32_embedding",
        "cosine_distance_to_relevance",
        "cosine_minimum_score_to_maximum_distance",
        "decode_ordered_metadata",
        "encode_ordered_metadata",
        "order_cosine_search_rows",
    }


def test_repository_annotations_contain_no_provider_client_types() -> None:
    annotations = " ".join(
        (
            str(get_type_hints(PostgreSQLEmbeddingProfileRepository.__init__)),
            str(get_type_hints(PostgreSQLEmbeddingProfileRepository.register_profile)),
            str(get_type_hints(PostgreSQLEmbeddingProfileRepository.get_profile)),
        )
    ).lower()

    for forbidden in ("psycopg", "cursor", "connection", "sqlalchemy"):
        assert forbidden not in annotations
