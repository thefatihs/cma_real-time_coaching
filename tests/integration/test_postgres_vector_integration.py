"""Opt-in real PostgreSQL/pgvector integration coverage."""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import psycopg
import pytest
from psycopg import Connection
from pydantic import SecretStr

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.vector_store.embedding_profile import (  # noqa: E402
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
)
from app.deployment import (  # noqa: E402
    PostgreSQLMigrationResult,
    PostgreSQLMigrationSettings,
    apply_postgres_vector_migrations,
)
from app.vector_store.models import (  # noqa: E402
    SearchRequest,
    VectorBatchWriteRequest,
    VectorRecord,
)
from app.vector_store.postgres.adapter import (  # noqa: E402
    ProfileBoundPostgreSQLVectorStore,
)
from app.vector_store.postgres.connection_factory import (  # noqa: E402
    PgvectorPsycopgConnectionFactory,
)
from app.vector_store.postgres.contracts import (  # noqa: E402
    PostgreSQLStoredVectorRow,
)
from app.vector_store.postgres.profile_repository import (  # noqa: E402
    PostgreSQLEmbeddingProfileRepository,
)
from app.vector_store.postgres.readiness import (  # noqa: E402
    PostgreSQLSchemaReadinessChecker,
)
from app.vector_store.postgres.runner import (  # noqa: E402
    PsycopgPostgreSQLVectorTransactionRunner,
)
from app.vector_store.postgres.transaction import (  # noqa: E402
    PsycopgPostgreSQLVectorTransaction,
)

pytestmark = pytest.mark.postgres_integration

EXPECTED_HOST = "127.0.0.1"
EXPECTED_DATABASE = "callmetric_vector_test"
EXPECTED_USER = "callmetric_test"


@dataclass(frozen=True, slots=True)
class PostgreSQLTestSettings:
    host: str
    port: int
    database: str
    user: str
    password: str
    connect_timeout: int


@pytest.fixture(scope="session")
def settings() -> PostgreSQLTestSettings:
    opt_in = os.environ.get("CALLMETRIC_POSTGRES_INTEGRATION")
    if opt_in is None:
        pytest.skip("requires the opt-in PostgreSQL integration runner")
    if opt_in != "1":
        pytest.fail("CALLMETRIC_POSTGRES_INTEGRATION must be exactly 1")
    host = os.environ.get("CALLMETRIC_POSTGRES_HOST")
    database = os.environ.get("CALLMETRIC_POSTGRES_DATABASE")
    user = os.environ.get("CALLMETRIC_POSTGRES_USER")
    password = os.environ.get("CALLMETRIC_POSTGRES_PASSWORD")
    raw_port = os.environ.get("CALLMETRIC_POSTGRES_PORT")
    raw_timeout = os.environ.get("CALLMETRIC_POSTGRES_CONNECT_TIMEOUT")
    if host != EXPECTED_HOST:
        pytest.fail("integration host must be exactly 127.0.0.1")
    if database != EXPECTED_DATABASE:
        pytest.fail("integration database identity is unsafe")
    if user != EXPECTED_USER:
        pytest.fail("integration user identity is unsafe")
    if not password:
        pytest.fail("integration password is missing")
    if raw_port is None or not raw_port.isascii() or not raw_port.isdigit():
        pytest.fail("integration port is malformed")
    if raw_timeout != "5":
        pytest.fail("integration connect timeout must be exactly 5")
    port = int(raw_port)
    if not 1 <= port <= 65535:
        pytest.fail("integration port is outside the valid range")
    return PostgreSQLTestSettings(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        connect_timeout=5,
    )


def _production_connect(
    settings: PostgreSQLTestSettings,
    **kwargs: object,
) -> Connection[Any]:
    expected_dsn = (
        f"postgresql://{settings.user}:{settings.password}@"
        f"{settings.host}:{settings.port}/{settings.database}"
    )
    expected_keyword_dsn = (
        f"host={settings.host} port={settings.port} "
        f"dbname={settings.database} user={settings.user} "
        f"password={settings.password}"
    )
    assert kwargs.pop("conninfo") in (expected_dsn, expected_keyword_dsn)
    assert kwargs.pop("sslmode") == "require"
    connect_kwargs = cast(dict[str, Any], kwargs)
    return psycopg.connect(
        host=settings.host,
        port=settings.port,
        dbname=settings.database,
        user=settings.user,
        password=settings.password,
        **connect_kwargs,
    )


@pytest.fixture(scope="session", autouse=True)
def migration_results(
    settings: PostgreSQLTestSettings,
) -> tuple[PostgreSQLMigrationResult, PostgreSQLMigrationResult]:
    settings_factory = cast(
        Callable[[], PostgreSQLMigrationSettings],
        PostgreSQLMigrationSettings,
    )
    migration_settings = settings_factory()
    barrier = threading.Barrier(2)
    results: list[PostgreSQLMigrationResult] = []
    failures: list[BaseException] = []

    def migrate() -> None:
        try:
            barrier.wait(timeout=5.0)
            result = apply_postgres_vector_migrations(
                settings=migration_settings,
                psycopg_connect=lambda **kwargs: _production_connect(
                    settings,
                    **kwargs,
                ),
            )
            results.append(result)
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=migrate) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20.0)
    assert all(not thread.is_alive() for thread in threads)
    assert failures == []
    assert sorted(result.value for result in results) == [
        PostgreSQLMigrationResult.ALREADY_APPLIED.value,
        PostgreSQLMigrationResult.APPLIED.value,
    ]
    return cast(
        tuple[PostgreSQLMigrationResult, PostgreSQLMigrationResult],
        tuple(results),
    )


def _connect(settings: PostgreSQLTestSettings) -> Connection[Any]:
    return psycopg.connect(
        host=settings.host,
        port=settings.port,
        dbname=settings.database,
        user=settings.user,
        password=settings.password,
        autocommit=False,
        connect_timeout=settings.connect_timeout,
        options="-c statement_timeout=10000 -c lock_timeout=5000",
    )


def _runner(
    settings: PostgreSQLTestSettings,
) -> PsycopgPostgreSQLVectorTransactionRunner:
    connection_factory = PgvectorPsycopgConnectionFactory(
        base_connection_factory=lambda: _connect(settings),
    )
    return PsycopgPostgreSQLVectorTransactionRunner(
        connection_factory=connection_factory,
        transaction_factory=PsycopgPostgreSQLVectorTransaction,
    )


def _profile(
    name: str,
    *,
    tenant_id: str | None = None,
    knowledge_base_id: str | None = None,
    vector_dimension: int = 2,
) -> KnowledgeBaseEmbeddingProfile:
    return KnowledgeBaseEmbeddingProfile(
        tenant_id=tenant_id or f"tenant-{name}",
        knowledge_base_id=knowledge_base_id or f"kb-{name}",
        model_id=f"model-{name}",
        vector_dimension=vector_dimension,
        normalize_embeddings=True,
        distance_metric=EmbeddingDistanceMetric.COSINE,
    )


def _record(
    profile: KnowledgeBaseEmbeddingProfile,
    *,
    document_id: str,
    chunk_id: str,
    text: str,
    embedding: tuple[float, ...] = (1.0, 0.0),
    metadata: tuple[tuple[str, str], ...] = (("kind", "synthetic"),),
) -> VectorRecord:
    return VectorRecord(
        tenant_id=profile.tenant_id,
        knowledge_base_id=profile.knowledge_base_id,
        document_id=document_id,
        chunk_id=chunk_id,
        text=text,
        embedding=embedding,
        metadata=metadata,
    )


def _repository(
    settings: PostgreSQLTestSettings,
) -> PostgreSQLEmbeddingProfileRepository:
    return PostgreSQLEmbeddingProfileRepository(_runner(settings))


def _store(
    settings: PostgreSQLTestSettings,
    profile: KnowledgeBaseEmbeddingProfile,
) -> ProfileBoundPostgreSQLVectorStore:
    return ProfileBoundPostgreSQLVectorStore(
        expected_profile=profile,
        transaction_runner=_runner(settings),
    )


def test_migration_extension_schema_ledger_and_tables(
    settings: PostgreSQLTestSettings,
) -> None:
    with _connect(settings) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            )
            """
        )
        assert cursor.fetchone() == (True,)
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.schemata
                WHERE schema_name = 'callmetric_vector'
            )
            """
        )
        assert cursor.fetchone() == (True,)
        cursor.execute(
            "SELECT version FROM callmetric_vector.schema_migrations ORDER BY version"
        )
        assert cursor.fetchall() == [("0001",), ("0002",), ("0003",)]
        cursor.execute(
            """
            SELECT is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'callmetric_vector'
              AND table_name = 'documents'
              AND column_name = 'storage_object_key'
            """
        )
        assert cursor.fetchall() == [("YES",)]
        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'callmetric_vector'
            ORDER BY table_name
            """
        )
        assert cursor.fetchall() == [
            ("document_ingestion_jobs",),
            ("documents",),
            ("embedding_profiles",),
            ("schema_migrations",),
            ("vector_records",),
        ]


def test_production_migration_serializes_and_is_idempotent(
    migration_results: tuple[
        PostgreSQLMigrationResult,
        PostgreSQLMigrationResult,
    ],
) -> None:
    assert sorted(result.value for result in migration_results) == [
        PostgreSQLMigrationResult.ALREADY_APPLIED.value,
        PostgreSQLMigrationResult.APPLIED.value,
    ]


def test_document_registry_scope_constraints_cascade_and_legacy_vector_isolation(
    settings: PostgreSQLTestSettings,
) -> None:
    scopes = (
        _profile("registry-primary"),
        _profile(
            "registry-tenant",
            tenant_id="tenant-registry-other",
            knowledge_base_id="kb-registry-primary",
        ),
        _profile(
            "registry-kb",
            tenant_id="tenant-registry-primary",
            knowledge_base_id="kb-registry-other",
        ),
    )
    repository = _repository(settings)
    for profile in scopes:
        repository.register_profile(profile)

    primary = scopes[0]
    _store(settings, primary).upsert(
        _record(
            primary,
            document_id="document-registry-primary",
            chunk_id="chunk-existing",
            text="Synthetic existing vector",
        )
    )
    digest = "a" * 64
    with _connect(settings) as connection, connection.cursor() as cursor:
        for index, profile in enumerate(scopes, start=1):
            cursor.execute(
                """
                INSERT INTO callmetric_vector.documents (
                    tenant_id, knowledge_base_id, document_id,
                    original_filename, media_type, byte_size, sha256_hex,
                    storage_object_key
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    profile.tenant_id,
                    profile.knowledge_base_id,
                    (
                        "document-registry-primary"
                        if index == 1
                        else f"document-registry-{index}"
                    ),
                    f"synthetic-{index}.txt",
                    "text/plain",
                    index,
                    digest,
                    f"registry/{index}",
                ),
            )
        cursor.execute(
            """
            INSERT INTO callmetric_vector.document_ingestion_jobs (
                tenant_id, knowledge_base_id, job_id, document_id,
                state, phase
            ) VALUES (%s, %s, %s, %s, 'QUEUED', 'VALIDATION')
            """,
            (
                primary.tenant_id,
                primary.knowledge_base_id,
                "job-registry-primary",
                "document-registry-primary",
            ),
        )

    with (
        _connect(settings) as connection,
        pytest.raises(psycopg.errors.UniqueViolation),
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """
            INSERT INTO callmetric_vector.documents (
                tenant_id, knowledge_base_id, document_id,
                original_filename, media_type, byte_size, sha256_hex,
                storage_object_key
            ) VALUES (%s, %s, %s, 'duplicate.txt', 'text/plain', 1, %s, %s)
            """,
            (
                primary.tenant_id,
                primary.knowledge_base_id,
                "document-registry-duplicate",
                digest,
                "registry/duplicate",
            ),
        )

    with (
        _connect(settings) as connection,
        pytest.raises(psycopg.errors.ForeignKeyViolation),
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """
            INSERT INTO callmetric_vector.document_ingestion_jobs (
                tenant_id, knowledge_base_id, job_id, document_id,
                state, phase
            ) VALUES (%s, %s, %s, %s, 'QUEUED', 'VALIDATION')
            """,
            (
                scopes[1].tenant_id,
                scopes[1].knowledge_base_id,
                "job-cross-scope",
                "document-registry-primary",
            ),
        )

    with _connect(settings) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM callmetric_vector.documents
            WHERE tenant_id = %s AND knowledge_base_id = %s AND document_id = %s
            """,
            (
                primary.tenant_id,
                primary.knowledge_base_id,
                "document-registry-primary",
            ),
        )
        cursor.execute(
            """
            SELECT count(*) FROM callmetric_vector.document_ingestion_jobs
            WHERE tenant_id = %s AND knowledge_base_id = %s AND document_id = %s
            """,
            (
                primary.tenant_id,
                primary.knowledge_base_id,
                "document-registry-primary",
            ),
        )
        assert cursor.fetchone() == (0,)
        cursor.execute(
            """
            SELECT text FROM callmetric_vector.vector_records
            WHERE tenant_id = %s AND knowledge_base_id = %s
              AND document_id = %s AND chunk_id = %s
            """,
            (
                primary.tenant_id,
                primary.knowledge_base_id,
                "document-registry-primary",
                "chunk-existing",
            ),
        )
        assert cursor.fetchone() == ("Synthetic existing vector",)


def test_schema_readiness_and_explicit_composed_profile_provisioning(
    settings: PostgreSQLTestSettings,
) -> None:
    from app.composition.postgres_rag import (
        KnowledgeBaseRAGProviderSettings,
        PostgreSQLVectorStoreSettings,
        compose_profile_bound_postgres_rag,
    )

    checker = PostgreSQLSchemaReadinessChecker(
        connection_factory=lambda: _connect(settings)
    )
    assert checker.verify() is None

    dsn = (
        f"host={settings.host} port={settings.port} "
        f"dbname={settings.database} user={settings.user} "
        f"password={settings.password}"
    )
    composition = compose_profile_bound_postgres_rag(
        postgres_settings=PostgreSQLVectorStoreSettings(
            dsn=SecretStr(dsn),
            connect_timeout_seconds=settings.connect_timeout,
            ssl_mode="require",
            application_name="callmetric-pr38-integration",
        ),
        knowledge_base_settings=KnowledgeBaseRAGProviderSettings(
            tenant_id="tenant-readiness",
            knowledge_base_id="kb-readiness",
            model_id="model-readiness",
            model_name_or_path="synthetic-local-model",
            vector_dimension=2,
            normalize_embeddings=True,
            device="cpu",
            local_files_only=True,
        ),
        psycopg_connect=lambda **kwargs: _production_connect(settings, **kwargs),
        embedding_backend_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("embedding backend must not load")
        ),
    )
    repository = composition.profile_repository
    assert (
        repository.get_profile(
            tenant_id=composition.profile.tenant_id,
            knowledge_base_id=composition.profile.knowledge_base_id,
        )
        is None
    )
    first = repository.register_profile(composition.profile)
    second = repository.register_profile(composition.profile)
    assert first is composition.profile
    assert second == composition.profile


def test_profile_registration_idempotency_conflict_and_persistence(
    settings: PostgreSQLTestSettings,
) -> None:
    profile = _profile("profile")
    repository = _repository(settings)
    assert (
        repository.get_profile(
            tenant_id=profile.tenant_id,
            knowledge_base_id=profile.knowledge_base_id,
        )
        is None
    )
    assert repository.register_profile(profile) == profile
    assert (
        _repository(settings).get_profile(
            tenant_id=profile.tenant_id,
            knowledge_base_id=profile.knowledge_base_id,
        )
        == profile
    )
    assert repository.register_profile(profile) == profile
    conflicting = profile.model_copy(update={"model_id": "model-conflicting"})
    with pytest.raises(ValueError, match="conflicts"):
        repository.register_profile(conflicting)
    assert (
        _repository(settings).get_profile(
            tenant_id=profile.tenant_id,
            knowledge_base_id=profile.knowledge_base_id,
        )
        == profile
    )


def test_scope_isolation_and_no_cross_scope_search_leakage(
    settings: PostgreSQLTestSettings,
) -> None:
    first = _profile("scope-first")
    second = _profile("scope-second")
    repository = _repository(settings)
    repository.register_profile(first)
    repository.register_profile(second)
    first_store = _store(settings, first)
    second_store = _store(settings, second)
    first_store.upsert(
        _record(
            first,
            document_id="document-first",
            chunk_id="chunk-shared",
            text="Synthetic first scope",
        )
    )
    second_store.upsert(
        _record(
            second,
            document_id="document-second",
            chunk_id="chunk-shared",
            text="Synthetic second scope",
        )
    )
    first_result = first_store.search(
        SearchRequest(
            tenant_id=first.tenant_id,
            knowledge_base_id=first.knowledge_base_id,
            query_embedding=(1.0, 0.0),
            top_k=10,
            minimum_score=0.0,
        )
    )
    assert [hit.record.text for hit in first_result.hits] == ["Synthetic first scope"]


def test_atomic_batch_idempotency_and_mixed_conflict_rollback(
    settings: PostgreSQLTestSettings,
) -> None:
    profile = _profile("batch")
    _repository(settings).register_profile(profile)
    store = _store(settings, profile)
    original = _record(
        profile,
        document_id="document-a",
        chunk_id="chunk-a",
        text="Synthetic original",
    )
    first = store.admit_batch(
        VectorBatchWriteRequest(
            tenant_id=profile.tenant_id,
            knowledge_base_id=profile.knowledge_base_id,
            records=(original,),
        )
    )
    assert [
        (item.document_id, item.chunk_id) for item in first.inserted_identities
    ] == [("document-a", "chunk-a")]
    repeated = store.admit_batch(
        VectorBatchWriteRequest(
            tenant_id=profile.tenant_id,
            knowledge_base_id=profile.knowledge_base_id,
            records=(original,),
        )
    )
    assert repeated.inserted_identities == ()
    assert len(repeated.unchanged_identities) == 1
    missing = _record(
        profile,
        document_id="document-b",
        chunk_id="chunk-b",
        text="Synthetic missing",
    )
    conflicting = original.model_copy(update={"text": "Synthetic conflict"})
    with pytest.raises(ValueError, match="conflicts"):
        store.admit_batch(
            VectorBatchWriteRequest(
                tenant_id=profile.tenant_id,
                knowledge_base_id=profile.knowledge_base_id,
                records=(missing, conflicting),
            )
        )
    fresh_result = _store(settings, profile).search(
        SearchRequest(
            tenant_id=profile.tenant_id,
            knowledge_base_id=profile.knowledge_base_id,
            query_embedding=(1.0, 0.0),
            top_k=10,
            minimum_score=0.0,
        )
    )
    assert [(hit.record.document_id, hit.record.text) for hit in fresh_result.hits] == [
        ("document-a", "Synthetic original")
    ]


def test_upsert_replaces_full_identity_and_preserves_cross_document_chunk(
    settings: PostgreSQLTestSettings,
) -> None:
    profile = _profile("upsert")
    _repository(settings).register_profile(profile)
    store = _store(settings, profile)
    store.upsert(
        _record(
            profile,
            document_id="document-a",
            chunk_id="chunk-shared",
            text="Synthetic before",
        )
    )
    store.upsert(
        _record(
            profile,
            document_id="document-a",
            chunk_id="chunk-shared",
            text="Synthetic after",
        )
    )
    store.upsert(
        _record(
            profile,
            document_id="document-b",
            chunk_id="chunk-shared",
            text="Synthetic other document",
        )
    )
    result = _store(settings, profile).search(
        SearchRequest(
            tenant_id=profile.tenant_id,
            knowledge_base_id=profile.knowledge_base_id,
            query_embedding=(1.0, 0.0),
            top_k=10,
            minimum_score=0.0,
        )
    )
    assert [(hit.record.document_id, hit.record.text) for hit in result.hits] == [
        ("document-a", "Synthetic after"),
        ("document-b", "Synthetic other document"),
    ]


def test_cosine_search_order_threshold_top_k_and_complete_mapping(
    settings: PostgreSQLTestSettings,
) -> None:
    profile = _profile("search")
    _repository(settings).register_profile(profile)
    store = _store(settings, profile)
    records = (
        _record(
            profile,
            document_id="document-b",
            chunk_id="chunk-b",
            text="Synthetic tie B",
            metadata=(("dil", "Türkçe"), ("sıra", "ikinci")),
        ),
        _record(
            profile,
            document_id="document-a",
            chunk_id="chunk-a",
            text="Synthetic tie A",
            metadata=(("dil", "Türkçe"), ("sıra", "birinci")),
        ),
        _record(
            profile,
            document_id="document-c",
            chunk_id="chunk-c",
            text="Synthetic orthogonal",
            embedding=(0.0, 1.0),
        ),
    )
    store.admit_batch(
        VectorBatchWriteRequest(
            tenant_id=profile.tenant_id,
            knowledge_base_id=profile.knowledge_base_id,
            records=records,
        )
    )
    result = _store(settings, profile).search(
        SearchRequest(
            tenant_id=profile.tenant_id,
            knowledge_base_id=profile.knowledge_base_id,
            query_embedding=(1.0, 0.0),
            top_k=2,
            minimum_score=0.75,
        )
    )
    assert [
        (hit.record.document_id, hit.record.chunk_id, hit.score) for hit in result.hits
    ] == [
        ("document-a", "chunk-a", 1.0),
        ("document-b", "chunk-b", 1.0),
    ]
    assert result.hits[0].record.text == "Synthetic tie A"
    assert result.hits[0].record.embedding == (1.0, 0.0)
    assert result.hits[0].record.metadata == (
        ("dil", "Türkçe"),
        ("sıra", "birinci"),
    )


def test_dimension_and_zero_vector_failures_leave_database_usable(
    settings: PostgreSQLTestSettings,
) -> None:
    profile = _profile("constraints")
    repository = _repository(settings)
    repository.register_profile(profile)
    store = _store(settings, profile)
    with pytest.raises(ValueError, match="dimension"):
        store.upsert(
            _record(
                profile,
                document_id="document-dimension",
                chunk_id="chunk-dimension",
                text="Synthetic dimension mismatch",
                embedding=(1.0, 0.0, 0.0),
            )
        )
    wrong_dimension_row = PostgreSQLStoredVectorRow(
        tenant_id=profile.tenant_id,
        knowledge_base_id=profile.knowledge_base_id,
        document_id="document-dimension-database",
        chunk_id="chunk-dimension-database",
        text="Synthetic database dimension mismatch",
        embedding=(1.0, 0.0, 0.0),
        metadata_json="[]",
    )
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        _runner(settings).run_in_transaction(
            lambda transaction: transaction.replace_record(wrong_dimension_row)
        )
    zero_row = PostgreSQLStoredVectorRow(
        tenant_id=profile.tenant_id,
        knowledge_base_id=profile.knowledge_base_id,
        document_id="document-zero",
        chunk_id="chunk-zero",
        text="Synthetic zero",
        embedding=(0.0, 0.0),
        metadata_json="[]",
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        _runner(settings).run_in_transaction(
            lambda transaction: transaction.replace_record(zero_row)
        )
    assert (
        _repository(settings).get_profile(
            tenant_id=profile.tenant_id,
            knowledge_base_id=profile.knowledge_base_id,
        )
        == profile
    )


def test_callback_write_then_exception_rolls_back(
    settings: PostgreSQLTestSettings,
) -> None:
    profile = _profile("callback-rollback")
    runner = _runner(settings)
    expected = RuntimeError("synthetic callback failure")

    def write_then_fail(transaction: Any) -> None:
        transaction.insert_profile(profile)
        raise expected

    with pytest.raises(RuntimeError) as raised:
        runner.run_in_transaction(write_then_fail)
    assert raised.value is expected
    assert (
        _repository(settings).get_profile(
            tenant_id=profile.tenant_id,
            knowledge_base_id=profile.knowledge_base_id,
        )
        is None
    )


def test_same_scope_advisory_lock_blocks_then_releases(
    settings: PostgreSQLTestSettings,
) -> None:
    tenant_id = "tenant-advisory-lock"
    knowledge_base_id = "kb-advisory-lock"
    first_connection = _connect(settings)
    second_connection = _connect(settings)
    first_transaction = PsycopgPostgreSQLVectorTransaction(first_connection)
    second_transaction = PsycopgPostgreSQLVectorTransaction(second_connection)
    attempted = threading.Event()
    acquired = threading.Event()
    failures: list[BaseException] = []

    def acquire_second() -> None:
        attempted.set()
        try:
            second_transaction.acquire_scope_lock(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
            )
            acquired.set()
            second_connection.commit()
        except BaseException as error:
            failures.append(error)

    try:
        first_transaction.acquire_scope_lock(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
        )
        thread = threading.Thread(target=acquire_second, daemon=True)
        thread.start()
        assert attempted.wait(timeout=2.0)
        time.sleep(0.25)
        assert not acquired.is_set()
        first_connection.commit()
        assert acquired.wait(timeout=4.0)
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert failures == []
    finally:
        first_connection.rollback()
        second_connection.rollback()
        first_connection.close()
        second_connection.close()
