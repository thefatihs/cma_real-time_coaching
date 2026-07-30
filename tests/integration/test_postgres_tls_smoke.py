"""Opt-in real TLS PostgreSQL RAG smoke coverage."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import psycopg
import pytest
from psycopg import Connection, OperationalError, sql

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.composition import (  # noqa: E402
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
)
from app.deployment import (  # noqa: E402
    PostgreSQLMigrationResult,
    PostgreSQLMigrationSettings,
    PostgreSQLRAGRetrievalRequest,
    apply_postgres_vector_migrations,
    ingest_profile_bound_postgres_rag,
    provision_profile_bound_postgres_rag,
    retrieve_profile_bound_postgres_rag,
)
from app.ingestion import DocumentChunkInput, DocumentIngestionRequest  # noqa: E402

pytestmark = pytest.mark.postgres_integration

DATABASE = "callmetric_vector_tls_smoke"
MIGRATION_USER = "callmetric_tls_migration"
APPLICATION_USER = "callmetric_tls_application"
TLS_HOST = "localhost"
MISMATCHED_HOST = "mismatch.invalid"
LOOPBACK_HOST = "127.0.0.1"


class _SyntheticBackend:
    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
    ) -> list[list[float]]:
        assert normalize_embeddings is True
        return [[1.0, 0.0] for _ in texts]


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail("TLS smoke environment is incomplete")
    return value


def _port() -> int:
    raw_port = _required_environment("CALLMETRIC_POSTGRES_TLS_PORT")
    if not raw_port.isascii() or not raw_port.isdigit():
        pytest.fail("TLS smoke port is malformed")
    port = int(raw_port)
    if not 1 <= port <= 65535:
        pytest.fail("TLS smoke port is outside the valid range")
    return port


def _certificate_path() -> Path:
    path = Path(_required_environment("CALLMETRIC_POSTGRES_TLS_CERT_DIR"))
    if not path.is_absolute() or not (path / "ca.crt").is_file():
        pytest.fail("TLS smoke trust root is unavailable")
    return path / "ca.crt"


def _connect(
    *,
    user: str,
    password: str,
    host: str = TLS_HOST,
    hostaddr: str | None = None,
    root_certificate: Path | None = None,
) -> Connection[tuple[object, ...]]:
    return psycopg.connect(
        host=host,
        hostaddr=hostaddr,
        port=_port(),
        dbname=DATABASE,
        user=user,
        password=password,
        sslmode="verify-full",
        sslrootcert=str(root_certificate) if root_certificate is not None else None,
        connect_timeout=5,
        autocommit=False,
    )


def _apply_migration_twice() -> None:
    settings_factory = cast(
        Callable[[], PostgreSQLMigrationSettings],
        PostgreSQLMigrationSettings,
    )
    settings = settings_factory()
    first = apply_postgres_vector_migrations(
        settings=settings,
        psycopg_connect=psycopg.connect,
    )
    second = apply_postgres_vector_migrations(
        settings=settings,
        psycopg_connect=psycopg.connect,
    )
    assert first is PostgreSQLMigrationResult.APPLIED
    assert second is PostgreSQLMigrationResult.ALREADY_APPLIED


def _create_application_role() -> None:
    migration_password = _required_environment(
        "CALLMETRIC_POSTGRES_TLS_MIGRATION_PASSWORD"
    )
    application_password = _required_environment(
        "CALLMETRIC_POSTGRES_TLS_APPLICATION_PASSWORD"
    )
    connection = _connect(
        user=MIGRATION_USER,
        password=migration_password,
        root_certificate=_certificate_path(),
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(APPLICATION_USER),
                    sql.Literal(application_password),
                )
            )
            cursor.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                    sql.Identifier(DATABASE),
                    sql.Identifier(APPLICATION_USER),
                )
            )
            cursor.execute(
                sql.SQL("GRANT USAGE ON SCHEMA callmetric_vector TO {}").format(
                    sql.Identifier(APPLICATION_USER)
                )
            )
            cursor.execute(
                sql.SQL(
                    "GRANT SELECT, INSERT, UPDATE ON ALL TABLES "
                    "IN SCHEMA callmetric_vector TO {}"
                ).format(sql.Identifier(APPLICATION_USER))
            )
        connection.commit()
    finally:
        connection.close()


def _require_tls_for_application_role() -> None:
    application_password = _required_environment(
        "CALLMETRIC_POSTGRES_TLS_APPLICATION_PASSWORD"
    )
    connection = _connect(
        user=APPLICATION_USER,
        password=application_password,
        root_certificate=_certificate_path(),
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()")
            assert cursor.fetchall() == [(True,)]
        connection.rollback()
    finally:
        connection.close()


def _require_tls_failures() -> None:
    migration_password = _required_environment(
        "CALLMETRIC_POSTGRES_TLS_MIGRATION_PASSWORD"
    )
    with pytest.raises(OperationalError):
        _connect(user=MIGRATION_USER, password=migration_password)
    with pytest.raises(OperationalError):
        _connect(
            user=MIGRATION_USER,
            password=migration_password,
            host=MISMATCHED_HOST,
            hostaddr=LOOPBACK_HOST,
            root_certificate=_certificate_path(),
        )


def _require_profile_ingestion_and_retrieval() -> None:
    settings_factory = cast(
        Callable[[], PostgreSQLVectorStoreSettings],
        PostgreSQLVectorStoreSettings,
    )
    postgres_settings = settings_factory()
    provider_settings = KnowledgeBaseRAGProviderSettings(
        tenant_id="tenant-tls-smoke",
        knowledge_base_id="kb-tls-smoke",
        model_id="synthetic-tls-smoke-model",
        model_name_or_path="synthetic-local-only-model",
        vector_dimension=2,
        normalize_embeddings=True,
        device="cpu",
        local_files_only=True,
    )

    def backend_factory(_config: object) -> _SyntheticBackend:
        return _SyntheticBackend()

    first_profile = provision_profile_bound_postgres_rag(
        postgres_settings=postgres_settings,
        knowledge_base_settings=provider_settings,
        psycopg_connect=psycopg.connect,
        embedding_backend_factory=backend_factory,
    )
    second_profile = provision_profile_bound_postgres_rag(
        postgres_settings=postgres_settings,
        knowledge_base_settings=provider_settings,
        psycopg_connect=psycopg.connect,
        embedding_backend_factory=backend_factory,
    )
    assert first_profile == second_profile

    ingestion_request = DocumentIngestionRequest(
        tenant_id=provider_settings.tenant_id,
        knowledge_base_id=provider_settings.knowledge_base_id,
        chunks=(
            DocumentChunkInput(
                document_id="synthetic-document",
                chunk_id="chunk_000001",
                text="Synthetic TLS retrieval evidence.",
                metadata=(("kind", "synthetic"),),
            ),
        ),
    )
    write_result = ingest_profile_bound_postgres_rag(
        postgres_settings=postgres_settings,
        knowledge_base_settings=provider_settings,
        request=ingestion_request,
        psycopg_connect=psycopg.connect,
        embedding_backend_factory=backend_factory,
    )
    assert len(write_result.inserted_identities) == 1
    assert write_result.unchanged_identities == ()

    retrieval = retrieve_profile_bound_postgres_rag(
        postgres_settings=postgres_settings,
        knowledge_base_settings=provider_settings,
        request=PostgreSQLRAGRetrievalRequest(
            tenant_id=provider_settings.tenant_id,
            knowledge_base_id=provider_settings.knowledge_base_id,
            query="Synthetic TLS query.",
            top_k=1,
            minimum_score=0.0,
        ),
        psycopg_connect=psycopg.connect,
        embedding_backend_factory=backend_factory,
    )
    assert tuple(
        (document.document_id, document.chunk_id) for document in retrieval.documents
    ) == (("synthetic-document", "chunk_000001"),)


def test_postgres_tls_smoke_end_to_end() -> None:
    if os.environ.get("CALLMETRIC_POSTGRES_TLS_SMOKE") != "1":
        pytest.skip("requires the opt-in PostgreSQL TLS smoke runner")

    _apply_migration_twice()
    _require_tls_failures()
    _create_application_role()
    _require_tls_for_application_role()
    _require_profile_ingestion_and_retrieval()
