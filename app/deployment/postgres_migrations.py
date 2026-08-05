"""Explicit forward-only PostgreSQL vector-store migration deployment."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from psycopg import Connection
from psycopg.pq import TransactionStatus
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.composition.postgres_rag import PsycopgConnect
from app.vector_store.postgres.readiness import PostgreSQLSchemaReadinessChecker

_APPLICATION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_MIGRATION_LOCK_KEY = -4795186792673390552
_EXPECTED_EXTENSION_VERSION = "0.8.5"


@dataclass(frozen=True, slots=True)
class _RegisteredMigration:
    version: str
    relative_path: str
    sha256: str
    expected_tables_after: tuple[str, ...]


_MIGRATIONS = (
    _RegisteredMigration(
        version="0001",
        relative_path="migrations/postgres/0001_vector_store.sql",
        sha256="deae3547544dac4d31c37b0b6e214cc1e54e5e2c164341323de8e1cf75c82aa7",
        expected_tables_after=(
            "embedding_profiles",
            "schema_migrations",
            "vector_records",
        ),
    ),
    _RegisteredMigration(
        version="0002",
        relative_path="migrations/postgres/0002_document_registry.sql",
        sha256="7312cd3675b08a3ba645d382d54426afa49542953f28ae23050e44ff7690b6fb",
        expected_tables_after=(
            "document_ingestion_jobs",
            "documents",
            "embedding_profiles",
            "schema_migrations",
            "vector_records",
        ),
    ),
    _RegisteredMigration(
        version="0003",
        relative_path="migrations/postgres/0003_ephemeral_document_sources.sql",
        sha256="0e62fa50728d6ccde59188f92fb98a4fa9d882f4fbd77228f589468844bd0d86",
        expected_tables_after=(
            "document_ingestion_jobs",
            "documents",
            "embedding_profiles",
            "schema_migrations",
            "vector_records",
        ),
    ),
)

# Compatibility aliases retained for callers and tests that verify immutable 0001.
_MIGRATION_VERSION = _MIGRATIONS[0].version
_MIGRATION_PATH = _REPOSITORY_ROOT / _MIGRATIONS[0].relative_path
_MIGRATION_SHA256 = _MIGRATIONS[0].sha256
_EXPECTED_TABLES = _MIGRATIONS[-1].expected_tables_after

_LOCK_SQL = "SELECT pg_catalog.pg_advisory_lock(%s)"
_EXTENSION_SQL = """
    SELECT extversion
    FROM pg_catalog.pg_extension
    WHERE extname = %s
    """
_SCHEMA_SQL = """
    SELECT nspname
    FROM pg_catalog.pg_namespace
    WHERE nspname = %s
    """
_TABLES_SQL = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = %s
    ORDER BY tablename
    """
_LEDGER_SQL = """
    SELECT version
    FROM callmetric_vector.schema_migrations
    ORDER BY version
    """


class PostgreSQLMigrationSettings(BaseSettings):
    """Secret-safe privileged migration connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="CALLMETRIC_POSTGRES_MIGRATION_",
        env_file=None,
        extra="forbid",
        frozen=True,
    )

    dsn: SecretStr
    connect_timeout_seconds: int
    ssl_mode: Literal["require", "verify-ca", "verify-full"]
    application_name: str
    lock_timeout_seconds: int
    statement_timeout_seconds: int

    @field_validator("dsn")
    @classmethod
    def validate_dsn(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()
        if not secret or secret != secret.strip():
            raise ValueError("migration DSN must be canonical and nonblank")
        return value

    @field_validator(
        "connect_timeout_seconds",
        "lock_timeout_seconds",
        "statement_timeout_seconds",
        mode="before",
    )
    @classmethod
    def validate_timeout(cls, value: object, info: object) -> int:
        if isinstance(value, str):
            if not value.isascii() or not value.isdigit():
                raise ValueError("migration timeout must be an integer")
            value = int(value)
        if type(value) is not int:
            raise ValueError("migration timeout must be an integer")
        field_name = getattr(info, "field_name", "")
        maximum = 60 if field_name == "connect_timeout_seconds" else 300
        if not 1 <= value <= maximum:
            raise ValueError(
                f"{field_name or 'timeout'} must be between 1 and {maximum} seconds"
            )
        return value

    @field_validator("application_name")
    @classmethod
    def validate_application_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("application_name cannot be empty")
        if cleaned != value:
            raise ValueError("application_name must be canonical")
        if not _APPLICATION_NAME_PATTERN.fullmatch(cleaned):
            raise ValueError("application_name contains unsafe characters")
        return cleaned


class PostgreSQLMigrationResult(str, Enum):
    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"


class PostgreSQLMigrationConfigurationError(ValueError):
    """Invalid local migration configuration or registered migration content."""


def apply_postgres_vector_migrations(
    *,
    settings: PostgreSQLMigrationSettings,
    psycopg_connect: PsycopgConnect,
) -> PostgreSQLMigrationResult:
    """Apply ordered registered migrations or validate an identical application."""
    if not isinstance(settings, PostgreSQLMigrationSettings):
        raise PostgreSQLMigrationConfigurationError(
            "settings must be PostgreSQLMigrationSettings"
        )
    if not callable(psycopg_connect):
        raise PostgreSQLMigrationConfigurationError("psycopg_connect must be callable")
    migration_sql = _load_registered_migrations()
    options = _connection_options(settings)

    def connect() -> Connection[Any]:
        return psycopg_connect(
            conninfo=settings.dsn.get_secret_value(),
            connect_timeout=settings.connect_timeout_seconds,
            sslmode=settings.ssl_mode,
            application_name=settings.application_name,
            options=options,
            autocommit=False,
        )

    connection = connect()
    try:
        if connection.autocommit is not False:
            raise ValueError("connection.autocommit must be exactly False")
        applied_versions = _inspect_locked_state(connection)
        connection.rollback()
        pending = tuple(
            (migration, sql_text)
            for migration, sql_text in zip(_MIGRATIONS, migration_sql, strict=True)
            if migration.version not in applied_versions
        )
        for _migration, sql_text in pending:
            with connection.cursor() as cursor:
                cursor.execute(cast(Any, sql_text), prepare=False)
            if connection.info.transaction_status is not TransactionStatus.IDLE:
                raise ValueError(
                    "migration did not complete its transaction boundary",
                )

        readiness_checker = PostgreSQLSchemaReadinessChecker(
            connection_factory=connect,
        )
        readiness_checker.verify()
    except BaseException as primary:
        _raise_after_cleanup(connection, primary)

    connection.close()
    return (
        PostgreSQLMigrationResult.APPLIED
        if pending
        else PostgreSQLMigrationResult.ALREADY_APPLIED
    )


def _connection_options(settings: PostgreSQLMigrationSettings) -> str:
    lock_milliseconds = _timeout_milliseconds(settings.lock_timeout_seconds)
    statement_milliseconds = _timeout_milliseconds(settings.statement_timeout_seconds)
    return (
        f"-c lock_timeout={lock_milliseconds}ms "
        f"-c statement_timeout={statement_milliseconds}ms"
    )


def _timeout_milliseconds(seconds: object) -> int:
    if type(seconds) is not int or not 1 <= seconds <= 300:
        raise PostgreSQLMigrationConfigurationError(
            "timeout seconds must be an integer between 1 and 300"
        )
    milliseconds = seconds * 1000
    if milliseconds // 1000 != seconds:
        raise PostgreSQLMigrationConfigurationError("timeout conversion overflowed")
    return milliseconds


def _load_registered_migration() -> str:
    """Load immutable 0001 for compatibility with existing focused tests."""
    return _load_registered_migration_entry(_MIGRATIONS[0])


def _load_registered_migrations() -> tuple[str, ...]:
    return tuple(_load_registered_migration_entry(item) for item in _MIGRATIONS)


def _load_registered_migration_entry(migration: _RegisteredMigration) -> str:
    path = (_REPOSITORY_ROOT / migration.relative_path).resolve()
    repository_root = _REPOSITORY_ROOT.resolve()
    if path.parent != (repository_root / "migrations" / "postgres").resolve():
        raise PostgreSQLMigrationConfigurationError(
            "registered migration path is invalid"
        )
    if not path.is_file():
        raise PostgreSQLMigrationConfigurationError(
            "registered migration file is missing"
        )
    content = path.read_bytes()
    if content.startswith(b"\xef\xbb\xbf"):
        raise PostgreSQLMigrationConfigurationError(
            "registered migration must not contain a UTF-8 BOM"
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PostgreSQLMigrationConfigurationError(
            "registered migration is not valid UTF-8"
        ) from error
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    if not canonical.strip():
        raise PostgreSQLMigrationConfigurationError("registered migration is empty")
    if any(
        line.startswith(("<" * 7, "=" * 7, ">" * 7)) for line in canonical.splitlines()
    ):
        raise PostgreSQLMigrationConfigurationError(
            "registered migration contains a conflict marker"
        )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != migration.sha256:
        raise PostgreSQLMigrationConfigurationError(
            "registered migration integrity check failed"
        )
    return canonical


def _inspect_locked_state(
    connection: Connection[Any],
) -> tuple[str, ...]:
    with connection.cursor() as cursor:
        cursor.execute(_LOCK_SQL, (_MIGRATION_LOCK_KEY,))
        _require_void_result(cursor.fetchall(), "migration advisory lock")

        cursor.execute(_EXTENSION_SQL, ("vector",))
        extension_rows = _text_rows(
            cursor.fetchall(),
            "vector extension",
        )
        if extension_rows not in ((), (_EXPECTED_EXTENSION_VERSION,)):
            raise ValueError("vector extension state is incompatible")

        cursor.execute(_SCHEMA_SQL, ("callmetric_vector",))
        schema_rows = _text_rows(cursor.fetchall(), "migration schema")
        if not schema_rows:
            return ()
        if schema_rows != ("callmetric_vector",):
            raise ValueError("migration schema state is malformed")
        if extension_rows != (_EXPECTED_EXTENSION_VERSION,):
            raise ValueError("vector extension is missing from applied schema")

        cursor.execute(_LEDGER_SQL)
        ledger_rows = _text_rows(cursor.fetchall(), "migration ledger")
        registered_versions = tuple(item.version for item in _MIGRATIONS)
        if not ledger_rows or ledger_rows != registered_versions[: len(ledger_rows)]:
            raise ValueError("migration ledger state is incompatible")
        expected_tables = _MIGRATIONS[len(ledger_rows) - 1].expected_tables_after

        cursor.execute(_TABLES_SQL, ("callmetric_vector",))
        table_rows = _text_rows(cursor.fetchall(), "migration tables")
        if table_rows != expected_tables:
            raise ValueError("migration table state is incomplete or unexpected")
        return ledger_rows


def _text_rows(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} returned malformed rows")
    result: list[str] = []
    for row in value:
        if not isinstance(row, tuple) or len(row) != 1:
            raise ValueError(f"{field_name} returned a malformed row")
        item = row[0]
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field_name} returned a malformed value")
        result.append(item)
    return tuple(result)


def _require_void_result(value: object, field_name: str) -> None:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{field_name} returned malformed rows")
    row = value[0]
    if not isinstance(row, tuple) or len(row) != 1 or row[0] not in (None, ""):
        raise ValueError(f"{field_name} returned a malformed row")


def _raise_after_cleanup(
    connection: Connection[Any],
    primary: BaseException,
) -> NoReturn:
    cleanup_failures: list[Exception] = []
    try:
        connection.rollback()
    except Exception as error:
        cleanup_failures.append(error)
    try:
        connection.close()
    except Exception as error:
        cleanup_failures.append(error)
    if cleanup_failures:
        raise primary from ExceptionGroup(
            "PostgreSQL migration cleanup failed",
            cleanup_failures,
        )
    raise primary
