"""Deterministic tests for explicit PostgreSQL vector-store migrations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from psycopg import Connection
from psycopg.pq import TransactionStatus
from pydantic import SecretStr, ValidationError

import app.deployment.postgres_migrations as migrations
from app.deployment import (
    PostgreSQLMigrationResult,
    PostgreSQLMigrationSettings,
    apply_postgres_vector_migrations,
)

_DSN = "postgresql://synthetic_migrator:synthetic_password@db.invalid/synthetic"
_ENVIRONMENT = {
    "CALLMETRIC_POSTGRES_MIGRATION_DSN": _DSN,
    "CALLMETRIC_POSTGRES_MIGRATION_CONNECT_TIMEOUT_SECONDS": "5",
    "CALLMETRIC_POSTGRES_MIGRATION_SSL_MODE": "verify-full",
    "CALLMETRIC_POSTGRES_MIGRATION_APPLICATION_NAME": "callmetric-migration",
    "CALLMETRIC_POSTGRES_MIGRATION_LOCK_TIMEOUT_SECONDS": "7",
    "CALLMETRIC_POSTGRES_MIGRATION_STATEMENT_TIMEOUT_SECONDS": "11",
}


def _settings(
    *,
    lock_timeout_seconds: int = 7,
    statement_timeout_seconds: int = 11,
) -> PostgreSQLMigrationSettings:
    return PostgreSQLMigrationSettings(
        dsn=SecretStr(_DSN),
        connect_timeout_seconds=5,
        ssl_mode="verify-full",
        application_name="callmetric-migration",
        lock_timeout_seconds=lock_timeout_seconds,
        statement_timeout_seconds=statement_timeout_seconds,
    )


@dataclass
class FakeInfo:
    transaction_status: TransactionStatus = TransactionStatus.IDLE


class FakeCursor:
    def __init__(
        self, responses: list[object], calls: list[tuple[object, ...]]
    ) -> None:
        self.responses = responses
        self.calls = calls
        self.response_index = -1
        self.last_is_query = False

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(
        self,
        query: object,
        parameters: object = None,
        *,
        prepare: bool | None = None,
    ) -> None:
        if prepare is None:
            call = (query,) if parameters is None else (query, parameters)
            self.last_is_query = True
            self.response_index += 1
        else:
            call = (query, "prepare", prepare)
            self.last_is_query = False
        self.calls.append(call)

    def fetchall(self) -> object:
        if not self.last_is_query:
            raise AssertionError("fetchall must follow a catalog query")
        return self.responses[self.response_index]


class FakeConnection:
    def __init__(
        self,
        responses: list[object],
        *,
        autocommit: object = False,
        rollback_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.autocommit = autocommit
        self.info = FakeInfo()
        self.responses = responses
        self.rollback_error = rollback_error
        self.close_error = close_error
        self.calls: list[tuple[object, ...]] = []
        self.lifecycle: list[str] = []
        self.rollback_calls = 0
        self.commit_calls = 0
        self.close_calls = 0

    def cursor(self) -> FakeCursor:
        self.lifecycle.append("cursor")
        return FakeCursor(self.responses, self.calls)

    def rollback(self) -> None:
        self.lifecycle.append("rollback")
        self.rollback_calls += 1
        if self.rollback_error is not None:
            raise self.rollback_error

    def commit(self) -> None:
        self.lifecycle.append("commit")
        self.commit_calls += 1

    def close(self) -> None:
        self.lifecycle.append("close")
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FakeConnect:
    def __init__(self, connections: list[FakeConnection]) -> None:
        self.connections = connections
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> Connection[Any]:
        self.calls.append(kwargs)
        return cast(Connection[Any], self.connections[len(self.calls) - 1])


def _fresh_responses() -> list[object]:
    return [[(None,)], [], []]


def _psycopg_fresh_responses() -> list[object]:
    return [[("",)], [], []]


def _applied_responses() -> list[object]:
    return [
        [(None,)],
        [("0.8.5",)],
        [("callmetric_vector",)],
        [("0001",), ("0002",), ("0003",)],
        [
            ("document_ingestion_jobs",),
            ("documents",),
            ("embedding_profiles",),
            ("schema_migrations",),
            ("vector_records",),
        ],
    ]


def _version_one_responses() -> list[object]:
    return [
        [(None,)],
        [("0.8.5",)],
        [("callmetric_vector",)],
        [("0001",)],
        [
            ("embedding_profiles",),
            ("schema_migrations",),
            ("vector_records",),
        ],
    ]


def _install_readiness(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> None:
    class FakeReadinessChecker:
        def __init__(self, *, connection_factory: Any) -> None:
            events.append("readiness-construction")
            self.connection_factory = connection_factory

        def verify(self) -> None:
            events.append("readiness")
            connection = self.connection_factory()
            connection.close()

    monkeypatch.setattr(
        migrations,
        "PostgreSQLSchemaReadinessChecker",
        FakeReadinessChecker,
    )


def test_settings_load_exact_environment_and_are_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in _ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    factory = cast(Any, PostgreSQLMigrationSettings)

    settings = factory()

    assert settings.dsn.get_secret_value() == _DSN
    assert settings.lock_timeout_seconds == 7
    assert settings.statement_timeout_seconds == 11
    assert _DSN not in repr(settings)
    assert _DSN not in str(settings.model_dump())
    with pytest.raises(ValidationError):
        settings.lock_timeout_seconds = 8


def test_runtime_postgres_environment_is_not_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CALLMETRIC_POSTGRES_DSN", _DSN)

    with pytest.raises(ValidationError):
        cast(Any, PostgreSQLMigrationSettings)()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("connect_timeout_seconds", 0),
        ("connect_timeout_seconds", 61),
        ("connect_timeout_seconds", True),
        ("lock_timeout_seconds", 0),
        ("lock_timeout_seconds", 301),
        ("lock_timeout_seconds", False),
        ("statement_timeout_seconds", " 5"),
        ("ssl_mode", "disable"),
        ("application_name", " unsafe "),
    ],
)
def test_settings_validation_is_strict(field: str, value: object) -> None:
    values = _settings().model_dump()
    values["dsn"] = _DSN
    values[field] = value

    with pytest.raises(ValidationError):
        PostgreSQLMigrationSettings.model_validate(values)


def test_registry_path_version_and_digest_are_fixed() -> None:
    assert migrations._MIGRATION_VERSION == "0001"  # noqa: SLF001
    assert (
        migrations._MIGRATION_PATH.relative_to(  # noqa: SLF001
            migrations._REPOSITORY_ROOT  # noqa: SLF001
        ).as_posix()
        == "migrations/postgres/0001_vector_store.sql"
    )
    assert migrations._MIGRATION_SHA256 == (  # noqa: SLF001
        "deae3547544dac4d31c37b0b6e214cc1e54e5e2c164341323de8e1cf75c82aa7"
    )
    assert migrations._MIGRATION_LOCK_KEY == -4795186792673390552  # noqa: SLF001
    assert [item.version for item in migrations._MIGRATIONS] == [  # noqa: SLF001
        "0001",
        "0002",
        "0003",
    ]
    assert [item.relative_path for item in migrations._MIGRATIONS] == [  # noqa: SLF001
        "migrations/postgres/0001_vector_store.sql",
        "migrations/postgres/0002_document_registry.sql",
        "migrations/postgres/0003_ephemeral_document_sources.sql",
    ]
    assert migrations._MIGRATIONS[1].sha256 == (  # noqa: SLF001
        "7312cd3675b08a3ba645d382d54426afa49542953f28ae23050e44ff7690b6fb"
    )
    assert migrations._MIGRATIONS[2].sha256 == (  # noqa: SLF001
        "0e62fa50728d6ccde59188f92fb98a4fa9d882f4fbd77228f589468844bd0d86"
    )


@pytest.mark.parametrize(
    ("settings", "connect"),
    [
        (object(), lambda **_kwargs: None),
        (_settings(), object()),
    ],
    ids=["wrong-settings-type", "noncallable-connect"],
)
def test_local_configuration_failures_use_dedicated_exception(
    settings: object,
    connect: object,
) -> None:
    with pytest.raises(migrations.PostgreSQLMigrationConfigurationError):
        apply_postgres_vector_migrations(
            settings=cast(Any, settings),
            psycopg_connect=cast(Any, connect),
        )


def _temporary_migration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    content: bytes | None,
) -> Path:
    root = tmp_path / "repository"
    path = root / "migrations" / "postgres" / "0001_vector_store.sql"
    path.parent.mkdir(parents=True)
    if content is not None:
        path.write_bytes(content)
    monkeypatch.setattr(migrations, "_REPOSITORY_ROOT", root)
    monkeypatch.setattr(migrations, "_MIGRATION_PATH", path)
    return path


def test_crlf_canonicalization_preserves_registered_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = (
        Path(__file__).resolve().parents[1]
        / "migrations"
        / "postgres"
        / "0001_vector_store.sql"
    ).read_text(encoding="utf-8")
    _temporary_migration(
        monkeypatch,
        tmp_path,
        original.replace("\n", "\r\n").encode("utf-8"),
    )

    assert migrations._load_registered_migration() == original  # noqa: SLF001


def test_document_registry_digest_mismatch_is_rejected() -> None:
    registered = migrations._MIGRATIONS[1]  # noqa: SLF001
    drifted = migrations._RegisteredMigration(  # noqa: SLF001
        version=registered.version,
        relative_path=registered.relative_path,
        sha256="0" * 64,
        expected_tables_after=registered.expected_tables_after,
    )

    with pytest.raises(
        migrations.PostgreSQLMigrationConfigurationError,
        match="integrity check failed",
    ):
        migrations._load_registered_migration_entry(drifted)  # noqa: SLF001


@pytest.mark.parametrize(
    "content",
    [
        None,
        b"",
        b"\xef\xbb\xbfBEGIN;",
        b"\xff",
        b"BEGIN;\nSELECT 1;\nCOMMIT;\n",
        b"BEGIN;\n<<<<<<< ours\nCOMMIT;\n",
    ],
    ids=["missing", "empty", "bom", "invalid-utf8", "altered", "conflict"],
)
def test_integrity_failures_precede_connection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    content: bytes | None,
) -> None:
    _temporary_migration(monkeypatch, tmp_path, content)
    connect_calls = 0

    def connect(**_kwargs: object) -> Connection[Any]:
        nonlocal connect_calls
        connect_calls += 1
        raise AssertionError("connection must remain deferred")

    with pytest.raises(migrations.PostgreSQLMigrationConfigurationError):
        apply_postgres_vector_migrations(
            settings=_settings(),
            psycopg_connect=connect,
        )

    assert connect_calls == 0


def test_fresh_state_executes_whole_file_after_rollback_then_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration_connection = FakeConnection(_fresh_responses())
    readiness_connection = FakeConnection([])
    connect = FakeConnect([migration_connection, readiness_connection])
    events: list[str] = []
    _install_readiness(monkeypatch, events)

    result = apply_postgres_vector_migrations(
        settings=_settings(),
        psycopg_connect=connect,
    )

    assert result is PostgreSQLMigrationResult.APPLIED
    assert migration_connection.calls[:3] == [
        (migrations._LOCK_SQL, (-4795186792673390552,)),  # noqa: SLF001
        (migrations._EXTENSION_SQL, ("vector",)),  # noqa: SLF001
        (migrations._SCHEMA_SQL, ("callmetric_vector",)),  # noqa: SLF001
    ]
    script_calls = migration_connection.calls[3:6]
    assert all(call[1:] == ("prepare", False) for call in script_calls)
    assert [call[0] for call in script_calls] == list(
        migrations._load_registered_migrations()  # noqa: SLF001
    )
    assert migration_connection.lifecycle.index("rollback") < len(
        migration_connection.calls
    )
    assert migration_connection.commit_calls == 0
    assert events == ["readiness-construction", "readiness"]
    assert migration_connection.lifecycle[-1] == "close"
    assert readiness_connection.close_calls == 1


def test_psycopg_void_advisory_lock_result_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration_connection = FakeConnection(_psycopg_fresh_responses())
    readiness_connection = FakeConnection([])
    connect = FakeConnect([migration_connection, readiness_connection])
    _install_readiness(monkeypatch, [])

    result = apply_postgres_vector_migrations(
        settings=_settings(),
        psycopg_connect=connect,
    )

    assert result is PostgreSQLMigrationResult.APPLIED


def test_applied_state_skips_script_and_returns_already_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration_connection = FakeConnection(_applied_responses())
    readiness_connection = FakeConnection([])
    connect = FakeConnect([migration_connection, readiness_connection])
    events: list[str] = []
    _install_readiness(monkeypatch, events)

    result = apply_postgres_vector_migrations(
        settings=_settings(),
        psycopg_connect=connect,
    )

    assert result is PostgreSQLMigrationResult.ALREADY_APPLIED
    assert all(
        call[0] not in migrations._load_registered_migrations()  # noqa: SLF001
        for call in migration_connection.calls
    )
    assert migration_connection.rollback_calls == 1
    assert migration_connection.commit_calls == 0
    assert events == ["readiness-construction", "readiness"]


def test_version_one_state_applies_remaining_migrations_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration_connection = FakeConnection(_version_one_responses())
    readiness_connection = FakeConnection([])
    connect = FakeConnect([migration_connection, readiness_connection])
    _install_readiness(monkeypatch, [])

    result = apply_postgres_vector_migrations(
        settings=_settings(),
        psycopg_connect=connect,
    )

    assert result is PostgreSQLMigrationResult.APPLIED
    script_calls = [
        call for call in migration_connection.calls if call[1:] == ("prepare", False)
    ]
    assert script_calls == [
        (script, "prepare", False)
        for script in migrations._load_registered_migrations()[1:]  # noqa: SLF001
    ]


def test_exact_generated_options_are_passed_to_both_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect = FakeConnect([FakeConnection(_fresh_responses()), FakeConnection([])])
    _install_readiness(monkeypatch, [])

    apply_postgres_vector_migrations(
        settings=_settings(lock_timeout_seconds=1, statement_timeout_seconds=300),
        psycopg_connect=connect,
    )

    expected = {
        "conninfo": _DSN,
        "connect_timeout": 5,
        "sslmode": "verify-full",
        "application_name": "callmetric-migration",
        "options": "-c lock_timeout=1000ms -c statement_timeout=300000ms",
        "autocommit": False,
    }
    assert connect.calls == [expected, expected]
    assert all(
        "SET" not in str(call[0]).upper() for call in connect.connections[0].calls
    )


@pytest.mark.parametrize(
    "seconds",
    [True, False, 0, 301, "5", "5 -c search_path=unsafe"],
)
def test_timeout_milliseconds_rejects_nonvalidated_values(seconds: object) -> None:
    with pytest.raises(migrations.PostgreSQLMigrationConfigurationError):
        migrations._timeout_milliseconds(seconds)  # noqa: SLF001


@pytest.mark.parametrize(
    "responses",
    [
        [[(None,)], [("0.8.4",)], []],
        [[(None,)], [], [("callmetric_vector",)], []],
        [
            [(None,)],
            [("0.8.5",)],
            [("callmetric_vector",)],
            [("0001",)],
            [("embedding_profiles",), ("vector_records",)],
        ],
        [
            [(None,)],
            [("0.8.5",)],
            [("callmetric_vector",)],
            [],
        ],
        [
            [(None,)],
            [("0.8.5",)],
            [("callmetric_vector",)],
            [("0001",), ("0003",)],
        ],
    ],
    ids=[
        "wrong-extension",
        "schema-without-extension",
        "partial-tables",
        "missing-ledger-version",
        "unknown-version",
    ],
)
def test_drift_states_fail_closed_without_script(
    responses: list[object],
) -> None:
    connection = FakeConnection(responses)

    with pytest.raises(ValueError):
        apply_postgres_vector_migrations(
            settings=_settings(),
            psycopg_connect=FakeConnect([connection]),
        )

    assert all(call[1:] != ("prepare", False) for call in connection.calls)
    assert connection.rollback_calls == 1
    assert connection.close_calls == 1


def test_readiness_failure_preserves_identity_and_closes_lock_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = RuntimeError("synthetic readiness failure")
    migration_connection = FakeConnection(_applied_responses())
    connect = FakeConnect([migration_connection])

    class FailingReadiness:
        def __init__(self, *, connection_factory: Any) -> None:
            del connection_factory

        def verify(self) -> None:
            raise expected

    monkeypatch.setattr(
        migrations,
        "PostgreSQLSchemaReadinessChecker",
        FailingReadiness,
    )

    with pytest.raises(RuntimeError) as raised:
        apply_postgres_vector_migrations(
            settings=_settings(),
            psycopg_connect=connect,
        )

    assert raised.value is expected
    assert migration_connection.close_calls == 1


def test_cleanup_failures_are_chained_without_replacing_primary() -> None:
    rollback = RuntimeError("synthetic rollback")
    close = RuntimeError("synthetic close")
    connection = FakeConnection(
        _fresh_responses(),
        rollback_error=rollback,
        close_error=close,
    )
    connection.responses[1] = [("0.8.4",)]

    with pytest.raises(ValueError) as raised:
        apply_postgres_vector_migrations(
            settings=_settings(),
            psycopg_connect=FakeConnect([connection]),
        )

    assert str(raised.value) == "vector extension state is incompatible"
    assert isinstance(raised.value.__cause__, ExceptionGroup)
    assert raised.value.__cause__.exceptions == (rollback, close)
