"""Deterministic tests for the explicit PostgreSQL migration CLI."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

import app.deployment as deployment
import app.deployment.postgres_migrations as migrations
import scripts.migrate_postgres_vector_store as cli
from app.deployment import PostgreSQLMigrationResult, PostgreSQLMigrationSettings

_DSN = "postgresql://synthetic_migrator:synthetic_password@db.invalid/synthetic"
_ENVIRONMENT = {
    "CALLMETRIC_POSTGRES_MIGRATION_DSN": _DSN,
    "CALLMETRIC_POSTGRES_MIGRATION_CONNECT_TIMEOUT_SECONDS": "5",
    "CALLMETRIC_POSTGRES_MIGRATION_SSL_MODE": "verify-full",
    "CALLMETRIC_POSTGRES_MIGRATION_APPLICATION_NAME": "callmetric-migration",
    "CALLMETRIC_POSTGRES_MIGRATION_LOCK_TIMEOUT_SECONDS": "7",
    "CALLMETRIC_POSTGRES_MIGRATION_STATEMENT_TIMEOUT_SECONDS": "11",
}


def _set_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            PostgreSQLMigrationResult.APPLIED,
            "PostgreSQL vector-store migration applied.\n",
        ),
        (
            PostgreSQLMigrationResult.ALREADY_APPLIED,
            "PostgreSQL vector-store migration is already applied.\n",
        ),
    ],
)
def test_success_results_are_fixed_and_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    result: PostgreSQLMigrationResult,
    message: str,
) -> None:
    _set_environment(monkeypatch)
    calls: list[dict[str, object]] = []

    def apply(**kwargs: object) -> PostgreSQLMigrationResult:
        calls.append(kwargs)
        return result

    monkeypatch.setattr(cli, "apply_postgres_vector_migrations", apply)

    assert cli.main([]) == 0

    output = capsys.readouterr()
    assert output.out == message
    assert output.err == ""
    assert len(calls) == 1
    assert calls[0]["psycopg_connect"] is cli.psycopg.connect
    settings = cast(PostgreSQLMigrationSettings, calls[0]["settings"])
    assert settings.dsn.get_secret_value() == _DSN


def test_unexpected_arguments_use_conventional_argparse_exit() -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["--dsn", "synthetic-secret"])

    assert raised.value.code == 2


def test_invalid_environment_returns_two_without_operation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    calls = 0

    def apply(**_kwargs: object) -> PostgreSQLMigrationResult:
        nonlocal calls
        calls += 1
        return PostgreSQLMigrationResult.APPLIED

    monkeypatch.setattr(cli, "apply_postgres_vector_migrations", apply)

    assert cli.main([]) == 2

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == (
        "PostgreSQL vector-store migration configuration is invalid.\n"
    )
    assert calls == 0


@pytest.mark.parametrize(
    ("content", "identifier"),
    [
        (None, "missing"),
        (b"BEGIN;\nSELECT 'altered';\nCOMMIT;\n", "altered"),
        (b"BEGIN;\n<<<<<<< ours\nCOMMIT;\n", "conflicted"),
    ],
)
def test_migration_integrity_failure_returns_two_without_connection_or_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: bytes | None,
    identifier: str,
) -> None:
    _set_environment(monkeypatch)
    root = tmp_path / identifier
    path = root / "migrations" / "postgres" / "0001_vector_store.sql"
    path.parent.mkdir(parents=True)
    if content is not None:
        path.write_bytes(content)
    monkeypatch.setattr(migrations, "_REPOSITORY_ROOT", root)
    monkeypatch.setattr(migrations, "_MIGRATION_PATH", path)
    connect_calls = 0

    def connect(**_kwargs: object) -> None:
        nonlocal connect_calls
        connect_calls += 1

    monkeypatch.setattr(cli.psycopg, "connect", connect)

    assert cli.main([]) == 2

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == (
        "PostgreSQL vector-store migration configuration is invalid.\n"
    )
    assert connect_calls == 0
    for private_value in (_DSN, str(path), "BEGIN", "SELECT", "digest", identifier):
        assert private_value not in output.err


@pytest.mark.parametrize(
    "error",
    [
        ValueError("synthetic database drift"),
        RuntimeError("synthetic readiness failure"),
        OSError("synthetic provider failure"),
    ],
    ids=["database-drift", "readiness", "provider"],
)
def test_operational_failure_returns_one_without_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    _set_environment(monkeypatch)

    def apply(**_kwargs: object) -> PostgreSQLMigrationResult:
        raise error

    monkeypatch.setattr(cli, "apply_postgres_vector_migrations", apply)

    assert cli.main([]) == 1

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "PostgreSQL vector-store migration failed.\n"
    assert _DSN not in output.err
    assert str(error) not in output.err


def test_configuration_exception_is_not_a_public_deployment_export() -> None:
    assert "PostgreSQLMigrationConfigurationError" not in deployment.__all__
    assert not hasattr(deployment, "PostgreSQLMigrationConfigurationError")


def test_keyboard_interrupt_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_environment(monkeypatch)

    def apply(**_kwargs: object) -> PostgreSQLMigrationResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "apply_postgres_vector_migrations", apply)

    with pytest.raises(KeyboardInterrupt):
        cli.main([])
