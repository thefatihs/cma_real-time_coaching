"""Deterministic tests for the PostgreSQL RAG provisioning CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.provision_postgres_rag as cli
from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
)

_DSN = "postgresql://synthetic_user:synthetic_password@db.invalid/synthetic"
_ENVIRONMENT = {
    "CALLMETRIC_POSTGRES_DSN": _DSN,
    "CALLMETRIC_POSTGRES_CONNECT_TIMEOUT_SECONDS": "5",
    "CALLMETRIC_POSTGRES_SSL_MODE": "verify-full",
    "CALLMETRIC_POSTGRES_APPLICATION_NAME": "callmetric-provision",
}
_PROVIDER = {
    "tenant_id": "tenant-synthetic",
    "knowledge_base_id": "kb-synthetic",
    "model_id": "model-synthetic",
    "model_name_or_path": "local/synthetic-model",
    "vector_dimension": 3,
    "normalize_embeddings": True,
    "device": "cpu",
    "local_files_only": True,
}


def _set_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_provider_settings_argument_is_required() -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main([])

    assert raised.value.code == 2


def test_success_is_fixed_safe_and_invokes_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_environment(monkeypatch)
    path = _write(tmp_path / "provider.json", _PROVIDER)
    calls: list[dict[str, object]] = []

    def provision(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(cli, "provision_profile_bound_postgres_rag", provision)
    argv = ["--provider-settings", str(path)]
    original = list(argv)

    result = cli.main(argv)

    output = capsys.readouterr()
    assert result == 0
    assert output.out == "PostgreSQL RAG profile provisioning succeeded.\n"
    assert output.err == ""
    assert len(calls) == 1
    provider_settings = calls[0]["knowledge_base_settings"]
    postgres_settings = calls[0]["postgres_settings"]
    assert isinstance(provider_settings, KnowledgeBaseRAGProviderSettings)
    assert isinstance(postgres_settings, PostgreSQLVectorStoreSettings)
    assert provider_settings.model_dump() == _PROVIDER
    assert postgres_settings.dsn.get_secret_value() == _DSN
    assert calls[0]["psycopg_connect"] is cli.psycopg.connect
    assert argv == original


@pytest.mark.parametrize(
    "content",
    [
        "",
        " ",
        "{",
        "[]",
        '"synthetic"',
        '{"tenant_id":"first","tenant_id":"second"}',
    ],
    ids=["empty", "blank", "malformed", "array", "scalar", "duplicate-key"],
)
def test_invalid_json_shapes_return_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: str,
) -> None:
    _set_environment(monkeypatch)
    path = tmp_path / "provider.json"
    path.write_text(content, encoding="utf-8")
    calls = 0

    def provision(**_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr(cli, "provision_profile_bound_postgres_rag", provision)

    result = cli.main(["--provider-settings", str(path)])

    output = capsys.readouterr()
    assert result == 2
    assert output.out == ""
    assert output.err == ("PostgreSQL RAG provisioning configuration is invalid.\n")
    assert calls == 0
    assert _DSN not in output.err
    if content.strip():
        assert content not in output.err


@pytest.mark.parametrize(
    "change",
    [
        {"remove": "tenant_id"},
        {"extra": ("unexpected", "synthetic")},
        {"extra": ("dsn", "synthetic-secret")},
        {"extra": ("password", "synthetic-secret")},
        {"replace": ("local_files_only", False)},
        {"replace": ("vector_dimension", 0)},
        {"replace": ("device", "mps")},
    ],
    ids=[
        "missing",
        "unknown",
        "dsn",
        "password",
        "download-enabled",
        "dimension",
        "device",
    ],
)
def test_invalid_provider_contract_returns_two_without_provisioning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    change: dict[str, object],
) -> None:
    _set_environment(monkeypatch)
    payload = dict(_PROVIDER)
    if "remove" in change:
        del payload[str(change["remove"])]
    if "extra" in change:
        key, value = change["extra"]  # type: ignore[misc]
        payload[key] = value
    if "replace" in change:
        key, value = change["replace"]  # type: ignore[misc]
        payload[key] = value
    path = _write(tmp_path / "provider.json", payload)
    calls = 0

    def provision(**_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr(cli, "provision_profile_bound_postgres_rag", provision)

    assert cli.main(["--provider-settings", str(path)]) == 2

    output = capsys.readouterr()
    assert output.err == ("PostgreSQL RAG provisioning configuration is invalid.\n")
    assert calls == 0
    assert "synthetic-secret" not in output.err


@pytest.mark.parametrize(
    "invalid_environment",
    [
        {},
        {
            **_ENVIRONMENT,
            "CALLMETRIC_POSTGRES_CONNECT_TIMEOUT_SECONDS": "0",
        },
        {**_ENVIRONMENT, "CALLMETRIC_POSTGRES_SSL_MODE": "disable"},
    ],
    ids=["missing", "timeout", "ssl"],
)
def test_missing_or_invalid_environment_returns_two(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    invalid_environment: dict[str, str],
) -> None:
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    for name, value in invalid_environment.items():
        monkeypatch.setenv(name, value)
    path = _write(tmp_path / "provider.json", _PROVIDER)
    calls = 0

    def provision(**_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr(cli, "provision_profile_bound_postgres_rag", provision)

    assert cli.main(["--provider-settings", str(path)]) == 2

    output = capsys.readouterr()
    assert output.err == ("PostgreSQL RAG provisioning configuration is invalid.\n")
    assert calls == 0


def test_missing_file_returns_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_environment(monkeypatch)

    assert cli.main(["--provider-settings", str(tmp_path / "missing.json")]) == 2

    assert capsys.readouterr().err == (
        "PostgreSQL RAG provisioning configuration is invalid.\n"
    )


def test_operational_failure_is_fixed_safe_and_returns_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_environment(monkeypatch)
    path = _write(tmp_path / "provider.json", _PROVIDER)
    secret_exception = RuntimeError(
        f"synthetic provider failure {_DSN} synthetic-secret"
    )

    def provision(**_kwargs: object) -> object:
        raise secret_exception

    monkeypatch.setattr(cli, "provision_profile_bound_postgres_rag", provision)

    result = cli.main(["--provider-settings", str(path)])

    output = capsys.readouterr()
    assert result == 1
    assert output.out == ""
    assert output.err == "PostgreSQL RAG profile provisioning failed.\n"
    assert _DSN not in output.err
    assert "synthetic-secret" not in output.err


def test_keyboard_interrupt_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_environment(monkeypatch)
    path = _write(tmp_path / "provider.json", _PROVIDER)

    def provision(**_kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "provision_profile_bound_postgres_rag", provision)

    with pytest.raises(KeyboardInterrupt):
        cli.main(["--provider-settings", str(path)])
