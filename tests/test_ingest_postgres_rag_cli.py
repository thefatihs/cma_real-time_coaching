"""Deterministic tests for the PostgreSQL RAG chunk-ingestion CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.ingest_postgres_rag as cli
from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
)
from app.ingestion.models import DocumentIngestionRequest

_DSN = "postgresql://synthetic_user:synthetic_password@db.invalid/synthetic"
_ENVIRONMENT = {
    "CALLMETRIC_POSTGRES_DSN": _DSN,
    "CALLMETRIC_POSTGRES_CONNECT_TIMEOUT_SECONDS": "5",
    "CALLMETRIC_POSTGRES_SSL_MODE": "verify-full",
    "CALLMETRIC_POSTGRES_APPLICATION_NAME": "callmetric-ingestion",
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
_INGESTION = {
    "tenant_id": "tenant-synthetic",
    "knowledge_base_id": "kb-synthetic",
    "chunks": [
        {
            "document_id": "document-synthetic",
            "chunk_id": "chunk-synthetic",
            "text": "Synthetic support guidance.",
            "metadata": [["category", "synthetic"], ["order", "first"]],
        }
    ],
}


def _set_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _write(tmp_path / "provider.json", _PROVIDER),
        _write(tmp_path / "ingestion.json", _INGESTION),
    )


def _argv(provider: Path, ingestion: Path) -> list[str]:
    return [
        "--provider-settings",
        str(provider),
        "--ingestion-request",
        str(ingestion),
    ]


def test_success_is_fixed_safe_and_invokes_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_environment(monkeypatch)
    provider, ingestion = _paths(tmp_path)
    calls: list[dict[str, object]] = []

    def ingest(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(cli, "ingest_profile_bound_postgres_rag", ingest)
    arguments = _argv(provider, ingestion)
    original = list(arguments)

    assert cli.main(arguments) == 0

    output = capsys.readouterr()
    assert output.out == "PostgreSQL RAG chunk ingestion succeeded.\n"
    assert output.err == ""
    assert len(calls) == 1
    assert isinstance(
        calls[0]["knowledge_base_settings"],
        KnowledgeBaseRAGProviderSettings,
    )
    assert isinstance(calls[0]["request"], DocumentIngestionRequest)
    assert isinstance(calls[0]["postgres_settings"], PostgreSQLVectorStoreSettings)
    assert calls[0]["psycopg_connect"] is cli.psycopg.connect
    request = calls[0]["request"]
    assert isinstance(request, DocumentIngestionRequest)
    assert request.chunks[0].metadata == (
        ("category", "synthetic"),
        ("order", "first"),
    )
    assert arguments == original


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--provider-settings", "synthetic.json"],
        ["--unknown", "synthetic"],
    ],
)
def test_argument_failures_are_fixed_and_identifier_free(
    arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(arguments) == 2

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "PostgreSQL RAG ingestion configuration is invalid.\n"
    assert "synthetic" not in output.err


@pytest.mark.parametrize(
    ("target", "content"),
    [
        ("provider", ""),
        ("provider", "{"),
        ("provider", "[]"),
        ("provider", '{"tenant_id":"first","tenant_id":"second"}'),
        ("provider", json.dumps({**_PROVIDER, "unexpected": "synthetic"})),
        ("provider", json.dumps({**_PROVIDER, "dsn": "synthetic-secret"})),
        (
            "provider",
            json.dumps(
                {key: value for key, value in _PROVIDER.items() if key != "model_id"}
            ),
        ),
        ("ingestion", ""),
        ("ingestion", "{"),
        ("ingestion", "[]"),
        ("ingestion", '{"tenant_id":"first","tenant_id":"second"}'),
        ("ingestion", json.dumps({**_INGESTION, "password": "synthetic-secret"})),
        (
            "ingestion",
            json.dumps(
                {key: value for key, value in _INGESTION.items() if key != "chunks"}
            ),
        ),
        (
            "ingestion",
            json.dumps(
                {
                    **_INGESTION,
                    "chunks": [{**_INGESTION["chunks"][0], "secret": "value"}],
                }
            ),
        ),
    ],
    ids=[
        "provider-empty",
        "provider-malformed",
        "provider-array",
        "provider-duplicate",
        "provider-unknown",
        "provider-secret",
        "provider-missing",
        "ingestion-empty",
        "ingestion-malformed",
        "ingestion-array",
        "ingestion-duplicate",
        "ingestion-secret",
        "ingestion-missing",
        "chunk-secret",
    ],
)
def test_invalid_json_is_configuration_failure_without_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target: str,
    content: str,
) -> None:
    _set_environment(monkeypatch)
    provider, ingestion = _paths(tmp_path)
    (provider if target == "provider" else ingestion).write_text(
        content,
        encoding="utf-8",
    )
    calls = 0

    def ingest(**_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr(cli, "ingest_profile_bound_postgres_rag", ingest)

    assert cli.main(_argv(provider, ingestion)) == 2

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "PostgreSQL RAG ingestion configuration is invalid.\n"
    assert calls == 0
    assert _DSN not in output.err
    assert "synthetic-secret" not in output.err
    assert str(provider) not in output.err
    assert str(ingestion) not in output.err


@pytest.mark.parametrize(
    "change",
    [
        {"tenant_id": "tenant-other"},
        {"knowledge_base_id": "kb-other"},
    ],
)
def test_mismatched_scope_returns_two_before_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    change: dict[str, str],
) -> None:
    _set_environment(monkeypatch)
    provider = _write(tmp_path / "provider.json", _PROVIDER)
    ingestion = _write(tmp_path / "ingestion.json", {**_INGESTION, **change})
    calls = 0

    def ingest(**_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr(cli, "ingest_profile_bound_postgres_rag", ingest)

    assert cli.main(_argv(provider, ingestion)) == 2
    assert calls == 0
    assert capsys.readouterr().err == (
        "PostgreSQL RAG ingestion configuration is invalid.\n"
    )


def test_invalid_environment_returns_two_without_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    provider, ingestion = _paths(tmp_path)
    calls = 0

    def ingest(**_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr(cli, "ingest_profile_bound_postgres_rag", ingest)

    assert cli.main(_argv(provider, ingestion)) == 2
    assert calls == 0
    assert capsys.readouterr().err == (
        "PostgreSQL RAG ingestion configuration is invalid.\n"
    )


def test_operational_failure_is_fixed_secret_safe_and_returns_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_environment(monkeypatch)
    provider, ingestion = _paths(tmp_path)

    def ingest(**_kwargs: object) -> object:
        raise RuntimeError(
            f"{_DSN} tenant-synthetic document-synthetic "
            "Synthetic support guidance. local/synthetic-model"
        )

    monkeypatch.setattr(cli, "ingest_profile_bound_postgres_rag", ingest)

    assert cli.main(_argv(provider, ingestion)) == 1

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "PostgreSQL RAG chunk ingestion failed.\n"
    for sensitive in (
        _DSN,
        "tenant-synthetic",
        "kb-synthetic",
        "document-synthetic",
        "chunk-synthetic",
        "Synthetic support guidance.",
        "local/synthetic-model",
        str(provider),
        str(ingestion),
    ):
        assert sensitive not in output.err


def test_keyboard_interrupt_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_environment(monkeypatch)
    provider, ingestion = _paths(tmp_path)

    def ingest(**_kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "ingest_profile_bound_postgres_rag", ingest)

    with pytest.raises(KeyboardInterrupt):
        cli.main(_argv(provider, ingestion))
