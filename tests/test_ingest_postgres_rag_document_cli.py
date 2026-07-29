"""Tests for explicit trusted UTF-8 PostgreSQL RAG document ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

import scripts.ingest_postgres_rag_document as cli
from app.ingestion import DocumentIngestionRequest

_DSN = "postgresql://synthetic_user:synthetic_password@db.invalid/synthetic"
_ENVIRONMENT = {
    "CALLMETRIC_POSTGRES_DSN": _DSN,
    "CALLMETRIC_POSTGRES_CONNECT_TIMEOUT_SECONDS": "5",
    "CALLMETRIC_POSTGRES_SSL_MODE": "verify-full",
    "CALLMETRIC_POSTGRES_APPLICATION_NAME": "callmetric-document",
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
_DOCUMENT_SETTINGS = {
    "document_id": "document-v1",
    "metadata": [["source", "synthetic"], ["version", "v1"]],
    "max_file_bytes": 1024,
    "max_document_characters": 512,
    "max_chunk_characters": 8,
}


def _set_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _files(
    tmp_path: Path,
    *,
    suffix: str = ".txt",
    content: bytes = b"Synthetic trusted document.",
    document_settings: object = _DOCUMENT_SETTINGS,
) -> tuple[Path, Path, Path]:
    provider = _write_json(tmp_path / "provider.json", _PROVIDER)
    settings = _write_json(tmp_path / "document-settings.json", document_settings)
    document = tmp_path / f"document{suffix}"
    document.write_bytes(content)
    return provider, settings, document


def _argv(provider: Path, settings: Path, document: Path) -> list[str]:
    return [
        "--provider-settings",
        str(provider),
        "--document-settings",
        str(settings),
        "--document",
        str(document),
    ]


@pytest.mark.parametrize("suffix", [".txt", ".md"])
def test_txt_and_markdown_success_delegate_exact_request_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    suffix: str,
) -> None:
    _set_environment(monkeypatch)
    provider, settings, document = _files(
        tmp_path,
        suffix=suffix,
        content="  Türkçe\ninternal  text\n".encode(),
    )
    calls: list[dict[str, object]] = []

    def ingest(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(cli, "ingest_profile_bound_postgres_rag", ingest)

    assert cli.main(_argv(provider, settings, document)) == 0

    output = capsys.readouterr()
    assert output.out == "PostgreSQL RAG document ingestion succeeded.\n"
    assert output.err == ""
    assert len(calls) == 1
    request = calls[0]["request"]
    assert isinstance(request, DocumentIngestionRequest)
    assert request.tenant_id == "tenant-synthetic"
    assert request.knowledge_base_id == "kb-synthetic"
    assert tuple(chunk.chunk_id for chunk in request.chunks) == (
        "chunk_000001",
        "chunk_000002",
        "chunk_000003",
    )
    assert "".join(chunk.text for chunk in request.chunks).replace(" ", "") == (
        "Türkçe\ninternaltext"
    )
    assert all(
        chunk.metadata == (("source", "synthetic"), ("version", "v1"))
        for chunk in request.chunks
    )
    assert calls[0]["psycopg_connect"] is cli.psycopg.connect
    assert not any(str(document) in repr(value) for value in calls[0].values())


def test_same_file_and_settings_produce_equal_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_environment(monkeypatch)
    provider, settings, document = _files(tmp_path)
    requests: list[DocumentIngestionRequest] = []

    def ingest(**kwargs: object) -> object:
        request = kwargs["request"]
        assert isinstance(request, DocumentIngestionRequest)
        requests.append(request)
        return object()

    monkeypatch.setattr(cli, "ingest_profile_bound_postgres_rag", ingest)
    arguments = _argv(provider, settings, document)

    assert cli.main(arguments) == 0
    assert cli.main(arguments) == 0
    assert requests[0] == requests[1]


@pytest.mark.parametrize(
    ("content", "settings_change"),
    [
        (b"", {}),
        (b" \n\t ", {}),
        (b"\xef\xbb\xbfSynthetic", {}),
        (b"Synthetic\x00text", {}),
        (b"\xff", {}),
        (b"12345", {"max_file_bytes": 4}),
        ("ğğğ".encode(), {"max_document_characters": 2}),
    ],
    ids=[
        "empty",
        "whitespace",
        "bom",
        "nul",
        "invalid-utf8",
        "byte-limit",
        "character-limit",
    ],
)
def test_invalid_document_content_fails_before_environment_or_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: bytes,
    settings_change: dict[str, object],
) -> None:
    provider, settings, document = _files(
        tmp_path,
        content=content,
        document_settings={**_DOCUMENT_SETTINGS, **settings_change},
    )
    monkeypatch.setattr(
        cli,
        "PostgreSQLVectorStoreSettings",
        lambda: pytest.fail("environment settings must remain deferred"),
    )
    monkeypatch.setattr(
        cli,
        "ingest_profile_bound_postgres_rag",
        lambda **kwargs: pytest.fail(f"unexpected operation: {tuple(kwargs)}"),
    )

    assert cli.main(_argv(provider, settings, document)) == 2
    assert capsys.readouterr().err == (
        "PostgreSQL RAG document configuration is invalid.\n"
    )


@pytest.mark.parametrize(
    "kind",
    ["missing", "directory", "traversal", "symlink", "unsupported"],
)
def test_invalid_path_kinds_are_rejected_before_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    provider, settings, valid = _files(tmp_path)
    if kind == "missing":
        document = tmp_path / "missing.txt"
    elif kind == "directory":
        document = tmp_path / "directory.txt"
        document.mkdir()
    elif kind == "traversal":
        intermediate = tmp_path / "nested"
        intermediate.mkdir()
        document = intermediate / ".." / valid.name
    elif kind == "symlink":
        document = valid
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: path == document or original_is_symlink(path),
        )
    else:
        document = tmp_path / "document.pdf"
        document.write_text("Synthetic document.", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "PostgreSQLVectorStoreSettings",
        lambda: pytest.fail("environment settings must remain deferred"),
    )

    assert cli.main(_argv(provider, settings, document)) == 2
    assert capsys.readouterr().err == (
        "PostgreSQL RAG document configuration is invalid.\n"
    )


def test_non_regular_opened_file_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider, settings, document = _files(tmp_path)
    monkeypatch.setattr(cli.stat, "S_ISREG", lambda _mode: False)
    monkeypatch.setattr(
        cli,
        "PostgreSQLVectorStoreSettings",
        lambda: pytest.fail("environment settings must remain deferred"),
    )

    assert cli.main(_argv(provider, settings, document)) == 2


@pytest.mark.parametrize(
    ("target", "content"),
    [
        ("provider", "[]"),
        ("provider", '{"tenant_id":"a","tenant_id":"b"}'),
        ("provider", json.dumps({**_PROVIDER, "password": "synthetic-secret"})),
        (
            "provider",
            json.dumps(
                {key: value for key, value in _PROVIDER.items() if key != "model_id"}
            ),
        ),
        ("settings", "[]"),
        ("settings", '{"document_id":"a","document_id":"b"}'),
        (
            "settings",
            json.dumps({**_DOCUMENT_SETTINGS, "dsn": "synthetic-secret"}),
        ),
        (
            "settings",
            json.dumps(
                {
                    key: value
                    for key, value in _DOCUMENT_SETTINGS.items()
                    if key != "metadata"
                }
            ),
        ),
        (
            "settings",
            json.dumps({**_DOCUMENT_SETTINGS, "metadata": [["key"]]}),
        ),
        (
            "settings",
            json.dumps(
                {
                    **_DOCUMENT_SETTINGS,
                    "metadata": [[" key ", "first"], ["key", "second"]],
                }
            ),
        ),
        (
            "settings",
            json.dumps({**_DOCUMENT_SETTINGS, "metadata": [[" ", "value"]]}),
        ),
        (
            "settings",
            json.dumps({**_DOCUMENT_SETTINGS, "max_file_bytes": True}),
        ),
    ],
    ids=[
        "provider-nonobject",
        "provider-duplicate",
        "provider-secret",
        "provider-missing",
        "settings-nonobject",
        "settings-duplicate",
        "settings-secret",
        "settings-missing",
        "metadata-malformed",
        "metadata-duplicate",
        "metadata-blank",
        "boolean-limit",
    ],
)
def test_invalid_json_and_metadata_fail_before_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target: str,
    content: str,
) -> None:
    provider, settings, document = _files(tmp_path)
    (provider if target == "provider" else settings).write_text(
        content,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "PostgreSQLVectorStoreSettings",
        lambda: pytest.fail("environment settings must remain deferred"),
    )

    assert cli.main(_argv(provider, settings, document)) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "PostgreSQL RAG document configuration is invalid.\n"
    assert "synthetic-secret" not in output.err


def test_missing_arguments_have_fixed_safe_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main([]) == 2
    assert capsys.readouterr().err == (
        "PostgreSQL RAG document configuration is invalid.\n"
    )


def test_exact_local_validation_environment_and_operation_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider, settings, document = _files(tmp_path)
    calls: list[str] = []
    original_provider_loader = cli._load_provider_settings
    original_settings_loader = cli._load_document_settings
    original_reader = cli._read_document
    original_builder = cli.build_fixed_character_document_ingestion_request

    def load_provider(path: Path) -> object:
        calls.append("provider")
        return original_provider_loader(path)

    def load_settings(path: Path) -> object:
        calls.append("document-settings")
        return original_settings_loader(path)

    def read_document(path: Path, *, max_file_bytes: int) -> str:
        calls.append("document")
        return original_reader(path, max_file_bytes=max_file_bytes)

    def build_request(**kwargs: object) -> object:
        calls.append("builder")
        return original_builder(**cast(Any, kwargs))

    def postgres_settings() -> object:
        calls.append("environment")
        return object()

    def ingest(**_kwargs: object) -> object:
        calls.append("operation")
        return object()

    monkeypatch.setattr(cli, "_load_provider_settings", load_provider)
    monkeypatch.setattr(cli, "_load_document_settings", load_settings)
    monkeypatch.setattr(cli, "_read_document", read_document)
    monkeypatch.setattr(
        cli,
        "build_fixed_character_document_ingestion_request",
        build_request,
    )
    monkeypatch.setattr(cli, "PostgreSQLVectorStoreSettings", postgres_settings)
    monkeypatch.setattr(cli, "ingest_profile_bound_postgres_rag", ingest)

    assert cli.main(_argv(provider, settings, document)) == 0
    assert calls == [
        "provider",
        "document-settings",
        "document",
        "builder",
        "environment",
        "operation",
    ]


def test_operational_failure_is_fixed_and_does_not_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_environment(monkeypatch)
    provider, settings, document = _files(tmp_path)

    def ingest(**_kwargs: object) -> object:
        raise RuntimeError(
            f"{_DSN} tenant-synthetic document-v1 "
            f"{document} Synthetic trusted document. local/synthetic-model"
        )

    monkeypatch.setattr(cli, "ingest_profile_bound_postgres_rag", ingest)

    assert cli.main(_argv(provider, settings, document)) == 1

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "PostgreSQL RAG document ingestion failed.\n"
    for sensitive in (
        _DSN,
        "tenant-synthetic",
        "kb-synthetic",
        "document-v1",
        str(document),
        "Synthetic trusted document.",
        "local/synthetic-model",
    ):
        assert sensitive not in output.err


def test_keyboard_interrupt_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_environment(monkeypatch)
    provider, settings, document = _files(tmp_path)

    def ingest(**_kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "ingest_profile_bound_postgres_rag", ingest)

    with pytest.raises(KeyboardInterrupt):
        cli.main(_argv(provider, settings, document))
