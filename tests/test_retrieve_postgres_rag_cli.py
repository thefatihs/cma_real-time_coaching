"""Deterministic tests for the secret-safe PostgreSQL RAG retrieval CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.retrieve_postgres_rag as cli
from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
)
from app.deployment import PostgreSQLRAGRetrievalRequest
from app.retrieval.models import RetrievalDocument, RetrievalResult

_DSN = "postgresql://synthetic_user:synthetic_password@db.invalid/synthetic"
_ENVIRONMENT = {
    "CALLMETRIC_POSTGRES_DSN": _DSN,
    "CALLMETRIC_POSTGRES_CONNECT_TIMEOUT_SECONDS": "7",
    "CALLMETRIC_POSTGRES_SSL_MODE": "verify-full",
    "CALLMETRIC_POSTGRES_APPLICATION_NAME": "callmetric-retrieval",
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
_REQUEST = {
    "tenant_id": "tenant-synthetic",
    "knowledge_base_id": "kb-synthetic",
    "query": "Synthetic Unicode question: çözüm",
    "top_k": 2,
    "minimum_score": 0.25,
}


def _result() -> RetrievalResult:
    return RetrievalResult(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        documents=(
            RetrievalDocument(
                tenant_id="tenant-synthetic",
                knowledge_base_id="kb-synthetic",
                document_id="document-b",
                chunk_id="chunk-2",
                text="Synthetic çözüm B",
                score=0.9,
            ),
            RetrievalDocument(
                tenant_id="tenant-synthetic",
                knowledge_base_id="kb-synthetic",
                document_id="document-a",
                chunk_id="chunk-1",
                text="Synthetic çözüm A",
                score=0.8,
            ),
        ),
    )


def _set_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        _write(tmp_path / "provider.json", _PROVIDER),
        _write(tmp_path / "request.json", _REQUEST),
        tmp_path / "result.json",
    )


def _argv(provider: Path, request: Path, output: Path) -> list[str]:
    return [
        "--provider-settings",
        str(provider),
        "--retrieval-request",
        str(request),
        "--output",
        str(output),
    ]


def test_success_writes_deterministic_result_only_to_explicit_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_environment(monkeypatch)
    provider, request_path, output_path = _paths(tmp_path)
    result = _result()
    calls: list[dict[str, object]] = []

    def retrieve(**kwargs: object) -> RetrievalResult:
        calls.append(kwargs)
        return result

    monkeypatch.setattr(cli, "retrieve_profile_bound_postgres_rag", retrieve)
    arguments = _argv(provider, request_path, output_path)
    original = list(arguments)

    assert cli.main(arguments) == 0

    console = capsys.readouterr()
    assert console.out == "PostgreSQL RAG retrieval succeeded.\n"
    assert console.err == ""
    assert len(calls) == 1
    assert isinstance(calls[0]["postgres_settings"], PostgreSQLVectorStoreSettings)
    assert isinstance(
        calls[0]["knowledge_base_settings"],
        KnowledgeBaseRAGProviderSettings,
    )
    assert isinstance(calls[0]["request"], PostgreSQLRAGRetrievalRequest)
    assert calls[0]["psycopg_connect"] is cli.psycopg.connect
    assert output_path.read_text(encoding="utf-8") == (
        '{"tenant_id":"tenant-synthetic","knowledge_base_id":"kb-synthetic",'
        '"documents":[{"tenant_id":"tenant-synthetic",'
        '"knowledge_base_id":"kb-synthetic","document_id":"document-b",'
        '"chunk_id":"chunk-2","text":"Synthetic çözüm B","score":0.9},'
        '{"tenant_id":"tenant-synthetic","knowledge_base_id":"kb-synthetic",'
        '"document_id":"document-a","chunk_id":"chunk-1",'
        '"text":"Synthetic çözüm A","score":0.8}]}\n'
    )
    assert arguments == original


def test_empty_result_is_written_safely(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_environment(monkeypatch)
    provider, request_path, output_path = _paths(tmp_path)

    def retrieve(**_kwargs: object) -> RetrievalResult:
        return RetrievalResult(
            tenant_id="tenant-synthetic",
            knowledge_base_id="kb-synthetic",
        )

    monkeypatch.setattr(cli, "retrieve_profile_bound_postgres_rag", retrieve)

    assert cli.main(_argv(provider, request_path, output_path)) == 0
    assert output_path.read_text(encoding="utf-8") == (
        '{"tenant_id":"tenant-synthetic","knowledge_base_id":"kb-synthetic",'
        '"documents":[]}\n'
    )


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--provider-settings", "synthetic.json"],
        ["--unknown", "synthetic"],
    ],
)
def test_argument_failure_is_fixed_and_identifier_free(
    arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(arguments) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "PostgreSQL RAG retrieval configuration is invalid.\n"
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
        ("request", ""),
        ("request", "{"),
        ("request", "[]"),
        ("request", '{"query":"first","query":"second"}'),
        ("request", json.dumps({**_REQUEST, "password": "synthetic-secret"})),
        ("request", json.dumps({**_REQUEST, "unexpected": "synthetic"})),
        (
            "request",
            json.dumps(
                {key: value for key, value in _REQUEST.items() if key != "query"}
            ),
        ),
    ],
)
def test_invalid_json_fails_before_environment_or_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target: str,
    content: str,
) -> None:
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    provider, request_path, output_path = _paths(tmp_path)
    (provider if target == "provider" else request_path).write_text(
        content,
        encoding="utf-8",
    )
    calls = 0

    def retrieve(**_kwargs: object) -> RetrievalResult:
        nonlocal calls
        calls += 1
        return _result()

    monkeypatch.setattr(cli, "retrieve_profile_bound_postgres_rag", retrieve)

    assert cli.main(_argv(provider, request_path, output_path)) == 2
    console = capsys.readouterr()
    assert console.out == ""
    assert console.err == "PostgreSQL RAG retrieval configuration is invalid.\n"
    assert calls == 0
    assert not output_path.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"tenant_id": "tenant-other"},
        {"knowledge_base_id": "kb-other"},
    ],
)
def test_scope_mismatch_fails_before_environment_or_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change: dict[str, str],
) -> None:
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    provider = _write(tmp_path / "provider.json", _PROVIDER)
    request_path = _write(tmp_path / "request.json", {**_REQUEST, **change})
    output_path = tmp_path / "result.json"
    calls = 0

    def retrieve(**_kwargs: object) -> RetrievalResult:
        nonlocal calls
        calls += 1
        return _result()

    monkeypatch.setattr(cli, "retrieve_profile_bound_postgres_rag", retrieve)

    assert cli.main(_argv(provider, request_path, output_path)) == 2
    assert calls == 0
    assert not output_path.exists()


def test_invalid_environment_fails_before_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    provider, request_path, output_path = _paths(tmp_path)
    calls = 0

    def retrieve(**_kwargs: object) -> RetrievalResult:
        nonlocal calls
        calls += 1
        return _result()

    monkeypatch.setattr(cli, "retrieve_profile_bound_postgres_rag", retrieve)

    assert cli.main(_argv(provider, request_path, output_path)) == 2
    assert calls == 0
    assert not output_path.exists()


@pytest.mark.parametrize("target_kind", ["file", "directory", "symlink"])
def test_existing_or_unsafe_output_is_rejected_before_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target_kind: str,
) -> None:
    _set_environment(monkeypatch)
    provider, request_path, output_path = _paths(tmp_path)
    if target_kind == "file":
        output_path.write_text("existing synthetic content", encoding="utf-8")
    elif target_kind == "directory":
        output_path.mkdir()
    else:
        target = tmp_path / "target.json"
        target.write_text("existing synthetic content", encoding="utf-8")
        try:
            output_path.symlink_to(target)
        except OSError:
            pytest.skip("symlink creation is unavailable")
    calls = 0

    def retrieve(**_kwargs: object) -> RetrievalResult:
        nonlocal calls
        calls += 1
        return _result()

    monkeypatch.setattr(cli, "retrieve_profile_bound_postgres_rag", retrieve)

    assert cli.main(_argv(provider, request_path, output_path)) == 2
    assert calls == 0


def test_operational_failure_is_fixed_secret_safe_and_creates_no_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_environment(monkeypatch)
    provider, request_path, output_path = _paths(tmp_path)

    def retrieve(**_kwargs: object) -> RetrievalResult:
        raise RuntimeError(
            f"{_DSN} tenant-synthetic document-synthetic "
            "Synthetic Unicode question: çözüm local/synthetic-model"
        )

    monkeypatch.setattr(cli, "retrieve_profile_bound_postgres_rag", retrieve)

    assert cli.main(_argv(provider, request_path, output_path)) == 1
    console = capsys.readouterr()
    assert console.out == ""
    assert console.err == "PostgreSQL RAG retrieval failed.\n"
    assert not output_path.exists()
    for sensitive in (
        _DSN,
        "tenant-synthetic",
        "kb-synthetic",
        "document-synthetic",
        "Synthetic Unicode question: çözüm",
        "local/synthetic-model",
        str(provider),
        str(request_path),
        str(output_path),
    ):
        assert sensitive not in console.err


def test_partial_output_is_removed_when_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "result.json"
    original_open = Path.open

    class FailingOutput:
        def __enter__(self) -> FailingOutput:
            self.output = original_open(
                output_path,
                "x",
                encoding="utf-8",
                newline="\n",
            )
            return self

        def write(self, content: str) -> int:
            self.output.write(content[:5])
            self.output.flush()
            raise OSError("synthetic write failure")

        def __exit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> None:
            del exc_type, exc_value, traceback
            self.output.close()

    def failing_open(
        _path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> FailingOutput:
        del mode, buffering, encoding, errors, newline
        return FailingOutput()

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(OSError, match="synthetic write failure"):
        cli._write_result(output_path, _result())

    assert not output_path.exists()


def test_keyboard_interrupt_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_environment(monkeypatch)
    provider, request_path, output_path = _paths(tmp_path)

    def retrieve(**_kwargs: object) -> RetrievalResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "retrieve_profile_bound_postgres_rag", retrieve)

    with pytest.raises(KeyboardInterrupt):
        cli.main(_argv(provider, request_path, output_path))
