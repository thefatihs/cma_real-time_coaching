"""Tests for the secret-safe PostgreSQL RAG orchestration CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.orchestrate_postgres_rag as cli
from app.orchestration.models import OrchestrationResult

_PROVIDER = {
    "tenant_id": "tenant-synthetic",
    "knowledge_base_id": "kb-synthetic",
    "model_id": "embedding-synthetic",
    "model_name_or_path": "local/synthetic",
    "vector_dimension": 3,
    "normalize_embeddings": True,
    "device": "cpu",
    "local_files_only": True,
}
_REQUEST = {
    "tenant_id": "tenant-synthetic",
    "call_id": "call-synthetic",
    "transcript_revision": 4,
    "knowledge_base_id": "kb-synthetic",
    "user_input": "Synthetic Unicode question",
    "top_k": 2,
    "minimum_score": 0.25,
}
_LIMITS = {
    "max_top_k": 5,
    "max_user_input_characters": 100,
    "max_prompt_characters": 500,
}
_ENV = {
    "CALLMETRIC_POSTGRES_DSN": "postgresql://synthetic:secret@db.invalid/synthetic",
    "CALLMETRIC_POSTGRES_CONNECT_TIMEOUT_SECONDS": "7",
    "CALLMETRIC_POSTGRES_SSL_MODE": "verify-full",
    "CALLMETRIC_POSTGRES_APPLICATION_NAME": "callmetric-orchestration",
    "CALLMETRIC_VLLM_BASE_URL": "https://vllm.invalid/v1",
    "CALLMETRIC_VLLM_MODEL_ID": "llm-synthetic",
    "CALLMETRIC_VLLM_CONNECT_TIMEOUT_SECONDS": "5",
    "CALLMETRIC_VLLM_READ_TIMEOUT_SECONDS": "20",
    "CALLMETRIC_VLLM_MAX_OUTPUT_TOKENS": "20",
    "CALLMETRIC_VLLM_TEMPERATURE": "0",
    "CALLMETRIC_VLLM_VERIFY_TLS": "true",
}


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _args(tmp_path: Path) -> tuple[list[str], Path]:
    output = tmp_path / "result.json"
    return (
        [
            "--provider-settings",
            str(_write(tmp_path / "provider.json", _PROVIDER)),
            "--orchestration-request",
            str(_write(tmp_path / "request.json", _REQUEST)),
            "--operation-limits",
            str(_write(tmp_path / "limits.json", _LIMITS)),
            "--output",
            str(output),
        ],
        output,
    )


def _environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)


def _result() -> OrchestrationResult:
    return OrchestrationResult.model_validate(
        {
            "tenant_id": "tenant-synthetic",
            "call_id": "call-synthetic",
            "transcript_revision": 4,
            "generated_text": "Synthetic Unicode answer",
            "citations": ({"document_id": "document-b", "chunk_id": "chunk-2"},),
        }
    )


def test_success_writes_only_deterministic_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _environment(monkeypatch)
    arguments, output = _args(tmp_path)
    result = _result()
    calls: list[dict[str, object]] = []

    def orchestrate(**kwargs: object) -> OrchestrationResult:
        calls.append(kwargs)
        return result

    monkeypatch.setattr(cli, "orchestrate_profile_bound_postgres_rag", orchestrate)
    assert cli.main(arguments) == 0
    assert output.read_text(encoding="utf-8") == (
        '{"tenant_id":"tenant-synthetic","call_id":"call-synthetic",'
        '"transcript_revision":4,"generated_text":"Synthetic Unicode answer",'
        '"citations":[{"document_id":"document-b","chunk_id":"chunk-2"}]}\n'
    )
    console = capsys.readouterr()
    assert console.out == "PostgreSQL RAG orchestration succeeded.\n"
    assert console.err == ""
    assert len(calls) == 1


def test_empty_retrieval_writes_exact_null(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _environment(monkeypatch)
    arguments, output = _args(tmp_path)
    monkeypatch.setattr(
        cli, "orchestrate_profile_bound_postgres_rag", lambda **_kwargs: None
    )
    assert cli.main(arguments) == 0
    assert output.read_text(encoding="utf-8") == "null\n"


@pytest.mark.parametrize(
    ("target", "content"),
    [
        ("provider", '{"tenant_id":"a","tenant_id":"b"}'),
        ("provider", json.dumps({**_PROVIDER, "api_token": "secret"})),
        ("request", json.dumps({**_REQUEST, "unknown": "value"})),
        ("request", json.dumps({k: v for k, v in _REQUEST.items() if k != "call_id"})),
        ("limits", json.dumps({**_LIMITS, "max_top_k": True})),
    ],
)
def test_invalid_json_fails_before_environment_or_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target: str,
    content: str,
) -> None:
    arguments, output = _args(tmp_path)
    names = {"provider": 1, "request": 3, "limits": 5}
    Path(arguments[names[target]]).write_text(content, encoding="utf-8")
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        cli,
        "orchestrate_profile_bound_postgres_rag",
        lambda **_kwargs: pytest.fail("operation must not run"),
    )
    assert cli.main(arguments) == 2
    assert not output.exists()


@pytest.mark.parametrize(
    ("request_change", "limit_change"),
    [
        ({"tenant_id": "other"}, {}),
        ({"knowledge_base_id": "other"}, {}),
        ({"top_k": 6}, {}),
        ({"user_input": "x" * 101}, {}),
    ],
)
def test_local_scope_and_limit_failures_precede_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request_change: dict[str, object],
    limit_change: dict[str, object],
) -> None:
    arguments, output = _args(tmp_path)
    _write(Path(arguments[3]), {**_REQUEST, **request_change})
    _write(Path(arguments[5]), {**_LIMITS, **limit_change})
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    assert cli.main(arguments) == 2
    assert not output.exists()


def test_existing_output_is_not_overwritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _environment(monkeypatch)
    arguments, output = _args(tmp_path)
    output.write_text("preserve", encoding="utf-8")
    assert cli.main(arguments) == 2
    assert output.read_text(encoding="utf-8") == "preserve"


def test_operational_failure_is_fixed_and_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _environment(monkeypatch)
    arguments, output = _args(tmp_path)
    monkeypatch.setattr(
        cli,
        "orchestrate_profile_bound_postgres_rag",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("secret synthetic text")),
    )
    assert cli.main(arguments) == 1
    console = capsys.readouterr()
    assert console.out == ""
    assert console.err == "PostgreSQL RAG orchestration failed.\n"
    assert "secret" not in console.err
    assert not output.exists()


def test_partial_output_is_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "result.json"
    original = Path.open

    class Failing:
        def __enter__(self) -> Failing:
            self.file = original(output, "x", encoding="utf-8", newline="\n")
            return self

        def write(self, value: str) -> int:
            self.file.write(value[:2])
            raise OSError("synthetic")

        def __exit__(self, *args: object) -> None:
            self.file.close()

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: Failing())
    with pytest.raises(OSError):
        cli._write_result(output, _result())
    assert not output.exists()
