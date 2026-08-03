import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from psycopg import OperationalError

from app.events.models import CoachingSuggestionSource
from app.orchestration.models import (
    OrchestrationCitationReference,
    OrchestrationResult,
)
from app.vector_store.models import VectorBatchWriteResult, VectorRecordIdentity
from scripts import run_windows_rag_vllm_e2e as subject


class Cursor:
    def __init__(self, calls: list[tuple[str, tuple[str, str]]]) -> None:
        self.calls = calls

    def __enter__(self) -> "Cursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, parameters: tuple[str, str]) -> None:
        self.calls.append((query, parameters))


class Connection:
    def __init__(self, calls: list[tuple[str, tuple[str, str]]]) -> None:
        self.calls = calls
        self.commits = 0
        self.rollbacks = 0
        self.closes = 0

    def cursor(self) -> Cursor:
        return Cursor(self.calls)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closes += 1


def environment(tmp_path: Path) -> dict[str, str]:
    token = tmp_path / "token file.txt"
    token.write_text("synthetic-secret-token\n", encoding="utf-8")
    ca = tmp_path / "local ca.crt"
    ca.write_text("synthetic-test-ca", encoding="utf-8")
    return {
        "CALLMETRIC_POSTGRES_DSN": "postgresql://synthetic-private-dsn",
        "CALLMETRIC_POSTGRES_CONNECT_TIMEOUT_SECONDS": "5",
        "CALLMETRIC_POSTGRES_SSL_MODE": "verify-full",
        "CALLMETRIC_POSTGRES_APPLICATION_NAME": "callmetric_windows_rag_e2e",
        "CALLMETRIC_VLLM_BASE_URL": "https://localhost:9443/v1",
        "CALLMETRIC_VLLM_MODEL_ID": "synthetic-vllm-model",
        "CALLMETRIC_VLLM_API_TOKEN_FILE": str(token),
        "CALLMETRIC_VLLM_CA_CERTIFICATE_PATH": str(ca),
        "CALLMETRIC_VLLM_CONNECT_TIMEOUT_SECONDS": "5",
        "CALLMETRIC_VLLM_READ_TIMEOUT_SECONDS": "30",
        "CALLMETRIC_VLLM_MAX_OUTPUT_TOKENS": "256",
        "CALLMETRIC_VLLM_TEMPERATURE": "0",
        "CALLMETRIC_VLLM_VERIFY_TLS": "true",
    }


def valid_result(**changes: object) -> OrchestrationResult:
    values: dict[str, object] = {
        "tenant_id": subject.TENANT_ID,
        "call_id": subject.CALL_ID,
        "transcript_revision": subject.TRANSCRIPT_REVISION,
        "generated_text": json.dumps(
            {
                "decision": "suggestion",
                "tenant_id": subject.TENANT_ID,
                "call_id": subject.CALL_ID,
                "revision": subject.TRANSCRIPT_REVISION,
                "action": "RAG_ACTION",
                "title": "Sentetik başlık",
                "suggestion": "Sentetik ürün için iade koşulunu açıklayın.",
                "priority": "HIGH",
                "citations": [
                    {
                        "document_id": subject.DOCUMENT_ID,
                        "chunk_id": subject.CHUNK_ID,
                    }
                ],
                "source": "llm",
            },
            ensure_ascii=False,
        ),
        "citations": (
            OrchestrationCitationReference(
                document_id=subject.DOCUMENT_ID,
                chunk_id=subject.CHUNK_ID,
            ),
        ),
    }
    values.update(changes)
    return OrchestrationResult.model_validate(values)


def valid_ingestion_result(
    *, document_id: str | None = None, chunk_id: str | None = None
) -> VectorBatchWriteResult:
    return VectorBatchWriteResult(
        tenant_id=subject.TENANT_ID,
        knowledge_base_id=subject.KNOWLEDGE_BASE_ID,
        inserted_identities=(
            VectorRecordIdentity(
                document_id=document_id or subject.DOCUMENT_ID,
                chunk_id=chunk_id or subject.CHUNK_ID,
            ),
        ),
        unchanged_identities=(),
    )


@pytest.fixture
def mocked_flow(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def cleanup(*_args: object, **_kwargs: object) -> None:
        calls.append("cleanup")

    def provision(**_kwargs: object) -> object:
        calls.append("provision")
        return SimpleNamespace(
            tenant_id=subject.TENANT_ID,
            knowledge_base_id=subject.KNOWLEDGE_BASE_ID,
            model_id=subject.MODEL_ID,
            vector_dimension=subject.VECTOR_DIMENSION,
        )

    def ingest(**_kwargs: object) -> VectorBatchWriteResult:
        calls.append("ingest")
        return valid_ingestion_result()

    def orchestrate(**_kwargs: object) -> OrchestrationResult:
        calls.append("orchestrate")
        return valid_result()

    monkeypatch.setattr(subject, "_cleanup_synthetic_scope", cleanup)
    monkeypatch.setattr(subject, "provision_profile_bound_postgres_rag", provision)
    monkeypatch.setattr(subject, "ingest_profile_bound_postgres_rag", ingest)
    monkeypatch.setattr(subject, "orchestrate_profile_bound_postgres_rag", orchestrate)
    return calls


def test_exact_synthetic_artifact_identity_and_existing_schemas() -> None:
    artifacts = subject._load_artifacts()

    assert artifacts.provider.tenant_id == "tenant_alpha"
    assert artifacts.provider.knowledge_base_id == "kb_smoke"
    assert artifacts.provider.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert artifacts.provider.vector_dimension == 384
    assert artifacts.provider.local_files_only is True
    assert artifacts.policy.rag_llm_enabled_labels == ("urun_bilgisi",)
    assert artifacts.policy.label_id == "urun_bilgisi"
    assert artifacts.document.chunks[0].document_id == subject.DOCUMENT_ID
    assert artifacts.document.chunks[0].chunk_id == subject.CHUNK_ID
    assert dict(artifacts.document.chunks[0].metadata)["data_class"] == "synthetic"


def test_preflight_validates_files_and_performs_no_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("preflight attempted a write or runtime operation")

    monkeypatch.setattr(subject, "_cleanup_synthetic_scope", forbidden)
    monkeypatch.setattr(subject, "provision_profile_bound_postgres_rag", forbidden)
    monkeypatch.setattr(subject, "ingest_profile_bound_postgres_rag", forbidden)
    monkeypatch.setattr(subject, "orchestrate_profile_bound_postgres_rag", forbidden)

    assert subject.run(preflight_only=True, environment=environment(tmp_path)) == (
        "PREFLIGHT_OK"
    )


def test_settings_require_verify_full_local_https_ca_and_token_file(
    tmp_path: Path,
) -> None:
    settings = subject._load_settings(environment(tmp_path))

    assert settings.postgres.ssl_mode == "verify-full"
    assert settings.vllm.base_url == "https://localhost:9443/v1"
    assert settings.vllm.verify_tls is True
    assert settings.vllm.api_token is not None
    assert settings.vllm.ca_certificate_path is not None
    assert settings.vllm.connect_timeout_seconds <= 60
    assert settings.vllm.read_timeout_seconds <= 600
    assert subject.TOTAL_E2E_TIMEOUT_SECONDS == 300


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("CALLMETRIC_POSTGRES_SSL_MODE", "require"),
        ("CALLMETRIC_VLLM_BASE_URL", "https://example.invalid/v1"),
        ("CALLMETRIC_VLLM_VERIFY_TLS", "false"),
        ("CALLMETRIC_VLLM_READ_TIMEOUT_SECONDS", "300"),
    ],
)
def test_insecure_or_nonlocal_settings_are_rejected(
    tmp_path: Path,
    key: str,
    value: str,
) -> None:
    values = environment(tmp_path)
    values[key] = value

    with pytest.raises(subject.WindowsRAGVLLME2EError, match="E_CONFIGURATION"):
        subject._load_settings(values)


def test_token_must_come_from_a_regular_absolute_file(tmp_path: Path) -> None:
    values = environment(tmp_path)
    values[subject.TOKEN_FILE_ENV] = "relative-token.txt"

    with pytest.raises(subject.WindowsRAGVLLME2EError, match="E_CONFIGURATION"):
        subject._load_settings(values)


def test_windows_compatible_paths_need_no_posix_permission_assumption(
    tmp_path: Path,
) -> None:
    values = environment(tmp_path)

    settings = subject._load_settings(values)

    assert settings.vllm.api_token is not None


def test_real_deployment_operations_are_wired_in_order(
    tmp_path: Path,
    mocked_flow: list[str],
) -> None:
    assert subject.run(preflight_only=False, environment=environment(tmp_path)) == (
        "E2E_OK"
    )
    assert mocked_flow == ["cleanup", "provision", "ingest", "orchestrate", "cleanup"]


def test_orchestration_receives_existing_settings_and_bounded_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocked_flow: list[str],
) -> None:
    captured: dict[str, object] = {}

    def orchestrate(**kwargs: object) -> OrchestrationResult:
        captured.update(kwargs)
        return valid_result()

    monkeypatch.setattr(subject, "orchestrate_profile_bound_postgres_rag", orchestrate)

    subject.run(preflight_only=False, environment=environment(tmp_path))

    request = captured["request"]
    limits = captured["limits"]
    assert request.tenant_id == subject.TENANT_ID  # type: ignore[attr-defined]
    assert request.knowledge_base_id == subject.KNOWLEDGE_BASE_ID  # type: ignore[attr-defined]
    assert request.top_k == 1  # type: ignore[attr-defined]
    assert limits.max_prompt_characters == 8_192  # type: ignore[attr-defined]


def test_exact_citation_scope_is_admitted_with_llm_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[CoachingSuggestionSource] = []
    original = subject.DeterministicLLMCoachingSuggestionFactory.create

    def create(self: Any, **kwargs: Any) -> Any:
        suggestion = original(self, **kwargs)
        assert suggestion is not None
        observed.append(suggestion.source)
        return suggestion

    monkeypatch.setattr(
        subject.DeterministicLLMCoachingSuggestionFactory, "create", create
    )
    artifacts = subject._load_artifacts()

    subject._admit_result(
        policy=artifacts.policy,
        event=subject._transcript_event(),
        result=valid_result(),
    )

    assert observed == [CoachingSuggestionSource.LLM]


@pytest.mark.parametrize(
    "result",
    [
        valid_result(tenant_id="tenant_other"),
        valid_result(call_id="call_other"),
        valid_result(
            citations=(
                OrchestrationCitationReference(
                    document_id="wrong_document", chunk_id=subject.CHUNK_ID
                ),
            )
        ),
        valid_result(generated_text="not-json"),
    ],
)
def test_invalid_json_scope_or_citation_is_rejected(
    result: OrchestrationResult,
) -> None:
    artifacts = subject._load_artifacts()

    with pytest.raises(subject.WindowsRAGVLLME2EError, match="E_ADMISSION"):
        subject._admit_result(
            policy=artifacts.policy,
            event=subject._transcript_event(),
            result=result,
        )


def test_retrieval_unavailable_has_deterministic_fallback_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocked_flow: list[str],
) -> None:
    monkeypatch.setattr(
        subject, "orchestrate_profile_bound_postgres_rag", lambda **_: None
    )

    with pytest.raises(subject.WindowsRAGVLLME2EError, match="E_RETRIEVAL_UNAVAILABLE"):
        subject.run(preflight_only=False, environment=environment(tmp_path))

    assert mocked_flow[-1] == "cleanup"


@pytest.mark.parametrize(
    ("stage", "expected_phase"),
    [
        ("cleanup", "E_INITIAL_CLEANUP"),
        ("provision", "E_PROVISIONING"),
        ("ingest", "E_INGESTION"),
        ("orchestrate", "E_RETRIEVAL_UNAVAILABLE"),
    ],
)
def test_psycopg_failures_have_distinct_fixed_phases_and_final_cleanup(
    stage: str,
    expected_phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocked_flow: list[str],
) -> None:
    if stage == "cleanup":
        cleanup_calls = 0

        def cleanup(*_args: object, **_kwargs: object) -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1
            mocked_flow.append("cleanup")
            if cleanup_calls == 1:
                raise OperationalError("synthetic private database detail")

        monkeypatch.setattr(subject, "_cleanup_synthetic_scope", cleanup)
    else:
        collaborator = {
            "provision": "provision_profile_bound_postgres_rag",
            "ingest": "ingest_profile_bound_postgres_rag",
            "orchestrate": "orchestrate_profile_bound_postgres_rag",
        }[stage]

        def unavailable(**_kwargs: object) -> None:
            raise OperationalError("synthetic private database detail")

        monkeypatch.setattr(subject, collaborator, unavailable)

    with pytest.raises(subject.WindowsRAGVLLME2EError, match=expected_phase):
        subject.run(preflight_only=False, environment=environment(tmp_path))

    assert mocked_flow[-1] == "cleanup"


def test_ingestion_must_report_exact_synthetic_identity_before_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocked_flow: list[str],
) -> None:
    monkeypatch.setattr(
        subject,
        "ingest_profile_bound_postgres_rag",
        lambda **_kwargs: valid_ingestion_result(document_id="unexpected_document"),
    )

    with pytest.raises(subject.WindowsRAGVLLME2EError, match="E_INGESTION"):
        subject.run(preflight_only=False, environment=environment(tmp_path))

    assert "orchestrate" not in mocked_flow
    assert mocked_flow[-1] == "cleanup"


def test_vllm_unavailable_has_deterministic_fallback_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocked_flow: list[str],
) -> None:
    request = httpx.Request("POST", "https://localhost:9443/v1/completions")

    def unavailable(**_kwargs: object) -> None:
        raise httpx.ConnectError("synthetic private detail", request=request)

    monkeypatch.setattr(subject, "orchestrate_profile_bound_postgres_rag", unavailable)

    with pytest.raises(subject.WindowsRAGVLLME2EError, match="E_VLLM_UNAVAILABLE"):
        subject.run(preflight_only=False, environment=environment(tmp_path))

    assert mocked_flow[-1] == "cleanup"


def test_cleanup_deletes_only_exact_synthetic_scope(tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[str, str]]] = []
    connections: list[Connection] = []

    def connect(**_kwargs: object) -> Connection:
        connection = Connection(calls)
        connections.append(connection)
        return connection

    settings = subject._load_settings(environment(tmp_path))
    subject._cleanup_synthetic_scope(settings.postgres, connect)

    assert len(calls) == 2
    assert all(parameters == ("tenant_alpha", "kb_smoke") for _, parameters in calls)
    assert "vector_records" in calls[0][0]
    assert "embedding_profiles" in calls[1][0]
    assert connections[0].commits == 1
    assert connections[0].closes == 1


def test_primary_failure_is_not_masked_by_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocked_flow: list[str],
) -> None:
    cleanup_calls = 0

    def cleanup(*_args: object, **_kwargs: object) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 2:
            raise RuntimeError("synthetic cleanup detail")

    monkeypatch.setattr(subject, "_cleanup_synthetic_scope", cleanup)
    monkeypatch.setattr(
        subject, "orchestrate_profile_bound_postgres_rag", lambda **_: None
    )

    with pytest.raises(subject.WindowsRAGVLLME2EError, match="E_RETRIEVAL_UNAVAILABLE"):
        subject.run(preflight_only=False, environment=environment(tmp_path))


def test_total_deadline_is_enforced(
    tmp_path: Path,
    mocked_flow: list[str],
) -> None:
    values: Iterator[float] = iter([0.0, 301.0])

    with pytest.raises(subject.WindowsRAGVLLME2EError, match="E_DEADLINE"):
        subject.run(
            preflight_only=False,
            environment=environment(tmp_path),
            clock=lambda: next(values),
        )

    assert mocked_flow == ["cleanup", "cleanup"]


def test_cli_output_never_leaks_secrets_or_private_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    values = environment(tmp_path)
    values["CALLMETRIC_POSTGRES_SSL_MODE"] = "disable"
    monkeypatch.setattr(subject.os, "environ", values)

    assert subject.main(["--preflight-only"]) == 1
    output = capsys.readouterr()
    combined = output.out + output.err
    assert combined.strip() == "E_CONFIGURATION"
    assert "synthetic-secret-token" not in combined
    assert "synthetic-private-dsn" not in combined
    assert str(tmp_path) not in combined
    assert subject.SYNTHETIC_TRANSCRIPT not in combined


@pytest.mark.parametrize(
    "phase",
    [
        "E_INITIAL_CLEANUP",
        "E_PROVISIONING",
        "E_INGESTION",
        "E_RETRIEVAL_UNAVAILABLE",
    ],
)
def test_database_phase_cli_output_is_fixed_and_secret_free(
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*, preflight_only: bool) -> str:
        del preflight_only
        try:
            raise OperationalError("postgresql://secret private/path document text")
        except OperationalError as error:
            raise subject.WindowsRAGVLLME2EError(phase) from error

    monkeypatch.setattr(subject, "run", fail)
    assert subject.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"{phase}\n"


def test_controller_owns_no_docker_aws_ssh_or_model_lifecycle() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")

    forbidden = (
        "subprocess",
        "docker compose",
        "ssh ",
        "boto3",
        "aws ",
        "local_files_only=False",
        "prune",
    )
    assert all(token not in source for token in forbidden)
