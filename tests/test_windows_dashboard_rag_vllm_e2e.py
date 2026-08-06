from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
from typing import cast

import pytest

import scripts.run_windows_dashboard_rag_vllm_e2e as subject
from app.coaching.coordinator import CoachingSourcePresentation, StableCoachingOutcome

HEAD = "4" * 40
BASELINE = "3" * 40
BRANCH = "feat/dashboard-rag-document-upload"


def environment(tmp_path: Path) -> dict[str, str]:
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    snapshot = tmp_path / subject.MINILM_MODEL.replace("/", "-")
    snapshot.mkdir()
    provider = tmp_path / "provider.json"
    provider.write_text(
        json.dumps(
            {
                "tenant_id": "tenant_alpha",
                "knowledge_base_id": "kb_smoke",
                "model_id": subject.MINILM_MODEL,
                "model_name_or_path": str(snapshot),
                "vector_dimension": 384,
                "normalize_embeddings": True,
                "device": "cpu",
                "local_files_only": True,
            }
        ),
        encoding="utf-8",
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "rag_llm_enabled_labels": ["urun_bilgisi"],
                "title": "Synthetic guidance",
                "action": "RAG_ACTION",
                "priority": "HIGH",
                "label_id": "urun_bilgisi",
                "expires_after_seconds": 60.0,
            }
        ),
        encoding="utf-8",
    )
    ca = tmp_path / "ca.crt"
    ca.write_text("synthetic-ca", encoding="utf-8")
    return {
        subject.BRANCH_ENV: BRANCH,
        subject.HEAD_ENV: HEAD,
        subject.BASELINE_ENV: BASELINE,
        subject.HANDOFF_ROOT_ENV: str(handoff),
        subject.PROVIDER_ENV: str(provider),
        subject.POLICY_ENV: str(policy),
        subject.TOKEN_ENV: "synthetic-private-token",
        subject.CA_ENV: str(ca),
        subject.TTL_ENV: "300",
        "CALLMETRIC_VLLM_BASE_URL": "https://localhost:9443/v1",
        "CALLMETRIC_VLLM_MODEL_ID": "synthetic-served-model",
        "CALLMETRIC_VLLM_CONNECT_TIMEOUT_SECONDS": "5",
        "CALLMETRIC_VLLM_READ_TIMEOUT_SECONDS": "30",
        "CALLMETRIC_VLLM_MAX_OUTPUT_TOKENS": "256",
        "CALLMETRIC_VLLM_TEMPERATURE": "0",
        "CALLMETRIC_VLLM_VERIFY_TLS": "true",
    }


def prepare_preflight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subject.sys, "platform", "win32")
    monkeypatch.setattr(subject.Path, "cwd", lambda: subject.REPOSITORY_ROOT)
    monkeypatch.setattr(subject.shutil, "which", lambda name: f"C:/{name}.exe")
    monkeypatch.setattr(
        subject,
        "validate_local_minilm_snapshot",
        lambda value: Path(value).resolve(),
    )

    def git_output(arguments: list[str]) -> str:
        return {
            ("branch", "--show-current"): BRANCH,
            ("rev-parse", "HEAD"): HEAD,
            ("rev-parse", f"origin/{BRANCH}"): HEAD,
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("merge-base", "--is-ancestor", BASELINE, HEAD): "",
        }[tuple(arguments)]

    monkeypatch.setattr(subject, "_git_output", git_output)


@dataclass
class FakeOperations:
    fail_phase: str | None = None
    cleanup_failure: bool = False
    events: list[str] = field(default_factory=list)

    def run_phase(self, phase: str) -> None:
        self.events.append(phase)
        if phase == self.fail_phase:
            raise RuntimeError("sensitive internal detail")

    def cleanup(self) -> None:
        self.events.append("E_CLEANUP")
        if self.cleanup_failure:
            raise RuntimeError("sensitive cleanup detail")


def test_preflight_validates_without_runtime_activity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    called = False

    def factory(_config: subject.ControllerConfig) -> FakeOperations:
        nonlocal called
        called = True
        return FakeOperations()

    assert (
        subject.run(
            preflight_only=True,
            environment=environment(tmp_path),
            operations_factory=factory,
        )
        == subject.PREFLIGHT_OK
    )
    assert called is False


def test_full_lifecycle_uses_exact_phase_order_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    operations = FakeOperations()

    assert (
        subject.run(
            preflight_only=False,
            environment=environment(tmp_path),
            operations_factory=lambda _config: operations,
        )
        == subject.E2E_OK
    )
    assert operations.events == [*subject.PHASES[1:-1], "E_CLEANUP"]


@pytest.mark.parametrize("phase", subject.PHASES[1:-1])
def test_every_phase_failure_cleans_once_and_stays_fixed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase: str
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    operations = FakeOperations(fail_phase=phase)
    with pytest.raises(subject.DashboardRAGVLLME2EError) as caught:
        subject.run(
            preflight_only=False,
            environment=environment(tmp_path),
            operations_factory=lambda _config: operations,
        )
    assert caught.value.phase == phase
    assert str(caught.value) == phase
    assert operations.events[-1] == "E_CLEANUP"


def test_primary_failure_is_not_masked_by_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    operations = FakeOperations(fail_phase="E_ORCHESTRATION", cleanup_failure=True)
    with pytest.raises(subject.DashboardRAGVLLME2EError) as caught:
        subject.run(
            preflight_only=False,
            environment=environment(tmp_path),
            operations_factory=lambda _config: operations,
        )
    assert caught.value.phase == "E_ORCHESTRATION"


def test_citation_primary_failure_is_not_masked_by_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    operations = FakeOperations(
        fail_phase="E_CITATION_PROJECTION", cleanup_failure=True
    )
    with pytest.raises(subject.DashboardRAGVLLME2EError) as caught:
        subject.run(
            preflight_only=False,
            environment=environment(tmp_path),
            operations_factory=lambda _config: operations,
        )
    assert caught.value.phase == "E_CITATION_PROJECTION"


@pytest.mark.parametrize("count", range(2, 6))
def test_citation_projection_accepts_two_through_five_safe_sources(
    count: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    sources = tuple(
        CoachingSourcePresentation(
            "synthetic-guide.txt" if index == 0 else "synthetic-other.txt", "TXT"
        )
        for index in range(count)
    )
    displayed = (object(),)
    lifecycle._outcome = cast(
        StableCoachingOutcome,
        SimpleNamespace(
            result=SimpleNamespace(displayed_suggestions=displayed), sources=sources
        ),
    )
    monkeypatch.setattr(
        "live_dashboard.view_models.suggestion_card",
        lambda _event, *, sources: SimpleNamespace(sources=sources, evidence_ids=()),
    )

    lifecycle._admission()
    lifecycle._citation_projection()


@pytest.mark.parametrize("count", [0, 6])
def test_citation_projection_rejects_zero_or_more_than_five_sources(
    count: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    lifecycle._outcome = cast(
        StableCoachingOutcome,
        SimpleNamespace(
            result=SimpleNamespace(displayed_suggestions=(object(),)),
            sources=tuple(
                CoachingSourcePresentation("synthetic-guide.txt", "TXT")
                for _ in range(count)
            ),
        ),
    )
    with pytest.raises(RuntimeError):
        lifecycle._citation_projection()


@pytest.mark.parametrize("count", [0, 2])
def test_admission_requires_exactly_one_displayed_suggestion(
    count: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    lifecycle._outcome = cast(
        StableCoachingOutcome,
        SimpleNamespace(
            result=SimpleNamespace(
                displayed_suggestions=tuple(object() for _ in range(count))
            ),
            sources=(),
        ),
    )
    with pytest.raises(RuntimeError):
        lifecycle._admission()


def test_citation_projection_rejects_internal_identity_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    source = SimpleNamespace(
        original_filename="synthetic-guide.txt", media_label="TXT", document_id="hidden"
    )
    lifecycle._outcome = cast(
        StableCoachingOutcome,
        SimpleNamespace(
            result=SimpleNamespace(displayed_suggestions=(object(),)), sources=(source,)
        ),
    )
    monkeypatch.setattr(
        "live_dashboard.view_models.suggestion_card",
        lambda _event, *, sources: SimpleNamespace(sources=sources, evidence_ids=()),
    )
    with pytest.raises(RuntimeError):
        lifecycle._citation_projection()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (subject.BRANCH_ENV, "../unsafe"),
        (subject.HEAD_ENV, "short"),
        ("CALLMETRIC_VLLM_BASE_URL", "http://localhost:9443/v1"),
        ("CALLMETRIC_VLLM_BASE_URL", "https://private.example/v1"),
        ("CALLMETRIC_VLLM_VERIFY_TLS", "false"),
        (subject.TTL_ENV, "True"),
        (subject.TTL_ENV, "299"),
    ],
)
def test_unsafe_configuration_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    key: str,
    value: str,
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    values[key] = value
    with pytest.raises(subject.DashboardRAGVLLME2EError, match="^E_PREFLIGHT$"):
        subject.preflight(values)


def test_missing_or_partial_environment_fails_without_value_disclosure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    secret = values.pop(subject.TOKEN_ENV)
    with pytest.raises(subject.DashboardRAGVLLME2EError) as caught:
        subject.preflight(values)
    assert str(caught.value) == "E_PREFLIGHT"
    assert secret not in str(caught.value)


def test_dirty_tree_and_ref_mismatch_fail_before_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subject,
        "_git_output",
        lambda arguments: "dirty" if arguments[0] == "status" else BRANCH,
    )
    with pytest.raises(subject.DashboardRAGVLLME2EError):
        subject.preflight(environment(tmp_path))


def test_main_prints_only_fixed_phase(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        subject,
        "run",
        lambda **kwargs: (_ for _ in ()).throw(
            subject.DashboardRAGVLLME2EError("E_ORCHESTRATION")
        ),
    )
    assert subject.main([]) == 1
    assert capsys.readouterr().out.strip() == "E_ORCHESTRATION"


class FakeServiceProcess:
    def __init__(self, *, poll_result: int | None = None, return_code: int = 0) -> None:
        self.signals: list[int] = []
        self.waits: list[float | None] = []
        self.poll_result = poll_result
        self.return_code = return_code

    def poll(self) -> int | None:
        return self.poll_result

    def send_signal(self, signal_number: int) -> None:
        self.signals.append(signal_number)

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        return self.return_code


def test_production_cleanup_requests_graceful_service_signal_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    config = subject.preflight(values)
    lifecycle = subject._ProductionLifecycle(config, values)
    process = FakeServiceProcess()
    lifecycle._service = cast(subprocess.Popen[bytes], process)
    lifecycle._postgres_project = "callmetric-pgvector-tls-123-abcdef123456"
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "docker")
    commands: list[list[str]] = []

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(subject.subprocess, "run", run)

    lifecycle.cleanup()

    assert process.signals == [signal.CTRL_BREAK_EVENT]
    assert process.waits == [150]
    assert len(commands) == 3
    assert all("--filter" in command for command in commands)
    assert all("prune" not in command and "rm" not in command for command in commands)


@pytest.mark.parametrize("resource", ["container", "network", "volume"])
def test_remaining_exact_project_resource_fails_cleanup(
    resource: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    lifecycle._service = cast(subprocess.Popen[bytes], FakeServiceProcess())
    lifecycle._postgres_project = "callmetric-pgvector-tls-123-abcdef123456"
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "docker")

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        output = "residue" if arguments[1] == resource else ""
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")

    monkeypatch.setattr(subject.subprocess, "run", run)
    with pytest.raises(RuntimeError):
        lifecycle.cleanup()


def test_remaining_handoff_fails_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    lifecycle._service = cast(subprocess.Popen[bytes], FakeServiceProcess())
    lifecycle._postgres_project = "callmetric-pgvector-tls-123-abcdef123456"
    lifecycle._handoff = tmp_path / "handoff-residue"
    lifecycle._handoff.mkdir()
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "docker")
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, stdout="", stderr=""
        ),
    )
    with pytest.raises(RuntimeError):
        lifecycle.cleanup()


def test_signal_failure_wait_timeout_and_abnormal_exit_fail_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)

    class FailingProcess(FakeServiceProcess):
        def send_signal(self, signal_number: int) -> None:
            del signal_number
            raise OSError

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            raise subprocess.TimeoutExpired([], 150)

    lifecycle._service = cast(subprocess.Popen[bytes], FailingProcess())
    monkeypatch.setattr(lifecycle, "_require_postgres_residue_absent", lambda: None)
    with pytest.raises(OSError):
        lifecycle.cleanup()

    lifecycle._service = cast(
        subprocess.Popen[bytes], FakeServiceProcess(poll_result=1, return_code=1)
    )
    with pytest.raises(RuntimeError):
        lifecycle.cleanup()
