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
from app.coaching.coordinator import (
    CoachingCoordinatorResult,
    CoachingProcessingStatus,
    CoachingSourcePresentation,
    StableCoachingOutcome,
)
from app.coaching.coordinator import _cooldown_available

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
    expected = (
        subject.E_COMPLETION_UNCLASSIFIED if phase == "E_COMPLETION_PUMP" else phase
    )
    assert caught.value.phase == expected
    assert str(caught.value) == expected
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


@pytest.mark.parametrize(
    "phase",
    [
        subject.E_CLEANUP_PROCESS_ACTION,
        subject.E_CLEANUP_PROJECT_ACTION,
        subject.E_CLEANUP_HANDOFF_ACTION,
        subject.E_CLEANUP_PROCESS_VERIFY,
        subject.E_CLEANUP_PROJECT_VERIFY,
        subject.E_CLEANUP_HANDOFF_VERIFY,
        subject.E_CLEANUP_PROTECTED_VERIFY,
        subject.E_CLEANUP_UNVERIFIABLE,
    ],
)
def test_cleanup_diagnostic_subphases_are_fixed_and_secret_safe(phase: str) -> None:
    error = subject._CleanupPhaseError(phase)
    assert error.phase == phase
    assert str(error) == phase


def test_full_functional_success_with_recovered_cleanup_returns_e2e_ok(
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
    assert operations.events[-1] == "E_CLEANUP"


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


class FakeCompletionProcessor:
    def __init__(self, outcomes: tuple[StableCoachingOutcome, ...]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def drain_completed(
        self, *, current_seconds: float
    ) -> tuple[StableCoachingOutcome, ...]:
        assert current_seconds == 1.0
        self.calls += 1
        return self.outcomes


def _completion_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> subject._ProductionLifecycle:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    return subject._ProductionLifecycle(subject.preflight(values), values)


def test_completion_pump_accepts_one_authoritative_processed_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle = _completion_lifecycle(monkeypatch, tmp_path)
    outcome = StableCoachingOutcome(
        status=CoachingProcessingStatus.PROCESSED,
        transcript_revision=1,
        result=cast(
            CoachingCoordinatorResult,
            SimpleNamespace(displayed_suggestions=(object(),)),
        ),
    )
    processor = FakeCompletionProcessor((outcome,))
    lifecycle._processor = cast(subject.RAGCoachingProcessorDecorator, processor)

    lifecycle._completion_pump()

    assert lifecycle._outcome is outcome
    assert processor.calls == 1


@pytest.mark.parametrize(
    ("outcomes", "phase"),
    [
        (
            (
                StableCoachingOutcome(
                    status=CoachingProcessingStatus.FAILED,
                    transcript_revision=1,
                    error_type="rag_orchestration",
                    error_code="background_failure",
                ),
            ),
            subject.E_COMPLETION_BACKGROUND_FAILED,
        ),
        (
            (
                StableCoachingOutcome(
                    status=CoachingProcessingStatus.PARTIAL_SKIPPED,
                    transcript_revision=1,
                ),
            ),
            subject.E_COMPLETION_NOT_PROCESSED,
        ),
        (
            (
                StableCoachingOutcome(
                    status=CoachingProcessingStatus.PROCESSED,
                    transcript_revision=1,
                ),
            ),
            subject.E_COMPLETION_RESULT_MISSING,
        ),
    ],
)
def test_completion_pump_reports_fixed_public_outcome_status(
    outcomes: tuple[StableCoachingOutcome, ...],
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lifecycle = _completion_lifecycle(monkeypatch, tmp_path)
    lifecycle._processor = cast(
        subject.RAGCoachingProcessorDecorator, FakeCompletionProcessor(outcomes)
    )

    with pytest.raises(subject._CompletionPumpError, match=f"^{phase}$"):
        lifecycle._completion_pump()


def test_completion_pump_rejects_multiple_authoritative_outcomes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle = _completion_lifecycle(monkeypatch, tmp_path)
    outcome = StableCoachingOutcome(
        status=CoachingProcessingStatus.PROCESSED,
        transcript_revision=1,
        result=cast(
            CoachingCoordinatorResult,
            SimpleNamespace(displayed_suggestions=(object(),)),
        ),
    )
    lifecycle._processor = cast(
        subject.RAGCoachingProcessorDecorator,
        FakeCompletionProcessor((outcome, outcome)),
    )

    with pytest.raises(
        subject._CompletionPumpError, match=f"^{subject.E_COMPLETION_CARDINALITY}$"
    ):
        lifecycle._completion_pump()


def test_completion_deadline_uses_provider_timeouts_and_bounded_margin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle = _completion_lifecycle(monkeypatch, tmp_path)
    assert lifecycle._completion_timeout_seconds() == 300.0
    lifecycle._processor = cast(
        subject.RAGCoachingProcessorDecorator, FakeCompletionProcessor(())
    )
    monotonic_values = iter((10.0, 310.0))
    monkeypatch.setattr(subject.time, "monotonic", lambda: next(monotonic_values))

    with pytest.raises(
        subject._CompletionPumpError,
        match=f"^{subject.E_COMPLETION_NO_AUTHORITATIVE_OUTCOME}$",
    ):
        lifecycle._completion_pump()


def test_completion_deadline_accounts_for_every_maximum_http_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    values["CALLMETRIC_VLLM_CONNECT_TIMEOUT_SECONDS"] = "60"
    values["CALLMETRIC_VLLM_READ_TIMEOUT_SECONDS"] = "600"
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)

    assert lifecycle._completion_timeout_seconds() == 1_380.0


def test_completion_arriving_just_before_legacy_deadline_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle = _completion_lifecycle(monkeypatch, tmp_path)
    outcome = StableCoachingOutcome(
        status=CoachingProcessingStatus.PROCESSED,
        transcript_revision=1,
        result=cast(
            CoachingCoordinatorResult,
            SimpleNamespace(displayed_suggestions=(object(),)),
        ),
    )
    processor = FakeCompletionProcessor((outcome,))
    lifecycle._processor = cast(subject.RAGCoachingProcessorDecorator, processor)
    monotonic_values = iter((0.0, 299.999))
    monkeypatch.setattr(subject.time, "monotonic", lambda: next(monotonic_values))

    lifecycle._completion_pump()

    assert processor.calls == 1


def test_completion_at_exact_deadline_exhausts_without_an_extra_poll(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle = _completion_lifecycle(monkeypatch, tmp_path)
    processor = FakeCompletionProcessor(())
    lifecycle._processor = cast(subject.RAGCoachingProcessorDecorator, processor)
    monotonic_values = iter((0.0, 300.0))
    monkeypatch.setattr(subject.time, "monotonic", lambda: next(monotonic_values))

    with pytest.raises(
        subject._CompletionPumpError,
        match=f"^{subject.E_COMPLETION_NO_AUTHORITATIVE_OUTCOME}$",
    ):
        lifecycle._completion_pump()

    assert processor.calls == 0


def test_terminal_failure_stops_without_a_second_wait_or_poll(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle = _completion_lifecycle(monkeypatch, tmp_path)
    outcome = StableCoachingOutcome(
        status=CoachingProcessingStatus.FAILED,
        transcript_revision=1,
        error_type="rag_orchestration",
        error_code="background_failure",
    )
    processor = FakeCompletionProcessor((outcome,))
    lifecycle._processor = cast(subject.RAGCoachingProcessorDecorator, processor)
    monotonic_calls = 0

    def monotonic() -> float:
        nonlocal monotonic_calls
        monotonic_calls += 1
        return 0.0

    monkeypatch.setattr(subject.time, "monotonic", monotonic)
    monkeypatch.setattr(
        subject.time,
        "sleep",
        lambda _seconds: pytest.fail("terminal outcome must not sleep"),
    )

    with pytest.raises(
        subject._CompletionPumpError,
        match=f"^{subject.E_COMPLETION_BACKGROUND_FAILED}$",
    ):
        lifecycle._completion_pump()

    assert processor.calls == 1
    assert monotonic_calls == 2


def test_completion_processor_must_exist_before_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle = _completion_lifecycle(monkeypatch, tmp_path)
    with pytest.raises(
        subject._CompletionPumpError,
        match=f"^{subject.E_COMPLETION_PROCESSOR_MISSING}$",
    ):
        lifecycle._completion_pump()


def test_e2e_zero_cooldown_allows_same_label_after_positive_cooldown_suppresses() -> (
    None
):
    assert _cooldown_available(1.0, 1.0, 8.0) is False
    assert _cooldown_available(1.0, 1.0, 0.0) is True
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert 'update={"enable_llm": True, "cooldown_seconds": 0.0}' in source


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
        ("CALLMETRIC_VLLM_MAX_OUTPUT_TOKENS", "255"),
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


@pytest.mark.parametrize(
    "completion_phase",
    sorted(subject.COMPLETION_FAILURE_PHASES - {subject.E_COMPLETION_UNCLASSIFIED}),
)
def test_public_main_prints_exact_safe_completion_category(
    completion_phase: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    config = subject.preflight(values)
    secret_like_detail = "private-token-path-endpoint-dsn"

    class CompletionOperations(FakeOperations):
        def run_phase(self, phase: str) -> None:
            self.events.append(phase)
            if phase == "E_COMPLETION_PUMP":
                raise subject._CompletionPumpError(completion_phase)

    operations = CompletionOperations()
    monkeypatch.setattr(subject, "_preflight", lambda _environment=None: config)
    monkeypatch.setattr(
        subject,
        "_ProductionLifecycle",
        lambda _config, _environment: operations,
    )
    monkeypatch.setenv("SYNTHETIC_SECRET_LIKE_VALUE", secret_like_detail)

    assert subject.main([]) == 1

    captured = capsys.readouterr()
    assert captured.out.strip() == completion_phase
    assert captured.err == ""
    assert secret_like_detail not in captured.out


@pytest.mark.parametrize("typed_error", [False, True])
def test_public_main_maps_unexpected_completion_exception_to_fixed_fallback(
    typed_error: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    config = subject.preflight(values)
    secret_like_detail = "raw-private-provider-exception"

    class UnexpectedCompletionOperations(FakeOperations):
        def run_phase(self, phase: str) -> None:
            self.events.append(phase)
            if phase == "E_COMPLETION_PUMP":
                if typed_error:
                    raise subject._CompletionPumpError(secret_like_detail)
                raise RuntimeError(secret_like_detail)

    operations = UnexpectedCompletionOperations()
    monkeypatch.setattr(subject, "_preflight", lambda _environment=None: config)
    monkeypatch.setattr(
        subject,
        "_ProductionLifecycle",
        lambda _config, _environment: operations,
    )

    assert subject.main([]) == 1

    captured = capsys.readouterr()
    assert captured.out.strip() == subject.E_COMPLETION_UNCLASSIFIED
    assert captured.err == ""
    assert secret_like_detail not in captured.out


def test_e2e_fixture_submits_exactly_one_orchestration_identity() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    orchestration = source.split("def _orchestration", 1)[1].split("def _admission", 1)[
        0
    ]
    assert orchestration.count("processor.process_safely(") == 1


class FakeServiceProcess:
    def __init__(self, *, poll_result: int | None = None, return_code: int = 0) -> None:
        self.signals: list[int] = []
        self.waits: list[float | None] = []
        self.poll_result = poll_result
        self.return_code = return_code
        self.pid = 123

    def poll(self) -> int | None:
        return self.poll_result

    def send_signal(self, signal_number: int) -> None:
        self.signals.append(signal_number)

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        self.poll_result = self.return_code
        return self.return_code


def _fake_process_tables(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle: subject._ProductionLifecycle,
    process: FakeServiceProcess,
) -> None:
    monkeypatch.setattr(
        lifecycle,
        "_windows_process_table",
        lambda: {process.pid: 1} if process.poll_result is None else {},
    )
    lifecycle._protected_resources = {
        "container": frozenset(),
        "network": frozenset(),
        "volume": frozenset(),
    }
    monkeypatch.setattr(
        lifecycle, "_require_protected_resources_unchanged", lambda: None
    )


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
    _fake_process_tables(monkeypatch, lifecycle, process)

    lifecycle.cleanup()

    assert process.signals == [signal.CTRL_BREAK_EVENT]
    assert process.waits == [150, 0]
    assert lifecycle._service is None
    assert len(commands) == 6
    assert all("--filter" in command for command in commands)
    assert all("prune" not in command and "rm" not in command for command in commands)


def test_graceful_signal_waits_for_tls_root_and_python_wrapper_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess()
    process.pid = 100
    lifecycle._service = cast(subprocess.Popen[bytes], process)
    lifecycle._postgres_project = "callmetric-pgvector-tls-123-abcdef123456"
    process_tables = [
        {100: 1, 200: 100, 900: 1},
        {900: 1},
        {900: 1},
        {900: 1},
        {900: 1},
    ]
    monkeypatch.setattr(
        lifecycle, "_windows_process_table", lambda: process_tables.pop(0)
    )
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "docker")
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, stdout="", stderr=""
        ),
    )
    lifecycle._protected_resources = {
        "container": frozenset({"unrelated-container"}),
        "network": frozenset({"unrelated-network"}),
        "volume": frozenset({"unrelated-volume"}),
    }
    monkeypatch.setattr(
        lifecycle, "_require_protected_resources_unchanged", lambda: None
    )

    lifecycle.cleanup()

    assert process.signals == [signal.CTRL_BREAK_EVENT]
    assert process.waits == [150, 0]
    assert lifecycle._service is None
    assert process_tables == []


@pytest.mark.parametrize("resource", ["container", "network", "volume"])
def test_remaining_exact_project_resource_fails_cleanup(
    resource: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    lifecycle._service = cast(subprocess.Popen[bytes], FakeServiceProcess())
    process = cast(FakeServiceProcess, lifecycle._service)
    lifecycle._postgres_project = "callmetric-pgvector-tls-123-abcdef123456"
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "docker")

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        output = "residue" if arguments[1] == resource else ""
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")

    monkeypatch.setattr(subject.subprocess, "run", run)
    _fake_process_tables(monkeypatch, lifecycle, process)
    monkeypatch.setattr(lifecycle, "_cleanup_exact_postgres_project", lambda: None)
    monkeypatch.setattr(lifecycle, "_cleanup_exact_handoff", lambda: None)
    with pytest.raises(RuntimeError):
        lifecycle.cleanup()


def test_remaining_handoff_fails_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    lifecycle._service = cast(subprocess.Popen[bytes], FakeServiceProcess())
    process = cast(FakeServiceProcess, lifecycle._service)
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
    _fake_process_tables(monkeypatch, lifecycle, process)
    monkeypatch.setattr(lifecycle, "_cleanup_exact_postgres_project", lambda: None)
    monkeypatch.setattr(lifecycle, "_cleanup_exact_handoff", lambda: None)
    with pytest.raises(RuntimeError):
        lifecycle.cleanup()


@pytest.mark.parametrize("failure", ["signal", "timeout", "abnormal"])
def test_graceful_failure_modes_each_trigger_every_fallback(
    failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)

    class FailingProcess(FakeServiceProcess):
        def send_signal(self, signal_number: int) -> None:
            if failure == "signal":
                raise OSError
            super().send_signal(signal_number)

        def wait(self, timeout: float | None = None) -> int:
            if failure == "timeout" and timeout == 150:
                raise subprocess.TimeoutExpired([], 150)
            return super().wait(timeout)

    process = FailingProcess(
        poll_result=1 if failure == "abnormal" else None,
        return_code=1 if failure == "abnormal" else 0,
    )
    lifecycle._service = cast(subprocess.Popen[bytes], process)
    failing = cast(FailingProcess, lifecycle._service)
    process_tables = [{failing.pid: 1}, {}]
    monkeypatch.setattr(
        lifecycle, "_windows_process_table", lambda: process_tables.pop(0)
    )
    monkeypatch.setattr(
        lifecycle, "_require_protected_resources_unchanged", lambda: None
    )
    fallbacks: list[str] = []

    def terminate(*_args: object) -> None:
        fallbacks.append("process")
        process.poll_result = 0

    monkeypatch.setattr(lifecycle, "_terminate_owned_process_tree", terminate)
    monkeypatch.setattr(
        lifecycle,
        "_cleanup_exact_postgres_project",
        lambda: fallbacks.append("project"),
    )
    monkeypatch.setattr(
        lifecycle,
        "_cleanup_exact_handoff",
        lambda: fallbacks.append("handoff"),
    )
    monkeypatch.setattr(lifecycle, "_require_postgres_residue_absent", lambda: None)
    lifecycle.cleanup()
    assert fallbacks == ["process", "project", "handoff"]


def test_early_owned_residue_triggers_recoverable_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess()
    lifecycle._service = cast(subprocess.Popen[bytes], process)
    _fake_process_tables(monkeypatch, lifecycle, process)
    verifications = 0
    fallbacks: list[str] = []

    def verify() -> None:
        nonlocal verifications
        verifications += 1
        if verifications == 1:
            raise RuntimeError

    monkeypatch.setattr(lifecycle, "_require_postgres_residue_absent", verify)
    monkeypatch.setattr(
        lifecycle,
        "_terminate_owned_process_tree",
        lambda *_args: fallbacks.append("process"),
    )
    monkeypatch.setattr(
        lifecycle,
        "_cleanup_exact_postgres_project",
        lambda: fallbacks.append("project"),
    )
    monkeypatch.setattr(
        lifecycle,
        "_cleanup_exact_handoff",
        lambda: fallbacks.append("handoff"),
    )

    lifecycle.cleanup()

    assert verifications == 2
    assert fallbacks == ["process", "project", "handoff"]


@pytest.mark.parametrize("failed_action", ["process", "project", "handoff"])
def test_nonrecoverable_fallback_validation_failure_remains_cleanup_failure(
    failed_action: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess(return_code=1)
    lifecycle._service = cast(subprocess.Popen[bytes], process)
    _fake_process_tables(monkeypatch, lifecycle, process)
    actions: list[str] = []

    def action(name: str) -> None:
        actions.append(name)
        if name == failed_action:
            raise RuntimeError

    monkeypatch.setattr(
        lifecycle, "_terminate_owned_process_tree", lambda *_args: action("process")
    )
    monkeypatch.setattr(
        lifecycle, "_cleanup_exact_postgres_project", lambda: action("project")
    )
    monkeypatch.setattr(lifecycle, "_cleanup_exact_handoff", lambda: action("handoff"))
    monkeypatch.setattr(lifecycle, "_require_postgres_residue_absent", lambda: None)

    with pytest.raises(RuntimeError):
        lifecycle.cleanup()

    assert actions == ["process", "project", "handoff"]


def test_unverifiable_or_changed_protected_resources_fail_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess()
    lifecycle._service = cast(subprocess.Popen[bytes], process)
    _fake_process_tables(monkeypatch, lifecycle, process)
    monkeypatch.setattr(lifecycle, "_require_postgres_residue_absent", lambda: None)
    monkeypatch.setattr(
        lifecycle,
        "_require_protected_resources_unchanged",
        lambda: (_ for _ in ()).throw(RuntimeError()),
    )

    with pytest.raises(RuntimeError):
        lifecycle.cleanup()


def test_remaining_owned_process_after_fallback_fails_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess()
    lifecycle._service = cast(subprocess.Popen[bytes], process)
    lifecycle._postgres_project = "callmetric-pgvector-tls-123-abcdef123456"
    monkeypatch.setattr(lifecycle, "_windows_process_table", lambda: {process.pid: 1})
    monkeypatch.setattr(lifecycle, "_terminate_owned_process_tree", lambda *_args: None)
    monkeypatch.setattr(lifecycle, "_cleanup_exact_postgres_project", lambda: None)
    monkeypatch.setattr(lifecycle, "_cleanup_exact_handoff", lambda: None)
    monkeypatch.setattr(
        lifecycle, "_require_protected_resources_unchanged", lambda: None
    )
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


def test_owned_process_tree_is_terminated_descendant_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess()
    process.pid = 100
    tables = [
        {100: 1, 200: 100, 300: 200, 900: 1},
        {100: 1, 900: 1},
        {100: 1, 900: 1},
        {900: 1},
    ]
    monkeypatch.setattr(
        lifecycle,
        "_windows_process_table",
        lambda: tables.pop(0),
    )
    monkeypatch.setattr(subject.shutil, "which", lambda name: name)
    terminated: list[int] = []

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        terminated.append(int(arguments[2]))
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(subject.subprocess, "run", run)
    lifecycle._terminate_owned_process_tree(
        cast(subprocess.Popen[bytes], process),
        {100: 1, 200: 100, 300: 200, 900: 1},
    )
    assert terminated == [300, 200, 100]


def test_final_owned_process_check_ignores_unrelated_initial_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess(poll_result=0)
    process.pid = 100
    lifecycle._service = cast(subprocess.Popen[bytes], process)
    monkeypatch.setattr(lifecycle, "_windows_process_table", lambda: {900: 1})

    lifecycle._require_initial_processes_absent({100: 1, 200: 100, 300: 200, 900: 1})


def test_descendant_disappearance_before_termination_is_recovered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess(poll_result=0)
    process.pid = 100
    tables = [{}, {}, {}]
    monkeypatch.setattr(lifecycle, "_windows_process_table", lambda: tables.pop(0))
    monkeypatch.setattr(
        subject.shutil,
        "which",
        lambda _name: pytest.fail("no termination tool required for absent targets"),
    )

    lifecycle._terminate_owned_process_tree(
        cast(subprocess.Popen[bytes], process), {100: 1, 200: 100}
    )


def test_root_disappearance_before_termination_is_recovered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess()
    process.pid = 100
    tables = [{}, {}, {}]
    monkeypatch.setattr(lifecycle, "_windows_process_table", lambda: tables.pop(0))
    monkeypatch.setattr(
        subject.shutil,
        "which",
        lambda _name: pytest.fail("no termination tool required for absent targets"),
    )

    lifecycle._terminate_owned_process_tree(
        cast(subprocess.Popen[bytes], process), {100: 1}
    )


def test_owned_descendants_are_terminated_after_root_disappears(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess(poll_result=0)
    process.pid = 100
    tables = [
        {200: 100, 300: 200, 900: 1},
        {900: 1},
        {900: 1},
        {900: 1},
    ]
    monkeypatch.setattr(lifecycle, "_windows_process_table", lambda: tables.pop(0))
    monkeypatch.setattr(subject.shutil, "which", lambda name: name)
    terminated: list[int] = []

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        terminated.append(int(arguments[2]))
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(subject.subprocess, "run", run)

    lifecycle._terminate_owned_process_tree(
        cast(subprocess.Popen[bytes], process), {100: 1, 200: 100, 300: 200}
    )

    assert terminated == [300, 200]


def test_process_ledger_captures_evolving_descendants_without_unrelated_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess()
    process.pid = 100
    lifecycle._service = cast(subprocess.Popen[bytes], process)
    tables = iter(({100: 1, 200: 100, 900: 1}, {100: 1, 200: 100, 300: 200, 900: 1}))
    monkeypatch.setattr(lifecycle, "_windows_process_table", lambda: next(tables))

    lifecycle._refresh_owned_process_ledger()
    lifecycle._refresh_owned_process_ledger()

    assert lifecycle._owned_processes == {100: 1, 200: 100, 300: 200}


def test_process_ledger_rejects_changed_lineage_or_pid_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess()
    process.pid = 100
    lifecycle._service = cast(subprocess.Popen[bytes], process)
    lifecycle._owned_processes = {100: 1, 200: 100}
    monkeypatch.setattr(lifecycle, "_windows_process_table", lambda: {100: 1, 200: 999})

    with pytest.raises(
        subject._CleanupPhaseError,
        match=f"^{subject.E_CLEANUP_PROCESS_VERIFY}$",
    ):
        lifecycle._refresh_owned_process_ledger()


def test_pid_reuse_or_non_descendant_is_never_terminated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess()
    process.pid = 100
    monkeypatch.setattr(lifecycle, "_windows_process_table", lambda: {100: 1, 200: 999})
    monkeypatch.setattr(subject.shutil, "which", lambda name: name)
    terminated: list[int] = []
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda arguments, **_kwargs: terminated.append(int(arguments[2])),
    )
    with pytest.raises(RuntimeError):
        lifecycle._terminate_owned_process_tree(
            cast(subprocess.Popen[bytes], process), {100: 1, 200: 100}
        )
    assert 200 not in terminated
