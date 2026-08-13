from __future__ import annotations

import json
from io import BytesIO
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
from typing import cast

import pytest

import scripts.run_postgres_tls_service as tls_subject
import scripts.run_windows_dashboard_rag_vllm_e2e as subject
from app.coaching.coordinator import (
    CoachingCoordinatorResult,
    CoachingProcessingStatus,
    CoachingSourcePresentation,
    StableCoachingOutcome,
)
from app.composition.postgres_rag import (
    RAGDiagnosticEvent,
    RAGDiagnosticFutureState,
    RAGDiagnosticSnapshot,
    RAGDiagnosticStage,
    RAGDiagnosticStatus,
    RAGDiagnosticSubmissionState,
)
from app.coaching.coordinator import _cooldown_available
from app.ingestion.document_background import (
    DocumentSubmissionResult,
    DocumentSubmissionStatus,
)
from app.ingestion.registry_models import (
    DocumentIngestionJob,
    DocumentIngestionPhase,
    DocumentIngestionState,
    DocumentReadiness,
    DocumentRegistryEntry,
    DocumentRegistryRecord,
)

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
                "rag_llm_enabled_labels": ["product_information"],
                "title": "Synthetic guidance",
                "action": "RAG_ACTION",
                "priority": "HIGH",
                "label_id": "product_information",
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
    expected = {
        "E_COMPLETION_PUMP": subject.E_COMPLETION_UNCLASSIFIED,
        "E_POSTGRES_START": subject.E_POSTGRES_UNCLASSIFIED,
        "E_DUPLICATE": subject.E_DUPLICATE_UNCLASSIFIED,
        "E_DOCUMENT_READY": subject.E_DOCUMENT_READY_UNCLASSIFIED,
    }.get(phase, phase)
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


@pytest.mark.parametrize(
    ("stage", "phase"),
    [
        (RAGDiagnosticStage.RUN, subject.E_COMPLETION_STALLED_RUN_ENTER),
        (RAGDiagnosticStage.EMBED, subject.E_COMPLETION_STALLED_EMBED_ENTER),
        (RAGDiagnosticStage.VECTOR, subject.E_COMPLETION_STALLED_VECTOR_ENTER),
        (RAGDiagnosticStage.PROMPT, subject.E_COMPLETION_STALLED_PROMPT_ENTER),
        (
            RAGDiagnosticStage.GATEWAY_FACTORY,
            subject.E_COMPLETION_STALLED_GATEWAY_FACTORY_ENTER,
        ),
        (RAGDiagnosticStage.HTTP, subject.E_COMPLETION_STALLED_HTTP_ENTER),
        (RAGDiagnosticStage.CALLBACK, subject.E_COMPLETION_STALLED_CALLBACK_ENTER),
    ],
)
def test_completion_stall_reports_exact_last_entered_stage(
    stage: RAGDiagnosticStage, phase: str
) -> None:
    snapshot = RAGDiagnosticSnapshot(
        events=(RAGDiagnosticEvent(stage, RAGDiagnosticStatus.ENTER),),
        worker_live=True,
        future_terminal=False,
        callback_entered=False,
        authoritative_completion_published=False,
        running_after_close=False,
        unclassified=False,
        submission_state=RAGDiagnosticSubmissionState.ACCEPTED,
        future_state=RAGDiagnosticFutureState.RUNNING,
        rag_worker_entered=True,
    )

    assert subject._stalled_completion_phase(snapshot) == phase


@pytest.mark.parametrize(
    ("submission_state", "future_state", "phase"),
    [
        (
            RAGDiagnosticSubmissionState.NOT_ATTEMPTED,
            RAGDiagnosticFutureState.ABSENT,
            subject.E_COMPLETION_SUBMISSION_NOT_ATTEMPTED,
        ),
        (
            RAGDiagnosticSubmissionState.ACCEPTED,
            RAGDiagnosticFutureState.QUEUED,
            subject.E_COMPLETION_FUTURE_QUEUED,
        ),
        (
            RAGDiagnosticSubmissionState.ACCEPTED,
            RAGDiagnosticFutureState.RUNNING,
            subject.E_COMPLETION_FUTURE_RUNNING_NO_STAGE,
        ),
        (
            RAGDiagnosticSubmissionState.ACCEPTED,
            RAGDiagnosticFutureState.TERMINAL,
            subject.E_COMPLETION_FUTURE_TERMINAL_NO_PUBLICATION,
        ),
    ],
)
def test_completion_stall_distinguishes_terminal_and_publication_states(
    submission_state: RAGDiagnosticSubmissionState,
    future_state: RAGDiagnosticFutureState,
    phase: str,
) -> None:
    snapshot = RAGDiagnosticSnapshot(
        (),
        future_state is RAGDiagnosticFutureState.RUNNING,
        future_state is RAGDiagnosticFutureState.TERMINAL,
        False,
        False,
        False,
        False,
        submission_state,
        future_state,
        False,
    )
    assert subject._stalled_completion_phase(snapshot) == phase


@pytest.mark.parametrize(
    ("submission_state", "phase"),
    [
        (
            RAGDiagnosticSubmissionState.REJECTED_DUPLICATE,
            subject.E_COMPLETION_SUBMISSION_REJECTED_DUPLICATE,
        ),
        (
            RAGDiagnosticSubmissionState.REJECTED_STALE,
            subject.E_COMPLETION_SUBMISSION_REJECTED_STALE,
        ),
        (
            RAGDiagnosticSubmissionState.REJECTED_CAPACITY_REJECTED,
            subject.E_COMPLETION_SUBMISSION_REJECTED_CAPACITY_REJECTED,
        ),
        (
            RAGDiagnosticSubmissionState.REJECTED_NOT_STARTED,
            subject.E_COMPLETION_SUBMISSION_REJECTED_NOT_STARTED,
        ),
        (
            RAGDiagnosticSubmissionState.REJECTED_CLOSED,
            subject.E_COMPLETION_SUBMISSION_REJECTED_CLOSED,
        ),
        (
            RAGDiagnosticSubmissionState.SUBMIT_FAILED,
            subject.E_COMPLETION_SUBMIT_FAILED,
        ),
    ],
)
def test_completion_stall_reports_exact_submission_failure(
    submission_state: RAGDiagnosticSubmissionState, phase: str
) -> None:
    snapshot = RAGDiagnosticSnapshot(
        (),
        False,
        False,
        False,
        False,
        False,
        False,
        submission_state,
        RAGDiagnosticFutureState.ABSENT,
        False,
    )

    assert subject._stalled_completion_phase(snapshot) == phase


def test_completion_diagnostic_snapshot_never_contains_injected_secret() -> None:
    snapshot = RAGDiagnosticSnapshot(
        events=(
            RAGDiagnosticEvent(RAGDiagnosticStage.HTTP, RAGDiagnosticStatus.FAILED),
        ),
        worker_live=False,
        future_terminal=True,
        callback_entered=True,
        authoritative_completion_published=False,
        running_after_close=True,
        unclassified=True,
    )

    phase = subject._stalled_completion_phase(snapshot)

    assert phase == subject.E_COMPLETION_DIAGNOSTIC_UNCLASSIFIED
    assert "injected-secret" not in repr(snapshot)
    assert "injected-secret" not in phase


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


@pytest.mark.parametrize("labels", [["urun_bilgisi"], ["no_action"]])
def test_non_triggering_policy_labels_fail_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    labels: list[str],
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    policy_path = Path(values[subject.POLICY_ENV])
    policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_payload["rag_llm_enabled_labels"] = labels
    policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")

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


@pytest.mark.parametrize(
    ("child_phase", "parent_phase"), sorted(subject.POSTGRES_CHILD_PHASES.items())
)
def test_public_main_propagates_exact_safe_tls_child_failure_and_cleans(
    child_phase: str,
    parent_phase: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    config = subject.preflight(values)

    class ChildFailureOperations(FakeOperations):
        def run_phase(self, phase: str) -> None:
            self.events.append(phase)
            if phase == "E_POSTGRES_START":
                raise subject._PostgresChildError(parent_phase)

    operations = ChildFailureOperations()
    monkeypatch.setattr(subject, "_preflight", lambda _environment=None: config)
    monkeypatch.setattr(
        subject,
        "_ProductionLifecycle",
        lambda _config, _environment: operations,
    )

    assert subject.main([]) == 1

    captured = capsys.readouterr()
    assert captured.out.strip() == parent_phase
    assert captured.err == ""
    assert operations.events == ["E_POSTGRES_START", "E_CLEANUP"]
    assert child_phase not in captured.out


def test_public_main_never_emits_unrecognized_tls_child_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    config = subject.preflight(values)
    secret_like_detail = "private-dsn-token-certificate-path"

    class UnknownChildOperations(FakeOperations):
        def run_phase(self, phase: str) -> None:
            self.events.append(phase)
            if phase == "E_POSTGRES_START":
                raise subject._PostgresChildError(secret_like_detail)

    operations = UnknownChildOperations()
    monkeypatch.setattr(subject, "_preflight", lambda _environment=None: config)
    monkeypatch.setattr(
        subject,
        "_ProductionLifecycle",
        lambda _config, _environment: operations,
    )

    assert subject.main([]) == 1

    captured = capsys.readouterr()
    assert captured.out.strip() == subject.E_POSTGRES_CHILD_UNCLASSIFIED
    assert captured.err == ""
    assert secret_like_detail not in captured.out
    assert operations.events[-1] == "E_CLEANUP"


@pytest.mark.parametrize(
    "startup_phase", sorted(subject.POSTGRES_STARTUP_FAILURE_PHASES)
)
def test_public_main_preserves_every_fixed_postgres_startup_phase(
    startup_phase: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    config = subject.preflight(values)

    class StartupFailureOperations(FakeOperations):
        def run_phase(self, phase: str) -> None:
            self.events.append(phase)
            if phase == "E_POSTGRES_START":
                raise subject._PostgresStartupError(startup_phase)

    operations = StartupFailureOperations()
    monkeypatch.setattr(subject, "_preflight", lambda _environment=None: config)
    monkeypatch.setattr(
        subject,
        "_ProductionLifecycle",
        lambda _config, _environment: operations,
    )

    assert subject.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == startup_phase
    assert captured.err == ""
    assert operations.events == ["E_POSTGRES_START", "E_CLEANUP"]


def test_unexpected_postgres_startup_exception_never_emits_generic_start_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    config = subject.preflight(values)
    secret_like_detail = "private-startup-path-token-dsn"

    class UnexpectedStartupOperations(FakeOperations):
        def run_phase(self, phase: str) -> None:
            if phase == "E_POSTGRES_START":
                raise RuntimeError(secret_like_detail)
            super().run_phase(phase)

    operations = UnexpectedStartupOperations()
    monkeypatch.setattr(subject, "_preflight", lambda _environment=None: config)
    monkeypatch.setattr(
        subject,
        "_ProductionLifecycle",
        lambda _config, _environment: operations,
    )

    assert subject.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == subject.E_POSTGRES_UNCLASSIFIED
    assert captured.out.strip() != "E_POSTGRES_START"
    assert captured.err == ""
    assert secret_like_detail not in captured.out
    assert secret_like_detail not in captured.err


@pytest.mark.parametrize("phase", sorted(subject.POSTGRES_STARTUP_FAILURE_PHASES))
def test_startup_boundary_maps_every_unexpected_source_to_its_fixed_phase(
    phase: str,
) -> None:
    def fail() -> None:
        raise RuntimeError("private-boundary-detail")

    with pytest.raises(subject._PostgresStartupError, match=f"^{phase}$"):
        subject._ProductionLifecycle._postgres_startup_call(phase, fail)


def test_postgres_startup_only_uses_postgres_preflight_and_same_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    for key in (
        subject.PROVIDER_ENV,
        subject.POLICY_ENV,
        subject.TOKEN_ENV,
        subject.CA_ENV,
        "CALLMETRIC_VLLM_BASE_URL",
        "CALLMETRIC_VLLM_MODEL_ID",
        "CALLMETRIC_VLLM_CONNECT_TIMEOUT_SECONDS",
        "CALLMETRIC_VLLM_READ_TIMEOUT_SECONDS",
        "CALLMETRIC_VLLM_MAX_OUTPUT_TOKENS",
        "CALLMETRIC_VLLM_TEMPERATURE",
        "CALLMETRIC_VLLM_VERIFY_TLS",
    ):
        values.pop(key)
    monkeypatch.setattr(
        subject,
        "validate_local_minilm_snapshot",
        lambda _value: pytest.fail("startup-only must not validate a model"),
    )
    monkeypatch.setattr(
        subject,
        "_read_json",
        lambda *_args: pytest.fail("startup-only must not read provider policy"),
    )
    operations = FakeOperations()

    assert (
        subject.run_postgres_startup_only(
            environment=values,
            operations_factory=lambda _config: operations,
        )
        == subject.POSTGRES_STARTUP_OK
    )
    assert operations.events == ["E_POSTGRES_START", "E_CLEANUP"]


def test_postgres_startup_only_dispatches_production_start_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    events: list[str] = []
    monkeypatch.setattr(
        subject._ProductionLifecycle,
        "_postgres_start",
        lambda _self: events.append("start"),
    )
    monkeypatch.setattr(
        subject._ProductionLifecycle,
        "cleanup",
        lambda _self: events.append("cleanup"),
    )

    assert (
        subject.run_postgres_startup_only(environment=environment(tmp_path))
        == subject.POSTGRES_STARTUP_OK
    )
    assert events == ["start", "cleanup"]


def test_postgres_startup_only_failure_cleans_and_preserves_exact_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)

    class OwnershipFailureOperations(FakeOperations):
        def run_phase(self, phase: str) -> None:
            self.events.append(phase)
            raise subject._PostgresStartupError(subject.E_POSTGRES_OWNERSHIP)

    operations = OwnershipFailureOperations(cleanup_failure=True)
    with pytest.raises(subject.DashboardRAGVLLME2EError) as caught:
        subject.run_postgres_startup_only(
            environment=environment(tmp_path),
            operations_factory=lambda _config: operations,
        )
    assert caught.value.phase == subject.E_POSTGRES_OWNERSHIP
    assert operations.events == ["E_POSTGRES_START", "E_CLEANUP"]


@pytest.mark.parametrize("phase", sorted(subject.CLEANUP_FAILURE_PHASES))
def test_postgres_startup_only_preserves_fixed_cleanup_subphase(
    phase: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)

    class CleanupFailureOperations(FakeOperations):
        def cleanup(self) -> None:
            self.events.append("E_CLEANUP")
            raise subject._CleanupPhaseError(phase)

    operations = CleanupFailureOperations()
    with pytest.raises(subject.DashboardRAGVLLME2EError) as caught:
        subject.run_postgres_startup_only(
            environment=environment(tmp_path),
            operations_factory=lambda _config: operations,
        )

    assert caught.value.phase == phase
    assert operations.events == ["E_POSTGRES_START", "E_CLEANUP"]


def test_postgres_startup_only_unknown_cleanup_failure_stays_generic_and_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    secret_like_text = "private-cleanup-token-dsn"

    class UnknownCleanupFailureOperations(FakeOperations):
        def cleanup(self) -> None:
            self.events.append("E_CLEANUP")
            raise RuntimeError(secret_like_text)

    operations = UnknownCleanupFailureOperations()
    run_startup_only = subject.run_postgres_startup_only
    monkeypatch.setattr(
        subject,
        "run_postgres_startup_only",
        lambda: run_startup_only(
            environment=values, operations_factory=lambda _config: operations
        ),
    )

    assert subject.main(["--postgres-startup-only"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "E_CLEANUP\n"
    assert captured.err == ""
    assert secret_like_text not in captured.out
    assert secret_like_text not in captured.err
    assert operations.events == ["E_POSTGRES_START", "E_CLEANUP"]


def test_postgres_startup_only_accepts_builtin_bridge_id_rotation_and_cleans_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    fingerprints = tuple(
        tls_subject.BuiltinNetworkFingerprint(
            name,
            {"bridge": "bridge", "host": "host", "none": "null"}[name],
            "local",
            False,
            False,
            False,
            "default",
        )
        for name in tls_subject.BUILTIN_NETWORK_NAMES
    )
    expected = tls_subject.ProtectedResourceSnapshot(
        frozenset({"container-old"}),
        frozenset({"volume-old"}),
        frozenset({"d" * 64}),
        fingerprints,
    )
    current_after_bridge_rotation = tls_subject.ProtectedResourceSnapshot(
        expected.container_ids,
        expected.volume_ids,
        expected.user_network_ids,
        fingerprints,
    )
    cleanup_count = 0

    class StartupOperations(FakeOperations):
        def cleanup(self) -> None:
            nonlocal cleanup_count
            cleanup_count += 1
            monkeypatch.setattr(
                tls_subject,
                "_resource_snapshot",
                lambda _docker: current_after_bridge_rotation,
            )
            tls_subject.require_protected_resources_unchanged("docker", expected)

    assert (
        subject.run_postgres_startup_only(
            environment=environment(tmp_path),
            operations_factory=lambda _config: StartupOperations(),
        )
        == subject.POSTGRES_STARTUP_OK
    )
    assert cleanup_count == 1


def test_user_network_mutation_maps_to_fixed_cleanup_protected_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    lifecycle._protected_resources = tls_subject.ProtectedResourceSnapshot(
        frozenset(),
        frozenset(),
        frozenset(),
        tuple(
            tls_subject.BuiltinNetworkFingerprint(
                name,
                {"bridge": "bridge", "host": "host", "none": "null"}[name],
                "local",
                False,
                False,
                False,
                "default",
            )
            for name in tls_subject.BUILTIN_NETWORK_NAMES
        ),
    )
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "docker")
    monkeypatch.setattr(
        tls_subject,
        "require_protected_resources_unchanged",
        lambda _docker, _expected: (_ for _ in ()).throw(
            tls_subject.PostgreSQLTLSServiceError(
                phase=tls_subject.E_PROTECTED_RESOURCES
            )
        ),
    )
    with pytest.raises(
        subject._CleanupPhaseError,
        match=f"^{subject.E_CLEANUP_PROTECTED_VERIFY}$",
    ):
        lifecycle._require_protected_resources_unchanged()


def test_main_postgres_startup_only_prints_only_fixed_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        subject, "run_postgres_startup_only", lambda: subject.POSTGRES_STARTUP_OK
    )
    assert subject.main(["--postgres-startup-only"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == subject.POSTGRES_STARTUP_OK
    assert captured.err == ""


def test_tls_child_reader_accepts_only_fixed_bounded_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle = _completion_lifecycle(monkeypatch, tmp_path)
    payload = (
        b"PR54 PostgreSQL TLS READY; TTL remaining: 300 seconds\n"
        b"E_STARTUP PR54 PostgreSQL TLS service failed\n"
        b"arbitrary-private-output\n"
        + b"x" * (subject._TLS_CHILD_LINE_LIMIT + 1)
        + b"\n"
    )

    lifecycle._read_tls_child_output(BytesIO(payload))

    events = lifecycle._drain_tls_child_events()
    assert events[:2] == (
        subject._TLSChildOutputEvent("ready"),
        subject._TLSChildOutputEvent("failure", subject.E_POSTGRES_CHILD_STARTUP),
    )
    assert all(event.kind == "malformed" for event in events[2:])
    assert all(event.phase is None for event in events[2:])


@pytest.mark.parametrize(
    ("ready_count", "failures", "malformed", "expected"),
    [
        (0, (), False, subject.E_POSTGRES_CHILD_UNCLASSIFIED),
        (0, (subject.E_POSTGRES_CHILD_TLS,), False, subject.E_POSTGRES_CHILD_TLS),
        (0, (), True, subject.E_POSTGRES_CHILD_UNCLASSIFIED),
        (
            0,
            (subject.E_POSTGRES_CHILD_TLS,) * 2,
            False,
            subject.E_POSTGRES_CHILD_UNCLASSIFIED,
        ),
        (1, (), False, subject.E_POSTGRES_CHILD_READY_EXIT),
        (
            1,
            (subject.E_POSTGRES_CHILD_TLS,),
            False,
            subject.E_POSTGRES_CHILD_UNCLASSIFIED,
        ),
    ],
)
def test_exited_tls_child_classification_is_fixed(
    ready_count: int,
    failures: tuple[str, ...],
    malformed: bool,
    expected: str,
) -> None:
    assert (
        subject._ProductionLifecycle._classify_exited_tls_child(
            ready_count, failures, malformed
        )
        == expected
    )


@pytest.mark.parametrize(
    ("ready_count", "failures", "malformed", "expected"),
    [
        (0, (), False, subject.E_POSTGRES_CHILD_TIMEOUT),
        (1, (), False, subject.E_POSTGRES_CHILD_HANDOFF_NOT_PRODUCED),
        (
            0,
            (subject.E_POSTGRES_CHILD_HANDOFF,),
            False,
            subject.E_POSTGRES_CHILD_HANDOFF,
        ),
        (2, (), False, subject.E_POSTGRES_CHILD_UNCLASSIFIED),
        (0, (), True, subject.E_POSTGRES_CHILD_UNCLASSIFIED),
    ],
)
def test_tls_child_timeout_classification_is_fixed(
    ready_count: int,
    failures: tuple[str, ...],
    malformed: bool,
    expected: str,
) -> None:
    assert (
        subject._ProductionLifecycle._classify_tls_child_timeout(
            ready_count, failures, malformed
        )
        == expected
    )


def test_tls_ready_requires_fixed_line_handoff_and_live_child() -> None:
    ready = subject._ProductionLifecycle._tls_child_startup_ready
    assert ready(
        ready_count=1,
        failure_count=0,
        malformed=False,
        handoff_count=1,
        child_running=True,
    )
    assert not ready(
        ready_count=0,
        failure_count=0,
        malformed=False,
        handoff_count=1,
        child_running=True,
    )
    assert not ready(
        ready_count=1,
        failure_count=0,
        malformed=False,
        handoff_count=0,
        child_running=True,
    )
    assert not ready(
        ready_count=1,
        failure_count=0,
        malformed=False,
        handoff_count=1,
        child_running=False,
    )


def test_e2e_fixture_submits_exactly_one_orchestration_identity() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    orchestration = source.split("def _orchestration", 1)[1].split("def _admission", 1)[
        0
    ]
    assert orchestration.count("processor.process_safely(") == 1


def _ready_document_entry(
    name: str, *, source_key: str | None = None
) -> DocumentRegistryEntry:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    document = DocumentRegistryRecord(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        document_id=f"document-{name}",
        original_filename={
            "target": "synthetic-guide.txt",
            "other": "synthetic-other.txt",
        }.get(name, f"{name}.txt"),
        media_type="text/plain",
        byte_size=10,
        storage_object_key=source_key,
        created_at_utc=now,
        ready_at_utc=now,
    )
    job = DocumentIngestionJob(
        tenant_id=document.tenant_id,
        knowledge_base_id=document.knowledge_base_id,
        document_id=document.document_id,
        job_id=f"job-{name}",
        state=DocumentIngestionState.SUCCEEDED,
        phase=DocumentIngestionPhase.FINALIZE,
        processed_chunks=1,
        total_chunks=1,
        attempt_count=1,
        created_at_utc=now,
        started_at_utc=now,
        updated_at_utc=now,
        finished_at_utc=now,
    )
    return DocumentRegistryEntry(
        document=document, job=job, readiness=DocumentReadiness.READY
    )


def _document_entry_with_state(
    name: str, state: DocumentIngestionState
) -> DocumentRegistryEntry:
    ready = _ready_document_entry(name)
    if state is DocumentIngestionState.SUCCEEDED:
        return ready
    readiness = {
        DocumentIngestionState.QUEUED: DocumentReadiness.PENDING,
        DocumentIngestionState.PROCESSING: DocumentReadiness.PENDING,
        DocumentIngestionState.FAILED: DocumentReadiness.FAILED,
        DocumentIngestionState.CANCELLED: DocumentReadiness.CANCELLED,
    }[state]
    started = (
        None if state is DocumentIngestionState.QUEUED else ready.job.started_at_utc
    )
    finished = (
        ready.job.finished_at_utc
        if state in {DocumentIngestionState.FAILED, DocumentIngestionState.CANCELLED}
        else None
    )
    job = ready.job.model_copy(
        update={
            "state": state,
            "phase": DocumentIngestionPhase.EXTRACTION,
            "processed_chunks": 0,
            "total_chunks": 0,
            "started_at_utc": started,
            "finished_at_utc": finished,
        }
    )
    document = ready.document.model_copy(update={"ready_at_utc": None})
    return DocumentRegistryEntry(document=document, job=job, readiness=readiness)


class _DocumentReadyRegistry:
    def __init__(self, results: tuple[object, ...]) -> None:
        self.results = list(results)
        self.calls = 0

    def list_documents(self, **_kwargs: object) -> object:
        self.calls += 1
        if not self.results:
            raise AssertionError("unexpected registry call")
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _document_ready_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *results: object,
) -> tuple[subject._ProductionLifecycle, _DocumentReadyRegistry]:
    lifecycle = _completion_lifecycle(monkeypatch, tmp_path)
    registry = _DocumentReadyRegistry(results)
    lifecycle._document_runtime = cast(
        subject.PostgreSQLDocumentIngestionRuntime,
        SimpleNamespace(
            tenant_id="tenant-synthetic",
            knowledge_base_id="kb-synthetic",
            registry=registry,
        ),
    )
    return lifecycle, registry


def test_document_ready_immediate_success_uses_real_models_without_sleep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = _ready_document_entry("target")
    other = _ready_document_entry("other")
    lifecycle, registry = _document_ready_lifecycle(
        monkeypatch, tmp_path, (other, target)
    )
    monkeypatch.setattr(subject.time, "monotonic", iter((0.0, 0.0)).__next__)
    monkeypatch.setattr(subject.time, "sleep", lambda _seconds: pytest.fail("slept"))

    lifecycle._document_ready()

    assert lifecycle._target_entry == target
    assert lifecycle._other_entry == other
    assert registry.calls == 1


def test_document_ready_mixed_pending_then_success_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = _ready_document_entry("target")
    other = _ready_document_entry("other")
    pending = _document_entry_with_state("other", DocumentIngestionState.PROCESSING)
    lifecycle, registry = _document_ready_lifecycle(
        monkeypatch, tmp_path, (target, pending), (target, other)
    )
    monkeypatch.setattr(subject.time, "monotonic", iter((0.0, 0.0, 0.1)).__next__)
    sleeps: list[float] = []
    monkeypatch.setattr(subject.time, "sleep", sleeps.append)

    lifecycle._document_ready()

    assert registry.calls == 2
    assert sleeps == [subject.POLL_INTERVAL_SECONDS]


@pytest.mark.parametrize(
    ("fixture", "state", "phase"),
    [
        (
            "target",
            DocumentIngestionState.FAILED,
            subject.E_DOCUMENT_READY_TARGET_FAILED,
        ),
        ("other", DocumentIngestionState.FAILED, subject.E_DOCUMENT_READY_OTHER_FAILED),
        (
            "target",
            DocumentIngestionState.CANCELLED,
            subject.E_DOCUMENT_READY_TARGET_CANCELLED,
        ),
        (
            "other",
            DocumentIngestionState.CANCELLED,
            subject.E_DOCUMENT_READY_OTHER_CANCELLED,
        ),
    ],
)
def test_document_ready_terminal_state_fails_without_sleep(
    fixture: str,
    state: DocumentIngestionState,
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = _document_entry_with_state(
        "target", state if fixture == "target" else DocumentIngestionState.PROCESSING
    )
    other = _document_entry_with_state(
        "other", state if fixture == "other" else DocumentIngestionState.PROCESSING
    )
    lifecycle, _ = _document_ready_lifecycle(monkeypatch, tmp_path, (target, other))
    monkeypatch.setattr(subject.time, "monotonic", iter((0.0, 0.0)).__next__)
    monkeypatch.setattr(subject.time, "sleep", lambda _seconds: pytest.fail("slept"))

    with pytest.raises(subject._DocumentReadyPhaseError, match=f"^{phase}$"):
        lifecycle._document_ready()


@pytest.mark.parametrize(
    ("entries", "phase"),
    [
        (
            (_ready_document_entry("unknown"), _ready_document_entry("other")),
            subject.E_DOCUMENT_READY_TARGET_MISSING,
        ),
        (
            (_ready_document_entry("target"), _ready_document_entry("unknown")),
            subject.E_DOCUMENT_READY_OTHER_MISSING,
        ),
        (
            (
                _ready_document_entry("target"),
                _ready_document_entry("other"),
                _ready_document_entry("unknown"),
            ),
            subject.E_DOCUMENT_READY_CARDINALITY,
        ),
        (
            (_ready_document_entry("target"), _ready_document_entry("target")),
            subject.E_DOCUMENT_READY_RESULT_SHAPE,
        ),
    ],
)
def test_document_ready_fixture_shape_has_exact_phase(
    entries: tuple[DocumentRegistryEntry, ...],
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lifecycle, _ = _document_ready_lifecycle(monkeypatch, tmp_path, entries)
    monkeypatch.setattr(subject.time, "monotonic", iter((0.0, 0.0)).__next__)
    with pytest.raises(subject._DocumentReadyPhaseError, match=f"^{phase}$"):
        lifecycle._document_ready()


@pytest.mark.parametrize("result", [[], (object(), object())])
def test_document_ready_rejects_malformed_registry_shape(
    result: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle, _ = _document_ready_lifecycle(monkeypatch, tmp_path, result)
    monkeypatch.setattr(subject.time, "monotonic", iter((0.0, 0.0)).__next__)
    with pytest.raises(
        subject._DocumentReadyPhaseError,
        match=f"^{subject.E_DOCUMENT_READY_RESULT_SHAPE}$",
    ):
        lifecycle._document_ready()


@pytest.mark.parametrize(
    ("fixture", "phase"),
    [
        ("target", subject.E_DOCUMENT_READY_TARGET_JOB),
        ("other", subject.E_DOCUMENT_READY_OTHER_JOB),
    ],
)
def test_document_ready_rejects_inconsistent_job_readiness(
    fixture: str, phase: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = _ready_document_entry("target")
    other = _ready_document_entry("other")
    if fixture == "target":
        target = target.model_copy(update={"readiness": DocumentReadiness.PENDING})
    else:
        other = other.model_copy(
            update={"job": other.job.model_copy(update={"document_id": "other-job"})}
        )
    lifecycle, _ = _document_ready_lifecycle(monkeypatch, tmp_path, (target, other))
    monkeypatch.setattr(subject.time, "monotonic", iter((0.0, 0.0)).__next__)
    with pytest.raises(subject._DocumentReadyPhaseError, match=f"^{phase}$"):
        lifecycle._document_ready()


def test_document_ready_rejects_non_null_source_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = _ready_document_entry("target", source_key="server-owned-key")
    lifecycle, _ = _document_ready_lifecycle(
        monkeypatch, tmp_path, (target, _ready_document_entry("other"))
    )
    monkeypatch.setattr(subject.time, "monotonic", iter((0.0, 0.0)).__next__)
    with pytest.raises(
        subject._DocumentReadyPhaseError,
        match=f"^{subject.E_DOCUMENT_READY_SOURCE_KEY}$",
    ):
        lifecycle._document_ready()


def test_document_ready_runtime_absence_has_exact_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle = _completion_lifecycle(monkeypatch, tmp_path)
    with pytest.raises(
        subject._DocumentReadyPhaseError,
        match=f"^{subject.E_DOCUMENT_READY_RUNTIME}$",
    ):
        lifecycle._document_ready()


def test_document_ready_registry_failure_is_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lifecycle, _ = _document_ready_lifecycle(
        monkeypatch, tmp_path, RuntimeError("private-secret")
    )
    monkeypatch.setattr(subject.time, "monotonic", iter((0.0, 0.0)).__next__)
    with pytest.raises(
        subject._DocumentReadyPhaseError,
        match=f"^{subject.E_DOCUMENT_READY_LIST}$",
    ):
        lifecycle._document_ready()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize("failure_at", [1, 2])
def test_document_ready_clock_failures_have_exact_phase(
    failure_at: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle, _ = _document_ready_lifecycle(
        monkeypatch,
        tmp_path,
        (
            _document_entry_with_state("target", DocumentIngestionState.PROCESSING),
            _document_entry_with_state("other", DocumentIngestionState.PROCESSING),
        ),
    )
    calls = 0

    def monotonic() -> float:
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise RuntimeError("private-secret")
        return 0.0

    monkeypatch.setattr(subject.time, "monotonic", monotonic)
    with pytest.raises(
        subject._DocumentReadyPhaseError,
        match=f"^{subject.E_DOCUMENT_READY_CLOCK}$",
    ):
        lifecycle._document_ready()


def test_document_ready_sleep_failure_has_exact_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pending = (
        _document_entry_with_state("target", DocumentIngestionState.PROCESSING),
        _document_entry_with_state("other", DocumentIngestionState.PROCESSING),
    )
    lifecycle, _ = _document_ready_lifecycle(monkeypatch, tmp_path, pending)
    monkeypatch.setattr(subject.time, "monotonic", iter((0.0, 0.0)).__next__)
    monkeypatch.setattr(
        subject.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(RuntimeError("private-secret")),
    )
    with pytest.raises(
        subject._DocumentReadyPhaseError,
        match=f"^{subject.E_DOCUMENT_READY_SLEEP}$",
    ):
        lifecycle._document_ready()


def test_document_ready_exact_deadline_times_out_without_registry_or_sleep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle, registry = _document_ready_lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subject.time,
        "monotonic",
        iter((10.0, 10.0 + subject.DOCUMENT_POLL_TIMEOUT_SECONDS)).__next__,
    )
    monkeypatch.setattr(subject.time, "sleep", lambda _seconds: pytest.fail("slept"))
    with pytest.raises(
        subject._DocumentReadyPhaseError,
        match=f"^{subject.E_DOCUMENT_READY_TIMEOUT}$",
    ):
        lifecycle._document_ready()
    assert registry.calls == 0


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (
            subject._DocumentReadyPhaseError(subject.E_DOCUMENT_READY_TARGET_FAILED),
            subject.E_DOCUMENT_READY_TARGET_FAILED,
        ),
        (RuntimeError("private-secret"), subject.E_DOCUMENT_READY_UNCLASSIFIED),
    ],
)
def test_document_ready_public_phase_and_cleanup_precedence_are_secret_safe(
    raised: BaseException,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    cleanup_count = 0

    class Operations(FakeOperations):
        def run_phase(self, phase: str) -> None:
            if phase == "E_DOCUMENT_READY":
                raise raised

        def cleanup(self) -> None:
            nonlocal cleanup_count
            cleanup_count += 1
            raise RuntimeError("cleanup-private-secret")

    with pytest.raises(subject.DashboardRAGVLLME2EError) as caught:
        subject.run(
            preflight_only=False,
            environment=values,
            operations_factory=lambda _config: Operations(),
        )
    assert caught.value.phase == expected
    monkeypatch.setattr(
        subject,
        "run",
        lambda *, preflight_only: (_ for _ in ()).throw(caught.value),
    )

    assert subject.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == f"{expected}\n"
    assert captured.err == ""
    assert "private-secret" not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err
    assert cleanup_count == 1


class _DuplicateRegistry:
    def __init__(
        self,
        before: tuple[DocumentRegistryEntry, ...],
        after: tuple[DocumentRegistryEntry, ...],
    ) -> None:
        self.before = before
        self.after = after
        self.list_calls = 0

    def list_documents(self, **_kwargs: object) -> tuple[DocumentRegistryEntry, ...]:
        self.list_calls += 1
        return self.before if self.list_calls == 1 else self.after

    def get_entry(
        self, *, document_id: str, **_kwargs: object
    ) -> DocumentRegistryEntry | None:
        entries = self.before if self.list_calls == 1 else self.after
        return next(
            (item for item in entries if item.document.document_id == document_id),
            None,
        )


class _DuplicateManager:
    def __init__(self, result: DocumentSubmissionResult | None = None) -> None:
        self.result = result or DocumentSubmissionResult(
            DocumentSubmissionStatus.ACCEPTED
        )
        self.calls = 0

    def submit(self, **_kwargs: object) -> DocumentSubmissionResult:
        self.calls += 1
        return self.result


class _FailingDuplicateManager(_DuplicateManager):
    def submit(self, **_kwargs: object) -> DocumentSubmissionResult:
        raise RuntimeError("secret-like-submit-value")


def _duplicate_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    after: tuple[DocumentRegistryEntry, ...] | None = None,
    manager: _DuplicateManager | None = None,
    vector_count: int = 1,
) -> tuple[subject._ProductionLifecycle, _DuplicateRegistry, _DuplicateManager]:
    lifecycle = _completion_lifecycle(monkeypatch, tmp_path)
    target = _ready_document_entry("target")
    other = _ready_document_entry("other")
    registry = _DuplicateRegistry((target, other), after or (target, other))
    actual_manager = manager or _DuplicateManager()
    lifecycle._document_runtime = cast(
        subject.PostgreSQLDocumentIngestionRuntime,
        SimpleNamespace(
            tenant_id=target.document.tenant_id,
            knowledge_base_id=target.document.knowledge_base_id,
            registry=registry,
            manager=actual_manager,
        ),
    )
    lifecycle._target_entry = target
    lifecycle._other_entry = other
    lifecycle._vector_count = 1
    vector_counts = iter((1, vector_count))
    monkeypatch.setattr(lifecycle, "_target_vector_count", lambda: next(vector_counts))
    return lifecycle, registry, actual_manager


def test_duplicate_replay_is_synchronous_and_does_not_poll(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle, registry, manager = _duplicate_lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(subject.time, "sleep", lambda _seconds: pytest.fail("polled"))

    lifecycle._duplicate()

    assert registry.list_calls == 2
    assert manager.calls == 1


@pytest.mark.parametrize(
    ("operation", "phase"),
    [
        ("before_list", subject.E_DUPLICATE_BEFORE_LIST),
        ("after_list", subject.E_DUPLICATE_AFTER_LIST),
        ("target_lookup", subject.E_DUPLICATE_TARGET_LOOKUP),
        ("other_lookup", subject.E_DUPLICATE_OTHER_LOOKUP),
        ("vector_query", subject.E_DUPLICATE_VECTOR_QUERY),
    ],
)
def test_duplicate_reviewed_helper_failures_have_exact_phases(
    operation: str,
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lifecycle, registry, _ = _duplicate_lifecycle(monkeypatch, tmp_path)
    original_list = registry.list_documents
    original_get = registry.get_entry
    if operation in {"before_list", "after_list"}:
        calls = 0

        def list_documents(**kwargs: object) -> tuple[DocumentRegistryEntry, ...]:
            nonlocal calls
            calls += 1
            if calls == (1 if operation == "before_list" else 2):
                raise RuntimeError("private-secret")
            return original_list(**kwargs)

        registry.list_documents = list_documents
    elif operation in {"target_lookup", "other_lookup"}:
        calls = 0

        def get_entry(**kwargs: object) -> DocumentRegistryEntry | None:
            nonlocal calls
            calls += 1
            if calls == (1 if operation == "target_lookup" else 2):
                raise RuntimeError("private-secret")
            document_id = kwargs.get("document_id")
            assert isinstance(document_id, str)
            return original_get(document_id=document_id)

        registry.get_entry = get_entry
    else:
        monkeypatch.setattr(
            lifecycle,
            "_target_vector_count",
            lambda: (_ for _ in ()).throw(RuntimeError("private-secret")),
        )

    with pytest.raises(subject._DuplicatePhaseError, match=f"^{phase}$"):
        lifecycle._duplicate()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize(
    "operation",
    [
        "before_list",
        "after_list",
        "target_lookup",
        "other_lookup",
        "submit",
        "vector_count",
    ],
)
def test_duplicate_malformed_helper_shapes_have_fixed_phase(
    operation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle, registry, manager = _duplicate_lifecycle(monkeypatch, tmp_path)
    original_list = registry.list_documents
    original_get = registry.get_entry
    if operation in {"before_list", "after_list"}:
        calls = 0

        def malformed_list(**kwargs: object) -> object:
            nonlocal calls
            calls += 1
            if calls == (1 if operation == "before_list" else 2):
                return list(original_list(**kwargs))
            return original_list(**kwargs)

        monkeypatch.setattr(registry, "list_documents", malformed_list)
    elif operation in {"target_lookup", "other_lookup"}:
        calls = 0

        def malformed_get(**kwargs: object) -> object:
            nonlocal calls
            calls += 1
            if calls == (1 if operation == "target_lookup" else 2):
                return object()
            document_id = kwargs.get("document_id")
            assert isinstance(document_id, str)
            return original_get(document_id=document_id)

        monkeypatch.setattr(registry, "get_entry", malformed_get)
    elif operation == "submit":
        monkeypatch.setattr(manager, "submit", lambda **_kwargs: object())
    else:
        counts = iter((1, -1))
        monkeypatch.setattr(lifecycle, "_target_vector_count", lambda: next(counts))

    with pytest.raises(
        subject._DuplicatePhaseError, match=f"^{subject.E_DUPLICATE_RESULT_SHAPE}$"
    ):
        lifecycle._duplicate()


@pytest.mark.parametrize("snapshot", ["before", "after"])
def test_duplicate_rejects_non_entry_tuple_members(
    snapshot: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle, registry, _ = _duplicate_lifecycle(monkeypatch, tmp_path)
    malformed = cast(tuple[DocumentRegistryEntry, ...], (object(), registry.before[1]))
    if snapshot == "before":
        registry.before = malformed
    else:
        registry.after = malformed
    with pytest.raises(
        subject._DuplicatePhaseError, match=f"^{subject.E_DUPLICATE_RESULT_SHAPE}$"
    ):
        lifecycle._duplicate()


def test_duplicate_rejects_malformed_submission_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    malformed = DocumentSubmissionResult(cast(DocumentSubmissionStatus, "accepted"))
    lifecycle, _, _ = _duplicate_lifecycle(
        monkeypatch, tmp_path, manager=_DuplicateManager(malformed)
    )
    with pytest.raises(
        subject._DuplicatePhaseError, match=f"^{subject.E_DUPLICATE_RESULT_SHAPE}$"
    ):
        lifecycle._duplicate()


def test_duplicate_uses_fresh_before_snapshots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle, _, _ = _duplicate_lifecycle(monkeypatch, tmp_path)
    assert lifecycle._target_entry is not None
    assert lifecycle._other_entry is not None
    lifecycle._target_entry = lifecycle._target_entry.model_copy(
        update={
            "document": lifecycle._target_entry.document.model_copy(
                update={"byte_size": 999}
            )
        }
    )
    lifecycle._other_entry = lifecycle._other_entry.model_copy(
        update={
            "job": lifecycle._other_entry.job.model_copy(update={"attempt_count": 2})
        }
    )

    lifecycle._duplicate()


class _VectorCountCursor:
    def __init__(
        self,
        row: object,
        *,
        execute_failure: bool = False,
        fetch_failure: bool = False,
    ) -> None:
        self.row = row
        self.execute_failure = execute_failure
        self.fetch_failure = fetch_failure
        self.executed = 0

    def __enter__(self) -> _VectorCountCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, *_args: object) -> None:
        self.executed += 1
        if self.execute_failure:
            raise RuntimeError("private-secret")

    def fetchone(self) -> object:
        if self.fetch_failure:
            raise RuntimeError("private-secret")
        return self.row


class _VectorCountConnection:
    def __init__(
        self,
        row: object,
        *,
        execute_failure: bool = False,
        cursor_failure: bool = False,
        fetch_failure: bool = False,
        close_failure: bool = False,
    ) -> None:
        self.cursor_instance = _VectorCountCursor(
            row, execute_failure=execute_failure, fetch_failure=fetch_failure
        )
        self.cursor_failure = cursor_failure
        self.close_failure = close_failure
        self.closed = 0

    def cursor(self) -> _VectorCountCursor:
        if self.cursor_failure:
            raise RuntimeError("private-secret")
        return self.cursor_instance

    def close(self) -> None:
        self.closed += 1
        if self.close_failure:
            raise RuntimeError("private-secret")


def test_target_vector_count_runs_production_shaped_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import psycopg

    lifecycle, _, _ = _duplicate_lifecycle(monkeypatch, tmp_path)
    monkeypatch.delattr(lifecycle, "_target_vector_count")
    connection = _VectorCountConnection((3,))
    monkeypatch.setattr(lifecycle, "_application_dsn", lambda: "private-dsn")
    monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: connection)

    assert lifecycle._target_vector_count() == 3
    assert connection.cursor_instance.executed == 1
    assert connection.closed == 1


@pytest.mark.parametrize("row", [None, (), (1, 2), (True,), (-1,), ("1",)])
def test_target_vector_count_rejects_malformed_production_rows(
    row: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import psycopg

    lifecycle, _, _ = _duplicate_lifecycle(monkeypatch, tmp_path)
    monkeypatch.delattr(lifecycle, "_target_vector_count")
    connection = _VectorCountConnection(row)
    monkeypatch.setattr(lifecycle, "_application_dsn", lambda: "private-dsn")
    monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: connection)
    with pytest.raises(subject._DuplicateResultShapeError):
        lifecycle._target_vector_count()
    assert connection.closed == 1


@pytest.mark.parametrize(
    "boundary", ["dsn", "connect", "cursor", "execute", "fetchone", "close"]
)
def test_duplicate_real_vector_query_boundaries_have_exact_phase(
    boundary: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import psycopg

    lifecycle, _, _ = _duplicate_lifecycle(monkeypatch, tmp_path)
    monkeypatch.delattr(lifecycle, "_target_vector_count")
    if boundary == "dsn":
        monkeypatch.setattr(
            lifecycle,
            "_application_dsn",
            lambda: (_ for _ in ()).throw(RuntimeError("private-secret")),
        )
    else:
        monkeypatch.setattr(lifecycle, "_application_dsn", lambda: "private-dsn")
    connection = _VectorCountConnection(
        (1,),
        cursor_failure=boundary == "cursor",
        execute_failure=boundary == "execute",
        fetch_failure=boundary == "fetchone",
        close_failure=boundary == "close",
    )
    if boundary == "connect":
        monkeypatch.setattr(
            psycopg,
            "connect",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("private-secret")
            ),
        )
    else:
        monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: connection)

    with pytest.raises(
        subject._DuplicatePhaseError,
        match=f"^{subject.E_DUPLICATE_VECTOR_QUERY}$",
    ):
        lifecycle._duplicate()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_duplicate_real_vector_row_shape_has_exact_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import psycopg

    lifecycle, _, _ = _duplicate_lifecycle(monkeypatch, tmp_path)
    monkeypatch.delattr(lifecycle, "_target_vector_count")
    monkeypatch.setattr(lifecycle, "_application_dsn", lambda: "private-dsn")
    monkeypatch.setattr(
        psycopg,
        "connect",
        lambda *_args, **_kwargs: _VectorCountConnection(("malformed",)),
    )
    with pytest.raises(
        subject._DuplicatePhaseError,
        match=f"^{subject.E_DUPLICATE_RESULT_SHAPE}$",
    ):
        lifecycle._duplicate()


def test_duplicate_missing_runtime_and_submit_failure_have_fixed_phases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lifecycle = _completion_lifecycle(monkeypatch, tmp_path)
    with pytest.raises(
        subject._DuplicatePhaseError, match=f"^{subject.E_DUPLICATE_RUNTIME}$"
    ):
        lifecycle._duplicate()
    (tmp_path / "submit").mkdir()
    lifecycle, _, _ = _duplicate_lifecycle(
        monkeypatch, tmp_path / "submit", manager=_FailingDuplicateManager()
    )
    with pytest.raises(
        subject._DuplicatePhaseError, match=f"^{subject.E_DUPLICATE_SUBMIT}$"
    ):
        lifecycle._duplicate()


@pytest.mark.parametrize(
    "phase",
    [
        subject.E_DUPLICATE_DOCUMENT_IDENTITY,
        subject.E_DUPLICATE_JOB_IDENTITY,
        subject.E_DUPLICATE_READINESS,
        subject.E_DUPLICATE_REGISTRY_CARDINALITY,
        subject.E_DUPLICATE_VECTOR_CARDINALITY,
        subject.E_DUPLICATE_OTHER_DOCUMENT,
        subject.E_DUPLICATE_SOURCE_KEY,
    ],
)
def test_duplicate_invariants_have_exact_fixed_phases(
    phase: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = _ready_document_entry("target")
    other = _ready_document_entry("other")
    after: tuple[DocumentRegistryEntry, ...] = (target, other)
    vector_count = 1
    if phase == subject.E_DUPLICATE_DOCUMENT_IDENTITY:
        after = (
            target.model_copy(
                update={
                    "document": target.document.model_copy(update={"byte_size": 11})
                }
            ),
            other,
        )
    elif phase == subject.E_DUPLICATE_JOB_IDENTITY:
        after = (
            target.model_copy(
                update={"job": target.job.model_copy(update={"attempt_count": 2})}
            ),
            other,
        )
    elif phase == subject.E_DUPLICATE_READINESS:
        after = (
            target.model_copy(update={"readiness": DocumentReadiness.PENDING}),
            other,
        )
    elif phase == subject.E_DUPLICATE_REGISTRY_CARDINALITY:
        after = (target,)
    elif phase == subject.E_DUPLICATE_VECTOR_CARDINALITY:
        vector_count = 2
    elif phase == subject.E_DUPLICATE_OTHER_DOCUMENT:
        after = (
            target,
            other.model_copy(
                update={"document": other.document.model_copy(update={"byte_size": 11})}
            ),
        )
    elif phase == subject.E_DUPLICATE_SOURCE_KEY:
        after = (
            target.model_copy(
                update={
                    "document": target.document.model_copy(
                        update={"storage_object_key": "objects/server-key"}
                    )
                }
            ),
            other,
        )
    lifecycle, _, _ = _duplicate_lifecycle(
        monkeypatch, tmp_path, after=after, vector_count=vector_count
    )

    with pytest.raises(subject._DuplicatePhaseError, match=f"^{phase}$"):
        lifecycle._duplicate()


def test_duplicate_rejection_and_query_failure_are_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rejected = _DuplicateManager(
        DocumentSubmissionResult(DocumentSubmissionStatus.BUSY)
    )
    lifecycle, registry, _ = _duplicate_lifecycle(
        monkeypatch, tmp_path, manager=rejected
    )
    with pytest.raises(
        subject._DuplicatePhaseError, match=f"^{subject.E_DUPLICATE_STATUS}$"
    ):
        lifecycle._duplicate()
    registry.list_documents = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("secret-like-value")
    )
    with pytest.raises(
        subject._DuplicatePhaseError,
        match=f"^{subject.E_DUPLICATE_BEFORE_LIST}$",
    ):
        lifecycle._duplicate()
    captured = capsys.readouterr()
    assert "secret-like-value" not in captured.out + captured.err


def test_duplicate_subphase_remains_primary_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)

    class Operations(FakeOperations):
        def run_phase(self, phase: str) -> None:
            if phase == "E_DUPLICATE":
                raise subject._DuplicatePhaseError(subject.E_DUPLICATE_STATUS)

    operations = Operations(cleanup_failure=True)
    with pytest.raises(
        subject.DashboardRAGVLLME2EError,
        match=f"^{subject.E_DUPLICATE_STATUS}$",
    ):
        subject.run(
            preflight_only=False,
            environment=environment(tmp_path),
            operations_factory=lambda _config: operations,
        )


class FakeServiceProcess:
    def __init__(self, *, poll_result: int | None = None, return_code: int = 0) -> None:
        self.signals: list[int] = []
        self.waits: list[float | None] = []
        self.poll_result = poll_result
        self.return_code = return_code
        self.pid = 123
        self.stdout: BytesIO | None = None

    def poll(self) -> int | None:
        return self.poll_result

    def send_signal(self, signal_number: int) -> None:
        self.signals.append(signal_number)

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        self.poll_result = self.return_code
        return self.return_code


class SynchronousThread:
    def __init__(
        self,
        *,
        target: object,
        args: tuple[object, ...],
        name: str,
        daemon: bool,
    ) -> None:
        self._target = cast(object, target)
        self._args = args
        self.name = name
        self.daemon = daemon

    def start(self) -> None:
        target = cast(object, self._target)
        assert callable(target)
        target(*self._args)

    def join(self, timeout: float | None = None) -> None:
        assert timeout is not None

    def is_alive(self) -> bool:
        return False


def test_postgres_start_uses_safe_child_pipe_and_requires_ready_handoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess()
    process.stdout = BytesIO(b"PR54 PostgreSQL TLS READY; TTL remaining: 300 seconds\n")
    captured: dict[str, object] = {}

    def popen(arguments: list[str], **kwargs: object) -> FakeServiceProcess:
        captured["arguments"] = arguments
        captured.update(kwargs)
        return process

    monkeypatch.setattr(subject.subprocess, "Popen", popen)
    monkeypatch.setattr(subject.threading, "Thread", SynchronousThread)
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "docker")
    monkeypatch.setattr(
        "scripts.run_postgres_tls_service.snapshot_protected_resources",
        lambda _docker: {
            "container": frozenset(),
            "network": frozenset(),
            "volume": frozenset(),
        },
    )
    ownership_calls = 0

    def discover() -> dict[int, int]:
        nonlocal ownership_calls
        ownership_calls += 1
        return {process.pid: 1}

    monkeypatch.setattr(lifecycle, "_discover_marker_processes", discover)
    handoff = (
        Path(values[subject.HANDOFF_ROOT_ENV]) / "callmetric-postgres-tls-abcdefgh"
    )
    monotonic_calls = 0

    def monotonic() -> float:
        nonlocal monotonic_calls
        monotonic_calls += 1
        if monotonic_calls == 2:
            handoff.mkdir()
            (handoff / "application.dsn").write_text("private", encoding="utf-8")
        return float(monotonic_calls)

    monkeypatch.setattr(subject.time, "monotonic", monotonic)

    lifecycle._postgres_start()

    assert lifecycle._handoff == handoff
    assert captured["cwd"] == subject.REPOSITORY_ROOT
    assert captured["stdout"] is subprocess.PIPE
    assert captured["stderr"] is subprocess.STDOUT
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["shell"] is False
    assert captured["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP
    child_environment = cast(dict[str, str], captured["env"])
    assert (
        child_environment["CALLMETRIC_POSTGRES_TLS_SERVICE_EXPECTED_BRANCH"] == BRANCH
    )
    assert child_environment["CALLMETRIC_POSTGRES_TLS_SERVICE_EXPECTED_HEAD"] == HEAD
    assert subject.POSTGRES_CHILD_PHASES
    arguments = cast(list[str], captured["arguments"])
    marker_index = arguments.index("--owner-marker")
    assert subject.OWNER_MARKER_PATTERN.fullmatch(arguments[marker_index + 1])
    assert ownership_calls == 1


def test_postgres_start_propagates_recognized_child_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess(poll_result=1, return_code=1)
    process.stdout = BytesIO(b"E_TLS PR54 PostgreSQL TLS service failed\n")
    monkeypatch.setattr(subject.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(subject.threading, "Thread", SynchronousThread)
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "docker")
    monkeypatch.setattr(
        "scripts.run_postgres_tls_service.snapshot_protected_resources",
        lambda _docker: {
            "container": frozenset(),
            "network": frozenset(),
            "volume": frozenset(),
        },
    )
    monkeypatch.setattr(lifecycle, "_windows_process_table", lambda: {process.pid: 1})

    with pytest.raises(
        subject._PostgresChildError, match=f"^{subject.E_POSTGRES_CHILD_TLS}$"
    ):
        lifecycle._postgres_start()


def test_postgres_ownership_initialization_failure_is_distinct_after_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess()
    process.stdout = BytesIO(b"PR54 PostgreSQL TLS READY; TTL remaining: 300 seconds\n")
    monkeypatch.setattr(subject.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(subject.threading, "Thread", SynchronousThread)
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "docker")
    monkeypatch.setattr(
        "scripts.run_postgres_tls_service.snapshot_protected_resources",
        lambda _docker: {
            "container": frozenset(),
            "network": frozenset(),
            "volume": frozenset(),
        },
    )
    handoff = (
        Path(values[subject.HANDOFF_ROOT_ENV]) / "callmetric-postgres-tls-abcdefgh"
    )
    monotonic_calls = 0

    def monotonic() -> float:
        nonlocal monotonic_calls
        monotonic_calls += 1
        if monotonic_calls == 2:
            handoff.mkdir()
            (handoff / "application.dsn").write_text("private", encoding="utf-8")
        return float(monotonic_calls)

    monkeypatch.setattr(subject.time, "monotonic", monotonic)
    monkeypatch.setattr(
        lifecycle,
        "_windows_process_table",
        lambda: (_ for _ in ()).throw(RuntimeError("private-wmi-detail")),
    )

    with pytest.raises(
        subject._PostgresStartupError, match=f"^{subject.E_POSTGRES_OWNERSHIP}$"
    ):
        lifecycle._postgres_start()

    assert lifecycle._handoff == handoff


def _marker_observation(
    *,
    process_id: int,
    parent_process_id: int,
    marker: str,
    executable_path: str | None = None,
    creation_time_utc: str = "2026-08-10T12:00:01+00:00",
) -> subject._WindowsProcessObservation:
    return subject._WindowsProcessObservation(
        process_id=process_id,
        parent_process_id=parent_process_id,
        executable_path=executable_path or subject.sys.executable,
        command_line=(
            f'python -m scripts.run_postgres_tls_service --owner-marker "{marker}" '
            "--ttl-seconds 300"
        ),
        creation_time_utc=creation_time_utc,
    )


def _windows_process_observations_from_payload(
    monkeypatch: pytest.MonkeyPatch, payload: object
) -> tuple[subject._WindowsProcessObservation, ...]:
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "powershell.exe")
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, stdout=json.dumps(payload).encode("utf-8"), stderr=b""
        ),
    )
    return subject._ProductionLifecycle._windows_process_observations()


def test_windows_process_observations_use_bounded_binary_utf8_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def run(
        arguments: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        captured.update(kwargs)
        captured["command"] = arguments[-1]
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=json.dumps(
                _wmi_row(1, 0, executable_path="C:/sentetik/çalıştırıcı.exe")
            ).encode("utf-8"),
            stderr=b"",
        )

    monkeypatch.setattr(subject.shutil, "which", lambda _name: "powershell.exe")
    monkeypatch.setattr(subject.subprocess, "run", run)

    observations = subject._ProductionLifecycle._windows_process_observations()

    assert observations[0].process_id == 1
    assert captured["text"] is False
    assert captured["capture_output"] is True
    assert "UTF8Encoding" in str(captured["command"])


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        (b"\x81", b""),
        (b"{}", b"\x81private-secret"),
    ],
)
def test_windows_process_observations_reject_malformed_or_oversized_bytes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stdout: bytes,
    stderr: bytes,
) -> None:
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "powershell.exe")
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, stdout=stdout, stderr=stderr
        ),
    )

    with pytest.raises(RuntimeError):
        subject._ProductionLifecycle._windows_process_observations()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert "private-secret" not in captured.out + captured.err


@pytest.mark.parametrize("oversized_stream", ["stdout", "stderr"])
def test_windows_process_observations_reject_oversized_output(
    monkeypatch: pytest.MonkeyPatch, oversized_stream: str
) -> None:
    stdout = b"{}"
    stderr = b""
    oversized = b"x" * (subject._MAX_WMI_OUTPUT_BYTES + 1)
    if oversized_stream == "stdout":
        stdout = oversized
    else:
        stderr = oversized
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "powershell.exe")
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, stdout=stdout, stderr=stderr
        ),
    )
    with pytest.raises(RuntimeError):
        subject._ProductionLifecycle._windows_process_observations()


def test_windows_process_observations_reject_nonzero_exit_without_output_leak(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "powershell.exe")

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, ["powershell"], b"\x81", b"secret")

    monkeypatch.setattr(subject.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        subject._ProductionLifecycle._windows_process_observations()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_malformed_wmi_bytes_emit_fixed_ownership_phase_and_cleanup_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    cleanup_count = 0
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "powershell.exe")
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, stdout=b"\x81", stderr=b"private-secret"
        ),
    )

    class Operations:
        def run_phase(self, phase: str) -> None:
            assert phase == "E_POSTGRES_START"
            subject._ProductionLifecycle._postgres_startup_call(
                subject.E_POSTGRES_OWNERSHIP,
                subject._ProductionLifecycle._windows_process_observations,
            )

        def cleanup(self) -> None:
            nonlocal cleanup_count
            cleanup_count += 1

    run_postgres_startup_only = subject.run_postgres_startup_only
    monkeypatch.setattr(
        subject,
        "run_postgres_startup_only",
        lambda: run_postgres_startup_only(
            environment=values, operations_factory=lambda _config: Operations()
        ),
    )

    assert subject.main(["--postgres-startup-only"]) == 1
    captured = capsys.readouterr()
    assert captured.out == f"{subject.E_POSTGRES_OWNERSHIP}\n"
    assert captured.err == ""
    assert "private-secret" not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err
    assert cleanup_count == 1


def _wmi_row(
    process_id: object,
    parent_process_id: object,
    *,
    executable_path: object = None,
    command_line: object = None,
    creation_date: object = None,
) -> dict[str, object]:
    return {
        "ProcessId": process_id,
        "ParentProcessId": parent_process_id,
        "ExecutablePath": executable_path,
        "CommandLine": command_line,
        "CreationDate": creation_date,
    }


def test_windows_process_observations_omit_canonical_pid_zero_and_keep_python_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "callmetric-owner-" + "a" * 32
    base_executable = getattr(subject.sys, "_base_executable", subject.sys.executable)
    payload = [
        _wmi_row(0, 0),
        _wmi_row(
            6052,
            30240,
            executable_path=subject.sys.executable,
            command_line=f"python --owner-marker {marker}",
            creation_date="2026-08-10T12:00:01+00:00",
        ),
        _wmi_row(
            32448,
            6052,
            executable_path=base_executable,
            command_line=f"python --owner-marker {marker}",
            creation_date="2026-08-10T12:00:02+00:00",
        ),
    ]

    observations = _windows_process_observations_from_payload(monkeypatch, payload)

    assert tuple(item.process_id for item in observations) == (6052, 32448)


def test_windows_process_observations_canonical_pid_zero_alone_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _windows_process_observations_from_payload(monkeypatch, _wmi_row(0, 0)) == ()


@pytest.mark.parametrize(
    "payload",
    [
        _wmi_row(0, 1),
        _wmi_row(-1, 0),
        [_wmi_row(0, 0), _wmi_row(0, 0)],
        _wmi_row(0, 0, executable_path=1),
        _wmi_row(0, 0, command_line=[]),
        _wmi_row(0, 0, creation_date=False),
        {"ProcessId": 0, "ParentProcessId": 0},
        _wmi_row("0", 0),
    ],
)
def test_windows_process_observations_reject_malformed_pid_zero_rows(
    monkeypatch: pytest.MonkeyPatch, payload: object
) -> None:
    with pytest.raises(RuntimeError):
        _windows_process_observations_from_payload(monkeypatch, payload)


def test_postgres_startup_only_accepts_pid_zero_and_cleans_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    secret_like_text = "private-command-line-token"
    payload = [
        _wmi_row(0, 0),
        _wmi_row(
            6052,
            1,
            executable_path=subject.sys.executable,
            command_line=secret_like_text,
            creation_date="2026-08-10T12:00:01+00:00",
        ),
    ]
    cleanup_count = 0

    class StartupOnlyOperations:
        def run_phase(self, phase: str) -> None:
            assert phase == "E_POSTGRES_START"
            observations = _windows_process_observations_from_payload(
                monkeypatch, payload
            )
            assert tuple(item.process_id for item in observations) == (6052,)

        def cleanup(self) -> None:
            nonlocal cleanup_count
            cleanup_count += 1

    run_postgres_startup_only = subject.run_postgres_startup_only
    monkeypatch.setattr(
        subject,
        "run_postgres_startup_only",
        lambda: run_postgres_startup_only(
            environment=values,
            operations_factory=lambda _config: StartupOnlyOperations(),
        ),
    )

    assert subject.main(["--postgres-startup-only"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "POSTGRES_STARTUP_OK\n"
    assert captured.err == ""
    assert secret_like_text not in captured.out
    assert secret_like_text not in captured.err
    assert cleanup_count == 1


def test_marker_ownership_finds_reparented_tls_interpreters_without_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    marker = "callmetric-owner-" + "a" * 32
    lifecycle._owner_marker = marker
    lifecycle._service = cast(subprocess.Popen[bytes], FakeServiceProcess())
    lifecycle._postgres_launch_boundary = subject.datetime.fromisoformat(
        "2026-08-10T12:00:00+00:00"
    )
    monkeypatch.setattr(
        lifecycle,
        "_windows_process_observations",
        lambda: (
            _marker_observation(
                process_id=6052, parent_process_id=30240, marker=marker
            ),
            _marker_observation(
                process_id=32448, parent_process_id=6052, marker=marker
            ),
        ),
    )
    monkeypatch.setattr(
        lifecycle,
        "_parse_windows_command_line",
        lambda line: tuple(line.replace('"', "").split()),
    )

    assert lifecycle._discover_marker_processes() == {6052: 30240, 32448: 6052}
    lifecycle._refresh_owned_process_ledger()

    assert lifecycle._owned_processes == {6052: 30240, 32448: 6052}


def test_marker_ownership_preserves_unmarked_and_different_marker_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    marker = "callmetric-owner-" + "a" * 32
    lifecycle._owner_marker = marker
    lifecycle._postgres_launch_boundary = subject.datetime.fromisoformat(
        "2026-08-10T12:00:00+00:00"
    )
    observations = (
        _marker_observation(process_id=101, parent_process_id=1, marker=marker),
        _marker_observation(
            process_id=102,
            parent_process_id=1,
            marker="callmetric-owner-" + "b" * 32,
        ),
        subject._WindowsProcessObservation(
            103,
            1,
            subject.sys.executable,
            "python -m scripts.run_postgres_tls_service --ttl-seconds 300",
            "2026-08-10T12:00:01+00:00",
        ),
    )
    monkeypatch.setattr(
        lifecycle, "_windows_process_observations", lambda: observations
    )
    monkeypatch.setattr(
        lifecycle,
        "_parse_windows_command_line",
        lambda line: tuple(line.replace('"', "").split()),
    )

    assert lifecycle._discover_marker_processes() == {101: 1}


@pytest.mark.parametrize(
    ("executable_path", "creation_time"),
    [
        ("C:/not-reviewed/python.exe", "2026-08-10T12:00:01+00:00"),
        (None, "2026-08-10T11:59:59+00:00"),
    ],
)
def test_marker_ownership_rejects_invalid_executable_or_creation_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    executable_path: str | None,
    creation_time: str,
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    marker = "callmetric-owner-" + "a" * 32
    lifecycle._owner_marker = marker
    lifecycle._postgres_launch_boundary = subject.datetime.fromisoformat(
        "2026-08-10T12:00:00+00:00"
    )
    observation = _marker_observation(
        process_id=101,
        parent_process_id=1,
        marker=marker,
        executable_path=executable_path,
        creation_time_utc=creation_time,
    )
    monkeypatch.setattr(
        lifecycle, "_windows_process_observations", lambda: (observation,)
    )
    monkeypatch.setattr(
        lifecycle,
        "_parse_windows_command_line",
        lambda line: tuple(line.replace('"', "").split()),
    )

    with pytest.raises(subject._CleanupPhaseError):
        lifecycle._discover_marker_processes()


def test_marker_candidate_with_unparsable_command_line_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    marker = "callmetric-owner-" + "a" * 32
    lifecycle._owner_marker = marker
    lifecycle._postgres_launch_boundary = subject.datetime.fromisoformat(
        "2026-08-10T12:00:00+00:00"
    )
    monkeypatch.setattr(
        lifecycle,
        "_windows_process_observations",
        lambda: (
            _marker_observation(process_id=101, parent_process_id=1, marker=marker),
        ),
    )
    monkeypatch.setattr(
        lifecycle,
        "_parse_windows_command_line",
        lambda _line: (_ for _ in ()).throw(ValueError("private-command-line")),
    )

    with pytest.raises(subject._CleanupPhaseError):
        lifecycle._discover_marker_processes()


class FailingChildStream(BytesIO):
    def readline(self, size: int | None = -1) -> bytes:
        raise OSError("private-reader-detail")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            b"E_TLS PR54 PostgreSQL TLS service failed\n",
            subject.E_POSTGRES_CHILD_TLS,
        ),
        (b"private-unknown-child-line\n", subject.E_POSTGRES_CHILD_UNCLASSIFIED),
        (b"\xff\n", subject.E_POSTGRES_CHILD_UNCLASSIFIED),
        (
            b"E_TLS PR54 PostgreSQL TLS service failed\n"
            b"E_STARTUP PR54 PostgreSQL TLS service failed\n",
            subject.E_POSTGRES_CHILD_UNCLASSIFIED,
        ),
        (b"", subject.E_POSTGRES_CHILD_UNCLASSIFIED),
        (
            b"PR54 PostgreSQL TLS READY; TTL remaining: 300 seconds\n",
            subject.E_POSTGRES_CHILD_READY_EXIT,
        ),
    ],
)
def test_public_main_classifies_buffered_child_output_before_absent_root_lookup(
    payload: bytes,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    config = subject.preflight(values)
    lifecycle = subject._ProductionLifecycle(config, values)
    process = FakeServiceProcess(poll_result=1, return_code=1)
    process.stdout = BytesIO(payload)
    cleanup_calls = 0

    def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        raise RuntimeError("private-cleanup-detail")

    monkeypatch.setattr(subject, "_preflight", lambda _environment=None: config)
    monkeypatch.setattr(
        subject, "_ProductionLifecycle", lambda _config, _environment: lifecycle
    )
    monkeypatch.setattr(subject.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(subject.threading, "Thread", SynchronousThread)
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "docker")
    monkeypatch.setattr(
        "scripts.run_postgres_tls_service.snapshot_protected_resources",
        lambda _docker: {
            "container": frozenset(),
            "network": frozenset(),
            "volume": frozenset(),
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "_windows_process_table",
        lambda: pytest.fail("exited child must be classified before WMI lookup"),
    )
    monkeypatch.setattr(lifecycle, "cleanup", cleanup)

    assert subject.main([]) == 1

    captured = capsys.readouterr()
    assert captured.out.strip() == expected
    assert captured.err == ""
    assert "private" not in captured.out
    assert cleanup_calls == 1


def test_public_main_classifies_reader_failure_before_absent_root_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    config = subject.preflight(values)
    lifecycle = subject._ProductionLifecycle(config, values)
    process = FakeServiceProcess(poll_result=1, return_code=1)
    process.stdout = FailingChildStream()
    cleanup_calls = 0

    def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    monkeypatch.setattr(subject, "_preflight", lambda _environment=None: config)
    monkeypatch.setattr(
        subject, "_ProductionLifecycle", lambda _config, _environment: lifecycle
    )
    monkeypatch.setattr(subject.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(subject.threading, "Thread", SynchronousThread)
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "docker")
    monkeypatch.setattr(
        "scripts.run_postgres_tls_service.snapshot_protected_resources",
        lambda _docker: {
            "container": frozenset(),
            "network": frozenset(),
            "volume": frozenset(),
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "_windows_process_table",
        lambda: pytest.fail("reader failure must be classified before WMI lookup"),
    )
    monkeypatch.setattr(lifecycle, "cleanup", cleanup)

    assert subject.main([]) == 1

    captured = capsys.readouterr()
    assert captured.out.strip() == subject.E_POSTGRES_CHILD_UNCLASSIFIED
    assert captured.err == ""
    assert "private-reader-detail" not in captured.out
    assert cleanup_calls == 1


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
    ) -> subprocess.CompletedProcess[bytes]:
        commands.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, stdout=b"", stderr=b"")

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
            arguments, 0, stdout=b"", stderr=b""
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
    ) -> subprocess.CompletedProcess[bytes]:
        output = "residue" if arguments[1] == resource else ""
        return subprocess.CompletedProcess(
            arguments, 0, stdout=output.encode("ascii"), stderr=b""
        )

    monkeypatch.setattr(subject.subprocess, "run", run)
    _fake_process_tables(monkeypatch, lifecycle, process)
    monkeypatch.setattr(lifecycle, "_cleanup_exact_postgres_project", lambda: None)
    monkeypatch.setattr(lifecycle, "_cleanup_exact_handoff", lambda: None)
    with pytest.raises(
        subject._CleanupPhaseError,
        match=f"^{subject.E_CLEANUP_PROJECT_VERIFY}$",
    ):
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
            arguments, 0, stdout=b"", stderr=b""
        ),
    )
    _fake_process_tables(monkeypatch, lifecycle, process)
    monkeypatch.setattr(lifecycle, "_cleanup_exact_postgres_project", lambda: None)
    monkeypatch.setattr(lifecycle, "_cleanup_exact_handoff", lambda: None)
    with pytest.raises(
        subject._CleanupPhaseError,
        match=f"^{subject.E_CLEANUP_HANDOFF_VERIFY}$",
    ):
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


def test_process_action_race_is_recovered_by_authoritative_final_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    process = FakeServiceProcess(return_code=1)
    lifecycle._service = cast(subprocess.Popen[bytes], process)
    _fake_process_tables(monkeypatch, lifecycle, process)
    actions: list[str] = []

    def process_race(*_args: object) -> None:
        actions.append("process")
        process.poll_result = 0
        raise subject._CleanupPhaseError(subject.E_CLEANUP_PROCESS_ACTION)

    monkeypatch.setattr(lifecycle, "_terminate_owned_process_tree", process_race)
    monkeypatch.setattr(
        lifecycle,
        "_cleanup_exact_postgres_project",
        lambda: actions.append("project-disappeared"),
    )
    monkeypatch.setattr(
        lifecycle,
        "_cleanup_exact_handoff",
        lambda: actions.append("handoff-disappeared"),
    )
    monkeypatch.setattr(lifecycle, "_require_postgres_residue_absent", lambda: None)

    lifecycle.cleanup()

    assert actions == ["process", "project-disappeared", "handoff-disappeared"]
    assert lifecycle._service is None


def test_internal_lifecycle_close_failure_is_never_recovered_by_external_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)

    class FailingManager:
        def close(self, *, wait: bool) -> None:
            assert wait is False
            raise RuntimeError("private-manager-close-detail")

    lifecycle._rag_manager = cast(subject.BoundedPostgreSQLRAGManager, FailingManager())
    monkeypatch.setattr(
        lifecycle, "_require_protected_resources_unchanged", lambda: None
    )

    with pytest.raises(RuntimeError):
        lifecycle.cleanup()


def test_handoff_root_or_sibling_mutation_is_final_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    root = Path(values[subject.HANDOFF_ROOT_ENV])
    lifecycle._protected_handoff_entries = frozenset(root.iterdir())
    (root / "unexpected-sibling").mkdir()

    with pytest.raises(
        subject._CleanupPhaseError,
        match=f"^{subject.E_CLEANUP_HANDOFF_VERIFY}$",
    ):
        lifecycle._require_handoff_root_unchanged()


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
        lambda: (_ for _ in ()).throw(
            subject._CleanupPhaseError(subject.E_CLEANUP_PROTECTED_VERIFY)
        ),
    )

    with pytest.raises(
        subject._CleanupPhaseError,
        match=f"^{subject.E_CLEANUP_PROTECTED_VERIFY}$",
    ):
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
            arguments, 0, stdout=b"", stderr=b""
        ),
    )

    with pytest.raises(
        subject._CleanupPhaseError,
        match=f"^{subject.E_CLEANUP_PROCESS_VERIFY}$",
    ):
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
    ) -> subprocess.CompletedProcess[bytes]:
        terminated.append(int(arguments[2]))
        return subprocess.CompletedProcess(arguments, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subject.subprocess, "run", run)
    lifecycle._terminate_owned_process_tree(
        cast(subprocess.Popen[bytes], process),
        {100: 1, 200: 100, 300: 200, 900: 1},
    )
    assert terminated == [300, 200, 100]


def test_marker_cleanup_rediscovers_and_terminates_reparented_processes_deepest_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_preflight(monkeypatch, tmp_path)
    values = environment(tmp_path)
    lifecycle = subject._ProductionLifecycle(subject.preflight(values), values)
    lifecycle._owner_marker = "callmetric-owner-" + "a" * 32
    lifecycle._postgres_launch_boundary = subject.datetime.fromisoformat(
        "2026-08-10T12:00:00+00:00"
    )
    process = FakeServiceProcess(poll_result=0)
    active = {6052: 30240, 32448: 6052}
    monkeypatch.setattr(lifecycle, "_discover_marker_processes", lambda: dict(active))
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "taskkill")
    terminated: list[int] = []

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        process_id = int(arguments[2])
        terminated.append(process_id)
        active.pop(process_id)
        if process_id == 32448:
            active[777] = 6052
        return subprocess.CompletedProcess(arguments, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subject.subprocess, "run", run)

    lifecycle._terminate_marker_processes(cast(subprocess.Popen[bytes], process))

    assert terminated == [32448, 6052, 777]


def test_windows_command_line_parser_preserves_exact_marker_argument() -> None:
    marker = "callmetric-owner-" + "a" * 32
    parsed = subject._ProductionLifecycle._parse_windows_command_line(
        f'"{subject.sys.executable}" -m scripts.run_postgres_tls_service '
        f'--owner-marker "{marker}" --ttl-seconds 300'
    )

    marker_index = parsed.index("--owner-marker")
    assert parsed[marker_index + 1] == marker


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
    ) -> subprocess.CompletedProcess[bytes]:
        terminated.append(int(arguments[2]))
        return subprocess.CompletedProcess(arguments, 0, stdout=b"", stderr=b"")

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
