"""Deterministic tests for bounded PostgreSQL RAG background execution."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import inspect
from threading import Event, Lock
import time
from typing import Any, cast

import pytest

import app.composition as composition_exports
import app.composition.postgres_rag_background as background_module
from app.composition.postgres_rag_background import (
    BoundedPostgreSQLRAGManager,
    RAGOrchestrationCompletion,
    RAGOrchestrationCompletionStatus,
    RAGOrchestrationIdentity,
    RAGOrchestrationSubmission,
    RAGOrchestrationSubmissionStatus,
)
from app.composition.postgres_rag_runtime import (
    ProfileVerifiedPostgreSQLRAGRunner,
)
from app.orchestration.models import (
    OrchestrationCitationReference,
    OrchestrationRequest,
    OrchestrationResult,
)


def _request(
    *,
    tenant_id: str = "tenant-synthetic",
    call_id: str = "call-synthetic",
    revision: int = 1,
) -> OrchestrationRequest:
    return OrchestrationRequest(
        tenant_id=tenant_id,
        call_id=call_id,
        transcript_revision=revision,
        knowledge_base_id="kb-synthetic",
        user_input=f"Synthetic question {revision}",
        top_k=2,
        minimum_score=0.25,
    )


def _identity(
    *,
    tenant_id: str = "tenant-synthetic",
    call_id: str = "call-synthetic",
    revision: int = 1,
) -> RAGOrchestrationIdentity:
    return RAGOrchestrationIdentity(
        tenant_id=tenant_id,
        call_id=call_id,
        transcript_revision=revision,
    )


def _result(request: OrchestrationRequest) -> OrchestrationResult:
    return OrchestrationResult(
        tenant_id=request.tenant_id,
        call_id=request.call_id,
        transcript_revision=request.transcript_revision,
        generated_text="Synthetic answer",
        citations=(
            OrchestrationCitationReference(
                document_id="document-b",
                chunk_id="chunk-2",
            ),
            OrchestrationCitationReference(
                document_id="document-a",
                chunk_id="chunk-1",
            ),
        ),
    )


class FakeRunner(ProfileVerifiedPostgreSQLRAGRunner):
    def __init__(
        self,
        *,
        prepare_errors: list[BaseException] | None = None,
        run_error: Exception | None = None,
        result_factory: Any = _result,
        prepare_entered: Event | None = None,
        prepare_release: Event | None = None,
        run_entered: Event | None = None,
        run_release: Event | None = None,
    ) -> None:
        self.prepare_errors = [] if prepare_errors is None else prepare_errors
        self.run_error = run_error
        self.result_factory = result_factory
        self.prepare_entered = prepare_entered
        self.prepare_release = prepare_release
        self.run_entered = run_entered
        self.run_release = run_release
        self.prepare_calls = 0
        self.run_calls: list[OrchestrationRequest] = []
        self._calls_lock = Lock()

    def prepare(self) -> None:
        with self._calls_lock:
            self.prepare_calls += 1
        if self.prepare_entered is not None:
            self.prepare_entered.set()
        if self.prepare_release is not None:
            assert self.prepare_release.wait(timeout=5)
        if self.prepare_errors:
            raise self.prepare_errors.pop(0)

    def run(
        self,
        request: OrchestrationRequest,
    ) -> OrchestrationResult | None:
        with self._calls_lock:
            self.run_calls.append(request)
        if self.run_entered is not None:
            self.run_entered.set()
        if self.run_release is not None:
            assert self.run_release.wait(timeout=5)
        if self.run_error is not None:
            raise self.run_error
        return cast(OrchestrationResult | None, self.result_factory(request))


def _manager(
    runner: FakeRunner | None = None,
    *,
    max_workers: int = 1,
    capacity: int = 2,
) -> tuple[BoundedPostgreSQLRAGManager, FakeRunner]:
    selected = FakeRunner() if runner is None else runner
    return (
        BoundedPostgreSQLRAGManager(
            runner=selected,
            max_workers=max_workers,
            capacity=capacity,
        ),
        selected,
    )


def _announce(
    manager: BoundedPostgreSQLRAGManager,
    *,
    tenant_id: str = "tenant-synthetic",
    call_id: str = "call-synthetic",
    revision: int = 1,
) -> None:
    manager.announce_current_revision(
        tenant_id=tenant_id,
        call_id=call_id,
        transcript_revision=revision,
    )


def _wait_until(predicate: Any) -> None:
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("synthetic completion was not observed")
        time.sleep(0.005)


def _poll_until(
    manager: BoundedPostgreSQLRAGManager,
    identity: RAGOrchestrationIdentity,
) -> RAGOrchestrationCompletion:
    completion: RAGOrchestrationCompletion | None = None

    def completed() -> bool:
        nonlocal completion
        completion = manager.poll(identity)
        return completion is not None

    _wait_until(completed)
    assert completion is not None
    return completion


def test_identity_and_result_models_are_canonical_frozen_and_slotted() -> None:
    identity = RAGOrchestrationIdentity(" tenant ", " call ", 2)
    submission = RAGOrchestrationSubmission(
        identity,
        RAGOrchestrationSubmissionStatus.ACCEPTED,
    )
    completion = RAGOrchestrationCompletion(
        identity,
        RAGOrchestrationCompletionStatus.EMPTY,
    )

    assert identity == RAGOrchestrationIdentity("tenant", "call", 2)
    assert not hasattr(identity, "__dict__")
    assert not hasattr(submission, "__dict__")
    assert not hasattr(completion, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(identity, "transcript_revision", 3)
    assert "result=" not in repr(completion)
    assert "error=" not in repr(completion)


@pytest.mark.parametrize(
    ("tenant_id", "call_id", "revision"),
    [
        ("", "call", 1),
        ("tenant", " ", 1),
        ("tenant", "call", -1),
        ("tenant", "call", True),
    ],
)
def test_identity_validation(
    tenant_id: str,
    call_id: str,
    revision: Any,
) -> None:
    with pytest.raises(ValueError):
        RAGOrchestrationIdentity(tenant_id, call_id, revision)


@pytest.mark.parametrize(
    ("runner", "workers", "capacity"),
    [
        (object(), 1, 1),
        (FakeRunner(), True, 1),
        (FakeRunner(), 0, 1),
        (FakeRunner(), 1, 1.0),
        (FakeRunner(), 2, 1),
    ],
)
def test_constructor_validation(
    runner: object,
    workers: Any,
    capacity: Any,
) -> None:
    with pytest.raises(ValueError):
        BoundedPostgreSQLRAGManager(
            runner=cast(Any, runner),
            max_workers=workers,
            capacity=capacity,
        )


def test_construction_has_zero_threads_or_provider_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions = 0

    def forbidden_executor(**_kwargs: object) -> Any:
        nonlocal constructions
        constructions += 1
        raise AssertionError("executor must not be constructed")

    monkeypatch.setattr(background_module, "ThreadPoolExecutor", forbidden_executor)
    manager, runner = _manager()

    assert manager._runner is runner  # noqa: SLF001
    assert constructions == runner.prepare_calls == 0


def test_start_failure_identity_is_retryable_and_executor_is_deferred() -> None:
    expected = RuntimeError("synthetic preparation failure")
    runner = FakeRunner(prepare_errors=[expected])
    manager, _runner = _manager(runner)

    with pytest.raises(RuntimeError) as raised:
        manager.start()
    assert raised.value is expected
    assert manager.submit(_request()).status is (
        RAGOrchestrationSubmissionStatus.NOT_STARTED
    )

    manager.start()
    assert runner.prepare_calls == 2
    manager.close()


def test_concurrent_start_prepares_and_constructs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = Event()
    release = Event()
    runner = FakeRunner(prepare_entered=entered, prepare_release=release)
    manager, _runner = _manager(runner)
    real_executor = ThreadPoolExecutor
    constructions = 0

    def executor_factory(**kwargs: object) -> ThreadPoolExecutor:
        nonlocal constructions
        constructions += 1
        return real_executor(**cast(Any, kwargs))

    monkeypatch.setattr(
        background_module,
        "ThreadPoolExecutor",
        executor_factory,
    )
    with ThreadPoolExecutor(max_workers=8) as callers:
        starts = tuple(callers.submit(manager.start) for _item in range(16))
        assert entered.wait(timeout=5)
        release.set()
        assert tuple(future.result(timeout=5) for future in starts) == (None,) * 16

    assert runner.prepare_calls == constructions == 1
    manager.close()


def test_submission_before_start_and_after_close_are_explicit() -> None:
    manager, runner = _manager()
    request = _request()
    assert manager.submit(request).status is (
        RAGOrchestrationSubmissionStatus.NOT_STARTED
    )
    manager.close()
    assert manager.submit(request).status is RAGOrchestrationSubmissionStatus.CLOSED
    with pytest.raises(RuntimeError, match="closed"):
        manager.start()
    assert runner.run_calls == []


def test_revision_state_is_independent_equal_idempotent_and_monotonic() -> None:
    manager, _runner = _manager()
    manager.start()
    _announce(manager, tenant_id="tenant-a", call_id="call-a", revision=2)
    _announce(manager, tenant_id="tenant-a", call_id="call-a", revision=2)
    _announce(manager, tenant_id="tenant-b", call_id="call-a", revision=1)
    _announce(manager, tenant_id="tenant-a", call_id="call-b", revision=1)

    with pytest.raises(ValueError, match="cannot decrease"):
        _announce(manager, tenant_id="tenant-a", call_id="call-a", revision=1)
    assert (
        manager.submit(
            _request(tenant_id="tenant-b", call_id="call-a", revision=1)
        ).status
        is RAGOrchestrationSubmissionStatus.ACCEPTED
    )
    manager.close()


def test_duplicate_and_completed_identity_cannot_be_resubmitted() -> None:
    manager, runner = _manager()
    manager.start()
    _announce(manager)
    request = _request()

    assert manager.submit(request).status is RAGOrchestrationSubmissionStatus.ACCEPTED
    assert manager.submit(request).status is RAGOrchestrationSubmissionStatus.DUPLICATE
    completion = _poll_until(manager, _identity())
    assert completion.status is RAGOrchestrationCompletionStatus.SUCCEEDED
    assert manager.submit(request).status is RAGOrchestrationSubmissionStatus.DUPLICATE
    assert runner.run_calls == [request]
    manager.close()


def test_capacity_counts_running_and_rejects_without_blocking() -> None:
    entered = Event()
    release = Event()
    runner = FakeRunner(run_entered=entered, run_release=release)
    manager, _runner = _manager(runner, capacity=1)
    manager.start()
    _announce(manager, call_id="call-a")
    _announce(manager, call_id="call-b")

    assert (
        manager.submit(_request(call_id="call-a")).status
        is RAGOrchestrationSubmissionStatus.ACCEPTED
    )
    assert entered.wait(timeout=5)
    started = time.monotonic()
    rejected = manager.submit(_request(call_id="call-b"))
    assert time.monotonic() - started < 0.25
    assert rejected.status is RAGOrchestrationSubmissionStatus.CAPACITY_REJECTED
    release.set()
    _poll_until(manager, _identity(call_id="call-a"))
    manager.close()


def test_undrained_completion_holds_capacity_until_poll() -> None:
    manager, _runner = _manager(capacity=1)
    manager.start()
    _announce(manager, call_id="call-a")
    _announce(manager, call_id="call-b")
    manager.submit(_request(call_id="call-a"))
    _wait_until(lambda: bool(manager._completions))  # noqa: SLF001

    assert manager.submit(_request(call_id="call-b")).status is (
        RAGOrchestrationSubmissionStatus.CAPACITY_REJECTED
    )
    _poll_until(manager, _identity(call_id="call-a"))
    assert manager.submit(_request(call_id="call-b")).status is (
        RAGOrchestrationSubmissionStatus.ACCEPTED
    )
    manager.close()


def test_queued_older_work_is_cancelled_and_releases_capacity() -> None:
    entered = Event()
    release = Event()
    runner = FakeRunner(run_entered=entered, run_release=release)
    manager, _runner = _manager(runner, max_workers=1, capacity=2)
    manager.start()
    _announce(manager, call_id="call-running")
    _announce(manager, call_id="call-queued")
    manager.submit(_request(call_id="call-running"))
    assert entered.wait(timeout=5)
    manager.submit(_request(call_id="call-queued"))

    _announce(manager, call_id="call-queued", revision=2)
    assert manager.submit(_request(call_id="call-queued", revision=2)).status is (
        RAGOrchestrationSubmissionStatus.ACCEPTED
    )
    release.set()
    _wait_until(lambda: len(runner.run_calls) == 2)
    assert runner.run_calls[1].transcript_revision == 2
    manager.close()


def test_running_stale_result_is_discarded_and_capacity_released() -> None:
    entered = Event()
    release = Event()
    runner = FakeRunner(run_entered=entered, run_release=release)
    manager, _runner = _manager(runner, capacity=1)
    manager.start()
    _announce(manager, revision=1)
    manager.submit(_request(revision=1))
    assert entered.wait(timeout=5)

    _announce(manager, revision=2)
    release.set()
    _wait_until(lambda: manager._reservations == 0)  # noqa: SLF001

    assert manager.poll(_identity(revision=1)) is None
    assert manager.submit(_request(revision=2)).status is (
        RAGOrchestrationSubmissionStatus.ACCEPTED
    )
    manager.close()


@pytest.mark.parametrize(
    ("factory", "status"),
    [
        (_result, RAGOrchestrationCompletionStatus.SUCCEEDED),
        (lambda _request: None, RAGOrchestrationCompletionStatus.EMPTY),
    ],
)
def test_success_and_empty_completion_are_explicit(
    factory: Any,
    status: RAGOrchestrationCompletionStatus,
) -> None:
    runner = FakeRunner(result_factory=factory)
    manager, _runner = _manager(runner)
    manager.start()
    _announce(manager)
    request = _request()
    manager.submit(request)

    completion = _poll_until(manager, _identity())

    assert completion.status is status
    if status is RAGOrchestrationCompletionStatus.SUCCEEDED:
        assert completion.result == _result(request)
    else:
        assert completion.result is None
    assert completion.error is None
    assert manager.poll(_identity()) is None
    manager.close()


def test_exact_result_and_exception_identity_are_delivered_once() -> None:
    request = _request()
    expected_result = _result(request)
    successful = FakeRunner(result_factory=lambda _request: expected_result)
    manager, _runner = _manager(successful)
    manager.start()
    _announce(manager)
    manager.submit(request)
    completion = _poll_until(manager, _identity())
    assert completion.result is expected_result
    manager.close()

    expected_error = RuntimeError("synthetic provider failure")
    failed = FakeRunner(run_error=expected_error)
    manager, _runner = _manager(failed)
    manager.start()
    _announce(manager)
    manager.submit(request)
    failure = _poll_until(manager, _identity())
    assert failure.status is RAGOrchestrationCompletionStatus.FAILED
    assert failure.error is expected_error
    assert manager.poll(_identity()) is None
    manager.close()


def test_poll_unknown_nonready_stale_and_drained_returns_none() -> None:
    manager, _runner = _manager()
    manager.start()
    _announce(manager, revision=2)

    assert manager.poll(_identity(revision=1)) is None
    assert manager.poll(_identity(revision=2)) is None
    assert manager.poll(_identity(call_id="unknown", revision=2)) is None
    manager.close()


def test_newer_revision_cannot_be_overwritten_by_older_exception() -> None:
    entered = Event()
    release = Event()
    expected = RuntimeError("synthetic stale provider failure")
    runner = FakeRunner(
        run_error=expected,
        run_entered=entered,
        run_release=release,
    )
    manager, _runner = _manager(runner)
    manager.start()
    _announce(manager, revision=1)
    manager.submit(_request(revision=1))
    assert entered.wait(timeout=5)

    _announce(manager, revision=2)
    release.set()
    _wait_until(lambda: manager._reservations == 0)  # noqa: SLF001

    assert manager.poll(_identity(revision=1)) is None
    assert manager.poll(_identity(revision=2)) is None
    manager.close()


def test_close_wait_false_discards_late_work_and_is_idempotent() -> None:
    entered = Event()
    release = Event()
    runner = FakeRunner(run_entered=entered, run_release=release)
    manager, _runner = _manager(runner)
    manager.start()
    _announce(manager)
    manager.submit(_request())
    assert entered.wait(timeout=5)

    started = time.monotonic()
    manager.close(wait=False)
    assert time.monotonic() - started < 0.25
    release.set()
    manager.close(wait=False)
    assert manager.poll(_identity()) is None
    assert manager.submit(_request()).status is (
        RAGOrchestrationSubmissionStatus.CLOSED
    )


def test_public_exports_and_no_deferred_features_are_exact() -> None:
    expected = {
        "BoundedPostgreSQLRAGManager": BoundedPostgreSQLRAGManager,
        "RAGOrchestrationCompletion": RAGOrchestrationCompletion,
        "RAGOrchestrationCompletionStatus": RAGOrchestrationCompletionStatus,
        "RAGOrchestrationIdentity": RAGOrchestrationIdentity,
        "RAGOrchestrationSubmission": RAGOrchestrationSubmission,
        "RAGOrchestrationSubmissionStatus": RAGOrchestrationSubmissionStatus,
    }
    for name, value in expected.items():
        assert getattr(composition_exports, name) is value
        assert composition_exports.__all__.count(name) == 1

    source = inspect.getsource(background_module)
    assert "logging" not in source
    assert "retry" not in source.casefold()
    assert "psycopg_pool" not in source
    assert "connection_pool" not in source
    assert "atexit" not in source
