from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread

import pytest

from app.coaching.coordinator import (
    CoachingProcessingStatus,
    StableCoachingOutcome,
)
from app.composition import BoundedPostgreSQLRAGManager
from app.events.models import (
    ClassificationResultEvent,
    CoachingAction,
    SuggestionPriority,
    TranscriptEvent,
)
from app.integration import (
    RAGCoachingIntegrationDependencies,
    RAGCoachingIntegrationPolicy,
)
from live_dashboard.runtime_wiring import (
    DashboardExecutionIdentity,
    DashboardExecutionResource,
    DashboardExecutionResourceRegistry,
    SnapshotPublisher,
    wait_for_live_cadence,
)
from live_dashboard.demo_data import tenant_demos
from live_dashboard.uploaded_audio import temporary_uploaded_audio
from live_dashboard.view_models import (
    DashboardExecutionSnapshot,
    DashboardExecutionStatus,
    create_local_execution,
    execution_snapshot,
)
from app.streaming.pipeline import StreamingASRStep

NOW = datetime(2026, 7, 29, tzinfo=UTC)


class FakeBackgroundManager(BoundedPostgreSQLRAGManager):
    def __init__(self) -> None:
        self.close_calls: list[bool] = []

    def close(self, *, wait: bool = False) -> None:
        self.close_calls.append(wait)


@dataclass
class FakeCompletionPump:
    outcomes: tuple[StableCoachingOutcome, ...]
    drain_calls: list[float] = field(default_factory=list)

    def process_safely(
        self,
        event: TranscriptEvent,
        current_seconds: float,
        *,
        classification_event: ClassificationResultEvent | None = None,
        active_labels: tuple[str, ...] | None = None,
    ) -> StableCoachingOutcome:
        del event, current_seconds, classification_event, active_labels
        raise AssertionError("base processing must not run")

    def drain_completed(
        self,
        *,
        current_seconds: float,
    ) -> tuple[StableCoachingOutcome, ...]:
        self.drain_calls.append(current_seconds)
        return self.outcomes


def integration(
    manager: BoundedPostgreSQLRAGManager,
) -> RAGCoachingIntegrationDependencies:
    return RAGCoachingIntegrationDependencies(
        background_manager=manager,
        policy=RAGCoachingIntegrationPolicy(
            rag_llm_enabled_labels=("product_information",),
            title="Synthetic guidance",
            action=CoachingAction.RAG_ACTION,
            priority=SuggestionPriority.HIGH,
            label_id="product_information",
            expires_after_seconds=30.0,
        ),
        suggestion_id_factory=lambda: "synthetic-suggestion",
        utc_datetime_factory=lambda: NOW,
    )


def identity(
    tenant_id: str = "tenant_alpha",
    call_id: str = "call_001",
) -> DashboardExecutionIdentity:
    return DashboardExecutionIdentity(tenant_id, call_id)


def test_identity_is_canonical_and_opaque() -> None:
    actual = DashboardExecutionIdentity(" tenant_alpha ", " call_001 ")

    assert actual == identity()
    assert len(actual.opaque_key) == 64
    assert "tenant_alpha" not in actual.opaque_key
    assert "call_001" not in actual.opaque_key


@pytest.mark.parametrize("capacity", [True, False, 0, -1, 1.0])
def test_registry_capacity_is_strict(capacity: object) -> None:
    with pytest.raises(ValueError, match="capacity"):
        DashboardExecutionResourceRegistry(capacity=capacity)  # type: ignore[arg-type]


def test_same_scope_rerun_returns_exact_resource() -> None:
    registry = DashboardExecutionResourceRegistry(capacity=2)

    first = registry.acquire(
        tenant_id="tenant_alpha",
        call_id="call_001",
        integration=None,
    )
    second = registry.acquire(
        tenant_id="tenant_alpha",
        call_id="call_001",
        integration=None,
    )

    assert second is first
    assert (
        registry.lookup(
            first.opaque_key,
            tenant_id="tenant_alpha",
            call_id="call_001",
        )
        is first
    )


def test_scope_isolation_and_wrong_scope_lookup_fail_closed() -> None:
    registry = DashboardExecutionResourceRegistry(capacity=2)
    first = registry.acquire(
        tenant_id="tenant_alpha",
        call_id="call_001",
        integration=None,
    )
    second = registry.acquire(
        tenant_id="tenant_beta",
        call_id="call_002",
        integration=None,
    )

    assert first is not second
    with pytest.raises(ValueError, match="scope"):
        registry.lookup(
            first.opaque_key,
            tenant_id="tenant_beta",
            call_id="call_002",
        )


def test_capacity_rejection_is_immediate_and_deterministic() -> None:
    registry = DashboardExecutionResourceRegistry(capacity=1)
    registry.acquire(
        tenant_id="tenant_alpha",
        call_id="call_001",
        integration=None,
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="capacity"):
            registry.acquire(
                tenant_id="tenant_beta",
                call_id="call_002",
                integration=None,
            )


def test_integration_cannot_be_replaced_for_retained_scope() -> None:
    registry = DashboardExecutionResourceRegistry(capacity=1)
    registry.acquire(
        tenant_id="tenant_alpha",
        call_id="call_001",
        integration=None,
    )

    with pytest.raises(RuntimeError, match="cannot be replaced"):
        registry.acquire(
            tenant_id="tenant_alpha",
            call_id="call_001",
            integration=integration(FakeBackgroundManager()),
        )


def test_late_completion_remains_explicitly_reachable() -> None:
    expected = (
        StableCoachingOutcome(
            status=CoachingProcessingStatus.PROCESSED,
            transcript_revision=3,
        ),
    )
    resource = DashboardExecutionResource(identity(), integration=None)
    pump = FakeCompletionPump(expected)
    resource.bind_completion_pump(pump)

    actual = resource.drain_completed(current_seconds=4.5)

    assert actual is expected
    assert pump.drain_calls == [4.5]


@pytest.mark.parametrize(
    "outcomes",
    [
        (),
        (
            StableCoachingOutcome(
                status=CoachingProcessingStatus.FAILED,
                transcript_revision=2,
                error_type="rag_orchestration",
                error_code="background_failure",
            ),
        ),
    ],
)
def test_empty_and_failed_completion_shapes_are_preserved(
    outcomes: tuple[StableCoachingOutcome, ...],
) -> None:
    resource = DashboardExecutionResource(identity(), integration=None)
    resource.bind_completion_pump(FakeCompletionPump(outcomes))

    assert resource.drain_completed(current_seconds=2.0) is outcomes


def test_base_only_resource_has_no_provider_or_pump_side_effect() -> None:
    resource = DashboardExecutionResource(identity(), integration=None)

    assert resource.integration is None
    assert resource.drain_completed(current_seconds=0.0) == ()


def test_close_is_idempotent_and_closes_manager_exactly_once() -> None:
    manager = FakeBackgroundManager()
    resource = DashboardExecutionResource(
        identity(),
        integration=integration(manager),
    )

    resource.close()
    resource.close()

    assert resource.closed
    assert manager.close_calls == [False]
    with pytest.raises(RuntimeError, match="closed"):
        resource.drain_completed(current_seconds=0.0)


def test_concurrent_close_calls_manager_exactly_once() -> None:
    manager = FakeBackgroundManager()
    resource = DashboardExecutionResource(
        identity(),
        integration=integration(manager),
    )
    threads = tuple(Thread(target=resource.close) for _ in range(8))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert resource.closed
    assert manager.close_calls == [False]


def test_closed_resource_cannot_be_reused_until_explicit_removal() -> None:
    registry = DashboardExecutionResourceRegistry(capacity=1)
    resource = registry.acquire(
        tenant_id="tenant_alpha",
        call_id="call_001",
        integration=None,
    )
    resource.close()

    with pytest.raises(RuntimeError, match="closed"):
        registry.acquire(
            tenant_id="tenant_alpha",
            call_id="call_001",
            integration=None,
        )
    registry.remove(tenant_id="tenant_alpha", call_id="call_001")
    fresh = registry.acquire(
        tenant_id="tenant_alpha",
        call_id="call_001",
        integration=None,
    )

    assert fresh is not resource


def test_open_resource_must_not_be_removed_or_replaced() -> None:
    registry = DashboardExecutionResourceRegistry(capacity=1)
    resource = registry.acquire(
        tenant_id="tenant_alpha",
        call_id="call_001",
        integration=None,
    )

    with pytest.raises(RuntimeError, match="closed"):
        registry.remove(tenant_id="tenant_alpha", call_id="call_001")

    registry.close_and_remove(resource.opaque_key)
    fresh = registry.acquire(
        tenant_id="tenant_alpha",
        call_id="call_001",
        integration=None,
    )
    assert fresh is not resource


def test_resource_and_registry_do_not_retain_payload_fields() -> None:
    resource = DashboardExecutionResource(identity(), integration=None)
    registry = DashboardExecutionResourceRegistry(capacity=1)
    registered = registry.acquire(
        tenant_id="tenant_alpha",
        call_id="call_001",
        integration=None,
    )

    forbidden = {
        "audio",
        "transcript",
        "prompt",
        "generated_text",
        "dsn",
        "token",
        "exception",
    }
    assert forbidden.isdisjoint(vars(resource))
    assert forbidden.isdisjoint(vars(registry))
    assert registered.opaque_key == identity().opaque_key


def _snapshot(
    revision: int,
    status: DashboardExecutionStatus,
    *,
    completed: int,
) -> DashboardExecutionSnapshot:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "call_001")
    state.status = "running"
    state.total_chunks = 2
    state.current_chunk = completed
    return execution_snapshot(
        state,
        revision=revision,
        lifecycle_status=status,
    )


def test_worker_publishes_first_chunk_before_terminal_and_retains_latest_only() -> None:
    resource = DashboardExecutionResource(identity(), integration=None)
    first_published = Event()
    release = Event()

    def task(
        cancel: Event,
        publish: SnapshotPublisher,
    ) -> DashboardExecutionSnapshot:
        del cancel
        publish(_snapshot(1, DashboardExecutionStatus.RUNNING, completed=1))
        publish(_snapshot(1, DashboardExecutionStatus.RUNNING, completed=1))
        first_published.set()
        assert release.wait(timeout=2)
        return _snapshot(2, DashboardExecutionStatus.COMPLETED, completed=2)

    assert resource.start_worker(
        _snapshot(0, DashboardExecutionStatus.RUNNING, completed=0),
        task,
    )
    assert first_published.wait(timeout=2)
    first = resource.latest_snapshot
    assert first is not None
    assert (first.revision, first.processed_chunks) == (1, 1)
    assert first.lifecycle_status is DashboardExecutionStatus.RUNNING
    assert not resource.start_worker(
        _snapshot(0, DashboardExecutionStatus.RUNNING, completed=0),
        task,
    )
    release.set()
    resource.join_worker()
    terminal = resource.latest_snapshot
    assert terminal is not None
    assert terminal.lifecycle_status is DashboardExecutionStatus.COMPLETED
    assert resource.diagnostics.published_snapshots == 3
    assert resource.diagnostics.rejected_snapshots == 1
    assert not hasattr(resource, "_snapshots")


def test_worker_failure_is_sanitized_and_terminal_is_published_once() -> None:
    resource = DashboardExecutionResource(identity(), integration=None)

    def task(
        _cancel: Event,
        _publish: SnapshotPublisher,
    ) -> DashboardExecutionSnapshot:
        raise RuntimeError("private path and provider response")

    resource.start_worker(
        _snapshot(0, DashboardExecutionStatus.RUNNING, completed=0),
        task,
    )
    resource.join_worker()
    snapshot = resource.latest_snapshot
    assert snapshot is not None
    assert snapshot.lifecycle_status is DashboardExecutionStatus.FAILED
    assert snapshot.failure_reason == "processing_failed"
    assert "private path" not in repr(snapshot)


def test_close_cancels_worker_then_closes_manager_exactly_once() -> None:
    manager = FakeBackgroundManager()
    resource = DashboardExecutionResource(identity(), integration=integration(manager))
    entered = Event()

    def task(
        cancel: Event,
        _publish: SnapshotPublisher,
    ) -> DashboardExecutionSnapshot:
        entered.set()
        cancel.wait()
        raise RuntimeError("cancelled")

    resource.start_worker(
        _snapshot(0, DashboardExecutionStatus.RUNNING, completed=0),
        task,
    )
    assert entered.wait(timeout=2)
    resource.close()
    resource.close()
    assert resource.latest_snapshot is not None
    assert (
        resource.latest_snapshot.lifecycle_status is DashboardExecutionStatus.CANCELLED
    )
    assert manager.close_calls == [False]


def test_live_cadence_uses_injected_clock_and_wait_without_sleep() -> None:
    step = StreamingASRStep(
        tenant_id="tenant_alpha",
        call_id="call_001",
        sequence_number=0,
        chunk_start_seconds=0.0,
        chunk_end_seconds=2.0,
        window_start_seconds=0.0,
        window_end_seconds=2.0,
        window_duration_seconds=2.0,
        raw_window_text="",
        transcript_events=(),
        stable_transcript="",
        partial_transcript="",
        transcription_time_seconds=0.25,
    )
    waits: list[float] = []

    assert not wait_for_live_cadence(
        step,
        started_at=10.0,
        realtime=True,
        clock=lambda: 10.5,
        cancellation_wait=lambda delay: waits.append(delay) or False,
    )
    assert waits == [1.5]
    waits.clear()
    assert not wait_for_live_cadence(
        step,
        started_at=10.0,
        realtime=False,
        clock=lambda: 0.0,
        cancellation_wait=lambda delay: waits.append(delay) or False,
    )
    assert waits == []


def test_worker_keeps_temporary_upload_until_processing_is_released() -> None:
    resource = DashboardExecutionResource(identity(), integration=None)
    entered = Event()
    release = Event()
    observed_paths: list[Path] = []

    def task(
        _cancel: Event,
        _publish: SnapshotPublisher,
    ) -> DashboardExecutionSnapshot:
        with temporary_uploaded_audio("synthetic.wav", b"synthetic") as path:
            observed_paths.append(path)
            entered.set()
            assert release.wait(timeout=2)
            assert path.exists()
        return _snapshot(1, DashboardExecutionStatus.COMPLETED, completed=0)

    resource.start_worker(
        _snapshot(0, DashboardExecutionStatus.RUNNING, completed=0),
        task,
    )
    assert entered.wait(timeout=2)
    assert observed_paths[0].exists()
    release.set()
    resource.join_worker()
    assert not observed_paths[0].exists()
