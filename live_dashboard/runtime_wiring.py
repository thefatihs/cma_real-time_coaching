"""Dashboard-only construction of existing classification and coaching services."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from threading import Event, Lock, Thread, current_thread

from app.calls.models import CallState
from app.classification.artifacts import load_training_metadata, sha256_file
from app.classification.calibration import sha256_directory
from app.classification.dataset import load_classification_taxonomy
from app.classification.encoding import MultiLabelEncoder
from app.classification.runtime import (
    DEFAULT_MODEL_DIR,
    DEFAULT_TAXONOMY,
    DEFAULT_THRESHOLD_PROFILE,
    RuntimeSetFitClassifier,
)
from app.classification.threshold_profiles import load_threshold_profile
from app.coaching.coordinator import CoachingCoordinator
from app.coaching.rule_engine import RuleBasedCoachingEngine
from app.integration import (
    CoachingCompletionPumpProtocol,
    RAGCoachingIntegrationDependencies,
    compose_rag_coaching_processor,
)
from app.coaching.coordinator import StableCoachingOutcome
from app.streaming.pipeline import (
    CoachingProcessorProtocol,
    StreamingASRPipeline,
    StreamingASRStep,
    WindowTranscriberProtocol,
)
from live_dashboard.demo_data import TenantDemo
from live_dashboard.view_models import (
    DashboardExecutionSnapshot,
    DashboardExecutionStage,
    DashboardExecutionStatus,
    DashboardRuntime,
)


@dataclass(frozen=True, slots=True)
class ArtifactAvailability:
    compatible: bool
    safe_message: str | None = None


@dataclass(frozen=True, slots=True)
class DashboardServiceSelection:
    enable_setfit: bool
    enable_coaching: bool


ClassifierProvider = Callable[[], RuntimeSetFitClassifier]
SnapshotPublisher = Callable[[DashboardExecutionSnapshot], None]
DashboardWorkerTask = Callable[
    [Event, SnapshotPublisher],
    DashboardExecutionSnapshot,
]
MonotonicClock = Callable[[], float]
CancellationWait = Callable[[float], bool]


@dataclass(frozen=True, slots=True)
class DashboardExecutionDiagnostics:
    published_snapshots: int
    rejected_snapshots: int
    worker_starts: int


def wait_for_live_cadence(
    step: StreamingASRStep,
    *,
    started_at: float,
    realtime: bool,
    clock: MonotonicClock,
    cancellation_wait: CancellationWait,
) -> bool:
    """Pace against audio time; return true when cancellation interrupts."""
    if not realtime:
        return False
    target = started_at + step.chunk_end_seconds
    delay = max(target - clock(), 0.0)
    return cancellation_wait(delay)


@dataclass(frozen=True, slots=True)
class DashboardExecutionIdentity:
    tenant_id: str
    call_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tenant_id", _required_text(self.tenant_id, "tenant_id")
        )
        object.__setattr__(self, "call_id", _required_text(self.call_id, "call_id"))

    @property
    def opaque_key(self) -> str:
        value = f"{self.tenant_id}\0{self.call_id}".encode()
        return sha256(value).hexdigest()


class DashboardExecutionResource:
    """Own one call-scoped dashboard pipeline and optional completion pump."""

    def __init__(
        self,
        identity: DashboardExecutionIdentity,
        *,
        integration: RAGCoachingIntegrationDependencies | None,
    ) -> None:
        if not isinstance(identity, DashboardExecutionIdentity):
            raise ValueError("identity must be DashboardExecutionIdentity")
        if integration is not None and not isinstance(
            integration,
            RAGCoachingIntegrationDependencies,
        ):
            raise ValueError(
                "integration must be RAGCoachingIntegrationDependencies or None"
            )
        self._identity = identity
        self._integration = integration
        self._manager = None if integration is None else integration.background_manager
        self._pipeline: StreamingASRPipeline | None = None
        self._completion_pump: CoachingCompletionPumpProtocol | None = None
        self._worker: Thread | None = None
        self._cancellation = Event()
        self._latest_snapshot: DashboardExecutionSnapshot | None = None
        self._terminal_snapshot: DashboardExecutionSnapshot | None = None
        self._published_snapshots = 0
        self._rejected_snapshots = 0
        self._worker_starts = 0
        self._lock = Lock()
        self._closed = False
        self._manager_closed = False

    @property
    def identity(self) -> DashboardExecutionIdentity:
        return self._identity

    @property
    def opaque_key(self) -> str:
        return self._identity.opaque_key

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def integration(self) -> RAGCoachingIntegrationDependencies | None:
        return self._integration

    @property
    def latest_snapshot(self) -> DashboardExecutionSnapshot | None:
        with self._lock:
            return self._latest_snapshot

    @property
    def worker_active(self) -> bool:
        with self._lock:
            worker = self._worker
            return worker is not None and worker.is_alive()

    @property
    def diagnostics(self) -> DashboardExecutionDiagnostics:
        with self._lock:
            return DashboardExecutionDiagnostics(
                published_snapshots=self._published_snapshots,
                rejected_snapshots=self._rejected_snapshots,
                worker_starts=self._worker_starts,
            )

    def start_worker(
        self,
        initial_snapshot: DashboardExecutionSnapshot,
        task: DashboardWorkerTask,
    ) -> bool:
        """Start at most one call worker and retain only its latest snapshot."""
        if not callable(task):
            raise ValueError("dashboard worker task must be callable")
        self._validate_snapshot_scope(initial_snapshot)
        if initial_snapshot.revision != 0:
            raise ValueError("initial snapshot revision must be zero")
        with self._lock:
            self._require_open()
            if self._worker is not None:
                return False
            self._latest_snapshot = initial_snapshot
            self._published_snapshots = 1
            self._cancellation.clear()
            worker = Thread(
                target=self._run_worker,
                args=(task,),
                name=f"dashboard-call-{self.opaque_key[:12]}",
                daemon=True,
            )
            self._worker = worker
            self._worker_starts = 1
            worker.start()
            return True

    def cancel(self) -> None:
        self._cancellation.set()

    def join_worker(self) -> None:
        with self._lock:
            worker = self._worker
        if worker is not None and worker is not current_thread():
            worker.join()

    def _run_worker(self, task: DashboardWorkerTask) -> None:
        try:
            terminal = task(self._cancellation, self._publish_snapshot)
            self._publish_snapshot(terminal)
        except Exception:
            with self._lock:
                latest = self._latest_snapshot
                terminal_exists = self._terminal_snapshot is not None
            if latest is not None and not terminal_exists:
                status = (
                    DashboardExecutionStatus.CANCELLED
                    if self._cancellation.is_set()
                    else DashboardExecutionStatus.FAILED
                )
                self._publish_snapshot(
                    replace(
                        latest,
                        revision=latest.revision + 1,
                        lifecycle_status=status,
                        execution_stage=(
                            DashboardExecutionStage.CANCELLED
                            if status is DashboardExecutionStatus.CANCELLED
                            else DashboardExecutionStage.FAILED
                        ),
                        failure_reason=(
                            None
                            if status is DashboardExecutionStatus.CANCELLED
                            else "processing_failed"
                        ),
                    )
                )

    def _publish_snapshot(self, snapshot: DashboardExecutionSnapshot) -> None:
        self._validate_snapshot_scope(snapshot)
        with self._lock:
            latest = self._latest_snapshot
            if (
                self._terminal_snapshot is not None
                or latest is None
                or snapshot.revision <= latest.revision
                or snapshot.processed_chunks < latest.processed_chunks
            ):
                self._rejected_snapshots += 1
                return
            self._latest_snapshot = snapshot
            self._published_snapshots += 1
            if snapshot.lifecycle_status in {
                DashboardExecutionStatus.COMPLETED,
                DashboardExecutionStatus.CANCELLED,
                DashboardExecutionStatus.FAILED,
            }:
                self._terminal_snapshot = snapshot

    def _validate_snapshot_scope(self, snapshot: DashboardExecutionSnapshot) -> None:
        if not isinstance(snapshot, DashboardExecutionSnapshot):
            raise ValueError("snapshot must be DashboardExecutionSnapshot")
        if (
            snapshot.tenant_id != self._identity.tenant_id
            or snapshot.call_id != self._identity.call_id
        ):
            raise ValueError("dashboard snapshot scope does not match")

    def attach_pipeline(self, pipeline: StreamingASRPipeline) -> None:
        if not isinstance(pipeline, StreamingASRPipeline):
            raise ValueError("pipeline must be StreamingASRPipeline")
        with self._lock:
            self._require_open()
            if self._pipeline is not None and self._pipeline is not pipeline:
                raise RuntimeError("dashboard execution pipeline is already attached")
            self._pipeline = pipeline

    def bind_completion_pump(
        self,
        processor: CoachingProcessorProtocol,
    ) -> None:
        if not isinstance(processor, CoachingCompletionPumpProtocol):
            return
        with self._lock:
            self._require_open()
            if (
                self._completion_pump is not None
                and self._completion_pump is not processor
            ):
                raise RuntimeError("dashboard completion pump is already attached")
            self._completion_pump = processor

    def drain_completed(
        self,
        *,
        current_seconds: float,
    ) -> tuple[StableCoachingOutcome, ...]:
        with self._lock:
            self._require_open()
            pump = self._completion_pump
        if pump is None:
            return ()
        return pump.drain_completed(current_seconds=current_seconds)

    def close(self) -> None:
        self._cancellation.set()
        self.join_worker()
        manager = None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pipeline = None
            self._completion_pump = None
            self._worker = None
            if not self._manager_closed:
                self._manager_closed = True
                manager = self._manager
            self._manager = None
            self._integration = None
        if manager is not None:
            manager.close(wait=False)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("dashboard execution resource is closed")


class DashboardExecutionResourceRegistry:
    """Bounded process-owned lookup for call-scoped execution resources."""

    def __init__(self, *, capacity: int) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        self._resources: dict[str, DashboardExecutionResource] = {}
        self._lock = Lock()

    def acquire(
        self,
        *,
        tenant_id: str,
        call_id: str,
        integration: RAGCoachingIntegrationDependencies | None,
    ) -> DashboardExecutionResource:
        identity = DashboardExecutionIdentity(tenant_id, call_id)
        key = identity.opaque_key
        with self._lock:
            existing = self._resources.get(key)
            if existing is not None:
                if existing.identity != identity:
                    raise ValueError("dashboard execution scope does not match")
                if existing.closed:
                    raise RuntimeError(
                        "closed dashboard execution resource is retained"
                    )
                if existing.integration is not integration:
                    raise RuntimeError(
                        "dashboard execution integration cannot be replaced"
                    )
                return existing
            if len(self._resources) >= self._capacity:
                raise RuntimeError("dashboard execution resource capacity is exhausted")
            resource = DashboardExecutionResource(
                identity,
                integration=integration,
            )
            self._resources[key] = resource
            return resource

    def find(
        self,
        *,
        tenant_id: str,
        call_id: str,
    ) -> DashboardExecutionResource | None:
        """Return an existing open exact-scope resource without creating one."""
        identity = DashboardExecutionIdentity(tenant_id, call_id)
        with self._lock:
            resource = self._resources.get(identity.opaque_key)
            if resource is None:
                return None
            if resource.identity != identity:
                raise ValueError("dashboard execution resource scope does not match")
            if resource.closed:
                raise RuntimeError("closed dashboard execution resource is retained")
            return resource

    def lookup(
        self,
        opaque_key: str,
        *,
        tenant_id: str,
        call_id: str,
    ) -> DashboardExecutionResource:
        key = _required_text(opaque_key, "opaque_key")
        identity = DashboardExecutionIdentity(tenant_id, call_id)
        with self._lock:
            resource = self._resources.get(key)
            if resource is None or resource.identity != identity:
                raise ValueError("dashboard execution resource scope does not match")
            if resource.closed:
                raise RuntimeError("dashboard execution resource is closed")
            return resource

    def remove(
        self,
        *,
        tenant_id: str,
        call_id: str,
    ) -> None:
        identity = DashboardExecutionIdentity(tenant_id, call_id)
        key = identity.opaque_key
        with self._lock:
            resource = self._resources.get(key)
            if resource is None:
                raise ValueError("dashboard execution resource does not exist")
            if resource.identity != identity:
                raise ValueError("dashboard execution resource scope does not match")
            if not resource.closed:
                raise RuntimeError(
                    "dashboard execution resource must be closed before removal"
                )
            self._resources.pop(key)

    def close_and_remove(self, opaque_key: str) -> None:
        key = _required_text(opaque_key, "opaque_key")
        with self._lock:
            resource = self._resources.get(key)
        if resource is None:
            raise ValueError("dashboard execution resource does not exist")
        resource.close()
        with self._lock:
            if self._resources.get(key) is not resource:
                raise RuntimeError("dashboard execution resource changed during close")
            self._resources.pop(key)


def inspect_default_artifacts(
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    threshold_profile: Path = DEFAULT_THRESHOLD_PROFILE,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
) -> ArtifactAvailability:
    """Validate local metadata/profile compatibility without loading model weights."""
    try:
        if not model_dir.is_dir() or not threshold_profile.is_file():
            raise ValueError("required artifact is missing")
        metadata = load_training_metadata(model_dir)
        taxonomy = load_classification_taxonomy(taxonomy_path)
        encoder = MultiLabelEncoder.from_taxonomy(taxonomy)
        if metadata.label_order != encoder.label_order:
            raise ValueError("artifact label order is incompatible")
        taxonomy_checksum = sha256_file(taxonomy_path)
        if metadata.taxonomy_checksum != taxonomy_checksum:
            raise ValueError("artifact taxonomy is stale")
        load_threshold_profile(
            threshold_profile,
            taxonomy=taxonomy,
            metadata=metadata,
            dataset_checksum=metadata.dataset_checksum,
            taxonomy_checksum=taxonomy_checksum,
            model_checksum=sha256_directory(model_dir),
        )
    except Exception:
        return ArtifactAvailability(
            compatible=False,
            safe_message=(
                "SetFit modeli şu anda kullanılamıyor; kural tabanlı koçluk "
                "ve ses işleme devam edecek."
            ),
        )
    return ArtifactAvailability(compatible=True)


def default_service_selection(
    availability: ArtifactAvailability,
    *,
    deterministic_rules_available: bool,
) -> DashboardServiceSelection:
    return DashboardServiceSelection(
        enable_setfit=availability.compatible,
        enable_coaching=availability.compatible or deterministic_rules_available,
    )


def build_live_pipeline(
    runtime: DashboardRuntime,
    window_transcriber: WindowTranscriberProtocol,
    *,
    selection: DashboardServiceSelection,
    availability: ArtifactAvailability,
    classifier_provider: ClassifierProvider,
    integration: RAGCoachingIntegrationDependencies | None = None,
    execution_resource: DashboardExecutionResource | None = None,
) -> StreamingASRPipeline:
    """Build one uploaded-audio pipeline with optional existing services."""
    classifier = (
        classifier_provider()
        if selection.enable_setfit and availability.compatible
        else None
    )
    if execution_resource is not None:
        expected_identity = DashboardExecutionIdentity(
            runtime.tenant.config.context.tenant_id,
            runtime.call_id,
        )
        if execution_resource.identity != expected_identity:
            raise ValueError(
                "dashboard execution resource scope does not match runtime"
            )
        if execution_resource.integration is not integration:
            raise ValueError("dashboard execution resource integration does not match")
    coaching_factory = (
        _coaching_factory(
            runtime.tenant,
            integration=integration,
            execution_resource=execution_resource,
        )
        if selection.enable_coaching
        else None
    )
    runtime.setfit_enabled = classifier is not None
    runtime.coaching_enabled = coaching_factory is not None
    runtime.rule_engine_enabled = coaching_factory is not None
    runtime.classification_failure = (
        selection.enable_setfit and not availability.compatible
    )
    runtime.service_status_message = (
        availability.safe_message if runtime.classification_failure else None
    )
    pipeline = StreamingASRPipeline(
        runtime.tenant.config.context,
        runtime.tenant.config.asr,
        window_transcriber,
        runtime_classifier=classifier,
        coaching_coordinator_factory=coaching_factory,
    )
    if execution_resource is not None:
        execution_resource.attach_pipeline(pipeline)
    return pipeline


def _coaching_factory(
    tenant: TenantDemo,
    *,
    integration: RAGCoachingIntegrationDependencies | None,
    execution_resource: DashboardExecutionResource | None,
) -> Callable[[CallState], CoachingProcessorProtocol]:
    def create(call_state: CallState) -> CoachingProcessorProtocol:
        coordinator = CoachingCoordinator(
            tenant.config,
            call_state,
            RuleBasedCoachingEngine(tenant.config, tenant.rules),
        )
        processor = compose_rag_coaching_processor(
            coordinator=coordinator,
            tenant_config=tenant.config,
            integration=integration,
        )
        if execution_resource is not None:
            execution_resource.bind_completion_pump(processor)
        return processor

    return create


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
