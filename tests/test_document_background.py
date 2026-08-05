"""Bounded document background-manager tests with synthetic collaborators."""

from __future__ import annotations

from datetime import UTC, datetime
from threading import Event
from time import monotonic, sleep
from typing import Any

from app.ingestion.document_background import (
    BoundedDocumentIngestionManager,
    DocumentBackgroundFailure,
    DocumentSubmissionStatus,
)
from app.ingestion.registry_models import (
    DocumentIngestionJob,
    DocumentIngestionPhase,
    DocumentIngestionState,
    DocumentReadiness,
    DocumentRegistryCreateResult,
    DocumentRegistryEntry,
    DocumentRegistryRecord,
)
from app.vector_store.models import VectorBatchWriteResult, VectorRecordIdentity


def _entry(
    request: Any,
    state: DocumentIngestionState,
    *,
    phase: DocumentIngestionPhase | None = None,
) -> DocumentRegistryEntry:
    now = datetime.now(UTC)
    started = None if state is DocumentIngestionState.QUEUED else now
    finished = now if state is DocumentIngestionState.SUCCEEDED else None
    total = 1 if state is DocumentIngestionState.SUCCEEDED else request.total_chunks
    return DocumentRegistryEntry(
        document=DocumentRegistryRecord(
            tenant_id=request.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
            document_id=request.document_id,
            original_filename=request.original_filename,
            media_type=request.media_type,
            byte_size=request.byte_size,
            storage_object_key=request.storage_object_key,
            created_at_utc=now,
            ready_at_utc=now if state is DocumentIngestionState.SUCCEEDED else None,
        ),
        job=DocumentIngestionJob(
            tenant_id=request.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
            job_id=request.job_id,
            document_id=request.document_id,
            state=state,
            phase=phase
            or (
                DocumentIngestionPhase.FINALIZE
                if state is DocumentIngestionState.SUCCEEDED
                else request.initial_phase
            ),
            processed_chunks=total if state is DocumentIngestionState.SUCCEEDED else 0,
            total_chunks=total,
            attempt_count=0 if state is DocumentIngestionState.QUEUED else 1,
            created_at_utc=now,
            started_at_utc=started,
            updated_at_utc=now,
            finished_at_utc=finished,
        ),
        readiness=DocumentReadiness.READY
        if state is DocumentIngestionState.SUCCEEDED
        else DocumentReadiness.PENDING,
    )


class _Storage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.write_calls = 0

    def write(self, content: bytes) -> str:
        self.write_calls += 1
        key = f"obj_{self.write_calls:064x}"
        self.objects[key] = content
        return key

    def read(self, key: str) -> bytes:
        return self.objects[key]

    def delete(self, key: str) -> object:
        self.deleted.append(key)
        self.objects.pop(key, None)
        return object()


class _FailingStorage(_Storage):
    def write(self, content: bytes) -> str:
        raise OSError("synthetic private detail")


class _Registry:
    def __init__(self, *, duplicate: bool = False) -> None:
        self.duplicate = duplicate
        self.request: Any = None
        self.entry: DocumentRegistryEntry | None = None
        self.calls: list[str] = []
        self.cancelled = Event()
        self.finished = Event()
        self.recovery_scopes: list[dict[str, str]] = []

    def fail_interrupted_jobs(self, **scope: str) -> int:
        self.recovery_scopes.append(scope)
        return 0

    def create_or_get(self, request: Any) -> DocumentRegistryCreateResult:
        self.calls.append("registry")
        self.request = request
        if self.duplicate:
            original = request.model_copy(
                update={"storage_object_key": "obj_" + "f" * 64}
            )
            return DocumentRegistryCreateResult(
                entry=_entry(original, DocumentIngestionState.QUEUED), created=False
            )
        self.entry = _entry(request, DocumentIngestionState.QUEUED)
        return DocumentRegistryCreateResult(entry=self.entry, created=True)

    def claim_queued_job(self, **scope: str) -> DocumentRegistryEntry | None:
        self.calls.append("claim")
        if self.cancelled.is_set() or self.entry is None:
            return None
        self.entry = _entry(self.request, DocumentIngestionState.PROCESSING)
        return self.entry

    def update_processing_progress(
        self, *, phase: DocumentIngestionPhase, **scope: object
    ) -> DocumentIngestionJob | None:
        self.calls.append(phase.value)
        return None if self.entry is None else self.entry.job

    def mark_cancelled(self, **scope: object) -> bool:
        self.calls.append("cancel")
        self.cancelled.set()
        self.finished.set()
        return True

    def mark_failed(self, **scope: object) -> bool:
        self.calls.append("failed")
        self.finished.set()
        return True

    def finalize_success(self, *, vector_operation: Any, **scope: object) -> Any:
        cancellation = scope.get("cancellation_requested")
        if callable(cancellation) and cancellation():
            from app.ingestion.registry_models import (
                DocumentOperationPhase,
                DocumentRegistryError,
            )

            raise DocumentRegistryError(DocumentOperationPhase.CANCEL)
        result = vector_operation(object())
        self.entry = _entry(self.request, DocumentIngestionState.SUCCEEDED)
        self.finished.set()
        return result

    def get_entry(self, **scope: str) -> DocumentRegistryEntry | None:
        return self.entry


class _Embedder:
    def __init__(self, gate: Event | None = None) -> None:
        self.calls = 0
        self.gate = gate

    def embed_documents(self, **request: object) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(timeout=2)
        texts = request["texts"]
        return tuple((0.1,) * 384 for _ in texts)  # type: ignore[union-attr]


class _Writer:
    def __init__(self) -> None:
        self.calls = 0

    def admit_batch_in_transaction(self, transaction: object, request: Any) -> Any:
        self.calls += 1
        return VectorBatchWriteResult(
            tenant_id=request.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
            inserted_identities=tuple(
                VectorRecordIdentity(document_id=row.document_id, chunk_id=row.chunk_id)
                for row in request.records
            ),
            unchanged_identities=(),
        )


def _manager(
    registry: _Registry,
    storage: _Storage,
    *,
    capacity: int = 1,
    embedder: _Embedder | None = None,
    writer: _Writer | None = None,
) -> BoundedDocumentIngestionManager:
    return BoundedDocumentIngestionManager(
        tenant_id="tenant-trusted",
        knowledge_base_id="kb-trusted",
        capacity=capacity,
        registry=registry,  # type: ignore[arg-type]
        document_embedder=embedder or _Embedder(),  # type: ignore[arg-type]
        vector_writer=writer or _Writer(),  # type: ignore[arg-type]
        close_timeout_seconds=0.5,
    )


def _submit(manager: BoundedDocumentIngestionManager, token: str = "token-1"):
    return manager.submit(
        submission_token=token,
        content=b"Synthetic source text.",
        original_filename="guide.txt",
        declared_media_type="text/plain",
    )


def test_construction_recovers_interrupted_jobs_once_in_exact_scope() -> None:
    registry = _Registry()
    manager = _manager(registry, _Storage())
    try:
        assert registry.recovery_scopes == [
            {"tenant_id": "tenant-trusted", "knowledge_base_id": "kb-trusted"}
        ]
    finally:
        manager.close(wait=True)


def test_success_uses_no_persistent_storage_and_releases_source() -> None:
    storage = _Storage()
    registry = _Registry()
    manager = _manager(registry, storage)
    try:
        result = _submit(manager)
        assert result.status is DocumentSubmissionStatus.ACCEPTED
        assert registry.finished.wait(timeout=2)
        assert not storage.objects
        assert storage.write_calls == 0
        assert manager.retained_source_bytes == 0
        assert registry.request.storage_object_key is None
        assert registry.calls[:4] == ["registry", "claim", "EXTRACTION", "CHUNKING"]
    finally:
        manager.close(wait=True)


def test_duplicate_removes_only_new_object_and_queues_nothing() -> None:
    storage = _Storage()
    registry = _Registry(duplicate=True)
    manager = _manager(registry, storage)
    try:
        first = _submit(manager)
        second = _submit(manager)
        assert first is second
        assert storage.deleted == []
        assert storage.write_calls == 0
        assert "claim" not in registry.calls
        assert manager.worker_count == 1
    finally:
        manager.close(wait=True)


def test_capacity_is_reserved_across_running_work_and_released() -> None:
    gate = Event()
    storage = _Storage()
    registry = _Registry()
    manager = _manager(registry, storage, embedder=_Embedder(gate))
    try:
        assert _submit(manager).status is DocumentSubmissionStatus.ACCEPTED
        deadline = monotonic() + 2
        while "CHUNKING" not in registry.calls and monotonic() < deadline:
            sleep(0.01)
        busy = _submit(manager, "token-2")
        assert busy.status is DocumentSubmissionStatus.BUSY
        assert busy.failure is DocumentBackgroundFailure.CAPACITY
        gate.set()
        assert registry.finished.wait(timeout=2)
    finally:
        gate.set()
        manager.close(wait=True)


def test_close_is_idempotent_bounded_and_refuses_new_work() -> None:
    manager = _manager(_Registry(), _Storage())
    manager.close(wait=True)
    started = monotonic()
    manager.close(wait=True)
    assert monotonic() - started < 0.2
    result = _submit(manager)
    assert result.status is DocumentSubmissionStatus.CLOSED
    assert result.failure is DocumentBackgroundFailure.CLOSED
    assert manager.worker_count == 0


def test_public_result_contains_no_internal_or_source_values() -> None:
    manager = _manager(_Registry(duplicate=True), _Storage())
    try:
        result = _submit(manager, "opaque-rerun-token")
        rendered = repr(result)
        assert "tenant-trusted" not in rendered
        assert "kb-trusted" not in rendered
        assert "guide.txt" not in rendered
        assert "Synthetic source" not in rendered
        assert "obj_" not in rendered
    finally:
        manager.close(wait=True)


def test_rejected_submission_retains_no_source() -> None:
    registry = _Registry()
    storage = _FailingStorage()
    manager = _manager(registry, storage)
    try:
        result = manager.submit(
            submission_token="token-invalid",
            content=b"",
            original_filename="guide.txt",
            declared_media_type="text/plain",
        )
        assert result.failure is DocumentBackgroundFailure.SUBMISSION
        assert registry.calls == []
        assert manager.retained_source_bytes == 0
        assert storage.write_calls == 0
    finally:
        manager.close(wait=True)


def test_registry_failure_retains_no_source() -> None:
    storage = _Storage()
    registry = _Registry()

    def fail_create(request: Any) -> DocumentRegistryCreateResult:
        raise RuntimeError("synthetic database detail")

    registry.create_or_get = fail_create  # type: ignore[method-assign]
    manager = _manager(registry, storage)
    try:
        result = _submit(manager)
        assert result.failure is DocumentBackgroundFailure.REGISTRY_CREATE
        assert storage.objects == {}
        assert storage.deleted == []
        assert manager.retained_source_bytes == 0
        assert "synthetic" not in repr(result)
    finally:
        manager.close(wait=True)


def test_running_cancellation_after_embedding_prevents_vector_commit() -> None:
    gate = Event()
    storage = _Storage()
    registry = _Registry()
    writer = _Writer()
    manager = _manager(registry, storage, embedder=_Embedder(gate), writer=writer)
    try:
        assert _submit(manager).status is DocumentSubmissionStatus.ACCEPTED
        deadline = monotonic() + 2
        while "EMBEDDING" not in registry.calls and monotonic() < deadline:
            sleep(0.01)
        assert manager.cancel(submission_token="token-1") is True
        gate.set()
        assert registry.finished.wait(timeout=2)
        assert writer.calls == 0
        assert storage.objects == {}
        deadline = monotonic() + 2
        while manager.retained_source_bytes and monotonic() < deadline:
            sleep(0.01)
        assert manager.retained_source_bytes == 0
    finally:
        gate.set()
        manager.close(wait=True)
