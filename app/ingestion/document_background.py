"""Bounded single-worker document ingestion without presentation dependencies."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Condition, Lock, Semaphore, Thread
from uuid import uuid4

from app.embeddings import DocumentEmbedder
from app.ingestion.registry import (
    DocumentRegistryRepository,
    TransactionAwareVectorBatchWriter,
)
from app.ingestion.registry_models import (
    DocumentIngestionPhase,
    DocumentOperationPhase,
    DocumentRegistryCreateRequest,
    DocumentRegistryEntry,
    DocumentRegistryError,
)
from app.ingestion.synchronous_orchestration import (
    SynchronousDocumentIngestionOrchestrator,
)
from app.ingestion.upload_preparation import (
    DEFAULT_UPLOAD_CHUNK_CHARACTERS,
    ValidatedUploadEnvelope,
    prepare_validated_upload_document,
    validate_upload_envelope,
)

_MAX_SUBMISSION_TOKEN_CHARACTERS = 128
_MAX_TOKEN_HISTORY = 256


class DocumentBackgroundFailure(str, Enum):
    SUBMISSION = "SUBMISSION"
    STORAGE = "STORAGE"
    REGISTRY_CREATE = "REGISTRY_CREATE"
    EXTRACTION = "EXTRACTION"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    VECTOR_WRITE = "VECTOR_WRITE"
    FINALIZE = "FINALIZE"
    CANCEL = "CANCEL"
    CAPACITY = "CAPACITY"
    CLOSED = "CLOSED"
    UNAVAILABLE = "UNAVAILABLE"


class DocumentSubmissionStatus(str, Enum):
    ACCEPTED = "accepted"
    BUSY = "busy"
    CLOSED = "closed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class DocumentSubmissionResult:
    status: DocumentSubmissionStatus
    failure: DocumentBackgroundFailure | None = None


@dataclass(frozen=True, slots=True)
class _Work:
    token: str
    document_id: str
    job_id: str
    envelope: ValidatedUploadEnvelope


class BoundedDocumentIngestionManager:
    """Own exactly one worker and at most ``capacity`` accepted operations."""

    def __init__(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        capacity: int,
        registry: DocumentRegistryRepository,
        document_embedder: DocumentEmbedder,
        vector_writer: TransactionAwareVectorBatchWriter,
        expected_vector_dimension: int = 384,
        max_chunk_characters: int = DEFAULT_UPLOAD_CHUNK_CHARACTERS,
        close_timeout_seconds: float = 5.0,
        availability_check: Callable[[], None] | None = None,
    ) -> None:
        if type(capacity) is not int or not 1 <= capacity <= 8:
            raise ValueError("capacity must be between 1 and 8")
        if type(expected_vector_dimension) is not int or expected_vector_dimension <= 0:
            raise ValueError("expected_vector_dimension must be positive")
        if close_timeout_seconds <= 0 or close_timeout_seconds > 30:
            raise ValueError("close_timeout_seconds is outside the allowed range")
        if availability_check is not None and not callable(availability_check):
            raise ValueError("availability_check must be callable")
        self._tenant_id = tenant_id
        self._knowledge_base_id = knowledge_base_id
        self._registry = registry
        self._embedder = document_embedder
        self._vector_writer = vector_writer
        self._expected_dimension = expected_vector_dimension
        self._max_chunk_characters = max_chunk_characters
        self._close_timeout = close_timeout_seconds
        self._availability_check = availability_check
        self._queue: deque[_Work] = deque()
        self._capacity = Semaphore(capacity)
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._submission_lock = Lock()
        self._closed = False
        self._results: dict[str, DocumentSubmissionResult] = {}
        self._cancelled: set[str] = set()
        self._work_by_token: dict[str, _Work] = {}
        recovered = self._registry.fail_interrupted_jobs(
            tenant_id=self._tenant_id,
            knowledge_base_id=self._knowledge_base_id,
        )
        if type(recovered) is not int or recovered < 0:
            raise ValueError("interrupted document recovery is invalid")
        self._worker = Thread(
            target=self._run, name="document-ingestion-worker", daemon=False
        )
        self._worker.start()

    def submit(
        self,
        *,
        submission_token: str,
        content: bytes,
        original_filename: str,
        declared_media_type: str,
    ) -> DocumentSubmissionResult:
        """Persist and register a bounded upload, then queue only new documents."""
        with self._submission_lock:
            return self._submit_once(
                submission_token=submission_token,
                content=content,
                original_filename=original_filename,
                declared_media_type=declared_media_type,
            )

    def _submit_once(
        self,
        *,
        submission_token: str,
        content: bytes,
        original_filename: str,
        declared_media_type: str,
    ) -> DocumentSubmissionResult:
        if (
            not isinstance(submission_token, str)
            or not 1 <= len(submission_token) <= _MAX_SUBMISSION_TOKEN_CHARACTERS
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in submission_token
            )
        ):
            return DocumentSubmissionResult(
                DocumentSubmissionStatus.UNAVAILABLE,
                DocumentBackgroundFailure.SUBMISSION,
            )
        with self._lock:
            prior = self._results.get(submission_token)
            if prior is not None:
                return prior
            closed = self._closed
        if closed:
            return self._remember(
                submission_token,
                DocumentSubmissionResult(
                    DocumentSubmissionStatus.CLOSED,
                    DocumentBackgroundFailure.CLOSED,
                ),
            )
        if not self._capacity.acquire(blocking=False):
            return DocumentSubmissionResult(
                DocumentSubmissionStatus.BUSY, DocumentBackgroundFailure.CAPACITY
            )
        try:
            envelope = validate_upload_envelope(
                content=content,
                original_filename=original_filename,
                declared_media_type=declared_media_type,
            )
        except Exception:
            self._capacity.release()
            return self._remember(
                submission_token,
                DocumentSubmissionResult(
                    DocumentSubmissionStatus.UNAVAILABLE,
                    DocumentBackgroundFailure.SUBMISSION,
                ),
            )
        try:
            if self._availability_check is not None:
                self._availability_check()
        except Exception:
            self._capacity.release()
            return self._remember(
                submission_token,
                DocumentSubmissionResult(
                    DocumentSubmissionStatus.UNAVAILABLE,
                    DocumentBackgroundFailure.UNAVAILABLE,
                ),
            )
        document_id = uuid4().hex
        job_id = uuid4().hex
        request = _registry_request(
            envelope,
            tenant_id=self._tenant_id,
            knowledge_base_id=self._knowledge_base_id,
            document_id=document_id,
            job_id=job_id,
            storage_object_key=None,
        )
        try:
            created = self._registry.create_or_get(request)
        except Exception:
            self._capacity.release()
            return self._remember(
                submission_token,
                DocumentSubmissionResult(
                    DocumentSubmissionStatus.UNAVAILABLE,
                    DocumentBackgroundFailure.REGISTRY_CREATE,
                ),
            )
        if not created.created:
            self._capacity.release()
            return self._remember(
                submission_token,
                DocumentSubmissionResult(DocumentSubmissionStatus.ACCEPTED),
            )
        work = _Work(
            token=submission_token,
            document_id=document_id,
            job_id=job_id,
            envelope=envelope,
        )
        with self._condition:
            if self._closed:
                self._cancelled.add(submission_token)
            self._work_by_token[submission_token] = work
            self._queue.append(work)
            self._condition.notify()
        return self._remember(
            submission_token,
            DocumentSubmissionResult(DocumentSubmissionStatus.ACCEPTED),
        )

    def cancel(self, *, submission_token: str) -> bool:
        with self._lock:
            work = self._work_by_token.get(submission_token)
            if work is None:
                return False
            self._cancelled.add(submission_token)
        try:
            changed = self._registry.mark_cancelled(
                tenant_id=self._tenant_id,
                knowledge_base_id=self._knowledge_base_id,
                job_id=work.job_id,
                phase=DocumentIngestionPhase.EXTRACTION,
            )
        except Exception:
            return False
        if not changed:
            return False
        with self._condition:
            queued = next(
                (item for item in self._queue if item.token == submission_token), None
            )
            if queued is not None:
                self._queue.remove(queued)
                self._work_by_token.pop(submission_token, None)
                self._capacity.release()
        return True

    def close(self, *, wait: bool = False) -> None:
        with self._submission_lock:
            with self._lock:
                if self._closed:
                    already_closed = True
                else:
                    already_closed = False
                    self._closed = True
                    self._cancelled.update(self._work_by_token)
        if already_closed:
            if wait:
                self._worker.join(timeout=self._close_timeout)
            return
        self._cancel_queued()
        with self._condition:
            self._condition.notify_all()
        if wait:
            self._worker.join(timeout=self._close_timeout)

    @property
    def worker_count(self) -> int:
        return 1 if self._worker.is_alive() else 0

    @property
    def retained_source_bytes(self) -> int:
        with self._lock:
            return sum(
                len(work.envelope.content) for work in self._work_by_token.values()
            )

    def _remember(
        self, token: str, result: DocumentSubmissionResult
    ) -> DocumentSubmissionResult:
        with self._lock:
            remembered = self._results.setdefault(token, result)
            while len(self._results) > _MAX_TOKEN_HISTORY:
                removable = next(
                    (
                        candidate
                        for candidate in self._results
                        if candidate not in self._work_by_token
                    ),
                    None,
                )
                if removable is None:
                    break
                self._results.pop(removable)
            return remembered

    def _is_cancelled(self, token: str) -> bool:
        with self._lock:
            return token in self._cancelled

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if not self._queue and self._closed:
                    return
                work = self._queue.popleft()
            try:
                self._process(work)
            finally:
                with self._lock:
                    self._work_by_token.pop(work.token, None)
                    self._cancelled.discard(work.token)
                self._capacity.release()

    def _process(self, work: _Work) -> None:
        if self._is_cancelled(work.token):
            return
        entry = self._claim(work)
        if entry is None:
            return
        try:
            self._progress(work, DocumentIngestionPhase.EXTRACTION, 0, 0)
            if self._is_cancelled(work.token):
                self._cancel_entry(work, DocumentIngestionPhase.EXTRACTION)
                return
            prepared = prepare_validated_upload_document(
                envelope=work.envelope,
                document_id=work.document_id,
                tenant_id=self._tenant_id,
                knowledge_base_id=self._knowledge_base_id,
                max_chunk_characters=self._max_chunk_characters,
            )
            self._progress(
                work, DocumentIngestionPhase.CHUNKING, 0, len(prepared.chunks)
            )
            if self._is_cancelled(work.token):
                self._cancel_entry(work, DocumentIngestionPhase.CHUNKING)
                return
            self._progress(
                work, DocumentIngestionPhase.EMBEDDING, 0, len(prepared.chunks)
            )
            orchestrator = SynchronousDocumentIngestionOrchestrator(
                registry=self._registry,
                document_embedder=self._embedder,
                vector_writer=self._vector_writer,
                expected_vector_dimension=self._expected_dimension,
                cancellation_requested=lambda: self._is_cancelled(work.token),
            )
            orchestrator.ingest_claimed(prepared, entry=entry)
        except DocumentRegistryError:
            return
        except Exception:
            try:
                self._registry.mark_failed(
                    tenant_id=self._tenant_id,
                    knowledge_base_id=self._knowledge_base_id,
                    job_id=work.job_id,
                    phase=DocumentIngestionPhase.EXTRACTION,
                )
            except Exception:
                pass

    def _claim(self, work: _Work) -> DocumentRegistryEntry | None:
        try:
            return self._registry.claim_queued_job(
                tenant_id=self._tenant_id,
                knowledge_base_id=self._knowledge_base_id,
                document_id=work.document_id,
                job_id=work.job_id,
            )
        except Exception:
            return None

    def _progress(
        self,
        work: _Work,
        phase: DocumentIngestionPhase,
        processed: int,
        total: int,
    ) -> None:
        updated = self._registry.update_processing_progress(
            tenant_id=self._tenant_id,
            knowledge_base_id=self._knowledge_base_id,
            job_id=work.job_id,
            phase=phase,
            processed_chunks=processed,
            total_chunks=total,
        )
        if updated is None:
            raise DocumentRegistryError(DocumentOperationPhase.PROGRESS)

    def _cancel_entry(self, work: _Work, phase: DocumentIngestionPhase) -> None:
        self._registry.mark_cancelled(
            tenant_id=self._tenant_id,
            knowledge_base_id=self._knowledge_base_id,
            job_id=work.job_id,
            phase=phase,
        )

    def _cancel_queued(self) -> None:
        with self._condition:
            queued = tuple(self._queue)
            self._queue.clear()
        for item in queued:
            try:
                self._cancel_entry(item, DocumentIngestionPhase.EXTRACTION)
            except Exception:
                pass
            with self._lock:
                self._work_by_token.pop(item.token, None)
                self._cancelled.discard(item.token)
            self._capacity.release()


def _registry_request(
    envelope: ValidatedUploadEnvelope,
    *,
    tenant_id: str,
    knowledge_base_id: str,
    document_id: str,
    job_id: str,
    storage_object_key: str | None,
) -> DocumentRegistryCreateRequest:
    return DocumentRegistryCreateRequest(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        job_id=job_id,
        original_filename=envelope.original_filename,
        media_type=envelope.media_type,
        byte_size=envelope.byte_size,
        sha256_hex=envelope.sha256_hex,
        storage_object_key=storage_object_key,
        total_chunks=0,
        initial_phase=DocumentIngestionPhase.EXTRACTION,
    )
