from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from app.ingestion.models import DocumentChunkInput, DocumentIngestionRequest
from app.ingestion.registry_models import (
    DocumentIngestionJob,
    DocumentIngestionPhase,
    DocumentIngestionState,
    DocumentOperationPhase,
    DocumentReadiness,
    DocumentRegistryCreateRequest,
    DocumentRegistryCreateResult,
    DocumentRegistryEntry,
    DocumentRegistryError,
    DocumentRegistryRecord,
)
from app.ingestion.synchronous_orchestration import (
    SynchronousDocumentIngestionOrchestrator,
)
from app.ingestion.upload_preparation import PreparedUploadDocument
from app.vector_store.models import (
    VectorBatchWriteRequest,
    VectorBatchWriteResult,
    VectorRecordIdentity,
)
from app.vector_store.postgres.contracts import PostgreSQLVectorTransaction


def prepared(media_type: str = "text/plain") -> PreparedUploadDocument:
    document_id = "doc-new"
    chunk = DocumentChunkInput(
        document_id=document_id,
        chunk_id="chunk_000001",
        text="synthetic document text",
    )
    return PreparedUploadDocument(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        document_id=document_id,
        original_filename="guide.pdf"
        if media_type == "application/pdf"
        else "guide.txt",
        media_type=media_type,
        byte_size=23,
        sha256_hex="a" * 64,
        ingestion_request=DocumentIngestionRequest(
            tenant_id="tenant-a", knowledge_base_id="kb-a", chunks=(chunk,)
        ),
    )


def entry(
    state: DocumentIngestionState,
    *,
    ready: bool = False,
    document_id: str = "doc-new",
    job_id: str = "job-new",
) -> DocumentRegistryEntry:
    now = datetime.now(UTC)
    started = now if state is not DocumentIngestionState.QUEUED else None
    finished = (
        now
        if state in {DocumentIngestionState.SUCCEEDED, DocumentIngestionState.FAILED}
        else None
    )
    document = DocumentRegistryRecord(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        document_id=document_id,
        original_filename="guide.txt",
        media_type="text/plain",
        byte_size=23,
        storage_object_key="objects/server-1",
        created_at_utc=now,
        ready_at_utc=now if ready else None,
    )
    job = DocumentIngestionJob(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        document_id=document_id,
        job_id=job_id,
        state=state,
        phase=(
            DocumentIngestionPhase.FINALIZE
            if state is DocumentIngestionState.SUCCEEDED
            else DocumentIngestionPhase.EMBEDDING
        ),
        processed_chunks=1 if state is DocumentIngestionState.SUCCEEDED else 0,
        total_chunks=1,
        attempt_count=1 if started else 0,
        created_at_utc=now,
        started_at_utc=started,
        updated_at_utc=now,
        finished_at_utc=finished,
    )
    readiness = (
        DocumentReadiness.READY
        if ready
        else DocumentReadiness.FAILED
        if state is DocumentIngestionState.FAILED
        else DocumentReadiness.PENDING
    )
    return DocumentRegistryEntry(document=document, job=job, readiness=readiness)


class FakeRegistry:
    def __init__(
        self, *, created: bool = True, initial: DocumentRegistryEntry | None = None
    ) -> None:
        self.created = created
        self.current = initial or entry(DocumentIngestionState.QUEUED)
        self.fail_finalize = False
        self.failures: list[DocumentIngestionPhase] = []
        self.transaction = object()

    def create_or_get(
        self, request: DocumentRegistryCreateRequest
    ) -> DocumentRegistryCreateResult:
        return DocumentRegistryCreateResult(entry=self.current, created=self.created)

    def claim_queued_job(self, **scope: str) -> DocumentRegistryEntry | None:
        self.current = entry(DocumentIngestionState.PROCESSING)
        return self.current

    def finalize_success(
        self,
        *,
        vector_operation: Callable[
            [PostgreSQLVectorTransaction], VectorBatchWriteResult
        ],
        **scope: object,
    ) -> VectorBatchWriteResult:
        result = vector_operation(self.transaction)  # type: ignore[arg-type]
        if self.fail_finalize:
            raise RuntimeError("synthetic finalization failure")
        self.current = entry(DocumentIngestionState.SUCCEEDED, ready=True)
        return result

    def mark_failed(self, *, phase: DocumentIngestionPhase, **scope: object) -> bool:
        self.failures.append(phase)
        self.current = entry(DocumentIngestionState.FAILED)
        return True

    def get_entry(self, **scope: str) -> DocumentRegistryEntry | None:
        return self.current


class FakeEmbedder:
    def __init__(self, output: object | None = None) -> None:
        self.output = output if output is not None else ((0.1,) * 384,)
        self.calls = 0

    def embed_documents(self, **request: object) -> object:
        self.calls += 1
        return self.output


class FakeWriter:
    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    def admit_batch_in_transaction(
        self, transaction: PostgreSQLVectorTransaction, request: VectorBatchWriteRequest
    ) -> VectorBatchWriteResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic vector failure")
        return VectorBatchWriteResult(
            tenant_id=request.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
            inserted_identities=tuple(
                VectorRecordIdentity(document_id=row.document_id, chunk_id=row.chunk_id)
                for row in request.records
            ),
            unchanged_identities=(),
        )


@pytest.mark.parametrize("media_type", ("text/plain", "application/pdf"))
def test_successful_prepared_ingestion_uses_atomic_vector_callback(
    media_type: str,
) -> None:
    registry = FakeRegistry()
    embedder = FakeEmbedder()
    writer = FakeWriter()
    result = SynchronousDocumentIngestionOrchestrator(
        registry=registry,  # type: ignore[arg-type]
        document_embedder=embedder,  # type: ignore[arg-type]
        vector_writer=writer,
        expected_vector_dimension=384,
    ).ingest(
        prepared(media_type), job_id="job-new", storage_object_key="objects/server-1"
    )
    assert result.entry.readiness is DocumentReadiness.READY
    assert embedder.calls == writer.calls == 1


@pytest.mark.parametrize("output", ((), ((0.1,) * 383,)))
def test_embedding_shape_failure_marks_fixed_embedding_phase(output: object) -> None:
    registry = FakeRegistry()
    with pytest.raises(DocumentRegistryError) as caught:
        SynchronousDocumentIngestionOrchestrator(
            registry=registry,  # type: ignore[arg-type]
            document_embedder=FakeEmbedder(output),  # type: ignore[arg-type]
            vector_writer=FakeWriter(),
            expected_vector_dimension=384,
        ).ingest(prepared(), job_id="job-new", storage_object_key="objects/server-1")
    assert caught.value.phase is DocumentOperationPhase.EMBEDDING
    assert registry.failures == [DocumentIngestionPhase.EMBEDDING]


def test_vector_failure_marks_vector_phase_without_masking() -> None:
    registry = FakeRegistry()
    writer = FakeWriter()
    writer.fail = True
    with pytest.raises(DocumentRegistryError) as caught:
        SynchronousDocumentIngestionOrchestrator(
            registry=registry,  # type: ignore[arg-type]
            document_embedder=FakeEmbedder(),  # type: ignore[arg-type]
            vector_writer=writer,
            expected_vector_dimension=384,
        ).ingest(prepared(), job_id="job-new", storage_object_key="objects/server-1")
    assert caught.value.phase is DocumentOperationPhase.VECTOR_WRITE
    assert registry.failures == [DocumentIngestionPhase.VECTOR_WRITE]


def test_succeeded_duplicate_skips_embedding_and_vector_write() -> None:
    registry = FakeRegistry(
        created=False, initial=entry(DocumentIngestionState.SUCCEEDED, ready=True)
    )
    embedder = FakeEmbedder()
    writer = FakeWriter()
    result = SynchronousDocumentIngestionOrchestrator(
        registry=registry,  # type: ignore[arg-type]
        document_embedder=embedder,  # type: ignore[arg-type]
        vector_writer=writer,
        expected_vector_dimension=384,
    ).ingest(prepared(), job_id="unused-job", storage_object_key="objects/new-key")
    assert result.created is False
    assert result.vector_result is None
    assert embedder.calls == writer.calls == 0
