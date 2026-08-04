"""Synchronous registry, embedding and atomic vector finalization flow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from app.embeddings import DocumentEmbedder
from app.ingestion.registry import (
    DocumentRegistryRepository,
    TransactionAwareVectorBatchWriter,
)
from app.ingestion.registry_models import (
    DocumentIngestionPhase,
    DocumentIngestionState,
    DocumentOperationPhase,
    DocumentReadiness,
    DocumentRegistryCreateRequest,
    DocumentRegistryEntry,
    DocumentRegistryError,
)
from app.ingestion.upload_preparation import PreparedUploadDocument
from app.vector_store.models import (
    VectorBatchWriteRequest,
    VectorBatchWriteResult,
    VectorRecord,
)
from app.vector_store.postgres.contracts import PostgreSQLVectorTransaction


@dataclass(frozen=True, slots=True)
class SynchronousDocumentIngestionResult:
    entry: DocumentRegistryEntry
    created: bool
    vector_result: VectorBatchWriteResult | None


class SynchronousDocumentIngestionOrchestrator:
    """Run one prepared document synchronously with atomic finalization."""

    def __init__(
        self,
        *,
        registry: DocumentRegistryRepository,
        document_embedder: DocumentEmbedder,
        vector_writer: TransactionAwareVectorBatchWriter,
        expected_vector_dimension: int,
    ) -> None:
        if not callable(getattr(registry, "create_or_get", None)):
            raise ValueError("registry is invalid")
        if not callable(getattr(document_embedder, "embed_documents", None)):
            raise ValueError("document_embedder is invalid")
        if not callable(getattr(vector_writer, "admit_batch_in_transaction", None)):
            raise ValueError("vector_writer is invalid")
        if type(expected_vector_dimension) is not int or expected_vector_dimension <= 0:
            raise ValueError("expected_vector_dimension must be positive")
        self._registry = registry
        self._document_embedder = document_embedder
        self._vector_writer = vector_writer
        self._expected_vector_dimension = expected_vector_dimension

    def ingest(
        self,
        prepared: PreparedUploadDocument,
        *,
        job_id: str,
        storage_object_key: str,
    ) -> SynchronousDocumentIngestionResult:
        request = _create_request(prepared, job_id, storage_object_key)
        try:
            created = self._registry.create_or_get(request)
        except DocumentRegistryError:
            raise
        except Exception:
            raise DocumentRegistryError(
                DocumentOperationPhase.REGISTRY_CREATE
            ) from None

        entry = created.entry
        if not created.created and entry.readiness is DocumentReadiness.READY:
            return SynchronousDocumentIngestionResult(
                entry=entry, created=False, vector_result=None
            )
        if not created.created and entry.job.state is DocumentIngestionState.FAILED:
            return SynchronousDocumentIngestionResult(
                entry=entry, created=False, vector_result=None
            )

        try:
            claimed = self._registry.claim_queued_job(
                tenant_id=entry.document.tenant_id,
                knowledge_base_id=entry.document.knowledge_base_id,
                document_id=entry.document.document_id,
                job_id=entry.job.job_id,
            )
            if claimed is None:
                raise DocumentRegistryError(DocumentOperationPhase.JOB_CLAIM)
        except DocumentRegistryError:
            raise
        except Exception:
            raise DocumentRegistryError(DocumentOperationPhase.JOB_CLAIM) from None

        try:
            output: object = self._document_embedder.embed_documents(
                tenant_id=prepared.tenant_id,
                knowledge_base_id=prepared.knowledge_base_id,
                texts=tuple(chunk.text for chunk in prepared.chunks),
            )
            vectors = _validated_vectors(
                output,
                expected_rows=len(prepared.chunks),
                expected_dimension=self._expected_vector_dimension,
            )
        except Exception:
            self._record_failure(entry, DocumentIngestionPhase.EMBEDDING)
            raise DocumentRegistryError(DocumentOperationPhase.EMBEDDING) from None

        batch = VectorBatchWriteRequest(
            tenant_id=prepared.tenant_id,
            knowledge_base_id=prepared.knowledge_base_id,
            records=tuple(
                VectorRecord(
                    tenant_id=prepared.tenant_id,
                    knowledge_base_id=prepared.knowledge_base_id,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    embedding=vector,
                    metadata=chunk.metadata,
                )
                for chunk, vector in zip(prepared.chunks, vectors, strict=True)
            ),
        )

        def write_vectors(
            transaction: PostgreSQLVectorTransaction,
        ) -> VectorBatchWriteResult:
            try:
                result = self._vector_writer.admit_batch_in_transaction(
                    transaction, batch
                )
                _validate_vector_result(result, batch)
                return result
            except Exception:
                raise DocumentRegistryError(
                    DocumentOperationPhase.VECTOR_WRITE
                ) from None

        try:
            vector_result = self._registry.finalize_success(
                tenant_id=entry.document.tenant_id,
                knowledge_base_id=entry.document.knowledge_base_id,
                document_id=entry.document.document_id,
                job_id=entry.job.job_id,
                total_chunks=len(prepared.chunks),
                vector_operation=write_vectors,
            )
        except DocumentRegistryError as error:
            failure_phase = (
                DocumentIngestionPhase.VECTOR_WRITE
                if error.phase is DocumentOperationPhase.VECTOR_WRITE
                else DocumentIngestionPhase.FINALIZE
            )
            self._record_failure(entry, failure_phase)
            raise
        except Exception:
            self._record_failure(entry, DocumentIngestionPhase.FINALIZE)
            raise DocumentRegistryError(DocumentOperationPhase.FINALIZE) from None

        final_entry = self._registry.get_entry(
            tenant_id=entry.document.tenant_id,
            knowledge_base_id=entry.document.knowledge_base_id,
            document_id=entry.document.document_id,
        )
        if final_entry is None or final_entry.readiness is not DocumentReadiness.READY:
            raise DocumentRegistryError(DocumentOperationPhase.FINALIZE)
        return SynchronousDocumentIngestionResult(
            entry=final_entry,
            created=created.created,
            vector_result=vector_result,
        )

    def _record_failure(
        self,
        entry: DocumentRegistryEntry,
        phase: DocumentIngestionPhase,
    ) -> None:
        try:
            self._registry.mark_failed(
                tenant_id=entry.document.tenant_id,
                knowledge_base_id=entry.document.knowledge_base_id,
                job_id=entry.job.job_id,
                phase=phase,
            )
        except Exception:
            pass


def _create_request(
    prepared: object,
    job_id: str,
    storage_object_key: str,
) -> DocumentRegistryCreateRequest:
    try:
        if not isinstance(prepared, PreparedUploadDocument):
            raise ValueError
        if any(chunk.document_id != prepared.document_id for chunk in prepared.chunks):
            raise ValueError
        return DocumentRegistryCreateRequest(
            tenant_id=prepared.tenant_id,
            knowledge_base_id=prepared.knowledge_base_id,
            document_id=prepared.document_id,
            job_id=job_id,
            original_filename=prepared.original_filename,
            media_type=prepared.media_type,
            byte_size=prepared.byte_size,
            sha256_hex=prepared.sha256_hex,
            storage_object_key=storage_object_key,
            total_chunks=len(prepared.chunks),
        )
    except Exception:
        raise DocumentRegistryError(DocumentOperationPhase.REGISTRY_CREATE) from None


def _validated_vectors(
    output: object,
    *,
    expected_rows: int,
    expected_dimension: int,
) -> tuple[tuple[float, ...], ...]:
    if (
        isinstance(output, (str, bytes))
        or not isinstance(output, Sequence)
        or len(output) != expected_rows
    ):
        raise ValueError
    vectors: list[tuple[float, ...]] = []
    for row in output:
        if (
            isinstance(row, (str, bytes))
            or not isinstance(row, Sequence)
            or len(row) != expected_dimension
        ):
            raise ValueError
        values: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, Real):
                raise ValueError
            value = float(item)
            if not isfinite(value):
                raise ValueError
            values.append(value)
        vectors.append(tuple(values))
    return tuple(vectors)


def _validate_vector_result(
    result: object,
    request: VectorBatchWriteRequest,
) -> None:
    if not isinstance(result, VectorBatchWriteResult):
        raise ValueError
    if (
        result.tenant_id != request.tenant_id
        or result.knowledge_base_id != request.knowledge_base_id
    ):
        raise ValueError
    inserted = tuple(
        (identity.document_id, identity.chunk_id)
        for identity in result.inserted_identities
    )
    unchanged = tuple(
        (identity.document_id, identity.chunk_id)
        for identity in result.unchanged_identities
    )
    requested = {(record.document_id, record.chunk_id) for record in request.records}
    if (
        len(inserted) != len(set(inserted))
        or len(unchanged) != len(set(unchanged))
        or set(inserted) & set(unchanged)
        or set(inserted) | set(unchanged) != requested
    ):
        raise ValueError
