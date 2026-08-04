"""Typed contracts for tenant-scoped document registry operations."""

from collections.abc import Callable
from typing import Protocol

from app.ingestion.registry_models import (
    DocumentDeletionResult,
    DocumentIngestionJob,
    DocumentIngestionPhase,
    DocumentListCursor,
    DocumentListPage,
    DocumentRegistryCreateRequest,
    DocumentRegistryCreateResult,
    DocumentRegistryEntry,
)
from app.vector_store.models import VectorBatchWriteRequest, VectorBatchWriteResult
from app.vector_store.postgres.contracts import PostgreSQLVectorTransaction


class DocumentRegistryRepository(Protocol):
    def create_or_get(
        self, request: DocumentRegistryCreateRequest
    ) -> DocumentRegistryCreateResult: ...

    def get_entry(
        self, *, tenant_id: str, knowledge_base_id: str, document_id: str
    ) -> DocumentRegistryEntry | None: ...

    def get_job(
        self, *, tenant_id: str, knowledge_base_id: str, job_id: str
    ) -> DocumentIngestionJob | None: ...

    def list_documents(
        self, *, tenant_id: str, knowledge_base_id: str
    ) -> tuple[DocumentRegistryEntry, ...]: ...

    def list_document_page(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        page_size: int = 50,
        cursor: DocumentListCursor | None = None,
    ) -> DocumentListPage: ...

    def update_processing_progress(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        job_id: str,
        phase: DocumentIngestionPhase,
        processed_chunks: int,
        total_chunks: int,
    ) -> DocumentIngestionJob | None: ...

    def mark_cancelled(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        job_id: str,
        phase: DocumentIngestionPhase,
    ) -> bool: ...

    def claim_queued_job(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_id: str,
        job_id: str,
    ) -> DocumentRegistryEntry | None: ...

    def retry_failed_job(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_id: str,
        job_id: str,
    ) -> DocumentRegistryEntry | None: ...

    def mark_failed(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        job_id: str,
        phase: DocumentIngestionPhase,
    ) -> bool: ...

    def finalize_success(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_id: str,
        job_id: str,
        total_chunks: int,
        vector_operation: Callable[
            [PostgreSQLVectorTransaction], VectorBatchWriteResult
        ],
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> VectorBatchWriteResult: ...

    def delete_document(
        self, *, tenant_id: str, knowledge_base_id: str, document_id: str
    ) -> DocumentDeletionResult | None: ...


class TransactionAwareVectorBatchWriter(Protocol):
    def admit_batch_in_transaction(
        self,
        transaction: PostgreSQLVectorTransaction,
        request: VectorBatchWriteRequest,
    ) -> VectorBatchWriteResult: ...
