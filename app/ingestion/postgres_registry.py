"""Tenant-scoped PostgreSQL document registry and transaction boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar
import unicodedata

from psycopg import Connection

from app.ingestion.registry_models import (
    MAX_INGESTION_ATTEMPTS,
    DocumentDeletionResult,
    DocumentIngestionJob,
    DocumentIngestionPhase,
    DocumentIngestionState,
    DocumentOperationPhase,
    DocumentRegistryCreateRequest,
    DocumentRegistryCreateResult,
    DocumentRegistryEntry,
    DocumentRegistryError,
    DocumentRegistryRecord,
    derive_document_readiness,
)
from app.vector_store.models import VectorBatchWriteResult
from app.vector_store.postgres.contracts import PostgreSQLVectorTransaction
from app.vector_store.postgres.transaction import PsycopgPostgreSQLVectorTransaction

PostgreSQLRegistryConnectionFactory = Callable[[], Connection[Any]]
VectorFinalizationOperation = Callable[
    [PostgreSQLVectorTransaction],
    VectorBatchWriteResult,
]
T = TypeVar("T")

_DOCUMENT_INSERT = """
    INSERT INTO callmetric_vector.documents (
        tenant_id, knowledge_base_id, document_id, original_filename,
        media_type, byte_size, sha256_hex, storage_object_key
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (tenant_id, knowledge_base_id, sha256_hex) DO NOTHING
    RETURNING document_id
    """
_JOB_INSERT = """
    INSERT INTO callmetric_vector.document_ingestion_jobs (
        tenant_id, knowledge_base_id, job_id, document_id, state, phase,
        processed_chunks, total_chunks, attempt_count
    )
    VALUES (%s, %s, %s, %s, 'QUEUED', 'EMBEDDING', 0, %s, 0)
    """
_ENTRY_COLUMNS = """
    documents.tenant_id, documents.knowledge_base_id, documents.document_id,
    documents.original_filename, documents.media_type, documents.byte_size,
    documents.storage_object_key, documents.created_at_utc,
    documents.ready_at_utc, jobs.job_id, jobs.state, jobs.phase,
    jobs.processed_chunks, jobs.total_chunks, jobs.attempt_count,
    jobs.created_at_utc, jobs.started_at_utc, jobs.updated_at_utc,
    jobs.finished_at_utc
    """
_ENTRY_FROM = """
    FROM callmetric_vector.documents AS documents
    INNER JOIN callmetric_vector.document_ingestion_jobs AS jobs
      ON jobs.tenant_id = documents.tenant_id
     AND jobs.knowledge_base_id = documents.knowledge_base_id
     AND jobs.document_id = documents.document_id
    """
_ENTRY_BY_ID = f"""
    SELECT {_ENTRY_COLUMNS}
    {_ENTRY_FROM}
    WHERE documents.tenant_id = %s
      AND documents.knowledge_base_id = %s
      AND documents.document_id = %s
    """
_ENTRY_BY_ID_FOR_UPDATE = _ENTRY_BY_ID + " FOR UPDATE OF documents, jobs"
_ENTRY_BY_SHA = f"""
    SELECT {_ENTRY_COLUMNS}
    {_ENTRY_FROM}
    WHERE documents.tenant_id = %s
      AND documents.knowledge_base_id = %s
      AND documents.sha256_hex = %s
    """
_ENTRY_LIST = f"""
    SELECT {_ENTRY_COLUMNS}
    {_ENTRY_FROM}
    WHERE documents.tenant_id = %s
      AND documents.knowledge_base_id = %s
    ORDER BY documents.created_at_utc DESC, documents.document_id
    """
_JOB_CLAIM = """
    UPDATE callmetric_vector.document_ingestion_jobs
    SET state = 'PROCESSING', phase = 'EMBEDDING',
        attempt_count = attempt_count + 1,
        started_at_utc = CURRENT_TIMESTAMP,
        updated_at_utc = CURRENT_TIMESTAMP,
        finished_at_utc = NULL
    WHERE tenant_id = %s AND knowledge_base_id = %s AND job_id = %s
      AND state = 'QUEUED' AND attempt_count < %s
    """
_JOB_RETRY = """
    UPDATE callmetric_vector.document_ingestion_jobs
    SET state = 'QUEUED', phase = 'EMBEDDING', processed_chunks = 0,
        started_at_utc = NULL, updated_at_utc = CURRENT_TIMESTAMP,
        finished_at_utc = NULL
    WHERE tenant_id = %s AND knowledge_base_id = %s AND job_id = %s
      AND state = 'FAILED' AND attempt_count < %s
    """
_JOB_FAIL = """
    UPDATE callmetric_vector.document_ingestion_jobs
    SET state = 'FAILED', phase = %s, updated_at_utc = CURRENT_TIMESTAMP,
        finished_at_utc = CURRENT_TIMESTAMP
    WHERE tenant_id = %s AND knowledge_base_id = %s AND job_id = %s
      AND state = 'PROCESSING'
    """
_JOB_SUCCEED = """
    UPDATE callmetric_vector.document_ingestion_jobs
    SET state = 'SUCCEEDED', phase = 'FINALIZE', processed_chunks = %s,
        total_chunks = %s, updated_at_utc = CURRENT_TIMESTAMP,
        finished_at_utc = CURRENT_TIMESTAMP
    WHERE tenant_id = %s AND knowledge_base_id = %s AND job_id = %s
      AND document_id = %s AND state = 'PROCESSING'
    """
_DOCUMENT_READY = """
    UPDATE callmetric_vector.documents
    SET ready_at_utc = CURRENT_TIMESTAMP
    WHERE tenant_id = %s AND knowledge_base_id = %s AND document_id = %s
      AND ready_at_utc IS NULL
    """
_VECTOR_DELETE = """
    DELETE FROM callmetric_vector.vector_records
    WHERE tenant_id = %s AND knowledge_base_id = %s AND document_id = %s
    """
_DOCUMENT_DELETE = """
    DELETE FROM callmetric_vector.documents
    WHERE tenant_id = %s AND knowledge_base_id = %s AND document_id = %s
    """


class PsycopgDocumentRegistryRepository:
    """Synchronous repository with one transaction per public mutation."""

    def __init__(
        self,
        *,
        connection_factory: PostgreSQLRegistryConnectionFactory,
    ) -> None:
        if not callable(connection_factory):
            raise ValueError("connection_factory must be callable")
        self._connection_factory = connection_factory

    def create_or_get(
        self,
        request: DocumentRegistryCreateRequest,
    ) -> DocumentRegistryCreateResult:
        if not isinstance(request, DocumentRegistryCreateRequest):
            raise DocumentRegistryError(DocumentOperationPhase.REGISTRY_CREATE)

        def operation(connection: Connection[Any]) -> DocumentRegistryCreateResult:
            with connection.cursor() as cursor:
                cursor.execute(
                    _DOCUMENT_INSERT,
                    (
                        request.tenant_id,
                        request.knowledge_base_id,
                        request.document_id,
                        request.original_filename,
                        request.media_type,
                        request.byte_size,
                        request.sha256_hex,
                        request.storage_object_key,
                    ),
                )
                inserted = cursor.fetchall()
                if len(inserted) > 1:
                    raise DocumentRegistryError(DocumentOperationPhase.REGISTRY_CREATE)
                created = bool(inserted)
                if created:
                    if inserted != [(request.document_id,)]:
                        raise DocumentRegistryError(
                            DocumentOperationPhase.REGISTRY_CREATE
                        )
                    cursor.execute(
                        _JOB_INSERT,
                        (
                            request.tenant_id,
                            request.knowledge_base_id,
                            request.job_id,
                            request.document_id,
                            request.total_chunks,
                        ),
                    )
                    entry = _fetch_one_entry(
                        cursor,
                        _ENTRY_BY_ID,
                        (
                            request.tenant_id,
                            request.knowledge_base_id,
                            request.document_id,
                        ),
                    )
                else:
                    entry = _fetch_one_entry(
                        cursor,
                        _ENTRY_BY_SHA,
                        (
                            request.tenant_id,
                            request.knowledge_base_id,
                            request.sha256_hex,
                        ),
                    )
            if entry is None:
                raise DocumentRegistryError(DocumentOperationPhase.REGISTRY_CREATE)
            return DocumentRegistryCreateResult(entry=entry, created=created)

        return self._run(DocumentOperationPhase.REGISTRY_CREATE, operation)

    def get_entry(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_id: str,
    ) -> DocumentRegistryEntry | None:
        parameters = _scope_document(tenant_id, knowledge_base_id, document_id)

        def operation(connection: Connection[Any]) -> DocumentRegistryEntry | None:
            with connection.cursor() as cursor:
                return _fetch_one_entry(cursor, _ENTRY_BY_ID, parameters)

        return self._run(DocumentOperationPhase.REGISTRY_CREATE, operation)

    def get_job(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        job_id: str,
    ) -> DocumentIngestionJob | None:
        tenant, knowledge_base, job = _scope_job(tenant_id, knowledge_base_id, job_id)
        query = f"""
            SELECT {_ENTRY_COLUMNS}
            {_ENTRY_FROM}
            WHERE jobs.tenant_id = %s AND jobs.knowledge_base_id = %s
              AND jobs.job_id = %s
            """

        def operation(connection: Connection[Any]) -> DocumentIngestionJob | None:
            with connection.cursor() as cursor:
                entry = _fetch_one_entry(cursor, query, (tenant, knowledge_base, job))
            return None if entry is None else entry.job

        return self._run(DocumentOperationPhase.JOB_CLAIM, operation)

    def list_documents(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> tuple[DocumentRegistryEntry, ...]:
        tenant, knowledge_base = _scope(tenant_id, knowledge_base_id)

        def operation(connection: Connection[Any]) -> tuple[DocumentRegistryEntry, ...]:
            with connection.cursor() as cursor:
                cursor.execute(_ENTRY_LIST, (tenant, knowledge_base))
                rows = cursor.fetchall()
            entries = tuple(_entry_from_row(row) for row in rows)
            ordering = tuple(
                (-entry.document.created_at_utc.timestamp(), entry.document.document_id)
                for entry in entries
            )
            if ordering != tuple(sorted(ordering)):
                raise DocumentRegistryError(DocumentOperationPhase.REGISTRY_CREATE)
            return entries

        return self._run(DocumentOperationPhase.REGISTRY_CREATE, operation)

    def claim_queued_job(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_id: str,
        job_id: str,
    ) -> DocumentRegistryEntry | None:
        document_scope = _scope_document(tenant_id, knowledge_base_id, document_id)
        job = _identifier(job_id)

        def operation(connection: Connection[Any]) -> DocumentRegistryEntry | None:
            with connection.cursor() as cursor:
                entry = _fetch_one_entry(
                    cursor,
                    _ENTRY_BY_ID_FOR_UPDATE,
                    document_scope,
                )
                if entry is None or entry.job.job_id != job:
                    return None
                if (
                    entry.job.state is not DocumentIngestionState.QUEUED
                    or entry.job.attempt_count >= MAX_INGESTION_ATTEMPTS
                ):
                    raise DocumentRegistryError(DocumentOperationPhase.JOB_CLAIM)
                cursor.execute(
                    _JOB_CLAIM,
                    (*document_scope[:2], job, MAX_INGESTION_ATTEMPTS),
                )
                _require_one_changed(cursor, DocumentOperationPhase.JOB_CLAIM)
                claimed = _fetch_one_entry(cursor, _ENTRY_BY_ID, document_scope)
            if claimed is None:
                raise DocumentRegistryError(DocumentOperationPhase.JOB_CLAIM)
            return claimed

        return self._run(DocumentOperationPhase.JOB_CLAIM, operation)

    def retry_failed_job(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_id: str,
        job_id: str,
    ) -> DocumentRegistryEntry | None:
        document_scope = _scope_document(tenant_id, knowledge_base_id, document_id)
        job = _identifier(job_id)

        def operation(connection: Connection[Any]) -> DocumentRegistryEntry | None:
            with connection.cursor() as cursor:
                entry = _fetch_one_entry(
                    cursor,
                    _ENTRY_BY_ID_FOR_UPDATE,
                    document_scope,
                )
                if entry is None or entry.job.job_id != job:
                    return None
                if (
                    entry.job.state is not DocumentIngestionState.FAILED
                    or entry.job.attempt_count >= MAX_INGESTION_ATTEMPTS
                ):
                    raise DocumentRegistryError(DocumentOperationPhase.RETRY)
                cursor.execute(
                    _JOB_RETRY,
                    (*document_scope[:2], job, MAX_INGESTION_ATTEMPTS),
                )
                _require_one_changed(cursor, DocumentOperationPhase.RETRY)
                retried = _fetch_one_entry(cursor, _ENTRY_BY_ID, document_scope)
            if retried is None:
                raise DocumentRegistryError(DocumentOperationPhase.RETRY)
            return retried

        return self._run(DocumentOperationPhase.RETRY, operation)

    def mark_failed(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        job_id: str,
        phase: DocumentIngestionPhase,
    ) -> bool:
        tenant, knowledge_base, job = _scope_job(tenant_id, knowledge_base_id, job_id)
        if not isinstance(phase, DocumentIngestionPhase):
            raise DocumentRegistryError(DocumentOperationPhase.FINALIZE)

        def operation(connection: Connection[Any]) -> bool:
            with connection.cursor() as cursor:
                cursor.execute(
                    _JOB_FAIL,
                    (phase.value, tenant, knowledge_base, job),
                )
                return cursor.rowcount == 1

        return self._run(DocumentOperationPhase.FINALIZE, operation)

    def finalize_success(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_id: str,
        job_id: str,
        total_chunks: int,
        vector_operation: VectorFinalizationOperation,
    ) -> VectorBatchWriteResult:
        document_scope = _scope_document(tenant_id, knowledge_base_id, document_id)
        job = _identifier(job_id)
        if type(total_chunks) is not int or total_chunks <= 0:
            raise DocumentRegistryError(DocumentOperationPhase.FINALIZE)
        if not callable(vector_operation):
            raise DocumentRegistryError(DocumentOperationPhase.FINALIZE)

        def operation(connection: Connection[Any]) -> VectorBatchWriteResult:
            with connection.cursor() as cursor:
                entry = _fetch_one_entry(
                    cursor,
                    _ENTRY_BY_ID_FOR_UPDATE,
                    document_scope,
                )
                if (
                    entry is None
                    or entry.job.job_id != job
                    or entry.job.state is not DocumentIngestionState.PROCESSING
                    or entry.document.ready_at_utc is not None
                ):
                    raise DocumentRegistryError(DocumentOperationPhase.FINALIZE)

            transaction = PsycopgPostgreSQLVectorTransaction(connection)
            result = vector_operation(transaction)

            with connection.cursor() as cursor:
                cursor.execute(
                    _JOB_SUCCEED,
                    (
                        total_chunks,
                        total_chunks,
                        *document_scope[:2],
                        job,
                        document_scope[2],
                    ),
                )
                _require_one_changed(cursor, DocumentOperationPhase.FINALIZE)
                cursor.execute(_DOCUMENT_READY, document_scope)
                _require_one_changed(cursor, DocumentOperationPhase.FINALIZE)
            return result

        return self._run(DocumentOperationPhase.FINALIZE, operation)

    def delete_document(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_id: str,
    ) -> DocumentDeletionResult | None:
        document_scope = _scope_document(tenant_id, knowledge_base_id, document_id)

        def operation(connection: Connection[Any]) -> DocumentDeletionResult | None:
            with connection.cursor() as cursor:
                entry = _fetch_one_entry(
                    cursor,
                    _ENTRY_BY_ID_FOR_UPDATE,
                    document_scope,
                )
                if entry is None:
                    return None
                cursor.execute(_VECTOR_DELETE, document_scope)
                cursor.execute(_DOCUMENT_DELETE, document_scope)
                _require_one_changed(cursor, DocumentOperationPhase.DELETE)
            return DocumentDeletionResult(
                storage_object_key=entry.document.storage_object_key
            )

        return self._run(DocumentOperationPhase.DELETE, operation)

    def _run(
        self,
        phase: DocumentOperationPhase,
        operation: Callable[[Connection[Any]], T],
    ) -> T:
        try:
            connection = self._connection_factory()
        except Exception:
            raise DocumentRegistryError(phase) from None
        try:
            if connection.autocommit is not False:
                raise DocumentRegistryError(phase)
            result = operation(connection)
            connection.commit()
        except DocumentRegistryError:
            _rollback_and_close(connection)
            raise
        except Exception:
            _rollback_and_close(connection)
            raise DocumentRegistryError(phase) from None
        try:
            connection.close()
        except Exception:
            raise DocumentRegistryError(phase) from None
        return result


def _fetch_one_entry(
    cursor: Any,
    query: str,
    parameters: tuple[object, ...],
) -> DocumentRegistryEntry | None:
    cursor.execute(query, parameters)
    rows = cursor.fetchall()
    if len(rows) > 1:
        raise DocumentRegistryError(DocumentOperationPhase.REGISTRY_CREATE)
    return None if not rows else _entry_from_row(rows[0])


def _entry_from_row(raw: object) -> DocumentRegistryEntry:
    if not isinstance(raw, tuple) or len(raw) != 19:
        raise DocumentRegistryError(DocumentOperationPhase.REGISTRY_CREATE)
    document = DocumentRegistryRecord(
        tenant_id=raw[0],
        knowledge_base_id=raw[1],
        document_id=raw[2],
        original_filename=raw[3],
        media_type=raw[4],
        byte_size=raw[5],
        storage_object_key=raw[6],
        created_at_utc=raw[7],
        ready_at_utc=raw[8],
    )
    job = DocumentIngestionJob(
        tenant_id=raw[0],
        knowledge_base_id=raw[1],
        document_id=raw[2],
        job_id=raw[9],
        state=raw[10],
        phase=raw[11],
        processed_chunks=raw[12],
        total_chunks=raw[13],
        attempt_count=raw[14],
        created_at_utc=raw[15],
        started_at_utc=raw[16],
        updated_at_utc=raw[17],
        finished_at_utc=raw[18],
    )
    return DocumentRegistryEntry(
        document=document,
        job=job,
        readiness=derive_document_readiness(document, job),
    )


def _scope(tenant_id: object, knowledge_base_id: object) -> tuple[str, str]:
    return (_identifier(tenant_id), _identifier(knowledge_base_id))


def _scope_document(
    tenant_id: object,
    knowledge_base_id: object,
    document_id: object,
) -> tuple[str, str, str]:
    return (*_scope(tenant_id, knowledge_base_id), _identifier(document_id))


def _scope_job(
    tenant_id: object,
    knowledge_base_id: object,
    job_id: object,
) -> tuple[str, str, str]:
    return (*_scope(tenant_id, knowledge_base_id), _identifier(job_id))


def _identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value) > 255
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise DocumentRegistryError(DocumentOperationPhase.REGISTRY_CREATE)
    return value


def _require_one_changed(cursor: Any, phase: DocumentOperationPhase) -> None:
    if cursor.rowcount != 1:
        raise DocumentRegistryError(phase)


def _rollback_and_close(connection: Connection[Any]) -> None:
    try:
        connection.rollback()
    except Exception:
        pass
    try:
        connection.close()
    except Exception:
        pass
