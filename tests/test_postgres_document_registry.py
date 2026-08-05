from datetime import UTC, datetime
from typing import Any, cast

import pytest

from app.ingestion.postgres_registry import PsycopgDocumentRegistryRepository
from app.ingestion.registry_models import (
    DocumentIngestionPhase,
    DocumentIngestionState,
    DocumentOperationPhase,
    DocumentRegistryCreateRequest,
    DocumentRegistryError,
)
from app.vector_store.models import VectorBatchWriteResult


def row(
    *,
    state: str = "QUEUED",
    ready: bool = False,
    phase: str | None = None,
    processed_chunks: int | None = None,
    total_chunks: int = 1,
    document_id: str = "doc-a",
    now: datetime | None = None,
    storage_object_key: str | None = "objects/server-1",
) -> tuple[object, ...]:
    now = datetime.now(UTC) if now is None else now
    started = now if state != "QUEUED" else None
    finished = now if state in {"SUCCEEDED", "FAILED"} else None
    return (
        "tenant-a",
        "kb-a",
        document_id,
        "guide.txt",
        "text/plain",
        10,
        storage_object_key,
        now,
        now if ready else None,
        "job-a",
        state,
        phase or ("FINALIZE" if state == "SUCCEEDED" else "EMBEDDING"),
        (1 if state == "SUCCEEDED" else 0)
        if processed_chunks is None
        else processed_chunks,
        total_chunks,
        1 if started else 0,
        now,
        started,
        now,
        finished,
    )


class FakeCursor:
    def __init__(self, responses: list[list[tuple[object, ...]]]) -> None:
        self.responses = responses
        self.executions: list[tuple[str, object]] = []
        self.rowcount = 1

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: str, parameters: object = None) -> None:
        self.executions.append((" ".join(query.split()), parameters))

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.responses.pop(0)


class FakeConnection:
    autocommit = False

    def __init__(self, responses: list[list[tuple[object, ...]]]) -> None:
        self.cursor_value = FakeCursor(responses)
        self.commits = 0
        self.rollbacks = 0
        self.closes = 0

    def cursor(self) -> FakeCursor:
        return self.cursor_value

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closes += 1


def request(**changes: object) -> DocumentRegistryCreateRequest:
    values: dict[str, object] = {
        "tenant_id": "tenant-a",
        "knowledge_base_id": "kb-a",
        "document_id": "doc-a",
        "job_id": "job-a",
        "original_filename": "guide.txt",
        "media_type": "text/plain",
        "byte_size": 10,
        "sha256_hex": "a" * 64,
        "storage_object_key": "objects/server-1",
        "total_chunks": 1,
    }
    values.update(changes)
    return DocumentRegistryCreateRequest.model_validate(values)


def repository(connection: FakeConnection) -> PsycopgDocumentRegistryRepository:
    return PsycopgDocumentRegistryRepository(
        connection_factory=lambda: cast(Any, connection)
    )


def test_create_document_and_job_commit_atomically_with_exact_scope() -> None:
    connection = FakeConnection([[("doc-a",)], [row()]])
    result = repository(connection).create_or_get(request())
    assert result.created is True
    assert connection.commits == connection.closes == 1
    assert connection.rollbacks == 0
    executions = connection.cursor_value.executions
    assert "ON CONFLICT (tenant_id, knowledge_base_id, sha256_hex)" in executions[0][0]
    assert executions[0][1] == (
        "tenant-a",
        "kb-a",
        "doc-a",
        "guide.txt",
        "text/plain",
        10,
        "a" * 64,
        "objects/server-1",
    )
    assert executions[1][1] == (
        "tenant-a",
        "kb-a",
        "job-a",
        "doc-a",
        "EMBEDDING",
        1,
    )


def test_create_fresh_document_persists_null_source_key() -> None:
    connection = FakeConnection([[("doc-a",)], [row(storage_object_key=None)]])
    result = repository(connection).create_or_get(request(storage_object_key=None))

    assert result.entry.document.storage_object_key is None
    parameters = connection.cursor_value.executions[0][1]
    assert isinstance(parameters, tuple)
    assert parameters[-1] is None


def test_duplicate_sha_returns_existing_identity_without_job_insert() -> None:
    connection = FakeConnection([[], [row()]])
    result = repository(connection).create_or_get(request(document_id="doc-other"))
    assert result.created is False
    assert result.entry.document.document_id == "doc-a"
    assert len(connection.cursor_value.executions) == 2
    assert connection.cursor_value.executions[1][1] == ("tenant-a", "kb-a", "a" * 64)


def test_exact_delete_orders_vector_before_parent_and_returns_opaque_key() -> None:
    connection = FakeConnection([[row()]])
    result = repository(connection).delete_document(
        tenant_id="tenant-a", knowledge_base_id="kb-a", document_id="doc-a"
    )
    assert result is not None and result.storage_object_key == "objects/server-1"
    queries = [query for query, _ in connection.cursor_value.executions]
    assert "FOR UPDATE OF documents, jobs" in queries[0]
    assert "DELETE FROM callmetric_vector.vector_records" in queries[1]
    assert "DELETE FROM callmetric_vector.documents" in queries[2]
    assert all(
        parameters == ("tenant-a", "kb-a", "doc-a")
        for _, parameters in connection.cursor_value.executions
    )


def test_cross_scope_delete_is_non_disclosing_and_changes_nothing() -> None:
    connection = FakeConnection([[]])
    result = repository(connection).delete_document(
        tenant_id="tenant-other", knowledge_base_id="kb-a", document_id="doc-a"
    )
    assert result is None
    assert len(connection.cursor_value.executions) == 1


def test_delete_supports_fresh_document_without_source_key() -> None:
    connection = FakeConnection([[row(storage_object_key=None)]])

    result = repository(connection).delete_document(
        tenant_id="tenant-a", knowledge_base_id="kb-a", document_id="doc-a"
    )

    assert result is not None and result.storage_object_key is None


def test_interrupted_job_recovery_is_exact_scoped_and_terminal_safe() -> None:
    connection = FakeConnection([])
    connection.cursor_value.rowcount = 2

    changed = repository(connection).fail_interrupted_jobs(
        tenant_id="tenant-a", knowledge_base_id="kb-a"
    )

    assert changed == 2
    query, parameters = connection.cursor_value.executions[0]
    assert "state IN ('QUEUED', 'PROCESSING')" in query
    assert "state = 'FAILED'" in query
    assert "phase = 'FINALIZE'" in query
    assert parameters == ("tenant-a", "kb-a")


def test_connection_failure_is_fixed_and_secret_free() -> None:
    def fail() -> Any:
        raise RuntimeError("sensitive database detail")

    subject = PsycopgDocumentRegistryRepository(connection_factory=fail)
    with pytest.raises(DocumentRegistryError) as caught:
        subject.get_entry(
            tenant_id="tenant-a", knowledge_base_id="kb-a", document_id="doc-a"
        )
    assert caught.value.phase is DocumentOperationPhase.REGISTRY_CREATE
    assert "sensitive" not in str(caught.value)


def test_claim_uses_lock_and_bounded_atomic_transition() -> None:
    connection = FakeConnection([[row()], [row(state="PROCESSING")]])
    claimed = repository(connection).claim_queued_job(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        document_id="doc-a",
        job_id="job-a",
    )
    assert claimed is not None
    assert claimed.job.state is DocumentIngestionState.PROCESSING
    queries = connection.cursor_value.executions
    assert "FOR UPDATE OF documents, jobs" in queries[0][0]
    assert "attempt_count < %s" in queries[1][0]
    assert queries[1][1] == ("tenant-a", "kb-a", "job-a", 10)


def test_retry_accepts_only_failed_job_below_bound() -> None:
    connection = FakeConnection([[row(state="FAILED")], [row()]])
    retried = repository(connection).retry_failed_job(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        document_id="doc-a",
        job_id="job-a",
    )
    assert retried is not None and retried.job.state is DocumentIngestionState.QUEUED
    assert connection.cursor_value.executions[1][1] == (
        "tenant-a",
        "kb-a",
        "job-a",
        10,
    )


def test_invalid_claim_transition_rolls_back() -> None:
    connection = FakeConnection([[row(state="FAILED")]])
    with pytest.raises(DocumentRegistryError) as caught:
        repository(connection).claim_queued_job(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            document_id="doc-a",
            job_id="job-a",
        )
    assert caught.value.phase is DocumentOperationPhase.JOB_CLAIM
    assert connection.commits == 0
    assert connection.rollbacks == connection.closes == 1


def test_vector_failure_rolls_back_before_success_updates() -> None:
    connection = FakeConnection([[row(state="PROCESSING")]])

    def fail(_transaction: object) -> VectorBatchWriteResult:
        raise DocumentRegistryError(DocumentOperationPhase.VECTOR_WRITE)

    with pytest.raises(DocumentRegistryError) as caught:
        repository(connection).finalize_success(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            document_id="doc-a",
            job_id="job-a",
            total_chunks=1,
            vector_operation=fail,  # type: ignore[arg-type]
        )
    assert caught.value.phase is DocumentOperationPhase.VECTOR_WRITE
    assert connection.rollbacks == 1 and connection.commits == 0
    assert len(connection.cursor_value.executions) == 1


def test_failure_status_stores_only_fixed_phase() -> None:
    connection = FakeConnection([])
    changed = repository(connection).mark_failed(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        job_id="job-a",
        phase=DocumentIngestionPhase.VECTOR_WRITE,
    )
    assert changed is True
    _, parameters = connection.cursor_value.executions[0]
    assert parameters == ("VECTOR_WRITE", "tenant-a", "kb-a", "job-a")


def test_success_commits_vectors_then_job_and_ready_together() -> None:
    connection = FakeConnection([[row(state="PROCESSING")]])
    callback_execution_counts: list[int] = []

    def write(_transaction: object) -> VectorBatchWriteResult:
        callback_execution_counts.append(len(connection.cursor_value.executions))
        return VectorBatchWriteResult(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            inserted_identities=(),
            unchanged_identities=(),
        )

    repository(connection).finalize_success(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        document_id="doc-a",
        job_id="job-a",
        total_chunks=1,
        vector_operation=write,  # type: ignore[arg-type]
    )
    queries = connection.cursor_value.executions
    assert callback_execution_counts == [1]
    assert "SET state = 'SUCCEEDED'" in queries[1][0]
    assert "SET ready_at_utc = CURRENT_TIMESTAMP" in queries[2][0]
    assert connection.commits == 1 and connection.rollbacks == 0


def test_listing_and_job_polling_are_scoped_and_deterministic() -> None:
    list_connection = FakeConnection([[row()]])
    listed = repository(list_connection).list_documents(
        tenant_id="tenant-a", knowledge_base_id="kb-a"
    )
    assert len(listed) == 1
    list_query, list_parameters = list_connection.cursor_value.executions[0]
    assert "ORDER BY documents.created_at_utc DESC, documents.document_id" in list_query
    assert list_parameters == ("tenant-a", "kb-a", 51)

    job_connection = FakeConnection([[row(state="PROCESSING")]])
    job = repository(job_connection).get_job(
        tenant_id="tenant-a", knowledge_base_id="kb-a", job_id="job-a"
    )
    assert job is not None and job.state is DocumentIngestionState.PROCESSING
    assert job_connection.cursor_value.executions[0][1] == (
        "tenant-a",
        "kb-a",
        "job-a",
    )


def test_create_supports_explicit_staged_initial_phase() -> None:
    connection = FakeConnection([[("doc-a",)], [row(phase="STORAGE", total_chunks=0)]])
    repository(connection).create_or_get(
        request(initial_phase=DocumentIngestionPhase.STORAGE, total_chunks=0)
    )
    assert connection.cursor_value.executions[1][1] == (
        "tenant-a",
        "kb-a",
        "job-a",
        "doc-a",
        "STORAGE",
        0,
    )


def test_processing_progress_moves_forward_and_sets_total_once() -> None:
    connection = FakeConnection(
        [
            [row(state="PROCESSING", phase="EXTRACTION", total_chunks=0)],
            [row(state="PROCESSING", phase="CHUNKING", total_chunks=2)],
        ]
    )
    updated = repository(connection).update_processing_progress(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        job_id="job-a",
        phase=DocumentIngestionPhase.CHUNKING,
        processed_chunks=0,
        total_chunks=2,
    )
    assert updated is not None and updated.phase is DocumentIngestionPhase.CHUNKING
    assert connection.cursor_value.executions[1][1] == (
        "CHUNKING",
        0,
        2,
        "tenant-a",
        "kb-a",
        "job-a",
    )


@pytest.mark.parametrize(
    ("current", "requested"),
    tuple(
        zip(
            tuple(DocumentIngestionPhase),
            tuple(DocumentIngestionPhase)[1:],
        )
    ),
)
def test_every_adjacent_processing_phase_transition_is_allowed(
    current: DocumentIngestionPhase,
    requested: DocumentIngestionPhase,
) -> None:
    connection = FakeConnection(
        [
            [row(state="PROCESSING", phase=current.value)],
            [row(state="PROCESSING", phase=requested.value)],
        ]
    )
    updated = repository(connection).update_processing_progress(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        job_id="job-a",
        phase=requested,
        processed_chunks=0,
        total_chunks=1,
    )
    assert updated is not None and updated.phase is requested


def test_processing_progress_cannot_decrease() -> None:
    connection = FakeConnection(
        [
            [
                row(
                    state="PROCESSING",
                    phase="EMBEDDING",
                    processed_chunks=1,
                    total_chunks=2,
                )
            ]
        ]
    )
    with pytest.raises(DocumentRegistryError) as caught:
        repository(connection).update_processing_progress(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            job_id="job-a",
            phase=DocumentIngestionPhase.EMBEDDING,
            processed_chunks=0,
            total_chunks=2,
        )
    assert caught.value.phase is DocumentOperationPhase.PROGRESS


@pytest.mark.parametrize(
    ("current", "requested", "processed", "total"),
    (
        ("EMBEDDING", DocumentIngestionPhase.CHUNKING, 0, 2),
        ("CHUNKING", DocumentIngestionPhase.CHUNKING, 0, 3),
        ("CHUNKING", DocumentIngestionPhase.EMBEDDING, -1, 2),
    ),
)
def test_invalid_progress_is_fixed_and_rolls_back(
    current: str,
    requested: DocumentIngestionPhase,
    processed: int,
    total: int,
) -> None:
    connection = FakeConnection(
        [[row(state="PROCESSING", phase=current, total_chunks=2)]]
    )
    with pytest.raises(DocumentRegistryError) as caught:
        repository(connection).update_processing_progress(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            job_id="job-a",
            phase=requested,
            processed_chunks=processed,
            total_chunks=total,
        )
    assert caught.value.phase is DocumentOperationPhase.PROGRESS
    assert connection.commits == 0


@pytest.mark.parametrize("state", ("SUCCEEDED", "FAILED", "CANCELLED"))
def test_terminal_jobs_reject_progress_and_cancellation(state: str) -> None:
    progress_connection = FakeConnection(
        [[row(state=state, ready=state == "SUCCEEDED")]]
    )
    with pytest.raises(DocumentRegistryError) as progress_error:
        repository(progress_connection).update_processing_progress(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            job_id="job-a",
            phase=DocumentIngestionPhase.FINALIZE,
            processed_chunks=1,
            total_chunks=1,
        )
    assert progress_error.value.phase is DocumentOperationPhase.PROGRESS

    cancel_connection = FakeConnection([[row(state=state, ready=state == "SUCCEEDED")]])
    with pytest.raises(DocumentRegistryError) as cancel_error:
        repository(cancel_connection).mark_cancelled(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            job_id="job-a",
            phase=DocumentIngestionPhase.FINALIZE,
        )
    assert cancel_error.value.phase is DocumentOperationPhase.CANCEL


@pytest.mark.parametrize("state", ("QUEUED", "PROCESSING"))
def test_active_job_cancellation_is_exact_scoped_and_reason_free(state: str) -> None:
    connection = FakeConnection([[row(state=state)]])
    assert repository(connection).mark_cancelled(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        job_id="job-a",
        phase=DocumentIngestionPhase.EMBEDDING,
    )
    query, parameters = connection.cursor_value.executions[1]
    assert "state IN ('QUEUED', 'PROCESSING')" in query
    assert parameters == ("EMBEDDING", "tenant-a", "kb-a", "job-a")
    assert "reason" not in query.casefold() and "exception" not in query.casefold()


def test_cross_scope_progress_and_cancel_are_non_disclosing() -> None:
    progress_connection = FakeConnection([[]])
    assert (
        repository(progress_connection).update_processing_progress(
            tenant_id="tenant-other",
            knowledge_base_id="kb-a",
            job_id="job-a",
            phase=DocumentIngestionPhase.EMBEDDING,
            processed_chunks=0,
            total_chunks=1,
        )
        is None
    )
    cancel_connection = FakeConnection([[]])
    assert not repository(cancel_connection).mark_cancelled(
        tenant_id="tenant-other",
        knowledge_base_id="kb-a",
        job_id="job-a",
        phase=DocumentIngestionPhase.EMBEDDING,
    )


def test_cancellation_inside_final_transaction_rolls_back_before_vectors() -> None:
    connection = FakeConnection([[row(state="PROCESSING")]])
    vector_calls: list[object] = []
    with pytest.raises(DocumentRegistryError) as caught:
        repository(connection).finalize_success(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            document_id="doc-a",
            job_id="job-a",
            total_chunks=1,
            vector_operation=lambda transaction: vector_calls.append(transaction),  # type: ignore[arg-type,return-value]
            cancellation_requested=lambda: True,
        )
    assert caught.value.phase is DocumentOperationPhase.CANCEL
    assert vector_calls == []
    assert connection.rollbacks == 1 and connection.commits == 0


def test_cancellation_after_vector_callback_rolls_back_complete_batch() -> None:
    connection = FakeConnection([[row(state="PROCESSING")]])
    checks = iter((False, True))
    vector_calls: list[object] = []

    def write(transaction: object) -> VectorBatchWriteResult:
        vector_calls.append(transaction)
        return VectorBatchWriteResult(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            inserted_identities=(),
            unchanged_identities=(),
        )

    with pytest.raises(DocumentRegistryError) as caught:
        repository(connection).finalize_success(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            document_id="doc-a",
            job_id="job-a",
            total_chunks=1,
            vector_operation=write,  # type: ignore[arg-type]
            cancellation_requested=lambda: next(checks),
        )
    assert caught.value.phase is DocumentOperationPhase.CANCEL
    assert len(vector_calls) == 1
    assert len(connection.cursor_value.executions) == 1
    assert connection.rollbacks == 1 and connection.commits == 0


def test_keyset_listing_is_bounded_and_cursor_repr_hides_identity() -> None:
    first_time = datetime(2026, 1, 2, tzinfo=UTC)
    second_time = datetime(2026, 1, 1, tzinfo=UTC)
    connection = FakeConnection(
        [
            [
                row(document_id="doc-a", now=first_time),
                row(document_id="doc-b", now=second_time),
                row(document_id="doc-c", now=second_time),
            ]
        ]
    )
    page = repository(connection).list_document_page(
        tenant_id="tenant-a", knowledge_base_id="kb-a", page_size=2
    )
    assert len(page.entries) == 2 and page.continuation is not None
    assert "doc-b" not in repr(page.continuation)

    next_connection = FakeConnection([[]])
    repository(next_connection).list_document_page(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        page_size=2,
        cursor=page.continuation,
    )
    query, parameters = next_connection.cursor_value.executions[0]
    assert "OFFSET" not in query
    assert parameters == (
        "tenant-a",
        "kb-a",
        second_time,
        second_time,
        "doc-b",
        3,
    )


@pytest.mark.parametrize("page_size", (0, 51, True))
def test_document_listing_rejects_invalid_page_size(page_size: object) -> None:
    with pytest.raises(DocumentRegistryError) as caught:
        repository(FakeConnection([])).list_document_page(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            page_size=page_size,  # type: ignore[arg-type]
        )
    assert caught.value.phase is DocumentOperationPhase.LIST
