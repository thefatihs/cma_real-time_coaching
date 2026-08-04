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


def row(*, state: str = "QUEUED", ready: bool = False) -> tuple[object, ...]:
    now = datetime.now(UTC)
    started = now if state != "QUEUED" else None
    finished = now if state in {"SUCCEEDED", "FAILED"} else None
    return (
        "tenant-a",
        "kb-a",
        "doc-a",
        "guide.txt",
        "text/plain",
        10,
        "objects/server-1",
        now,
        now if ready else None,
        "job-a",
        state,
        "FINALIZE" if state == "SUCCEEDED" else "EMBEDDING",
        1 if state == "SUCCEEDED" else 0,
        1,
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
    assert executions[1][1] == ("tenant-a", "kb-a", "job-a", "doc-a", 1)


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
    assert list_parameters == ("tenant-a", "kb-a")

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
