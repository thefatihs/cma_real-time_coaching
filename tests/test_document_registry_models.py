from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.ingestion.registry_models import (
    DocumentIngestionJob,
    DocumentIngestionPhase,
    DocumentIngestionState,
    DocumentReadiness,
    DocumentRegistryCreateRequest,
    DocumentRegistryEntry,
    DocumentRegistryRecord,
    validate_storage_object_key,
)


def create_request(**changes: object) -> DocumentRegistryCreateRequest:
    values: dict[str, object] = {
        "tenant_id": "tenant-a",
        "knowledge_base_id": "kb-a",
        "document_id": "doc-a",
        "job_id": "job-a",
        "original_filename": "guide.txt",
        "media_type": "text/plain",
        "byte_size": 10,
        "sha256_hex": "a" * 64,
        "storage_object_key": "documents/server-object-1",
        "total_chunks": 1,
    }
    values.update(changes)
    return DocumentRegistryCreateRequest.model_validate(values)


@pytest.mark.parametrize(
    "value",
    ("", "/absolute", "C:/drive", "a\\b", "a/../b", "a//b", "a/", "a\x00b"),
)
def test_storage_object_key_rejects_path_semantics(value: str) -> None:
    with pytest.raises(ValueError, match="storage_object_key is invalid"):
        validate_storage_object_key(value)


def test_create_request_accepts_only_scoped_lowercase_digest() -> None:
    assert create_request().sha256_hex == "a" * 64
    with pytest.raises(ValidationError):
        create_request(sha256_hex="A" * 64)
    with pytest.raises(ValidationError):
        create_request(media_type="application/octet-stream")


def test_registry_entry_derives_ready_state_without_digest_exposure() -> None:
    now = datetime.now(UTC)
    document = DocumentRegistryRecord(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        document_id="doc-a",
        original_filename="guide.pdf",
        media_type="application/pdf",
        byte_size=10,
        storage_object_key="objects/server-1",
        created_at_utc=now,
        ready_at_utc=now,
    )
    job = DocumentIngestionJob(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        document_id="doc-a",
        job_id="job-a",
        state=DocumentIngestionState.SUCCEEDED,
        phase=DocumentIngestionPhase.FINALIZE,
        processed_chunks=1,
        total_chunks=1,
        attempt_count=1,
        created_at_utc=now,
        started_at_utc=now,
        updated_at_utc=now,
        finished_at_utc=now,
    )
    entry = DocumentRegistryEntry(
        document=document, job=job, readiness=DocumentReadiness.READY
    )
    assert "sha256" not in entry.model_dump()["document"]
