"""Safe document presentation projection tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.ingestion.registry_models import (
    DocumentIngestionJob,
    DocumentIngestionPhase,
    DocumentIngestionState,
    DocumentReadiness,
    DocumentRegistryEntry,
    DocumentRegistryRecord,
)
from live_dashboard.document_view_models import (
    DocumentRuntimeStatus,
    RUNTIME_STATUS_MESSAGES,
    project_document,
    project_progress,
    safe_upload_selection,
)


def _entry(
    state: DocumentIngestionState,
    phase: DocumentIngestionPhase = DocumentIngestionPhase.EXTRACTION,
) -> DocumentRegistryEntry:
    now = datetime(2026, 8, 5, 10, 30, tzinfo=UTC)
    terminal = state in {
        DocumentIngestionState.SUCCEEDED,
        DocumentIngestionState.FAILED,
        DocumentIngestionState.CANCELLED,
    }
    return DocumentRegistryEntry(
        document=DocumentRegistryRecord(
            tenant_id="tenant-private",
            knowledge_base_id="kb-private",
            document_id="document-private",
            original_filename="rehber.pdf",
            media_type="application/pdf",
            byte_size=2048,
            storage_object_key="obj_" + "a" * 64,
            created_at_utc=now,
            ready_at_utc=now if state is DocumentIngestionState.SUCCEEDED else None,
        ),
        job=DocumentIngestionJob(
            tenant_id="tenant-private",
            knowledge_base_id="kb-private",
            job_id="job-private",
            document_id="document-private",
            state=state,
            phase=DocumentIngestionPhase.FINALIZE
            if state is DocumentIngestionState.SUCCEEDED
            else phase,
            processed_chunks=3 if state is DocumentIngestionState.SUCCEEDED else 1,
            total_chunks=3,
            attempt_count=0 if state is DocumentIngestionState.QUEUED else 1,
            created_at_utc=now,
            started_at_utc=None if state is DocumentIngestionState.QUEUED else now,
            updated_at_utc=now,
            finished_at_utc=now if terminal else None,
        ),
        readiness={
            DocumentIngestionState.SUCCEEDED: DocumentReadiness.READY,
            DocumentIngestionState.FAILED: DocumentReadiness.FAILED,
            DocumentIngestionState.CANCELLED: DocumentReadiness.CANCELLED,
        }.get(state, DocumentReadiness.PENDING),
    )


@pytest.mark.parametrize(
    ("filename", "media_type", "label"),
    [
        ("a.pdf", "application/pdf", "PDF"),
        ("a.txt", "text/plain", "TXT"),
        ("a.md", "text/markdown", "Markdown"),
    ],
)
def test_valid_upload_projection_is_display_only(
    filename: str, media_type: str, label: str
) -> None:
    result = safe_upload_selection(
        filename=filename, media_type=media_type, byte_size=1024
    )
    assert result.media_label == label
    assert result.formatted_size == "1.0 KiB"
    assert not hasattr(result, "content")
    assert not hasattr(result, "digest")


@pytest.mark.parametrize(
    ("filename", "media_type", "size", "message"),
    [
        ("../a.pdf", "application/pdf", 1, "Belge adı geçersiz."),
        ("a.exe", "application/octet-stream", 1, "Yalnızca PDF"),
        ("a.txt", "text/plain", 10 * 1024 * 1024 + 1, "10 MiB"),
    ],
)
def test_invalid_upload_messages_are_fixed(
    filename: str, media_type: str, size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        safe_upload_selection(filename=filename, media_type=media_type, byte_size=size)


def test_authoritative_progress_and_document_projection_hide_internal_values() -> None:
    entry = _entry(DocumentIngestionState.PROCESSING, DocumentIngestionPhase.EMBEDDING)
    progress = project_progress(entry)
    document = project_document(entry, action_token="opaque-action")
    rendered = repr((progress, document))
    assert progress.label == "Embedding oluşturuluyor"
    assert (progress.processed_chunks, progress.total_chunks) == (1, 3)
    assert document.readiness_label == "İşleniyor"
    assert document.created_at_utc == "2026-08-05 10:30 UTC"
    for private in (
        "tenant-private",
        "kb-private",
        "document-private",
        "job-private",
        "obj_",
    ):
        assert private not in rendered


def test_runtime_messages_are_exact_and_do_not_claim_retrieval() -> None:
    assert RUNTIME_STATUS_MESSAGES == {
        DocumentRuntimeStatus.READY: "Bilgi tabanı hazır",
        DocumentRuntimeStatus.DISABLED: "Bilgi tabanı yapılandırılmadı",
        DocumentRuntimeStatus.UNAVAILABLE: (
            "Bilgi tabanı geçici olarak kullanılamıyor; "
            "temel görüşme analizi devam ediyor"
        ),
    }
    assert all(
        "retrieval" not in value.casefold()
        for value in RUNTIME_STATUS_MESSAGES.values()
    )
