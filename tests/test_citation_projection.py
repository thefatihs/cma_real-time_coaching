from datetime import UTC, datetime

import pytest

from app.coaching.coordinator import CoachingSourcePresentation
from app.ingestion.registry_models import (
    DocumentIngestionJob,
    DocumentIngestionPhase,
    DocumentIngestionState,
    DocumentReadiness,
    DocumentRegistryEntry,
    DocumentRegistryRecord,
)
from app.integration.citation_projection import SafeCoachingCitationProjector

NOW = datetime(2026, 8, 5, tzinfo=UTC)


def entry(document_id: str, filename: str, media_type: str) -> DocumentRegistryEntry:
    document = DocumentRegistryRecord(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        document_id=document_id,
        original_filename=filename,
        media_type=media_type,
        byte_size=10,
        storage_object_key=None,
        created_at_utc=NOW,
        ready_at_utc=NOW,
    )
    job = DocumentIngestionJob(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        document_id=document_id,
        job_id=f"job-{document_id}",
        state=DocumentIngestionState.SUCCEEDED,
        phase=DocumentIngestionPhase.FINALIZE,
        processed_chunks=1,
        total_chunks=1,
        attempt_count=1,
        created_at_utc=NOW,
        started_at_utc=NOW,
        updated_at_utc=NOW,
        finished_at_utc=NOW,
    )
    return DocumentRegistryEntry(
        document=document,
        job=job,
        readiness=DocumentReadiness.READY,
    )


class FakeRepository:
    def __init__(self, entries: tuple[DocumentRegistryEntry, ...]) -> None:
        self.entries = entries
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def get_entries_by_document_ids(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_ids: tuple[str, ...],
    ) -> tuple[DocumentRegistryEntry, ...]:
        self.calls.append((tenant_id, knowledge_base_id, document_ids))
        by_id = {item.document.document_id: item for item in self.entries}
        return tuple(by_id[item] for item in document_ids if item in by_id)


def test_projection_is_exact_scope_ordered_deduplicated_and_safe() -> None:
    repository = FakeRepository(
        (
            entry("doc-a", "guide.pdf", "application/pdf"),
            entry("doc-b", "notes.txt", "text/plain"),
            entry("doc-c", "readme.md", "text/markdown"),
        )
    )
    projector = SafeCoachingCitationProjector(repository)

    sources = projector.project(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        citation_document_ids=("doc-b", "doc-a", "doc-b", "missing", "doc-c"),
    )

    assert repository.calls == [
        ("tenant-a", "kb-a", ("doc-b", "doc-a", "missing", "doc-c"))
    ]
    assert tuple((item.original_filename, item.media_label) for item in sources) == (
        ("notes.txt", "TXT"),
        ("guide.pdf", "PDF"),
        ("readme.md", "Markdown"),
    )
    assert all(not hasattr(item, "document_id") for item in sources)


def test_projection_bounds_display_to_five() -> None:
    repository = FakeRepository(
        tuple(
            entry(f"doc-{index}", f"file-{index}.txt", "text/plain")
            for index in range(6)
        )
    )
    projector = SafeCoachingCitationProjector(repository)
    sources = projector.project(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        citation_document_ids=tuple(f"doc-{index}" for index in range(6)),
    )
    assert len(sources) == 5


@pytest.mark.parametrize("count", [0, 21])
def test_projection_rejects_out_of_bound_citation_count(count: int) -> None:
    repository = FakeRepository(())
    projector = SafeCoachingCitationProjector(repository)
    with pytest.raises(ValueError, match="citation input"):
        projector.project(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            citation_document_ids=tuple(f"doc-{index}" for index in range(count)),
        )
    assert repository.calls == []


@pytest.mark.parametrize("filename", ["../guide.pdf", "folder/guide.pdf", "bad\n.txt"])
def test_safe_presentation_rejects_path_or_control_filename(filename: str) -> None:
    with pytest.raises(ValueError, match="source filename"):
        CoachingSourcePresentation(filename, "PDF")
