"""Bounded projection of internal citations into safe coaching sources."""

from dataclasses import dataclass
from typing import Protocol

from app.coaching.coordinator import CoachingSourcePresentation
from app.events.models import CoachingSuggestionEvent
from app.ingestion.registry_models import DocumentRegistryEntry

MAX_INTERNAL_CITATIONS = 20
MAX_PRESENTED_SOURCES = 5
_MEDIA_LABELS = {
    "application/pdf": "PDF",
    "text/plain": "TXT",
    "text/markdown": "Markdown",
}


class CitationRegistryReader(Protocol):
    def get_entries_by_document_ids(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_ids: tuple[str, ...],
    ) -> tuple[DocumentRegistryEntry, ...]: ...


@dataclass(frozen=True, slots=True)
class GroundedCoachingSuggestion:
    event: CoachingSuggestionEvent
    citation_document_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.citation_document_ids or len(self.citation_document_ids) > 20:
            raise ValueError("grounded citations are invalid")


class SafeCoachingCitationProjector:
    def __init__(self, repository: CitationRegistryReader) -> None:
        self._repository = repository

    def project(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        citation_document_ids: tuple[str, ...],
    ) -> tuple[CoachingSourcePresentation, ...]:
        if (
            not citation_document_ids
            or len(citation_document_ids) > MAX_INTERNAL_CITATIONS
        ):
            raise ValueError("citation input is invalid")
        unique_ids = tuple(dict.fromkeys(citation_document_ids))
        entries = self._repository.get_entries_by_document_ids(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            document_ids=unique_ids,
        )
        return tuple(
            CoachingSourcePresentation(
                original_filename=entry.document.original_filename,
                media_label=_MEDIA_LABELS[entry.document.media_type],
            )
            for entry in entries[:MAX_PRESENTED_SOURCES]
        )
