"""Deterministic synthetic-only in-memory retrieval implementation."""

from collections.abc import Iterable

from app.retrieval.models import RetrievalDocument, RetrievalResult


class InMemoryRetriever:
    def __init__(self, documents: Iterable[RetrievalDocument] = ()) -> None:
        stored = tuple(documents)
        identities: set[tuple[str, str, str]] = set()
        for document in stored:
            identity = (
                document.tenant_id,
                document.knowledge_base_id,
                document.chunk_id,
            )
            if identity in identities:
                raise ValueError("duplicate chunk_id within tenant and knowledge base")
            identities.add(identity)
        self._documents = stored

    def retrieve(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        query: str,
        top_k: int,
        minimum_score: float = 0.0,
    ) -> RetrievalResult:
        tenant = _required_text(tenant_id, "tenant_id")
        knowledge_base = _required_text(
            knowledge_base_id,
            "knowledge_base_id",
        )
        _required_text(query, "query")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0 <= minimum_score <= 1:
            raise ValueError("minimum_score must be between 0 and 1")

        matches = (
            document
            for document in self._documents
            if document.tenant_id == tenant
            and document.knowledge_base_id == knowledge_base
            and document.score >= minimum_score
        )
        ordered = sorted(
            matches,
            key=lambda document: (
                -document.score,
                document.document_id,
                document.chunk_id,
            ),
        )
        return RetrievalResult(
            tenant_id=tenant,
            knowledge_base_id=knowledge_base,
            documents=tuple(ordered[:top_k]),
        )


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
