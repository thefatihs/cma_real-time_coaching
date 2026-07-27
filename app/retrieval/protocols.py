"""Provider-neutral retrieval interface."""

from typing import Protocol

from app.retrieval.models import RetrievalResult


class Retriever(Protocol):
    def retrieve(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        query: str,
        top_k: int,
        minimum_score: float = 0.0,
    ) -> RetrievalResult: ...
