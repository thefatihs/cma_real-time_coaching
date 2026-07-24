"""Provider-neutral vector-store interface."""

from typing import Protocol

from app.vector_store.models import SearchRequest, SearchResult, VectorRecord


class VectorStore(Protocol):
    def upsert(self, record: VectorRecord) -> None: ...

    def search(self, request: SearchRequest) -> SearchResult: ...
