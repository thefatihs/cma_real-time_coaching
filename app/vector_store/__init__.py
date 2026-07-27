"""Provider-neutral vector-store foundation."""

from app.vector_store.in_memory import InMemoryVectorStore
from app.vector_store.models import (
    SearchRequest,
    SearchResult,
    VectorRecord,
    VectorSearchHit,
)
from app.vector_store.protocols import VectorStore

__all__ = [
    "InMemoryVectorStore",
    "SearchRequest",
    "SearchResult",
    "VectorRecord",
    "VectorSearchHit",
    "VectorStore",
]
