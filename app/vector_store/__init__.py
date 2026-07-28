"""Provider-neutral vector-store foundation."""

from app.vector_store.embedding_profile import (
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
)
from app.vector_store.in_memory import InMemoryVectorStore
from app.vector_store.models import (
    SearchRequest,
    SearchResult,
    VectorBatchWriteRequest,
    VectorBatchWriteResult,
    VectorRecord,
    VectorRecordIdentity,
    VectorSearchHit,
)
from app.vector_store.protocols import AtomicVectorBatchWriter, VectorStore

__all__ = [
    "AtomicVectorBatchWriter",
    "EmbeddingDistanceMetric",
    "InMemoryVectorStore",
    "KnowledgeBaseEmbeddingProfile",
    "SearchRequest",
    "SearchResult",
    "VectorBatchWriteRequest",
    "VectorBatchWriteResult",
    "VectorRecord",
    "VectorRecordIdentity",
    "VectorSearchHit",
    "VectorStore",
]
