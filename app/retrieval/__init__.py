"""Tenant-aware retrieval boundaries and in-memory implementation."""

from app.retrieval.in_memory import InMemoryRetriever
from app.retrieval.models import RetrievalDocument, RetrievalResult
from app.retrieval.protocols import Retriever
from app.retrieval.vector_backed import VectorBackedRetriever

__all__ = [
    "InMemoryRetriever",
    "RetrievalDocument",
    "RetrievalResult",
    "Retriever",
    "VectorBackedRetriever",
]
