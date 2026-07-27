"""Provider-neutral embedding boundaries."""

from app.embeddings.protocols import DocumentEmbedder, QueryEmbedder
from app.embeddings.sentence_transformers import (
    SentenceTransformerBackend,
    SentenceTransformerQueryEmbedder,
    SentenceTransformerQueryEmbedderConfig,
)

__all__ = [
    "DocumentEmbedder",
    "QueryEmbedder",
    "SentenceTransformerBackend",
    "SentenceTransformerQueryEmbedder",
    "SentenceTransformerQueryEmbedderConfig",
]
