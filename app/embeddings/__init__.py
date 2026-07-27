"""Provider-neutral embedding boundaries."""

from app.embeddings.protocols import QueryEmbedder
from app.embeddings.sentence_transformers import (
    SentenceTransformerBackend,
    SentenceTransformerQueryEmbedder,
    SentenceTransformerQueryEmbedderConfig,
)

__all__ = [
    "QueryEmbedder",
    "SentenceTransformerBackend",
    "SentenceTransformerQueryEmbedder",
    "SentenceTransformerQueryEmbedderConfig",
]
