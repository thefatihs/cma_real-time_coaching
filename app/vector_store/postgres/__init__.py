"""SQL-free PostgreSQL vector boundary contracts and codecs."""

from app.vector_store.postgres.codecs import (
    canonicalize_float32_embedding,
    cosine_distance_to_relevance,
    cosine_minimum_score_to_maximum_distance,
    decode_ordered_metadata,
    encode_ordered_metadata,
    order_cosine_search_rows,
)
from app.vector_store.postgres.contracts import (
    PostgreSQLCosineSearchRow,
    PostgreSQLStoredVectorRow,
    PostgreSQLVectorTransaction,
    PostgreSQLVectorTransactionRunner,
)

__all__ = [
    "PostgreSQLCosineSearchRow",
    "PostgreSQLStoredVectorRow",
    "PostgreSQLVectorTransaction",
    "PostgreSQLVectorTransactionRunner",
    "canonicalize_float32_embedding",
    "cosine_distance_to_relevance",
    "cosine_minimum_score_to_maximum_distance",
    "decode_ordered_metadata",
    "encode_ordered_metadata",
    "order_cosine_search_rows",
]
