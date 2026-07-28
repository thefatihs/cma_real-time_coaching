"""SQL-free PostgreSQL vector boundary contracts and codecs."""

from app.vector_store.postgres.adapter import (
    ProfileBoundPostgreSQLVectorStore as ProfileBoundPostgreSQLVectorStore,
)
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
from app.vector_store.postgres.profile_repository import (
    PostgreSQLEmbeddingProfileRepository as PostgreSQLEmbeddingProfileRepository,
)
from app.vector_store.postgres.readiness import (
    PostgreSQLSchemaReadinessChecker as PostgreSQLSchemaReadinessChecker,
)

__all__ = [
    "PostgreSQLCosineSearchRow",
    "PostgreSQLSchemaReadinessChecker",
    "PostgreSQLStoredVectorRow",
    "PostgreSQLVectorTransaction",
    "PostgreSQLVectorTransactionRunner",
    "ProfileBoundPostgreSQLVectorStore",
    "canonicalize_float32_embedding",
    "cosine_distance_to_relevance",
    "cosine_minimum_score_to_maximum_distance",
    "decode_ordered_metadata",
    "encode_ordered_metadata",
    "order_cosine_search_rows",
]
