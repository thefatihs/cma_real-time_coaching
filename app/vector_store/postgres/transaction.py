"""Complete Psycopg transaction operations for the fixed vector-store schema."""

import hashlib
import math
from numbers import Real
from typing import Any

from pgvector import Vector
from psycopg import Connection, sql

from app.vector_store.embedding_profile import (
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
)
from app.vector_store.models import VectorRecordIdentity
from app.vector_store.postgres.codecs import (
    canonicalize_float32_embedding,
    decode_ordered_metadata,
    encode_ordered_metadata,
)
from app.vector_store.postgres.contracts import (
    PostgreSQLCosineSearchRow,
    PostgreSQLStoredVectorRow,
)

_PROFILE_TABLE = sql.Identifier("callmetric_vector", "embedding_profiles")
_RECORD_TABLE = sql.Identifier("callmetric_vector", "vector_records")
RecordParameters = tuple[str, str, str, str, str, int, Vector, str]

_LOCK_SQL = sql.SQL("SELECT pg_advisory_xact_lock(%s)")
_PROFILE_SELECT = sql.SQL(
    """
    SELECT tenant_id, knowledge_base_id, model_id, vector_dimension,
           normalize_embeddings, distance_metric
    FROM {table}
    WHERE tenant_id = %s AND knowledge_base_id = %s
    """
).format(table=_PROFILE_TABLE)
_PROFILE_INSERT = sql.SQL(
    """
    INSERT INTO {table} (
        tenant_id, knowledge_base_id, model_id, vector_dimension,
        normalize_embeddings, distance_metric
    )
    VALUES (%s, %s, %s, %s, %s, %s)
    """
).format(table=_PROFILE_TABLE)
_RECORD_SELECT = sql.SQL(
    """
    WITH requested(document_id, chunk_id, ordinal) AS (
        SELECT *
        FROM unnest(%s::text[], %s::text[]) WITH ORDINALITY
    )
    SELECT records.tenant_id, records.knowledge_base_id, records.document_id,
           records.chunk_id, records.text, records.vector_dimension,
           records.embedding, records.metadata_json::text
    FROM {table} AS records
    INNER JOIN requested
        ON requested.document_id = records.document_id
       AND requested.chunk_id = records.chunk_id
    WHERE records.tenant_id = %s AND records.knowledge_base_id = %s
    ORDER BY records.document_id, records.chunk_id
    """
).format(table=_RECORD_TABLE)
_RECORD_INSERT = sql.SQL(
    """
    INSERT INTO {table} (
        tenant_id, knowledge_base_id, document_id, chunk_id, text,
        vector_dimension, embedding, metadata_json
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
    """
).format(table=_RECORD_TABLE)
_RECORD_REPLACE = sql.SQL(
    """
    INSERT INTO {table} (
        tenant_id, knowledge_base_id, document_id, chunk_id, text,
        vector_dimension, embedding, metadata_json
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
    ON CONFLICT (tenant_id, knowledge_base_id, document_id, chunk_id)
    DO UPDATE SET
        text = EXCLUDED.text,
        vector_dimension = EXCLUDED.vector_dimension,
        embedding = EXCLUDED.embedding,
        metadata_json = EXCLUDED.metadata_json
    """
).format(table=_RECORD_TABLE)
_COSINE_SEARCH = sql.SQL(
    """
    WITH scored AS (
        SELECT tenant_id, knowledge_base_id, document_id, chunk_id, text,
               vector_dimension, embedding, metadata_json::text,
               embedding <=> %s AS cosine_distance
        FROM {table}
        WHERE tenant_id = %s AND knowledge_base_id = %s
    )
    SELECT tenant_id, knowledge_base_id, document_id, chunk_id, text,
           vector_dimension, embedding, metadata_json, cosine_distance
    FROM scored
    WHERE cosine_distance <= %s
    ORDER BY cosine_distance, document_id, chunk_id
    LIMIT %s
    """
).format(table=_RECORD_TABLE)


class PsycopgPostgreSQLVectorTransaction:
    """Execute SQL within a runner-owned Psycopg transaction."""

    def __init__(self, connection: Connection[Any]) -> None:
        if not callable(getattr(connection, "cursor", None)):
            raise ValueError("connection.cursor must be callable")
        self._connection = connection

    def acquire_scope_lock(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> None:
        tenant = _canonical_text(tenant_id, "tenant_id")
        knowledge_base = _canonical_text(knowledge_base_id, "knowledge_base_id")
        lock_key = _scope_lock_key(tenant, knowledge_base)
        with self._connection.cursor() as cursor:
            cursor.execute(_LOCK_SQL, (lock_key,))

    def get_profile(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        for_update: bool,
    ) -> KnowledgeBaseEmbeddingProfile | None:
        tenant = _canonical_text(tenant_id, "tenant_id")
        knowledge_base = _canonical_text(knowledge_base_id, "knowledge_base_id")
        if type(for_update) is not bool:
            raise ValueError("for_update must be a boolean")
        query = (
            _PROFILE_SELECT + sql.SQL(" FOR UPDATE") if for_update else _PROFILE_SELECT
        )
        with self._connection.cursor() as cursor:
            cursor.execute(query, (tenant, knowledge_base))
            rows = cursor.fetchall()
        if len(rows) > 1:
            raise ValueError("profile query returned more than one row")
        if not rows:
            return None
        return _profile_from_row(rows[0])

    def insert_profile(
        self,
        profile: KnowledgeBaseEmbeddingProfile,
    ) -> None:
        values = _profile_values(profile)
        with self._connection.cursor() as cursor:
            cursor.execute(_PROFILE_INSERT, values)

    def get_records(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        identities: tuple[VectorRecordIdentity, ...],
    ) -> tuple[PostgreSQLStoredVectorRow, ...]:
        tenant = _canonical_text(tenant_id, "tenant_id")
        knowledge_base = _canonical_text(knowledge_base_id, "knowledge_base_id")
        identity_values = _identity_values(identities)
        if not identity_values:
            return ()
        requested = set(identity_values)
        document_ids = [identity[0] for identity in identity_values]
        chunk_ids = [identity[1] for identity in identity_values]
        with self._connection.cursor() as cursor:
            cursor.execute(
                _RECORD_SELECT,
                (document_ids, chunk_ids, tenant, knowledge_base),
            )
            rows = cursor.fetchall()
        result: list[PostgreSQLStoredVectorRow] = []
        returned: set[tuple[str, str]] = set()
        for raw_row in rows:
            row = _stored_row_from_database(
                raw_row,
                tenant_id=tenant,
                knowledge_base_id=knowledge_base,
            )
            identity = (row.document_id, row.chunk_id)
            if identity not in requested:
                raise ValueError("record query returned an unexpected identity")
            if identity in returned:
                raise ValueError("record query returned a duplicate identity")
            returned.add(identity)
            result.append(row)
        expected_order = sorted(
            result,
            key=lambda row: (row.document_id, row.chunk_id),
        )
        if result != expected_order:
            raise ValueError("record query rows are not deterministically ordered")
        return tuple(result)

    def insert_records(
        self,
        rows: tuple[PostgreSQLStoredVectorRow, ...],
    ) -> None:
        values = _batch_record_values(rows)
        with self._connection.cursor() as cursor:
            cursor.executemany(_RECORD_INSERT, values)

    def replace_record(
        self,
        row: PostgreSQLStoredVectorRow,
    ) -> None:
        values = _record_values(row)
        with self._connection.cursor() as cursor:
            cursor.execute(_RECORD_REPLACE, values)

    def search_cosine(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        query_embedding: tuple[float, ...],
        top_k: int,
        maximum_cosine_distance: float,
    ) -> tuple[PostgreSQLCosineSearchRow, ...]:
        tenant = _canonical_text(tenant_id, "tenant_id")
        knowledge_base = _canonical_text(knowledge_base_id, "knowledge_base_id")
        query = _canonical_embedding(query_embedding, "query_embedding")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        maximum_distance = _cosine_distance(
            maximum_cosine_distance,
            "maximum_cosine_distance",
        )
        with self._connection.cursor() as cursor:
            cursor.execute(
                _COSINE_SEARCH,
                (
                    Vector(list(query)),
                    tenant,
                    knowledge_base,
                    maximum_distance,
                    top_k,
                ),
            )
            rows = cursor.fetchall()
        if len(rows) > top_k:
            raise ValueError("cosine search returned more than top_k rows")
        result: list[PostgreSQLCosineSearchRow] = []
        identities: set[tuple[str, str]] = set()
        for raw_row in rows:
            row = _search_row_from_database(
                raw_row,
                tenant_id=tenant,
                knowledge_base_id=knowledge_base,
            )
            identity = (row.document_id, row.chunk_id)
            if identity in identities:
                raise ValueError("cosine search returned a duplicate identity")
            identities.add(identity)
            result.append(row)
        order = [(row.cosine_distance, row.document_id, row.chunk_id) for row in result]
        if order != sorted(order):
            raise ValueError("cosine search rows are not deterministically ordered")
        return tuple(result)


def _scope_lock_key(tenant_id: str, knowledge_base_id: str) -> int:
    tenant_bytes = tenant_id.encode("utf-8")
    knowledge_base_bytes = knowledge_base_id.encode("utf-8")
    payload = (
        len(tenant_bytes).to_bytes(8, "big")
        + tenant_bytes
        + len(knowledge_base_bytes).to_bytes(8, "big")
        + knowledge_base_bytes
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def _profile_values(profile: object) -> tuple[str, str, str, int, bool, str]:
    if not isinstance(profile, KnowledgeBaseEmbeddingProfile):
        raise ValueError("profile must be a KnowledgeBaseEmbeddingProfile")
    tenant_id = _canonical_text(profile.tenant_id, "profile tenant_id")
    knowledge_base_id = _canonical_text(
        profile.knowledge_base_id,
        "profile knowledge_base_id",
    )
    model_id = _canonical_text(profile.model_id, "profile model_id")
    vector_dimension = _positive_integer(
        profile.vector_dimension,
        "profile vector_dimension",
    )
    if type(profile.normalize_embeddings) is not bool:
        raise ValueError("profile normalize_embeddings must be a boolean")
    if profile.distance_metric is not EmbeddingDistanceMetric.COSINE:
        raise ValueError("profile distance_metric must be cosine")
    return (
        tenant_id,
        knowledge_base_id,
        model_id,
        vector_dimension,
        profile.normalize_embeddings,
        profile.distance_metric.value,
    )


def _profile_from_row(row: object) -> KnowledgeBaseEmbeddingProfile:
    values = _row_values(row, expected_arity=6, field_name="profile row")
    tenant_id = _canonical_text(values[0], "stored profile tenant_id")
    knowledge_base_id = _canonical_text(
        values[1],
        "stored profile knowledge_base_id",
    )
    model_id = _canonical_text(values[2], "stored profile model_id")
    vector_dimension = _positive_integer(
        values[3],
        "stored profile vector_dimension",
    )
    if type(values[4]) is not bool:
        raise ValueError("stored profile normalize_embeddings must be a boolean")
    if values[5] != EmbeddingDistanceMetric.COSINE.value:
        raise ValueError("stored profile distance_metric must be cosine")
    return KnowledgeBaseEmbeddingProfile(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        model_id=model_id,
        vector_dimension=vector_dimension,
        normalize_embeddings=values[4],
        distance_metric=EmbeddingDistanceMetric.COSINE,
    )


def _identity_values(
    identities: object,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(identities, tuple):
        raise ValueError("identities must be a tuple")
    values: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for identity in identities:
        if not isinstance(identity, VectorRecordIdentity):
            raise ValueError("identities must contain VectorRecordIdentity objects")
        value = (
            _canonical_text(identity.document_id, "identity document_id"),
            _canonical_text(identity.chunk_id, "identity chunk_id"),
        )
        if value in seen:
            raise ValueError("record identities must be unique")
        seen.add(value)
        values.append(value)
    return tuple(values)


def _batch_record_values(
    rows: object,
) -> tuple[RecordParameters, ...]:
    if not isinstance(rows, tuple) or not rows:
        raise ValueError("rows must be a nonempty tuple")
    canonical: list[tuple[tuple[str, str], tuple[str, str], int, RecordParameters]] = []
    scope: tuple[str, str] | None = None
    dimension: int | None = None
    identities: set[tuple[str, str]] = set()
    for row in rows:
        values = _record_values(row)
        row_scope = (values[0], values[1])
        identity = (values[2], values[3])
        row_dimension = values[5]
        if scope is None:
            scope = row_scope
            dimension = row_dimension
        if row_scope != scope:
            raise ValueError("record rows must share one tenant and knowledge base")
        if row_dimension != dimension:
            raise ValueError("record rows must share one vector dimension")
        if identity in identities:
            raise ValueError("record row identities must be unique")
        identities.add(identity)
        canonical.append((identity, row_scope, row_dimension, values))
    canonical.sort(key=lambda item: item[0])
    return tuple(item[3] for item in canonical)


def _record_values(row: object) -> RecordParameters:
    if not isinstance(row, PostgreSQLStoredVectorRow):
        raise ValueError("row must be a PostgreSQLStoredVectorRow")
    tenant_id = _canonical_text(row.tenant_id, "row tenant_id")
    knowledge_base_id = _canonical_text(
        row.knowledge_base_id,
        "row knowledge_base_id",
    )
    document_id = _canonical_text(row.document_id, "row document_id")
    chunk_id = _canonical_text(row.chunk_id, "row chunk_id")
    text = _canonical_text(row.text, "row text")
    embedding = _canonical_embedding(row.embedding, "row embedding")
    metadata_json = _canonical_metadata_json(row.metadata_json)
    return (
        tenant_id,
        knowledge_base_id,
        document_id,
        chunk_id,
        text,
        len(embedding),
        Vector(list(embedding)),
        metadata_json,
    )


def _stored_row_from_database(
    raw_row: object,
    *,
    tenant_id: str,
    knowledge_base_id: str,
) -> PostgreSQLStoredVectorRow:
    values = _row_values(raw_row, expected_arity=8, field_name="stored record row")
    stored_tenant = _canonical_text(values[0], "stored row tenant_id")
    stored_knowledge_base = _canonical_text(
        values[1],
        "stored row knowledge_base_id",
    )
    if stored_tenant != tenant_id or stored_knowledge_base != knowledge_base_id:
        raise ValueError("stored row scope does not match requested scope")
    document_id = _canonical_text(values[2], "stored row document_id")
    chunk_id = _canonical_text(values[3], "stored row chunk_id")
    text = _canonical_text(values[4], "stored row text")
    dimension = _positive_integer(values[5], "stored row vector_dimension")
    embedding = _database_embedding(values[6], expected_dimension=dimension)
    metadata_json = _normalize_database_metadata_json(values[7])
    return PostgreSQLStoredVectorRow(
        tenant_id=stored_tenant,
        knowledge_base_id=stored_knowledge_base,
        document_id=document_id,
        chunk_id=chunk_id,
        text=text,
        embedding=embedding,
        metadata_json=metadata_json,
    )


def _search_row_from_database(
    raw_row: object,
    *,
    tenant_id: str,
    knowledge_base_id: str,
) -> PostgreSQLCosineSearchRow:
    values = _row_values(raw_row, expected_arity=9, field_name="cosine search row")
    stored = _stored_row_from_database(
        values[:8],
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
    )
    return PostgreSQLCosineSearchRow(
        tenant_id=stored.tenant_id,
        knowledge_base_id=stored.knowledge_base_id,
        document_id=stored.document_id,
        chunk_id=stored.chunk_id,
        text=stored.text,
        embedding=stored.embedding,
        metadata_json=stored.metadata_json,
        cosine_distance=_cosine_distance(values[8], "cosine_distance"),
    )


def _database_embedding(
    value: object,
    *,
    expected_dimension: int,
) -> tuple[float, ...]:
    if not isinstance(value, Vector):
        raise ValueError("database embedding must be a pgvector.Vector")
    return canonicalize_float32_embedding(
        value.to_list(),
        expected_dimension=expected_dimension,
    )


def _canonical_embedding(value: object, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field_name} must be a nonempty tuple")
    canonical = canonicalize_float32_embedding(
        value,
        expected_dimension=len(value),
    )
    if any(type(item) is not float for item in value) or canonical != value:
        raise ValueError(f"{field_name} must contain canonical float32 values")
    return canonical


def _canonical_metadata_json(value: object) -> str:
    metadata = decode_ordered_metadata(value)
    canonical = encode_ordered_metadata(metadata)
    if value != canonical:
        raise ValueError("metadata_json must be canonical ordered metadata JSON")
    return canonical


def _normalize_database_metadata_json(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("database metadata_json must be a string")
    return encode_ordered_metadata(decode_ordered_metadata(value))


def _row_values(
    row: object,
    *,
    expected_arity: int,
    field_name: str,
) -> tuple[object, ...]:
    if not isinstance(row, tuple) or len(row) != expected_arity:
        raise ValueError(f"{field_name} must be a {expected_arity}-element tuple")
    return row


def _canonical_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    if cleaned != value:
        raise ValueError(f"{field_name} must be canonical")
    return cleaned


def _positive_integer(value: object, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _cosine_distance(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a real numeric value")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 2.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 2")
    return result
