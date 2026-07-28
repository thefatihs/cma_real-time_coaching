"""Deterministic fake-cursor tests for complete PostgreSQL vector SQL."""

import hashlib
from typing import Any, cast

import pytest
from pgvector import Vector
from psycopg import Connection
from psycopg.sql import Composable

from app.vector_store.embedding_profile import (
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
)
from app.vector_store.models import VectorRecordIdentity
from app.vector_store.postgres.contracts import (
    PostgreSQLStoredVectorRow,
    PostgreSQLVectorTransaction,
)
from app.vector_store.postgres.transaction import (
    PsycopgPostgreSQLVectorTransaction,
)


class SyntheticCursor:
    def __init__(
        self,
        events: list[str],
        *,
        rows: list[tuple[object, ...]] | None = None,
        execute_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.rows = rows if rows is not None else []
        self.execute_error = execute_error
        self.executions: list[tuple[str, tuple[object, ...] | None]] = []
        self.executemany_calls: list[tuple[str, tuple[tuple[object, ...], ...]]] = []

    def __enter__(self) -> "SyntheticCursor":
        self.events.append("cursor_enter")
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        del exc_type, exc_value, traceback
        self.events.append("cursor_exit")

    def execute(
        self,
        query: Composable,
        params: tuple[object, ...] | None = None,
    ) -> None:
        self.events.append("execute")
        self.executions.append((query.as_string(), params))
        if self.execute_error is not None:
            raise self.execute_error

    def executemany(
        self,
        query: Composable,
        params: tuple[tuple[object, ...], ...],
    ) -> None:
        self.events.append("executemany")
        self.executemany_calls.append((query.as_string(), params))
        if self.execute_error is not None:
            raise self.execute_error

    def fetchall(self) -> list[tuple[object, ...]]:
        self.events.append("fetchall")
        return self.rows


class SyntheticConnection:
    def __init__(self, cursors: list[SyntheticCursor]) -> None:
        self.cursors = cursors
        self.cursor_calls = 0
        self.commit_calls = 0
        self.rollback_calls = 0
        self.close_calls = 0

    def cursor(self) -> SyntheticCursor:
        cursor = self.cursors[self.cursor_calls]
        self.cursor_calls += 1
        return cursor

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _connection(value: SyntheticConnection) -> Connection[Any]:
    return cast(Connection[Any], value)


def _transaction(
    *cursors: SyntheticCursor,
) -> tuple[PsycopgPostgreSQLVectorTransaction, SyntheticConnection]:
    connection = SyntheticConnection(list(cursors))
    return PsycopgPostgreSQLVectorTransaction(_connection(connection)), connection


def _profile() -> KnowledgeBaseEmbeddingProfile:
    return KnowledgeBaseEmbeddingProfile(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        model_id="model-synthetic",
        vector_dimension=2,
        normalize_embeddings=True,
        distance_metric=EmbeddingDistanceMetric.COSINE,
    )


def _stored_row(
    *,
    document_id: str = "document-b",
    chunk_id: str = "chunk-b",
    text: str = "Synthetic text",
    metadata_json: str = '[["kind","synthetic"]]',
) -> PostgreSQLStoredVectorRow:
    return PostgreSQLStoredVectorRow(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        document_id=document_id,
        chunk_id=chunk_id,
        text=text,
        embedding=(1.0, 0.0),
        metadata_json=metadata_json,
    )


def _database_row(
    *,
    document_id: str = "document-a",
    chunk_id: str = "chunk-a",
    metadata_json: object = '[["kind","synthetic"]]',
    distance: float | None = None,
) -> tuple[object, ...]:
    values: tuple[object, ...] = (
        "tenant-synthetic",
        "kb-synthetic",
        document_id,
        chunk_id,
        "Synthetic text",
        2,
        Vector([1.0, 0.0]),
        metadata_json,
    )
    return values if distance is None else (*values, distance)


def test_transaction_structurally_satisfies_protocol() -> None:
    transaction: PostgreSQLVectorTransaction = _transaction()[0]
    assert callable(transaction.search_cosine)


def test_constructor_is_side_effect_free_and_retains_structural_connection() -> None:
    connection = SyntheticConnection([])

    PsycopgPostgreSQLVectorTransaction(_connection(connection))

    assert connection.cursor_calls == 0


def test_constructor_rejects_missing_callable_cursor() -> None:
    with pytest.raises(ValueError, match="connection.cursor"):
        PsycopgPostgreSQLVectorTransaction(cast(Connection[Any], object()))


@pytest.mark.parametrize(
    ("tenant_id", "knowledge_base_id"),
    [("", "kb-synthetic"), (" tenant-synthetic", "kb-synthetic")],
)
def test_invalid_lock_scope_fails_before_cursor(
    tenant_id: str,
    knowledge_base_id: str,
) -> None:
    transaction, connection = _transaction()

    with pytest.raises(ValueError):
        transaction.acquire_scope_lock(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
        )

    assert connection.cursor_calls == 0


def test_lock_uses_stable_length_prefixed_sha256_key_and_bound_parameter() -> None:
    events: list[str] = []
    cursor = SyntheticCursor(events)
    transaction, _ = _transaction(cursor)
    tenant = "tenant-synthetic"
    knowledge_base = "kb-synthetic"
    tenant_bytes = tenant.encode()
    kb_bytes = knowledge_base.encode()
    expected = int.from_bytes(
        hashlib.sha256(
            len(tenant_bytes).to_bytes(8, "big")
            + tenant_bytes
            + len(kb_bytes).to_bytes(8, "big")
            + kb_bytes
        ).digest()[:8],
        "big",
        signed=True,
    )

    transaction.acquire_scope_lock(
        tenant_id=tenant,
        knowledge_base_id=knowledge_base,
    )

    query, params = cursor.executions[0]
    assert query == "SELECT pg_advisory_xact_lock(%s)"
    assert params == (expected,)
    assert tenant not in query
    assert knowledge_base not in query
    assert events == ["cursor_enter", "execute", "cursor_exit"]


@pytest.mark.parametrize("for_update", [False, True])
def test_profile_lookup_uses_bound_scope_and_optional_static_lock(
    for_update: bool,
) -> None:
    cursor = SyntheticCursor(
        [],
        rows=[
            (
                "tenant-synthetic",
                "kb-synthetic",
                "model-synthetic",
                2,
                True,
                "cosine",
            )
        ],
    )
    transaction, _ = _transaction(cursor)

    result = transaction.get_profile(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        for_update=for_update,
    )

    query, params = cursor.executions[0]
    assert ("FOR UPDATE" in query) is for_update
    assert params == ("tenant-synthetic", "kb-synthetic")
    assert result == _profile()


def test_missing_profile_returns_none_and_extra_or_malformed_rows_fail() -> None:
    missing = SyntheticCursor([], rows=[])
    extra = SyntheticCursor(
        [],
        rows=[
            ("tenant-synthetic", "kb-synthetic", "model", 2, True, "cosine"),
            ("tenant-synthetic", "kb-synthetic", "model", 2, True, "cosine"),
        ],
    )
    malformed = SyntheticCursor(
        [],
        rows=[("tenant-synthetic", "kb-synthetic", "model", True, True, "cosine")],
    )
    transaction, _ = _transaction(missing, extra, malformed)

    assert (
        transaction.get_profile(
            tenant_id="tenant-synthetic",
            knowledge_base_id="kb-synthetic",
            for_update=False,
        )
        is None
    )
    with pytest.raises(ValueError, match="more than one"):
        transaction.get_profile(
            tenant_id="tenant-synthetic",
            knowledge_base_id="kb-synthetic",
            for_update=False,
        )
    with pytest.raises(ValueError, match="positive integer"):
        transaction.get_profile(
            tenant_id="tenant-synthetic",
            knowledge_base_id="kb-synthetic",
            for_update=False,
        )


def test_profile_insert_uses_all_bound_values_without_conflict_clause() -> None:
    cursor = SyntheticCursor([])
    transaction, _ = _transaction(cursor)

    transaction.insert_profile(_profile())

    query, params = cursor.executions[0]
    assert 'INSERT INTO "callmetric_vector"."embedding_profiles"' in query
    assert "ON CONFLICT" not in query
    assert params == (
        "tenant-synthetic",
        "kb-synthetic",
        "model-synthetic",
        2,
        True,
        "cosine",
    )


def test_empty_record_lookup_returns_without_cursor() -> None:
    transaction, connection = _transaction()

    assert (
        transaction.get_records(
            tenant_id="tenant-synthetic",
            knowledge_base_id="kb-synthetic",
            identities=(),
        )
        == ()
    )
    assert connection.cursor_calls == 0


def test_record_lookup_binds_paired_arrays_and_maps_complete_vector() -> None:
    cursor = SyntheticCursor([], rows=[_database_row()])
    transaction, _ = _transaction(cursor)
    identities = (
        VectorRecordIdentity(document_id="document-a", chunk_id="chunk-a"),
        VectorRecordIdentity(document_id="document-b", chunk_id="chunk-b"),
    )

    result = transaction.get_records(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        identities=identities,
    )

    query, params = cursor.executions[0]
    assert "unnest(%s::text[], %s::text[]) WITH ORDINALITY" in query
    assert params == (
        ["document-a", "document-b"],
        ["chunk-a", "chunk-b"],
        "tenant-synthetic",
        "kb-synthetic",
    )
    assert result[0].embedding == (1.0, 0.0)
    assert result[0].metadata_json == '[["kind","synthetic"]]'


def test_database_metadata_whitespace_is_normalized_without_mutating_rows() -> None:
    metadata_json = '[["zihin", "ölçüm"], ["kind", "synthetic value"]]'
    raw_row = _database_row(metadata_json=metadata_json)
    rows = [raw_row]
    cursor = SyntheticCursor([], rows=rows)
    transaction, _ = _transaction(cursor)

    result = transaction.get_records(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        identities=(
            VectorRecordIdentity(document_id="document-a", chunk_id="chunk-a"),
        ),
    )

    assert result[0].metadata_json == ('[["zihin","ölçüm"],["kind","synthetic value"]]')
    assert rows == [raw_row]
    assert rows[0][7] is metadata_json


@pytest.mark.parametrize(
    "metadata_json",
    [
        42,
        "not-json",
        '{"kind":"synthetic"}',
        '[["kind"]]',
        '[[" ","synthetic"]]',
        '[["kind","one"],["kind","two"]]',
    ],
)
def test_database_metadata_malformed_values_fail_closed(
    metadata_json: object,
) -> None:
    transaction, _ = _transaction(
        SyntheticCursor([], rows=[_database_row(metadata_json=metadata_json)]),
    )

    with pytest.raises(ValueError):
        transaction.get_records(
            tenant_id="tenant-synthetic",
            knowledge_base_id="kb-synthetic",
            identities=(
                VectorRecordIdentity(
                    document_id="document-a",
                    chunk_id="chunk-a",
                ),
            ),
        )


def test_duplicate_requested_identity_fails_before_cursor() -> None:
    transaction, connection = _transaction()
    identity = VectorRecordIdentity(document_id="document-a", chunk_id="chunk-a")

    with pytest.raises(ValueError, match="unique"):
        transaction.get_records(
            tenant_id="tenant-synthetic",
            knowledge_base_id="kb-synthetic",
            identities=(identity, identity),
        )

    assert connection.cursor_calls == 0


@pytest.mark.parametrize(
    "rows",
    [
        [_database_row(), _database_row()],
        [
            _database_row(document_id="unexpected-document"),
        ],
        [
            (
                "tenant-synthetic",
                "kb-synthetic",
                "document-a",
                "chunk-a",
                "Synthetic text",
                2,
                [1.0, 0.0],
                "[]",
            )
        ],
    ],
)
def test_record_lookup_rejects_duplicate_unexpected_or_non_vector_rows(
    rows: list[tuple[object, ...]],
) -> None:
    transaction, _ = _transaction(SyntheticCursor([], rows=rows))

    with pytest.raises(ValueError):
        transaction.get_records(
            tenant_id="tenant-synthetic",
            knowledge_base_id="kb-synthetic",
            identities=(
                VectorRecordIdentity(
                    document_id="document-a",
                    chunk_id="chunk-a",
                ),
            ),
        )


def test_batch_is_fully_validated_then_executed_in_canonical_order() -> None:
    cursor = SyntheticCursor([])
    transaction, _ = _transaction(cursor)
    rows = (
        _stored_row(document_id="document-b", chunk_id="chunk-b"),
        _stored_row(document_id="document-a", chunk_id="chunk-a"),
    )

    transaction.insert_records(rows)

    query, params = cursor.executemany_calls[0]
    assert "%s::jsonb" in query
    assert [values[2:4] for values in params] == [
        ("document-a", "chunk-a"),
        ("document-b", "chunk-b"),
    ]
    assert all(isinstance(values[6], Vector) for values in params)
    assert rows[0].document_id == "document-b"


@pytest.mark.parametrize(
    "rows",
    [
        (),
        (
            _stored_row(),
            _stored_row(document_id="document-b", chunk_id="chunk-b"),
        ),
        (
            _stored_row(),
            PostgreSQLStoredVectorRow(
                tenant_id="other-tenant",
                knowledge_base_id="kb-synthetic",
                document_id="document-a",
                chunk_id="chunk-a",
                text="Synthetic text",
                embedding=(1.0, 0.0),
                metadata_json="[]",
            ),
        ),
    ],
)
def test_invalid_batch_fails_before_cursor(
    rows: tuple[PostgreSQLStoredVectorRow, ...],
) -> None:
    transaction, connection = _transaction()

    with pytest.raises(ValueError):
        transaction.insert_records(rows)

    assert connection.cursor_calls == 0


def test_replace_uses_full_identity_conflict_and_updates_only_content() -> None:
    cursor = SyntheticCursor([])
    transaction, _ = _transaction(cursor)

    transaction.replace_record(_stored_row())

    query, params = cursor.executions[0]
    assert params is not None
    assert "ON CONFLICT (tenant_id, knowledge_base_id, document_id, chunk_id)" in query
    update_clause = query.split("DO UPDATE SET", 1)[1]
    assert "text = EXCLUDED.text" in update_clause
    assert "vector_dimension = EXCLUDED.vector_dimension" in update_clause
    assert "embedding = EXCLUDED.embedding" in update_clause
    assert "metadata_json = EXCLUDED.metadata_json" in update_clause
    assert "tenant_id =" not in update_clause
    assert isinstance(params[6], Vector)


@pytest.mark.parametrize("operation", ["insert", "replace"])
def test_write_side_noncanonical_metadata_fails_before_cursor(
    operation: str,
) -> None:
    transaction, connection = _transaction()
    row = _stored_row(metadata_json='[["kind", "synthetic"]]')

    with pytest.raises(ValueError, match="canonical ordered metadata"):
        if operation == "insert":
            transaction.insert_records((row,))
        else:
            transaction.replace_record(row)

    assert connection.cursor_calls == 0


def test_cosine_search_uses_one_computation_and_complete_mapping() -> None:
    cursor = SyntheticCursor(
        [],
        rows=[_database_row(distance=0.25)],
    )
    transaction, _ = _transaction(cursor)

    result = transaction.search_cosine(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        query_embedding=(1.0, 0.0),
        top_k=3,
        maximum_cosine_distance=1.0,
    )

    query, params = cursor.executions[0]
    assert params is not None
    assert query.count("<=> %s") == 1
    assert "WHERE cosine_distance <= %s" in query
    assert "ORDER BY cosine_distance, document_id, chunk_id" in query
    assert "LIMIT %s" in query
    assert isinstance(params[0], Vector)
    assert params[1:] == ("tenant-synthetic", "kb-synthetic", 1.0, 3)
    assert result[0].cosine_distance == 0.25
    assert result[0].embedding == (1.0, 0.0)


def test_cosine_search_normalizes_database_metadata() -> None:
    metadata_json = '[["zihin", "ölçüm"], ["kind", "synthetic"]]'
    raw_row = _database_row(metadata_json=metadata_json, distance=0.25)
    rows = [raw_row]
    transaction, _ = _transaction(SyntheticCursor([], rows=rows))

    result = transaction.search_cosine(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        query_embedding=(1.0, 0.0),
        top_k=1,
        maximum_cosine_distance=1.0,
    )

    assert result[0].metadata_json == '[["zihin","ölçüm"],["kind","synthetic"]]'
    assert rows == [raw_row]
    assert rows[0][7] is metadata_json


@pytest.mark.parametrize(
    ("query_embedding", "top_k", "maximum_distance"),
    [
        ((), 1, 1.0),
        ((1.0, 0.0), True, 1.0),
        ((1.0, 0.0), 1, float("nan")),
        ((1.0, 0.0), 1, 2.1),
    ],
)
def test_invalid_search_arguments_fail_before_cursor(
    query_embedding: tuple[float, ...],
    top_k: int,
    maximum_distance: float,
) -> None:
    transaction, connection = _transaction()

    with pytest.raises(ValueError):
        transaction.search_cosine(
            tenant_id="tenant-synthetic",
            knowledge_base_id="kb-synthetic",
            query_embedding=query_embedding,
            top_k=top_k,
            maximum_cosine_distance=maximum_distance,
        )

    assert connection.cursor_calls == 0


@pytest.mark.parametrize(
    "rows",
    [
        [_database_row(distance=float("nan"))],
        [_database_row(distance=0.2), _database_row(distance=0.1)],
        [_database_row(distance=0.1), _database_row(distance=0.1)],
        [
            (
                "tenant-synthetic",
                "kb-synthetic",
                "document-a",
                "chunk-a",
                "Synthetic text",
                3,
                Vector([1.0, 0.0]),
                "[]",
                0.1,
            )
        ],
    ],
)
def test_search_rejects_malformed_distance_order_duplicate_or_dimension(
    rows: list[tuple[object, ...]],
) -> None:
    transaction, _ = _transaction(SyntheticCursor([], rows=rows))

    with pytest.raises(ValueError):
        transaction.search_cosine(
            tenant_id="tenant-synthetic",
            knowledge_base_id="kb-synthetic",
            query_embedding=(1.0, 0.0),
            top_k=3,
            maximum_cosine_distance=1.0,
        )


def test_sql_injection_shaped_values_remain_bound_parameters() -> None:
    cursor = SyntheticCursor([])
    transaction, _ = _transaction(cursor)
    tenant = "tenant'; SELECT synthetic"
    knowledge_base = "kb--synthetic"

    transaction.acquire_scope_lock(
        tenant_id=tenant,
        knowledge_base_id=knowledge_base,
    )

    query, params = cursor.executions[0]
    assert params is not None
    assert tenant not in query
    assert knowledge_base not in query
    assert len(params) == 1


def test_sql_injection_shaped_text_and_metadata_remain_bound_parameters() -> None:
    cursor = SyntheticCursor([])
    transaction, _ = _transaction(cursor)
    text = "Synthetic'; DROP TABLE vector_records"
    metadata_json = '[["kind","synthetic ); DROP TABLE vector_records"]]'

    transaction.replace_record(
        _stored_row(text=text, metadata_json=metadata_json),
    )

    query, params = cursor.executions[0]
    assert params is not None
    assert text not in query
    assert metadata_json not in query
    assert params[4] == text
    assert params[7] == metadata_json


def test_provider_exception_identity_and_cursor_close_are_preserved() -> None:
    primary = OSError("synthetic provider failure")
    events: list[str] = []
    cursor = SyntheticCursor(events, execute_error=primary)
    transaction, connection = _transaction(cursor)

    with pytest.raises(OSError) as raised:
        transaction.acquire_scope_lock(
            tenant_id="tenant-synthetic",
            knowledge_base_id="kb-synthetic",
        )

    assert raised.value is primary
    assert events == ["cursor_enter", "execute", "cursor_exit"]
    assert connection.commit_calls == 0
    assert connection.rollback_calls == 0
    assert connection.close_calls == 0


def test_repeated_calls_are_deterministic_and_never_manage_connection() -> None:
    first = SyntheticCursor([], rows=[_database_row(distance=0.1)])
    second = SyntheticCursor([], rows=[_database_row(distance=0.1)])
    transaction, connection = _transaction(first, second)
    arguments = {
        "tenant_id": "tenant-synthetic",
        "knowledge_base_id": "kb-synthetic",
        "query_embedding": (1.0, 0.0),
        "top_k": 1,
        "maximum_cosine_distance": 1.0,
    }

    one = transaction.search_cosine(**arguments)
    two = transaction.search_cosine(**arguments)

    assert one == two
    assert first.executions == second.executions
    assert connection.commit_calls == 0
    assert connection.rollback_calls == 0
    assert connection.close_calls == 0
