"""Focused tests for read-only PostgreSQL schema readiness verification."""

from __future__ import annotations

from inspect import signature
from typing import Any, cast

import pytest
from psycopg import Connection

import app.vector_store as vector_store
import app.vector_store.postgres as postgres
import app.vector_store.postgres.readiness as readiness
from app.vector_store.postgres import PostgreSQLSchemaReadinessChecker


def _required_column_rows() -> list[tuple[object, ...]]:
    return [
        (table, column)
        for table, columns in readiness._REQUIRED_COLUMNS.items()  # noqa: SLF001
        for column in sorted(columns)
    ]


def _required_constraint_rows() -> list[tuple[object, ...]]:
    return [
        (table, constraint)
        for table, constraints in readiness._REQUIRED_CONSTRAINTS.items()  # noqa: SLF001
        for constraint in sorted(constraints)
    ]


def _required_index_rows() -> list[tuple[object, ...]]:
    return [
        (table, index)
        for table, indexes in readiness._REQUIRED_INDEXES.items()  # noqa: SLF001
        for index in sorted(indexes)
    ]


def _ready_responses() -> list[object]:
    responses: list[object] = [
        [(1,)],
        [("0.8.5",)],
        [("callmetric_vector",)],
        [
            ("document_ingestion_jobs",),
            ("documents",),
            ("embedding_profiles",),
            ("schema_migrations",),
            ("vector_records",),
        ],
        [("0001",), ("0002",), ("0003",)],
        _required_column_rows(),
        [("YES",)],
        _required_constraint_rows(),
        _required_index_rows(),
    ]
    return responses


class FakeCursor:
    def __init__(
        self,
        responses: list[object],
        *,
        execute_error: BaseException | None = None,
        execute_error_index: int = -1,
        exit_error: BaseException | None = None,
    ) -> None:
        self.responses = responses
        self.execute_error = execute_error
        self.execute_error_index = execute_error_index
        self.exit_error = exit_error
        self.calls: list[tuple[object, ...]] = []
        self.response_index = -1
        self.entered = 0
        self.exited = 0

    def __enter__(self) -> FakeCursor:
        self.entered += 1
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        del exception_type, exception, traceback
        self.exited += 1
        if self.exit_error is not None:
            raise self.exit_error

    def execute(
        self,
        query: object,
        parameters: object = None,
    ) -> None:
        call = (query,) if parameters is None else (query, parameters)
        self.calls.append(call)
        call_index = len(self.calls) - 1
        if self.execute_error is not None and call_index == self.execute_error_index:
            raise self.execute_error
        if query != readiness._READ_ONLY_SQL:  # noqa: SLF001
            self.response_index += 1

    def fetchall(self) -> object:
        return self.responses[self.response_index]


class FakeConnection:
    def __init__(
        self,
        responses: list[object] | None = None,
        *,
        autocommit: object = False,
        cursor_error: BaseException | None = None,
        execute_error: BaseException | None = None,
        execute_error_index: int = -1,
        cursor_exit_error: BaseException | None = None,
        rollback_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.autocommit = autocommit
        self.cursor_error = cursor_error
        self.rollback_error = rollback_error
        self.close_error = close_error
        self.cursor_value = FakeCursor(
            responses or _ready_responses(),
            execute_error=execute_error,
            execute_error_index=execute_error_index,
            exit_error=cursor_exit_error,
        )
        self.cursor_calls = 0
        self.rollback_calls = 0
        self.commit_calls = 0
        self.close_calls = 0
        self.lifecycle: list[str] = []

    def cursor(self) -> FakeCursor:
        self.cursor_calls += 1
        self.lifecycle.append("cursor")
        if self.cursor_error is not None:
            raise self.cursor_error
        return self.cursor_value

    def rollback(self) -> None:
        self.rollback_calls += 1
        self.lifecycle.append("rollback")
        if self.rollback_error is not None:
            raise self.rollback_error

    def commit(self) -> None:
        self.commit_calls += 1
        self.lifecycle.append("commit")

    def close(self) -> None:
        self.close_calls += 1
        self.lifecycle.append("close")
        if self.close_error is not None:
            raise self.close_error


class FalseyFactory:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.calls = 0

    def __bool__(self) -> bool:
        return False

    def __call__(self) -> Connection[Any]:
        self.calls += 1
        return cast(Connection[Any], self.connection)


def _checker(connection: FakeConnection) -> PostgreSQLSchemaReadinessChecker:
    return PostgreSQLSchemaReadinessChecker(
        connection_factory=lambda: cast(Connection[Any], connection)
    )


def test_public_signature_is_minimal_and_synchronous() -> None:
    constructor = signature(PostgreSQLSchemaReadinessChecker)
    verify = signature(PostgreSQLSchemaReadinessChecker.verify)

    assert tuple(constructor.parameters) == ("connection_factory",)
    assert constructor.parameters["connection_factory"].kind.name == "KEYWORD_ONLY"
    assert tuple(verify.parameters) == ("self",)
    assert verify.return_annotation in (None, "None")


def test_constructor_is_side_effect_free_and_preserves_falsey_callable() -> None:
    connection = FakeConnection()
    factory = FalseyFactory(connection)

    checker = PostgreSQLSchemaReadinessChecker(connection_factory=factory)

    assert checker._connection_factory is factory  # noqa: SLF001
    assert factory.calls == 0
    assert connection.cursor_calls == 0


@pytest.mark.parametrize("factory", [None, object(), False, 0])
def test_constructor_rejects_noncallable_factory(factory: object) -> None:
    with pytest.raises(ValueError, match="connection_factory must be callable"):
        PostgreSQLSchemaReadinessChecker(
            connection_factory=cast(Any, factory),
        )


def test_success_uses_exact_fixed_order_then_rolls_back_and_closes() -> None:
    connection = FakeConnection()

    result = _checker(connection).verify()

    assert result is None
    assert connection.cursor_value.calls == [
        (readiness._READ_ONLY_SQL,),  # noqa: SLF001
        (readiness._USABILITY_SQL,),  # noqa: SLF001
        (readiness._EXTENSION_SQL, ("vector",)),  # noqa: SLF001
        (readiness._SCHEMA_SQL, ("callmetric_vector",)),  # noqa: SLF001
        (
            readiness._TABLES_SQL,  # noqa: SLF001
            (
                "callmetric_vector",
                [
                    "document_ingestion_jobs",
                    "documents",
                    "embedding_profiles",
                    "schema_migrations",
                    "vector_records",
                ],
            ),
        ),
        (readiness._MIGRATION_SQL,),  # noqa: SLF001
        (
            readiness._COLUMNS_SQL,  # noqa: SLF001
            (
                "callmetric_vector",
                [
                    "embedding_profiles",
                    "documents",
                    "document_ingestion_jobs",
                    "vector_records",
                ],
            ),
        ),
        (
            readiness._NULLABILITY_SQL,  # noqa: SLF001
            ("callmetric_vector", "documents", "storage_object_key"),
        ),
        (
            readiness._CONSTRAINTS_SQL,  # noqa: SLF001
            (
                "callmetric_vector",
                [
                    "documents",
                    "document_ingestion_jobs",
                    "embedding_profiles",
                    "vector_records",
                ],
            ),
        ),
        (
            readiness._INDEXES_SQL,  # noqa: SLF001
            (
                "callmetric_vector",
                ["documents", "document_ingestion_jobs"],
            ),
        ),
    ]
    assert connection.cursor_value.entered == 1
    assert connection.cursor_value.exited == 1
    assert connection.rollback_calls == 1
    assert connection.close_calls == 1
    assert connection.commit_calls == 0
    assert connection.lifecycle == ["cursor", "rollback", "close"]


@pytest.mark.parametrize(
    ("response_index", "value", "message"),
    [
        (0, [], "connection usability"),
        (1, [], "pgvector extension"),
        (1, [("0.8.4",)], "pgvector extension"),
        (1, [("0.8.5",), ("0.8.5",)], "pgvector extension"),
        (2, [], "PostgreSQL schema"),
        (3, [("embedding_profiles",), ("vector_records",)], "PostgreSQL tables"),
        (4, [], "migration ledger"),
        (4, [("0001",), ("0001",)], "migration ledger"),
    ],
    ids=[
        "unusable",
        "missing-extension",
        "wrong-version",
        "duplicate-extension",
        "missing-schema",
        "missing-table",
        "missing-migration",
        "duplicate-migration",
    ],
)
def test_exact_catalog_checks_fail_closed(
    response_index: int,
    value: object,
    message: str,
) -> None:
    responses: list[object] = _ready_responses()
    responses[response_index] = value
    connection = FakeConnection(responses)

    with pytest.raises(ValueError, match=message):
        _checker(connection).verify()

    assert connection.rollback_calls == 1
    assert connection.close_calls == 1
    assert connection.commit_calls == 0


@pytest.mark.parametrize(
    ("response_index", "value", "message"),
    [
        (5, [], "columns"),
        (
            5,
            [
                row
                for row in _required_column_rows()
                if row != ("vector_records", "embedding")
            ],
            "columns",
        ),
        (5, _required_column_rows() + [_required_column_rows()[0]], "duplicate"),
        (6, [], "document source nullability"),
        (6, [("NO",)], "document source nullability"),
        (7, [], "constraints"),
        (
            7,
            [
                row
                for row in _required_constraint_rows()
                if row != ("vector_records", "vector_records_primary_key")
            ],
            "constraints",
        ),
        (
            7,
            _required_constraint_rows() + [_required_constraint_rows()[0]],
            "duplicate",
        ),
        (8, [], "indexes"),
        (
            8,
            [
                row
                for row in _required_index_rows()
                if row != ("documents", "documents_scope_created_document_index")
            ],
            "indexes",
        ),
    ],
    ids=[
        "no-columns",
        "missing-column",
        "duplicate-column",
        "missing-nullability",
        "wrong-nullability",
        "no-constraints",
        "missing-constraint",
        "duplicate-constraint",
        "no-indexes",
        "missing-index",
    ],
)
def test_required_column_and_constraint_subsets_fail_closed(
    response_index: int,
    value: object,
    message: str,
) -> None:
    responses: list[object] = _ready_responses()
    responses[response_index] = value

    with pytest.raises(ValueError, match=message):
        _checker(FakeConnection(responses)).verify()


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        (),
        [None],
        [()],
        [(None,)],
        [(1,)],
        [["0.8.5"]],
    ],
    ids=[
        "none",
        "tuple-collection",
        "non-tuple-row",
        "empty-row",
        "null",
        "non-text",
        "list-row",
    ],
)
def test_malformed_extension_rows_fail_closed(malformed: object) -> None:
    responses: list[object] = _ready_responses()
    responses[1] = malformed

    with pytest.raises(ValueError):
        _checker(FakeConnection(responses)).verify()


def test_read_only_setup_precedes_query_failure_and_exception_identity_survives() -> (
    None
):
    expected = RuntimeError("synthetic provider failure")
    connection = FakeConnection(execute_error=expected, execute_error_index=1)

    with pytest.raises(RuntimeError) as raised:
        _checker(connection).verify()

    assert raised.value is expected
    assert connection.cursor_value.calls[0] == (readiness._READ_ONLY_SQL,)  # noqa: SLF001
    assert connection.rollback_calls == 1
    assert connection.close_calls == 1


@pytest.mark.parametrize("stage", ["cursor", "cursor-exit"])
def test_cursor_failures_preserve_exact_exception(stage: str) -> None:
    expected = RuntimeError("synthetic cursor failure")
    connection = FakeConnection(
        cursor_error=expected if stage == "cursor" else None,
        cursor_exit_error=expected if stage == "cursor-exit" else None,
    )

    with pytest.raises(RuntimeError) as raised:
        _checker(connection).verify()

    assert raised.value is expected
    assert connection.rollback_calls == 1
    assert connection.close_calls == 1


def test_primary_failure_keeps_cleanup_failures_in_cause() -> None:
    primary = RuntimeError("synthetic primary")
    rollback = RuntimeError("synthetic rollback")
    close = RuntimeError("synthetic close")
    connection = FakeConnection(
        execute_error=primary,
        execute_error_index=1,
        rollback_error=rollback,
        close_error=close,
    )

    with pytest.raises(RuntimeError) as raised:
        _checker(connection).verify()

    assert raised.value is primary
    assert isinstance(raised.value.__cause__, ExceptionGroup)
    assert raised.value.__cause__.exceptions == (rollback, close)


def test_success_rollback_failure_is_primary_and_close_is_attempted() -> None:
    rollback = RuntimeError("synthetic rollback")
    connection = FakeConnection(rollback_error=rollback)

    with pytest.raises(RuntimeError) as raised:
        _checker(connection).verify()

    assert raised.value is rollback
    assert connection.close_calls == 1
    assert connection.commit_calls == 0


def test_success_close_failure_propagates_unchanged() -> None:
    close = RuntimeError("synthetic close")
    connection = FakeConnection(close_error=close)

    with pytest.raises(RuntimeError) as raised:
        _checker(connection).verify()

    assert raised.value is close
    assert connection.rollback_calls == 1


def test_invalid_autocommit_fails_before_cursor_and_cleans_up() -> None:
    connection = FakeConnection(autocommit=True)

    with pytest.raises(ValueError, match="autocommit"):
        _checker(connection).verify()

    assert connection.cursor_calls == 0
    assert connection.rollback_calls == 1
    assert connection.close_calls == 1


def test_repeated_verification_uses_fresh_connections() -> None:
    connections = [FakeConnection(), FakeConnection()]
    calls = 0

    def factory() -> Connection[Any]:
        nonlocal calls
        value = connections[calls]
        calls += 1
        return cast(Connection[Any], value)

    checker = PostgreSQLSchemaReadinessChecker(connection_factory=factory)

    checker.verify()
    checker.verify()

    assert calls == 2
    assert [connection.close_calls for connection in connections] == [1, 1]


def test_queries_are_fixed_read_only_and_contain_no_sensitive_values() -> None:
    queries = (
        readiness._READ_ONLY_SQL,  # noqa: SLF001
        readiness._USABILITY_SQL,  # noqa: SLF001
        readiness._EXTENSION_SQL,  # noqa: SLF001
        readiness._SCHEMA_SQL,  # noqa: SLF001
        readiness._TABLES_SQL,  # noqa: SLF001
        readiness._MIGRATION_SQL,  # noqa: SLF001
        readiness._COLUMNS_SQL,  # noqa: SLF001
        readiness._NULLABILITY_SQL,  # noqa: SLF001
        readiness._CONSTRAINTS_SQL,  # noqa: SLF001
    )
    combined = " ".join(queries).lower()

    assert "synthetic-secret" not in repr(_checker(FakeConnection()))
    assert all(token not in combined for token in ("insert ", "update ", "delete "))
    assert all(token not in combined for token in ("create ", "drop ", "truncate "))
    assert "callmetric_vector" in combined
    assert "set transaction read only" in combined


def test_postgres_export_is_exact_and_top_level_remains_private() -> None:
    assert postgres.PostgreSQLSchemaReadinessChecker is PostgreSQLSchemaReadinessChecker
    assert postgres.__all__.count("PostgreSQLSchemaReadinessChecker") == 1
    assert "PostgreSQLSchemaReadinessChecker" not in vector_store.__all__
    assert not hasattr(vector_store, "PostgreSQLSchemaReadinessChecker")
