"""Focused tests for the synchronous Psycopg transaction runner."""

from collections.abc import Callable
from typing import Any, cast

import pytest
from psycopg import Connection

from app.vector_store.postgres.contracts import (
    PostgreSQLVectorTransaction,
    PostgreSQLVectorTransactionRunner,
)
from app.vector_store.postgres.runner import (
    PsycopgPostgreSQLVectorTransactionRunner,
)


class SyntheticTransaction:
    pass


class SyntheticConnection:
    def __init__(
        self,
        *,
        autocommit: object = False,
        events: list[str] | None = None,
        commit_error: Exception | None = None,
        rollback_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.autocommit = autocommit
        self.events = events if events is not None else []
        self.commit_error = commit_error
        self.rollback_error = rollback_error
        self.close_error = close_error
        self.commit_calls = 0
        self.rollback_calls = 0
        self.close_calls = 0

    def commit(self) -> None:
        self.events.append("commit")
        self.commit_calls += 1
        if self.commit_error is not None:
            raise self.commit_error

    def rollback(self) -> None:
        self.events.append("rollback")
        self.rollback_calls += 1
        if self.rollback_error is not None:
            raise self.rollback_error

    def close(self) -> None:
        self.events.append("close")
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FalseyCallable:
    def __init__(self, callback: Callable[..., Any]) -> None:
        self.callback = callback
        self.calls = 0

    def __bool__(self) -> bool:
        return False

    def __call__(self, *args: object) -> Any:
        self.calls += 1
        return self.callback(*args)


def _as_connection(connection: SyntheticConnection) -> Connection[Any]:
    return cast(Connection[Any], connection)


def _runner(
    connection: SyntheticConnection,
    *,
    events: list[str] | None = None,
) -> PsycopgPostgreSQLVectorTransactionRunner:
    transaction = SyntheticTransaction()

    def connection_factory() -> Connection[Any]:
        if events is not None:
            events.append("connect")
        return _as_connection(connection)

    def transaction_factory(
        supplied: Connection[Any],
    ) -> PostgreSQLVectorTransaction:
        assert supplied is _as_connection(connection)
        if events is not None:
            events.append("transaction")
        return cast(PostgreSQLVectorTransaction, transaction)

    return PsycopgPostgreSQLVectorTransactionRunner(
        connection_factory=connection_factory,
        transaction_factory=transaction_factory,
    )


def test_runner_structurally_satisfies_protocol() -> None:
    runner: PostgreSQLVectorTransactionRunner = _runner(SyntheticConnection())
    assert callable(runner.run_in_transaction)


@pytest.mark.parametrize("field", ["connection_factory", "transaction_factory"])
def test_constructor_rejects_non_callable_factory(field: str) -> None:
    values: dict[str, object] = {
        "connection_factory": lambda: None,
        "transaction_factory": lambda _connection: None,
    }
    values[field] = None

    with pytest.raises(ValueError, match=field):
        PsycopgPostgreSQLVectorTransactionRunner(**values)  # type: ignore[arg-type]


def test_constructor_does_not_invoke_factories() -> None:
    connection_factory = FalseyCallable(lambda: None)
    transaction_factory = FalseyCallable(lambda _connection: None)

    PsycopgPostgreSQLVectorTransactionRunner(
        connection_factory=connection_factory,
        transaction_factory=transaction_factory,
    )

    assert connection_factory.calls == 0
    assert transaction_factory.calls == 0


def test_falsey_callable_factories_are_retained_and_invoked() -> None:
    connection = SyntheticConnection()
    transaction = SyntheticTransaction()
    connection_factory = FalseyCallable(lambda: _as_connection(connection))
    transaction_factory = FalseyCallable(
        lambda _connection: cast(PostgreSQLVectorTransaction, transaction)
    )
    runner = PsycopgPostgreSQLVectorTransactionRunner(
        connection_factory=connection_factory,
        transaction_factory=transaction_factory,
    )

    assert runner.run_in_transaction(lambda supplied: supplied) is transaction
    assert connection_factory.calls == 1
    assert transaction_factory.calls == 1


def test_non_callable_operation_is_rejected_before_connection() -> None:
    connection_factory = FalseyCallable(lambda: None)
    runner = PsycopgPostgreSQLVectorTransactionRunner(
        connection_factory=connection_factory,
        transaction_factory=lambda _connection: cast(
            PostgreSQLVectorTransaction, SyntheticTransaction()
        ),
    )

    with pytest.raises(ValueError, match="operation"):
        runner.run_in_transaction(None)  # type: ignore[arg-type]

    assert connection_factory.calls == 0


def test_success_has_exact_order_counts_and_result_identity() -> None:
    events: list[str] = []
    connection = SyntheticConnection(events=events)
    runner = _runner(connection, events=events)
    expected = object()
    callback_calls = 0

    def operation(_transaction: PostgreSQLVectorTransaction) -> object:
        nonlocal callback_calls
        callback_calls += 1
        events.append("operation")
        return expected

    assert runner.run_in_transaction(operation) is expected
    assert events == ["connect", "transaction", "operation", "commit", "close"]
    assert callback_calls == 1
    assert connection.commit_calls == 1
    assert connection.rollback_calls == 0
    assert connection.close_calls == 1


def test_autocommit_true_is_primary_failure() -> None:
    events: list[str] = []
    connection = SyntheticConnection(autocommit=True, events=events)
    runner = _runner(connection, events=events)

    with pytest.raises(ValueError, match="autocommit"):
        runner.run_in_transaction(lambda _transaction: None)

    assert events == ["connect", "rollback", "close"]
    assert connection.commit_calls == 0


def test_transaction_factory_failure_preserves_exception() -> None:
    connection = SyntheticConnection()
    expected = RuntimeError("synthetic transaction factory failure")

    def fail(_connection: Connection[Any]) -> PostgreSQLVectorTransaction:
        raise expected

    runner = PsycopgPostgreSQLVectorTransactionRunner(
        connection_factory=lambda: _as_connection(connection),
        transaction_factory=fail,
    )

    with pytest.raises(RuntimeError) as raised:
        runner.run_in_transaction(lambda _transaction: None)

    assert raised.value is expected
    assert connection.commit_calls == 0
    assert connection.rollback_calls == 1
    assert connection.close_calls == 1


def test_callback_failure_preserves_exception_and_operation() -> None:
    connection = SyntheticConnection()
    runner = _runner(connection)
    expected = LookupError("synthetic callback failure")

    def operation(_transaction: PostgreSQLVectorTransaction) -> None:
        raise expected

    with pytest.raises(LookupError) as raised:
        runner.run_in_transaction(operation)

    assert raised.value is expected
    assert operation.__name__ == "operation"
    assert connection.commit_calls == 0
    assert connection.rollback_calls == 1
    assert connection.close_calls == 1


def test_commit_failure_rolls_back_closes_and_preserves_exception() -> None:
    expected = RuntimeError("synthetic commit failure")
    connection = SyntheticConnection(commit_error=expected)
    runner = _runner(connection)

    with pytest.raises(RuntimeError) as raised:
        runner.run_in_transaction(lambda _transaction: None)

    assert raised.value is expected
    assert connection.commit_calls == 1
    assert connection.rollback_calls == 1
    assert connection.close_calls == 1


def test_rollback_failure_is_attached_to_primary_as_group_cause() -> None:
    primary = RuntimeError("synthetic primary failure")
    rollback = OSError("synthetic rollback failure")
    connection = SyntheticConnection(rollback_error=rollback)
    runner = _runner(connection)

    with pytest.raises(RuntimeError) as raised:
        runner.run_in_transaction(lambda _transaction: (_ for _ in ()).throw(primary))

    assert raised.value is primary
    cause = raised.value.__cause__
    assert isinstance(cause, ExceptionGroup)
    assert cause.exceptions == (rollback,)
    assert connection.close_calls == 1


def test_close_only_failure_after_commit_propagates_exactly() -> None:
    expected = OSError("synthetic close failure")
    connection = SyntheticConnection(close_error=expected)
    runner = _runner(connection)

    with pytest.raises(OSError) as raised:
        runner.run_in_transaction(lambda _transaction: None)

    assert raised.value is expected
    assert connection.commit_calls == 1
    assert connection.rollback_calls == 0
    assert connection.close_calls == 1


def test_rollback_and_close_failures_preserve_order_in_group_cause() -> None:
    primary = RuntimeError("synthetic primary failure")
    rollback = OSError("synthetic rollback failure")
    close = OSError("synthetic close failure")
    connection = SyntheticConnection(
        events=[],
        rollback_error=rollback,
        close_error=close,
    )
    runner = _runner(connection)

    def fail(_transaction: PostgreSQLVectorTransaction) -> None:
        raise primary

    with pytest.raises(RuntimeError) as raised:
        runner.run_in_transaction(fail)

    assert raised.value is primary
    cause = raised.value.__cause__
    assert isinstance(cause, ExceptionGroup)
    assert cause.exceptions == (rollback, close)
    assert connection.events == ["rollback", "close"]


def test_connection_factory_failure_has_no_cleanup() -> None:
    expected = OSError("synthetic connection failure")
    transaction_calls = 0

    def connection_factory() -> Connection[Any]:
        raise expected

    def transaction_factory(
        _connection: Connection[Any],
    ) -> PostgreSQLVectorTransaction:
        nonlocal transaction_calls
        transaction_calls += 1
        return cast(PostgreSQLVectorTransaction, SyntheticTransaction())

    runner = PsycopgPostgreSQLVectorTransactionRunner(
        connection_factory=connection_factory,
        transaction_factory=transaction_factory,
    )

    with pytest.raises(OSError) as raised:
        runner.run_in_transaction(lambda _transaction: None)

    assert raised.value is expected
    assert transaction_calls == 0


def test_repeated_operations_use_independent_fresh_connections() -> None:
    connections: list[SyntheticConnection] = []

    def connection_factory() -> Connection[Any]:
        connection = SyntheticConnection()
        connections.append(connection)
        return _as_connection(connection)

    runner = PsycopgPostgreSQLVectorTransactionRunner(
        connection_factory=connection_factory,
        transaction_factory=lambda _connection: cast(
            PostgreSQLVectorTransaction, SyntheticTransaction()
        ),
    )

    assert runner.run_in_transaction(lambda _transaction: "first") == "first"
    assert runner.run_in_transaction(lambda _transaction: "second") == "second"
    assert len(connections) == 2
    assert connections[0] is not connections[1]
    assert all(connection.close_calls == 1 for connection in connections)


def test_runner_does_not_access_cursor_or_execute_sql() -> None:
    class CursorGuardConnection(SyntheticConnection):
        def cursor(self) -> None:
            raise AssertionError("runner must not create cursors")

        def execute(self, _query: object) -> None:
            raise AssertionError("runner must not execute SQL")

    connection = CursorGuardConnection()
    runner = _runner(connection)

    assert runner.run_in_transaction(lambda _transaction: "synthetic") == "synthetic"
