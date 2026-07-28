"""Focused tests for pgvector registration on fresh Psycopg connections."""

from collections.abc import Callable
from typing import Any, cast

import pytest
from psycopg import Connection

import app.vector_store.postgres.connection_factory as factory_module
from app.vector_store.postgres.connection_factory import (
    PgvectorPsycopgConnectionFactory,
)


class SyntheticConnection:
    def __init__(
        self,
        *,
        autocommit: object = False,
        close_error: Exception | None = None,
    ) -> None:
        self.autocommit = autocommit
        self.close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FalseyCallable:
    def __init__(self, callback: Callable[[], Connection[Any]]) -> None:
        self._callback = callback
        self.calls = 0

    def __bool__(self) -> bool:
        return False

    def __call__(self) -> Connection[Any]:
        self.calls += 1
        return self._callback()


def _connection(value: SyntheticConnection) -> Connection[Any]:
    return cast(Connection[Any], value)


def test_factory_is_callable_and_constructor_has_no_side_effects() -> None:
    base = FalseyCallable(lambda: _connection(SyntheticConnection()))

    factory = PgvectorPsycopgConnectionFactory(base_connection_factory=base)

    assert callable(factory)
    assert base.calls == 0


def test_constructor_rejects_noncallable_factory() -> None:
    with pytest.raises(ValueError, match="base_connection_factory"):
        PgvectorPsycopgConnectionFactory(
            base_connection_factory=None,  # type: ignore[arg-type]
        )


def test_falsey_factory_is_retained_and_registration_returns_exact_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = SyntheticConnection()
    base = FalseyCallable(lambda: _connection(connection))
    registered: list[Connection[Any]] = []
    monkeypatch.setattr(
        factory_module,
        "register_vector",
        lambda supplied: registered.append(supplied),
    )
    factory = PgvectorPsycopgConnectionFactory(base_connection_factory=base)

    result = factory()

    assert result is _connection(connection)
    assert base.calls == 1
    assert registered == [_connection(connection)]
    assert connection.close_calls == 0


def test_autocommit_validation_closes_without_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = SyntheticConnection(autocommit=True)
    registrations = 0

    def register(_connection: Connection[Any]) -> None:
        nonlocal registrations
        registrations += 1

    monkeypatch.setattr(factory_module, "register_vector", register)
    factory = PgvectorPsycopgConnectionFactory(
        base_connection_factory=lambda: _connection(connection)
    )

    with pytest.raises(ValueError, match="autocommit"):
        factory()

    assert registrations == 0
    assert connection.close_calls == 1


def test_registration_failure_closes_and_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = SyntheticConnection()
    primary = RuntimeError("synthetic registration failure")
    monkeypatch.setattr(
        factory_module,
        "register_vector",
        lambda _connection: (_ for _ in ()).throw(primary),
    )
    factory = PgvectorPsycopgConnectionFactory(
        base_connection_factory=lambda: _connection(connection)
    )

    with pytest.raises(RuntimeError) as raised:
        factory()

    assert raised.value is primary
    assert connection.close_calls == 1


def test_close_failure_is_group_cause_without_replacing_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = RuntimeError("synthetic registration failure")
    close = OSError("synthetic close failure")
    connection = SyntheticConnection(close_error=close)
    monkeypatch.setattr(
        factory_module,
        "register_vector",
        lambda _connection: (_ for _ in ()).throw(primary),
    )
    factory = PgvectorPsycopgConnectionFactory(
        base_connection_factory=lambda: _connection(connection)
    )

    with pytest.raises(RuntimeError) as raised:
        factory()

    assert raised.value is primary
    assert isinstance(raised.value.__cause__, ExceptionGroup)
    assert raised.value.__cause__.exceptions == (close,)
    assert connection.close_calls == 1


def test_base_factory_failure_has_no_cleanup_or_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = OSError("synthetic acquisition failure")
    registrations = 0

    def fail() -> Connection[Any]:
        raise primary

    def register(_connection: Connection[Any]) -> None:
        nonlocal registrations
        registrations += 1

    monkeypatch.setattr(factory_module, "register_vector", register)
    factory = PgvectorPsycopgConnectionFactory(base_connection_factory=fail)

    with pytest.raises(OSError) as raised:
        factory()

    assert raised.value is primary
    assert registrations == 0


def test_repeated_calls_acquire_and_register_fresh_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[SyntheticConnection] = []
    registered: list[Connection[Any]] = []

    def base() -> Connection[Any]:
        connection = SyntheticConnection()
        connections.append(connection)
        return _connection(connection)

    monkeypatch.setattr(
        factory_module,
        "register_vector",
        lambda supplied: registered.append(supplied),
    )
    factory = PgvectorPsycopgConnectionFactory(base_connection_factory=base)

    first = factory()
    second = factory()

    assert first is not second
    assert registered == [first, second]
    assert all(connection.close_calls == 0 for connection in connections)


def test_factory_owns_no_configuration_or_pooling_surface() -> None:
    public = {
        name
        for name in vars(PgvectorPsycopgConnectionFactory)
        if not name.startswith("_")
    }

    assert public == set()
