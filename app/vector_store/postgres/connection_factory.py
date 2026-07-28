"""Pgvector type registration for freshly acquired Psycopg connections."""

from collections.abc import Callable
from typing import Any

from pgvector.psycopg import register_vector
from psycopg import Connection


class PgvectorPsycopgConnectionFactory:
    """Register pgvector types on one fresh connection per call."""

    def __init__(
        self,
        *,
        base_connection_factory: Callable[[], Connection[Any]],
    ) -> None:
        if not callable(base_connection_factory):
            raise ValueError("base_connection_factory must be callable")
        self._base_connection_factory = base_connection_factory

    def __call__(self) -> Connection[Any]:
        connection = self._base_connection_factory()
        try:
            if connection.autocommit is not False:
                raise ValueError("connection.autocommit must be exactly False")
            register_vector(connection)
        except Exception as primary:
            try:
                connection.close()
            except Exception as close_error:
                raise primary from ExceptionGroup(
                    "Pgvector connection cleanup failed",
                    [close_error],
                )
            raise
        return connection
