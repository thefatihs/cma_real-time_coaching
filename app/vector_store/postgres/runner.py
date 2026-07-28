"""Synchronous Psycopg transaction lifecycle for PostgreSQL vector operations."""

from collections.abc import Callable
from typing import Any, TypeVar

from psycopg import Connection

from app.vector_store.postgres.contracts import PostgreSQLVectorTransaction

PsycopgConnectionFactory = Callable[[], Connection[Any]]
PsycopgTransactionFactory = Callable[
    [Connection[Any]],
    PostgreSQLVectorTransaction,
]

T = TypeVar("T")


class PsycopgPostgreSQLVectorTransactionRunner:
    """Run one callback inside one synchronous Psycopg transaction."""

    def __init__(
        self,
        *,
        connection_factory: PsycopgConnectionFactory,
        transaction_factory: PsycopgTransactionFactory,
    ) -> None:
        if not callable(connection_factory):
            raise ValueError("connection_factory must be callable")
        if not callable(transaction_factory):
            raise ValueError("transaction_factory must be callable")
        self._connection_factory = connection_factory
        self._transaction_factory = transaction_factory

    def run_in_transaction(
        self,
        operation: Callable[[PostgreSQLVectorTransaction], T],
    ) -> T:
        if not callable(operation):
            raise ValueError("operation must be callable")

        connection = self._connection_factory()
        try:
            if connection.autocommit is not False:
                raise ValueError("connection.autocommit must be exactly False")
            transaction = self._transaction_factory(connection)
            result = operation(transaction)
            connection.commit()
        except Exception as primary:
            cleanup_failures: list[Exception] = []
            try:
                connection.rollback()
            except Exception as error:
                cleanup_failures.append(error)
            try:
                connection.close()
            except Exception as error:
                cleanup_failures.append(error)
            if cleanup_failures:
                raise primary from ExceptionGroup(
                    "PostgreSQL transaction cleanup failed",
                    cleanup_failures,
                )
            raise

        connection.close()
        return result
