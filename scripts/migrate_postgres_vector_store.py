"""Explicitly apply the fixed PostgreSQL vector-store migration."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Any, cast

import psycopg
from pydantic import ValidationError

from app.deployment import (
    PostgreSQLMigrationResult,
    PostgreSQLMigrationSettings,
    apply_postgres_vector_migrations,
)
from app.deployment.postgres_migrations import (
    PostgreSQLMigrationConfigurationError,
)

_APPLIED = "PostgreSQL vector-store migration applied."
_ALREADY_APPLIED = "PostgreSQL vector-store migration is already applied."
_CONFIGURATION_FAILURE = "PostgreSQL vector-store migration configuration is invalid."
_OPERATIONAL_FAILURE = "PostgreSQL vector-store migration failed."


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Apply the fixed PostgreSQL vector-store migration.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    _build_parser().parse_args(argv)
    try:
        settings_factory = cast(
            Callable[[], PostgreSQLMigrationSettings],
            PostgreSQLMigrationSettings,
        )
        settings = settings_factory()
    except (ValidationError, ValueError):
        print(_CONFIGURATION_FAILURE, file=sys.stderr)
        return 2

    try:
        result = apply_postgres_vector_migrations(
            settings=settings,
            psycopg_connect=cast(Any, psycopg.connect),
        )
    except PostgreSQLMigrationConfigurationError:
        print(_CONFIGURATION_FAILURE, file=sys.stderr)
        return 2
    except Exception:
        print(_OPERATIONAL_FAILURE, file=sys.stderr)
        return 1

    print(_APPLIED if result is PostgreSQLMigrationResult.APPLIED else _ALREADY_APPLIED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
