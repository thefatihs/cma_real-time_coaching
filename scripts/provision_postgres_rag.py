"""Explicitly verify PostgreSQL RAG readiness and provision one profile."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import psycopg
from pydantic import ValidationError

from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
)
from app.deployment import provision_profile_bound_postgres_rag

_PROVIDER_KEYS = frozenset(
    {
        "tenant_id",
        "knowledge_base_id",
        "model_id",
        "model_name_or_path",
        "vector_dimension",
        "normalize_embeddings",
        "device",
        "local_files_only",
    }
)
_CONNECTION_KEYS = frozenset(
    {
        "dsn",
        "password",
        "credential",
        "secret",
        "host",
        "port",
        "database",
        "user",
        "username",
        "ssl_mode",
        "connect_timeout_seconds",
        "application_name",
    }
)
_CONFIGURATION_FAILURE = "PostgreSQL RAG provisioning configuration is invalid."
_OPERATIONAL_FAILURE = "PostgreSQL RAG profile provisioning failed."
_SUCCESS = "PostgreSQL RAG profile provisioning succeeded."


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify PostgreSQL RAG readiness and provision one profile.",
    )
    parser.add_argument(
        "--provider-settings",
        required=True,
        type=Path,
        metavar="PATH",
    )
    return parser


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("provider settings contain a duplicate key")
        result[key] = value
    return result


def _load_provider_settings(path: Path) -> KnowledgeBaseRAGProviderSettings:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError("provider settings file cannot be empty")
    payload = json.loads(content, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("provider settings must be a JSON object")
    keys = set(payload)
    if keys & _CONNECTION_KEYS:
        raise ValueError("provider settings contain a prohibited connection key")
    if keys != _PROVIDER_KEYS:
        raise ValueError("provider settings keys are invalid")
    return KnowledgeBaseRAGProviderSettings.model_validate(payload)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        provider_settings = _load_provider_settings(arguments.provider_settings)
        postgres_settings_factory = cast(
            Callable[[], PostgreSQLVectorStoreSettings],
            PostgreSQLVectorStoreSettings,
        )
        postgres_settings = postgres_settings_factory()
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError):
        print(_CONFIGURATION_FAILURE, file=sys.stderr)
        return 2

    try:
        provision_profile_bound_postgres_rag(
            postgres_settings=postgres_settings,
            knowledge_base_settings=provider_settings,
            psycopg_connect=cast(Any, psycopg.connect),
        )
    except Exception:
        print(_OPERATIONAL_FAILURE, file=sys.stderr)
        return 1

    print(_SUCCESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
