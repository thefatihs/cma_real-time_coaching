"""Explicitly ingest preconstructed chunks into one PostgreSQL RAG scope."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

import psycopg
from pydantic import ValidationError

from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
)
from app.deployment import ingest_profile_bound_postgres_rag
from app.ingestion.models import DocumentIngestionRequest

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
_INGESTION_KEYS = frozenset({"tenant_id", "knowledge_base_id", "chunks"})
_CHUNK_KEYS = frozenset({"document_id", "chunk_id", "text", "metadata"})
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
_CONFIGURATION_FAILURE = "PostgreSQL RAG ingestion configuration is invalid."
_OPERATIONAL_FAILURE = "PostgreSQL RAG chunk ingestion failed."
_SUCCESS = "PostgreSQL RAG chunk ingestion succeeded."


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise ValueError("invalid command arguments")


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Ingest preconstructed chunks into one PostgreSQL RAG scope.",
    )
    parser.add_argument(
        "--provider-settings",
        required=True,
        type=Path,
        metavar="PATH",
    )
    parser.add_argument(
        "--ingestion-request",
        required=True,
        type=Path,
        metavar="PATH",
    )
    return parser


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains a duplicate key")
        result[key] = value
    return result


def _load_json_object(path: Path) -> dict[str, object]:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError("JSON file cannot be empty")
    payload = json.loads(content, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("JSON value must be an object")
    return payload


def _require_exact_keys(
    payload: dict[str, object],
    expected: frozenset[str],
) -> None:
    keys = set(payload)
    if keys & _CONNECTION_KEYS or keys != expected:
        raise ValueError("JSON keys are invalid")


def _load_provider_settings(path: Path) -> KnowledgeBaseRAGProviderSettings:
    payload = _load_json_object(path)
    _require_exact_keys(payload, _PROVIDER_KEYS)
    return KnowledgeBaseRAGProviderSettings.model_validate(payload)


def _load_ingestion_request(path: Path) -> DocumentIngestionRequest:
    payload = _load_json_object(path)
    _require_exact_keys(payload, _INGESTION_KEYS)
    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("chunks must be a nonempty array")
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            raise ValueError("each chunk must be an object")
        _require_exact_keys(raw_chunk, _CHUNK_KEYS)
    return DocumentIngestionRequest.model_validate(payload)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _build_parser().parse_args(argv)
        provider_settings = _load_provider_settings(arguments.provider_settings)
        ingestion_request = _load_ingestion_request(arguments.ingestion_request)
        if ingestion_request.tenant_id != provider_settings.tenant_id:
            raise ValueError("ingestion tenant scope does not match provider scope")
        if ingestion_request.knowledge_base_id != provider_settings.knowledge_base_id:
            raise ValueError(
                "ingestion knowledge-base scope does not match provider scope"
            )
        postgres_settings_factory = cast(
            Callable[[], PostgreSQLVectorStoreSettings],
            PostgreSQLVectorStoreSettings,
        )
        postgres_settings = postgres_settings_factory()
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ):
        print(_CONFIGURATION_FAILURE, file=sys.stderr)
        return 2

    try:
        ingest_profile_bound_postgres_rag(
            postgres_settings=postgres_settings,
            knowledge_base_settings=provider_settings,
            request=ingestion_request,
            psycopg_connect=cast(Any, psycopg.connect),
        )
    except Exception:
        print(_OPERATIONAL_FAILURE, file=sys.stderr)
        return 1

    print(_SUCCESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
