"""Explicitly retrieve from one ready, profile-bound PostgreSQL RAG scope."""

from __future__ import annotations

import argparse
import json
import os
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
from app.deployment import (
    PostgreSQLRAGRetrievalRequest,
    retrieve_profile_bound_postgres_rag,
)
from app.retrieval.models import RetrievalResult

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
_RETRIEVAL_KEYS = frozenset(
    {
        "tenant_id",
        "knowledge_base_id",
        "query",
        "top_k",
        "minimum_score",
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
_CONFIGURATION_FAILURE = "PostgreSQL RAG retrieval configuration is invalid."
_OPERATIONAL_FAILURE = "PostgreSQL RAG retrieval failed."
_SUCCESS = "PostgreSQL RAG retrieval succeeded."


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise ValueError("invalid command arguments")


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Retrieve from one ready PostgreSQL RAG scope.",
    )
    parser.add_argument(
        "--provider-settings",
        required=True,
        type=Path,
        metavar="PATH",
    )
    parser.add_argument(
        "--retrieval-request",
        required=True,
        type=Path,
        metavar="PATH",
    )
    parser.add_argument(
        "--output",
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


def _load_retrieval_request(path: Path) -> PostgreSQLRAGRetrievalRequest:
    payload = _load_json_object(path)
    _require_exact_keys(payload, _RETRIEVAL_KEYS)
    return PostgreSQLRAGRetrievalRequest.model_validate(payload)


def _validate_output_path(path: Path) -> None:
    if os.path.lexists(path):
        raise ValueError("output path must not exist")
    parent = path.parent
    if not parent.is_dir():
        raise ValueError("output parent must be a directory")


def _write_result(path: Path, result: RetrievalResult) -> None:
    content = (
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    created = False
    try:
        with path.open("x", encoding="utf-8", newline="\n") as output:
            created = True
            output.write(content)
    except BaseException:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _build_parser().parse_args(argv)
        provider_settings = _load_provider_settings(arguments.provider_settings)
        retrieval_request = _load_retrieval_request(arguments.retrieval_request)
        if retrieval_request.tenant_id != provider_settings.tenant_id:
            raise ValueError("retrieval tenant scope does not match provider scope")
        if retrieval_request.knowledge_base_id != provider_settings.knowledge_base_id:
            raise ValueError(
                "retrieval knowledge-base scope does not match provider scope"
            )
        _validate_output_path(arguments.output)
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
        result = retrieve_profile_bound_postgres_rag(
            postgres_settings=postgres_settings,
            knowledge_base_settings=provider_settings,
            request=retrieval_request,
            psycopg_connect=cast(Any, psycopg.connect),
        )
        _write_result(arguments.output, result)
    except Exception:
        print(_OPERATIONAL_FAILURE, file=sys.stderr)
        return 1

    print(_SUCCESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
