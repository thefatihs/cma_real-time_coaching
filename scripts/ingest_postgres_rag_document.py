"""Explicitly ingest one trusted UTF-8 document into PostgreSQL RAG."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

import psycopg
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
)

from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
)
from app.deployment import ingest_profile_bound_postgres_rag
from app.ingestion import (
    TextDocumentSource,
    build_fixed_character_document_ingestion_request,
)
from app.vector_store.models import Metadata

_UTF8_BOM = b"\xef\xbb\xbf"
_SUPPORTED_SUFFIXES = frozenset({".md", ".txt"})
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
_DOCUMENT_KEYS = frozenset(
    {
        "document_id",
        "metadata",
        "max_file_bytes",
        "max_document_characters",
        "max_chunk_characters",
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
_CONFIGURATION_FAILURE = "PostgreSQL RAG document configuration is invalid."
_OPERATIONAL_FAILURE = "PostgreSQL RAG document ingestion failed."
_SUCCESS = "PostgreSQL RAG document ingestion succeeded."


class _DocumentSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    metadata: Metadata
    max_file_bytes: int
    max_document_characters: int
    max_chunk_characters: int

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("document_id cannot be empty")
        if cleaned != value:
            raise ValueError("document_id must be canonical")
        return value

    @field_validator(
        "max_file_bytes",
        "max_document_characters",
        "max_chunk_characters",
        mode="before",
    )
    @classmethod
    def validate_limit(cls, value: object, info: object) -> int:
        if type(value) is not int:
            raise ValueError(
                f"{getattr(info, 'field_name', 'limit')} must be an integer"
            )
        if value <= 0:
            raise ValueError(f"{getattr(info, 'field_name', 'limit')} must be positive")
        return value


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise ValueError("invalid command arguments")


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Ingest one trusted UTF-8 TXT or Markdown document.",
    )
    parser.add_argument(
        "--provider-settings",
        required=True,
        type=Path,
        metavar="PATH",
    )
    parser.add_argument(
        "--document-settings",
        required=True,
        type=Path,
        metavar="PATH",
    )
    parser.add_argument(
        "--document",
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


def _load_document_settings(path: Path) -> _DocumentSettings:
    payload = _load_json_object(path)
    _require_exact_keys(payload, _DOCUMENT_KEYS)
    metadata = payload.get("metadata")
    if not isinstance(metadata, list):
        raise ValueError("metadata must be an array")
    if any(
        not isinstance(pair, list)
        or len(pair) != 2
        or any(not isinstance(item, str) for item in pair)
        for pair in metadata
    ):
        raise ValueError("metadata must contain text pairs")
    return _DocumentSettings.model_validate(payload)


def _read_document(path: Path, *, max_file_bytes: int) -> str:
    if any(part == ".." for part in path.parts):
        raise ValueError("document path cannot contain traversal components")
    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise ValueError("document extension is unsupported")
    if path.is_symlink():
        raise ValueError("document path cannot be a symlink")
    if not path.is_file():
        raise ValueError("document path must be a regular file")

    with path.open("rb") as stream:
        file_status = os.fstat(stream.fileno())
        if not stat.S_ISREG(file_status.st_mode):
            raise ValueError("document path must be a regular file")
        if file_status.st_size > max_file_bytes:
            raise ValueError("document file exceeds max_file_bytes")
        content = stream.read(max_file_bytes + 1)
    if len(content) > max_file_bytes:
        raise ValueError("document file exceeds max_file_bytes")
    if content.startswith(_UTF8_BOM):
        raise ValueError("UTF-8 BOM is not supported")
    text = content.decode("utf-8", errors="strict")
    if "\x00" in text:
        raise ValueError("document text cannot contain NUL")
    return text


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _build_parser().parse_args(argv)
        provider_settings = _load_provider_settings(arguments.provider_settings)
        document_settings = _load_document_settings(arguments.document_settings)
        text = _read_document(
            arguments.document,
            max_file_bytes=document_settings.max_file_bytes,
        )
        document = TextDocumentSource(
            document_id=document_settings.document_id,
            text=text,
            metadata=document_settings.metadata,
        )
        request = build_fixed_character_document_ingestion_request(
            tenant_id=provider_settings.tenant_id,
            knowledge_base_id=provider_settings.knowledge_base_id,
            document=document,
            max_chunk_characters=document_settings.max_chunk_characters,
            max_document_characters=document_settings.max_document_characters,
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
            request=request,
            psycopg_connect=cast(Any, psycopg.connect),
        )
    except Exception:
        print(_OPERATIONAL_FAILURE, file=sys.stderr)
        return 1

    print(_SUCCESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
