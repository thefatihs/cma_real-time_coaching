"""Run one readiness-verified PostgreSQL retrieval-to-vLLM orchestration."""

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

from app.coaching.llm_result_gate import coaching_wire_json_schema
from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
)
from app.deployment import (
    PostgreSQLRAGOrchestrationLimits,
    orchestrate_profile_bound_postgres_rag,
)
from app.llm.vllm_openai_compatible import VLLMOpenAICompatibleSettings
from app.orchestration.models import OrchestrationRequest, OrchestrationResult

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
_REQUEST_KEYS = frozenset(
    {
        "tenant_id",
        "call_id",
        "transcript_revision",
        "knowledge_base_id",
        "user_input",
        "top_k",
        "minimum_score",
    }
)
_LIMIT_KEYS = frozenset(
    {"max_top_k", "max_user_input_characters", "max_prompt_characters"}
)
_SECRET_KEYS = frozenset(
    {
        "dsn",
        "password",
        "credential",
        "secret",
        "token",
        "api_token",
        "endpoint",
        "base_url",
        "host",
        "port",
        "database",
        "user",
        "username",
        "ssl_mode",
        "connect_timeout_seconds",
        "read_timeout_seconds",
        "application_name",
        "verify_tls",
    }
)
_CONFIGURATION_FAILURE = "PostgreSQL RAG orchestration configuration is invalid."
_OPERATIONAL_FAILURE = "PostgreSQL RAG orchestration failed."
_SUCCESS = "PostgreSQL RAG orchestration succeeded."


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise ValueError("invalid command arguments")


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description="Run one PostgreSQL RAG orchestration.")
    parser.add_argument("--provider-settings", required=True, type=Path, metavar="PATH")
    parser.add_argument(
        "--orchestration-request", required=True, type=Path, metavar="PATH"
    )
    parser.add_argument("--operation-limits", required=True, type=Path, metavar="PATH")
    parser.add_argument("--output", required=True, type=Path, metavar="PATH")
    return parser


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains a duplicate key")
        result[key] = value
    return result


def _load_exact_json(path: Path, keys: frozenset[str]) -> dict[str, object]:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError("JSON file cannot be empty")
    payload = json.loads(content, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("JSON value must be an object")
    actual = set(payload)
    if actual & _SECRET_KEYS or actual != keys:
        raise ValueError("JSON keys are invalid")
    return payload


def _validate_output_path(path: Path) -> None:
    if os.path.lexists(path):
        raise ValueError("output path must not exist")
    if not path.parent.is_dir():
        raise ValueError("output parent must be a directory")


def _write_result(path: Path, result: OrchestrationResult | None) -> None:
    payload = None if result is None else result.model_dump(mode="json")
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
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
        provider = KnowledgeBaseRAGProviderSettings.model_validate(
            _load_exact_json(arguments.provider_settings, _PROVIDER_KEYS)
        )
        request = OrchestrationRequest.model_validate(
            _load_exact_json(arguments.orchestration_request, _REQUEST_KEYS)
        )
        limits = PostgreSQLRAGOrchestrationLimits.model_validate(
            _load_exact_json(arguments.operation_limits, _LIMIT_KEYS)
        )
        if request.tenant_id != provider.tenant_id:
            raise ValueError("request tenant scope does not match provider scope")
        if request.knowledge_base_id != provider.knowledge_base_id:
            raise ValueError(
                "request knowledge-base scope does not match provider scope"
            )
        if request.top_k > limits.max_top_k:
            raise ValueError("request top_k exceeds operation limit")
        if len(request.user_input) > limits.max_user_input_characters:
            raise ValueError("request user_input exceeds operation limit")
        _validate_output_path(arguments.output)
        postgres_factory = cast(
            Callable[[], PostgreSQLVectorStoreSettings],
            PostgreSQLVectorStoreSettings,
        )
        vllm_factory = cast(
            Callable[[], VLLMOpenAICompatibleSettings],
            VLLMOpenAICompatibleSettings,
        )
        postgres_settings = postgres_factory()
        vllm_settings = vllm_factory()
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
        result = orchestrate_profile_bound_postgres_rag(
            postgres_settings=postgres_settings,
            knowledge_base_settings=provider,
            vllm_settings=vllm_settings,
            request=request,
            limits=limits,
            psycopg_connect=cast(Any, psycopg.connect),
            structured_output_json_schema=coaching_wire_json_schema(),
        )
        _write_result(arguments.output, result)
    except Exception:
        print(_OPERATIONAL_FAILURE, file=sys.stderr)
        return 1
    print(_SUCCESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
