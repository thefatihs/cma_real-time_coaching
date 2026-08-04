"""Run one bounded synthetic Windows PostgreSQL-RAG-to-vLLM contract check."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx
from psycopg import Error as PsycopgError
from psycopg import connect as psycopg_connect
from pydantic import SecretStr

from app.coaching.llm_result_gate import (
    LLMCoachingGateStatus,
    LLMCoachingRejectionReason,
    LLMCoachingResultGate,
    coaching_wire_json_schema,
)
from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
)
from app.deployment import (
    ingest_profile_bound_postgres_rag,
    orchestrate_profile_bound_postgres_rag,
    provision_profile_bound_postgres_rag,
)
from app.deployment.postgres_orchestration import PostgreSQLRAGOrchestrationLimits
from app.events.models import (
    ClassificationLabel,
    ClassificationResultEvent,
    CoachingAction,
    CoachingSuggestionSource,
    TranscriptEvent,
    TranscriptKind,
)
from app.ingestion.models import DocumentIngestionRequest
from app.integration.llm_suggestion_factory import (
    DeterministicLLMCoachingSuggestionFactory,
)
from app.integration.policy import RAGCoachingIntegrationPolicy
from app.llm.vllm_openai_compatible import VLLMOpenAICompatibleSettings
from app.orchestration.models import OrchestrationRequest, OrchestrationResult
from app.vector_store.models import VectorBatchWriteResult

TENANT_ID = "tenant_alpha"
KNOWLEDGE_BASE_ID = "kb_smoke"
CATEGORY = "urun_bilgisi"
DOCUMENT_ID = "sentetik_urun_iade_politikasi_v1"
CHUNK_ID = "sentetik_urun_iade_001"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384
CALL_ID = "call_windows_rag_vllm_e2e"
TRANSCRIPT_EVENT_ID = "transcript_windows_rag_vllm_e2e_1"
TRANSCRIPT_REVISION = 1
VLLM_BASE_URL = "https://localhost:9443/v1"
TOTAL_E2E_TIMEOUT_SECONDS = 300.0
MAX_CONFIGURATION_BYTES = 65_536
MAX_TOKEN_BYTES = 4_096

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROVIDER_PATH = REPOSITORY_ROOT / "docs/examples/windows-rag-vllm-e2e-provider.json"
POLICY_PATH = REPOSITORY_ROOT / "docs/examples/windows-rag-vllm-e2e-policy.json"
DOCUMENT_PATH = REPOSITORY_ROOT / "docs/examples/windows-rag-vllm-e2e-document.json"

PROVIDER_KEYS = frozenset(
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
POLICY_KEYS = frozenset(
    {
        "rag_llm_enabled_labels",
        "title",
        "action",
        "priority",
        "label_id",
        "expires_after_seconds",
    }
)
DOCUMENT_KEYS = frozenset({"tenant_id", "knowledge_base_id", "category", "chunks"})
TOKEN_FILE_ENV = "CALLMETRIC_VLLM_API_TOKEN_FILE"

SYNTHETIC_TRANSCRIPT = (
    "Sentetik deneme ürününü iade etmek için süre ve gerekli belge nedir?"
)
FIXED_TIME = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)

GATE_REJECTION_PHASES = {
    LLMCoachingRejectionReason.INVALID_JSON: "E_ADMISSION_INVALID_JSON",
    LLMCoachingRejectionReason.DUPLICATE_KEY: "E_ADMISSION_DUPLICATE_KEY",
    LLMCoachingRejectionReason.PAYLOAD_TOO_LARGE: "E_ADMISSION_PAYLOAD_TOO_LARGE",
    LLMCoachingRejectionReason.PAYLOAD_TOO_DEEP: "E_ADMISSION_PAYLOAD_TOO_DEEP",
    LLMCoachingRejectionReason.SCHEMA_VALIDATION_FAILED: "E_ADMISSION_SCHEMA",
    LLMCoachingRejectionReason.SCOPE_MISMATCH: "E_ADMISSION_GATE_SCOPE",
    LLMCoachingRejectionReason.UNSUPPORTED_DECISION: "E_ADMISSION_DECISION",
    LLMCoachingRejectionReason.CITATION_NOT_ALLOWED: (
        "E_ADMISSION_CITATION_NOT_ALLOWED"
    ),
    LLMCoachingRejectionReason.DUPLICATE_CITATION: ("E_ADMISSION_DUPLICATE_CITATION"),
}


class WindowsRAGVLLME2EError(RuntimeError):
    """A fixed, secret-free E2E phase failure."""


class ConnectionFactory(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class E2EArtifacts:
    provider: KnowledgeBaseRAGProviderSettings
    policy: RAGCoachingIntegrationPolicy
    document: DocumentIngestionRequest


@dataclass(frozen=True, slots=True)
class E2ESettings:
    postgres: PostgreSQLVectorStoreSettings
    vllm: VLLMOpenAICompatibleSettings


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise WindowsRAGVLLME2EError("E_CONFIGURATION")
        result[key] = value
    return result


def _load_exact_json(path: Path, keys: frozenset[str]) -> dict[str, object]:
    try:
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise WindowsRAGVLLME2EError("E_CONFIGURATION")
        raw = path.read_bytes()
        if (
            not raw
            or len(raw) > MAX_CONFIGURATION_BYTES
            or raw.startswith(b"\xef\xbb\xbf")
            or b"\0" in raw
        ):
            raise WindowsRAGVLLME2EError("E_CONFIGURATION")
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except WindowsRAGVLLME2EError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise WindowsRAGVLLME2EError("E_CONFIGURATION") from None
    if not isinstance(value, dict) or set(value) != keys:
        raise WindowsRAGVLLME2EError("E_CONFIGURATION")
    return value


def _load_artifacts() -> E2EArtifacts:
    try:
        provider = KnowledgeBaseRAGProviderSettings.model_validate(
            _load_exact_json(PROVIDER_PATH, PROVIDER_KEYS)
        )
        policy = RAGCoachingIntegrationPolicy.model_validate(
            _load_exact_json(POLICY_PATH, POLICY_KEYS)
        )
        raw_document = _load_exact_json(DOCUMENT_PATH, DOCUMENT_KEYS)
        category = raw_document.pop("category", None)
        document = DocumentIngestionRequest.model_validate(raw_document)
    except WindowsRAGVLLME2EError:
        raise
    except Exception:
        raise WindowsRAGVLLME2EError("E_CONFIGURATION") from None
    if (
        provider.tenant_id != TENANT_ID
        or provider.knowledge_base_id != KNOWLEDGE_BASE_ID
        or provider.model_id != MODEL_ID
        or provider.model_name_or_path != MODEL_ID
        or provider.vector_dimension != VECTOR_DIMENSION
        or provider.local_files_only is not True
        or document.tenant_id != TENANT_ID
        or document.knowledge_base_id != KNOWLEDGE_BASE_ID
        or category != CATEGORY
        or len(document.chunks) != 1
        or document.chunks[0].document_id != DOCUMENT_ID
        or document.chunks[0].chunk_id != CHUNK_ID
        or policy.rag_llm_enabled_labels != (CATEGORY,)
        or policy.label_id != CATEGORY
        or policy.action is not CoachingAction.RAG_ACTION
    ):
        raise WindowsRAGVLLME2EError("E_SCOPE")
    return E2EArtifacts(provider=provider, policy=policy, document=document)


def _required_environment(environment: Mapping[str, str], key: str) -> str:
    value = environment.get(key)
    if value is None or not value or value != value.strip() or "\0" in value:
        raise WindowsRAGVLLME2EError("E_CONFIGURATION")
    return value


def _validated_private_file(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise WindowsRAGVLLME2EError("E_CONFIGURATION")
    return path


def _read_token(path: Path) -> SecretStr:
    try:
        raw = path.read_bytes()
        if not raw or len(raw) > MAX_TOKEN_BYTES or b"\0" in raw:
            raise WindowsRAGVLLME2EError("E_CONFIGURATION")
        token = raw.decode("utf-8", errors="strict").strip("\r\n")
    except WindowsRAGVLLME2EError:
        raise
    except (OSError, UnicodeError):
        raise WindowsRAGVLLME2EError("E_CONFIGURATION") from None
    if not token or token != token.strip() or any(char.isspace() for char in token):
        raise WindowsRAGVLLME2EError("E_CONFIGURATION")
    return SecretStr(token)


def _load_settings(environment: Mapping[str, str]) -> E2ESettings:
    token_path = _validated_private_file(
        _required_environment(environment, TOKEN_FILE_ENV)
    )
    ca_path = _validated_private_file(
        _required_environment(environment, "CALLMETRIC_VLLM_CA_CERTIFICATE_PATH")
    )
    try:
        postgres = PostgreSQLVectorStoreSettings.model_validate(
            {
                "dsn": SecretStr(
                    _required_environment(environment, "CALLMETRIC_POSTGRES_DSN")
                ),
                "connect_timeout_seconds": _required_environment(
                    environment, "CALLMETRIC_POSTGRES_CONNECT_TIMEOUT_SECONDS"
                ),
                "ssl_mode": _required_environment(
                    environment, "CALLMETRIC_POSTGRES_SSL_MODE"
                ),
                "application_name": _required_environment(
                    environment, "CALLMETRIC_POSTGRES_APPLICATION_NAME"
                ),
            }
        )
        vllm = VLLMOpenAICompatibleSettings.model_validate(
            {
                "base_url": _required_environment(
                    environment, "CALLMETRIC_VLLM_BASE_URL"
                ),
                "model_id": _required_environment(
                    environment, "CALLMETRIC_VLLM_MODEL_ID"
                ),
                "api_token": _read_token(token_path),
                "ca_certificate_path": SecretStr(str(ca_path)),
                "connect_timeout_seconds": _required_environment(
                    environment, "CALLMETRIC_VLLM_CONNECT_TIMEOUT_SECONDS"
                ),
                "read_timeout_seconds": _required_environment(
                    environment, "CALLMETRIC_VLLM_READ_TIMEOUT_SECONDS"
                ),
                "max_output_tokens": _required_environment(
                    environment, "CALLMETRIC_VLLM_MAX_OUTPUT_TOKENS"
                ),
                "temperature": _required_environment(
                    environment, "CALLMETRIC_VLLM_TEMPERATURE"
                ),
                "verify_tls": _required_environment(
                    environment, "CALLMETRIC_VLLM_VERIFY_TLS"
                ),
            }
        )
    except WindowsRAGVLLME2EError:
        raise
    except Exception:
        raise WindowsRAGVLLME2EError("E_CONFIGURATION") from None
    if (
        postgres.ssl_mode != "verify-full"
        or vllm.base_url != VLLM_BASE_URL
        or vllm.connect_timeout_seconds + vllm.read_timeout_seconds
        >= TOTAL_E2E_TIMEOUT_SECONDS
    ):
        raise WindowsRAGVLLME2EError("E_CONFIGURATION")
    return E2ESettings(postgres=postgres, vllm=vllm)


def _classification_context(
    policy: RAGCoachingIntegrationPolicy,
) -> ClassificationResultEvent:
    if CATEGORY not in policy.rag_llm_enabled_labels:
        raise WindowsRAGVLLME2EError("E_SCOPE")
    return ClassificationResultEvent(
        tenant_id=TENANT_ID,
        call_id=CALL_ID,
        transcript_event_id=TRANSCRIPT_EVENT_ID,
        labels=[ClassificationLabel(name=CATEGORY, score=1.0)],
        action=CoachingAction.RAG_ACTION,
        model_id="synthetic-fixed-label-context-v1",
        probabilities={CATEGORY: 1.0},
        thresholds={CATEGORY: 1.0},
        created_at_utc=FIXED_TIME,
    )


def _transcript_event() -> TranscriptEvent:
    return TranscriptEvent(
        tenant_id=TENANT_ID,
        call_id=CALL_ID,
        event_id=TRANSCRIPT_EVENT_ID,
        kind=TranscriptKind.STABLE,
        text=SYNTHETIC_TRANSCRIPT,
        start_seconds=0,
        end_seconds=5,
        revision=TRANSCRIPT_REVISION,
        created_at_utc=FIXED_TIME,
    )


def _orchestration_request(event: TranscriptEvent) -> OrchestrationRequest:
    return OrchestrationRequest(
        tenant_id=event.tenant_id,
        call_id=event.call_id,
        transcript_revision=event.revision,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        user_input=event.text,
        top_k=1,
        minimum_score=0.0,
    )


def _admit_result(
    *,
    policy: RAGCoachingIntegrationPolicy,
    event: TranscriptEvent,
    result: OrchestrationResult,
) -> None:
    if (
        result.tenant_id != TENANT_ID
        or result.call_id != CALL_ID
        or result.transcript_revision != TRANSCRIPT_REVISION
    ):
        raise WindowsRAGVLLME2EError("E_ADMISSION_SCOPE")
    citation_identities = tuple(
        (item.document_id, item.chunk_id) for item in result.citations
    )
    if citation_identities != ((DOCUMENT_ID, CHUNK_ID),):
        raise WindowsRAGVLLME2EError("E_ADMISSION_CITATION")
    gate_result = LLMCoachingResultGate().evaluate(
        tenant_id=event.tenant_id,
        call_id=event.call_id,
        revision=event.revision,
        raw_output=result.generated_text,
        allowed_citations=set(citation_identities),
    )
    if gate_result.status is LLMCoachingGateStatus.VALID_NO_SUGGESTION:
        raise WindowsRAGVLLME2EError("E_ADMISSION_NO_SUGGESTION")
    if gate_result.status is LLMCoachingGateStatus.REJECTED:
        rejection_reason = gate_result.rejection_reason
        if rejection_reason is None:
            raise WindowsRAGVLLME2EError("E_ADMISSION")
        phase = GATE_REJECTION_PHASES.get(rejection_reason)
        if phase is None:
            raise WindowsRAGVLLME2EError("E_ADMISSION")
        raise WindowsRAGVLLME2EError(phase)
    if (
        gate_result.status is not LLMCoachingGateStatus.VALID_SUGGESTION
        or gate_result.suggestion is None
    ):
        raise WindowsRAGVLLME2EError("E_ADMISSION")
    factory = DeterministicLLMCoachingSuggestionFactory(
        title=policy.title,
        action=policy.action,
        priority=policy.priority,
        label_id=policy.label_id,
        expires_after_seconds=policy.expires_after_seconds,
        suggestion_id_factory=lambda: "suggestion_windows_rag_vllm_e2e_1",
        utc_datetime_factory=lambda: FIXED_TIME,
    )
    suggestion = factory.create(
        event=event,
        orchestration_result=result,
        current_seconds=event.end_seconds,
    )
    if (
        suggestion is None
        or suggestion.tenant_id != TENANT_ID
        or suggestion.call_id != CALL_ID
        or suggestion.label_id != CATEGORY
        or suggestion.source is not CoachingSuggestionSource.LLM
        or not suggestion.suggestion.strip()
    ):
        raise WindowsRAGVLLME2EError("E_ADMISSION_SUGGESTION")


def _cleanup_synthetic_scope(
    settings: PostgreSQLVectorStoreSettings,
    connect: ConnectionFactory,
) -> None:
    connection = connect(
        conninfo=settings.dsn.get_secret_value(),
        connect_timeout=settings.connect_timeout_seconds,
        sslmode=settings.ssl_mode,
        application_name=settings.application_name,
        autocommit=False,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM callmetric_vector.vector_records "
                "WHERE tenant_id = %s AND knowledge_base_id = %s",
                (TENANT_ID, KNOWLEDGE_BASE_ID),
            )
            cursor.execute(
                "DELETE FROM callmetric_vector.embedding_profiles "
                "WHERE tenant_id = %s AND knowledge_base_id = %s",
                (TENANT_ID, KNOWLEDGE_BASE_ID),
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _check_deadline(deadline: float, clock: Callable[[], float]) -> None:
    if clock() >= deadline:
        raise WindowsRAGVLLME2EError("E_DEADLINE")


def _database_phase(code: str, operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except PsycopgError:
        raise WindowsRAGVLLME2EError(code) from None


def _require_exact_ingestion(result: object) -> None:
    if not isinstance(result, VectorBatchWriteResult):
        raise WindowsRAGVLLME2EError("E_INGESTION")
    identities = tuple(
        (identity.document_id, identity.chunk_id)
        for identity in (*result.inserted_identities, *result.unchanged_identities)
    )
    if (
        result.tenant_id != TENANT_ID
        or result.knowledge_base_id != KNOWLEDGE_BASE_ID
        or identities != ((DOCUMENT_ID, CHUNK_ID),)
    ):
        raise WindowsRAGVLLME2EError("E_INGESTION")


def run(
    *,
    preflight_only: bool,
    environment: Mapping[str, str] | None = None,
    connect: ConnectionFactory = psycopg_connect,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Validate or run the fixed synthetic flow without owning external lifecycle."""
    artifacts = _load_artifacts()
    settings = _load_settings(os.environ if environment is None else environment)
    _classification_context(artifacts.policy)
    event = _transcript_event()
    if preflight_only:
        return "PREFLIGHT_OK"

    deadline = clock() + TOTAL_E2E_TIMEOUT_SECONDS
    primary: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        _database_phase(
            "E_INITIAL_CLEANUP",
            lambda: _cleanup_synthetic_scope(settings.postgres, connect),
        )
        _check_deadline(deadline, clock)
        profile = _database_phase(
            "E_PROVISIONING",
            lambda: provision_profile_bound_postgres_rag(
                postgres_settings=settings.postgres,
                knowledge_base_settings=artifacts.provider,
                psycopg_connect=connect,
            ),
        )
        if (
            profile.tenant_id != TENANT_ID
            or profile.knowledge_base_id != KNOWLEDGE_BASE_ID
            or profile.model_id != MODEL_ID
            or profile.vector_dimension != VECTOR_DIMENSION
        ):
            raise WindowsRAGVLLME2EError("E_PROFILE")
        _check_deadline(deadline, clock)
        ingestion = _database_phase(
            "E_INGESTION",
            lambda: ingest_profile_bound_postgres_rag(
                postgres_settings=settings.postgres,
                knowledge_base_settings=artifacts.provider,
                request=artifacts.document,
                psycopg_connect=connect,
            ),
        )
        _require_exact_ingestion(ingestion)
        _check_deadline(deadline, clock)
        result = _database_phase(
            "E_RETRIEVAL_UNAVAILABLE",
            lambda: orchestrate_profile_bound_postgres_rag(
                postgres_settings=settings.postgres,
                knowledge_base_settings=artifacts.provider,
                vllm_settings=settings.vllm,
                request=_orchestration_request(event),
                limits=PostgreSQLRAGOrchestrationLimits(
                    max_top_k=1,
                    max_user_input_characters=512,
                    max_prompt_characters=8_192,
                ),
                psycopg_connect=connect,
                structured_output_json_schema=coaching_wire_json_schema(),
            ),
        )
        if result is None:
            raise WindowsRAGVLLME2EError("E_RETRIEVAL_UNAVAILABLE")
        _check_deadline(deadline, clock)
        _admit_result(policy=artifacts.policy, event=event, result=result)
    except BaseException as error:
        primary = error
    finally:
        try:
            _cleanup_synthetic_scope(settings.postgres, connect)
        except BaseException as error:
            cleanup_error = error

    if primary is not None:
        if isinstance(primary, WindowsRAGVLLME2EError):
            raise primary
        if isinstance(primary, httpx.HTTPError):
            raise WindowsRAGVLLME2EError("E_VLLM_UNAVAILABLE") from None
        raise WindowsRAGVLLME2EError("E_E2E") from None
    if cleanup_error is not None:
        raise WindowsRAGVLLME2EError("E_CLEANUP") from None
    return "E2E_OK"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or run the fixed synthetic Windows RAG/vLLM E2E"
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        print(run(preflight_only=arguments.preflight_only))
    except WindowsRAGVLLME2EError as error:
        print(str(error), file=sys.stderr)
        return 1
    except BaseException:
        print("E_E2E", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
