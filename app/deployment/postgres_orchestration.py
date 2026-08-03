"""Explicit readiness-verified PostgreSQL RAG orchestration operation."""

from __future__ import annotations

from collections.abc import Mapping
from threading import Lock
from typing import Any

import httpx
from psycopg import Connection
from pydantic import BaseModel, ConfigDict, field_validator

from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
    PsycopgConnect,
)
from app.composition.postgres_rag_orchestration import (
    compose_profile_bound_postgres_rag_orchestration,
)
from app.embeddings.sentence_transformers import BackendFactory
from app.llm.models import LLMRequest, LLMResponse
from app.llm.protocols import LLMGateway
from app.llm.vllm_openai_compatible import (
    VLLMOpenAICompatibleGateway,
    VLLMOpenAICompatibleSettings,
)
from app.orchestration.models import OrchestrationRequest, OrchestrationResult
from app.vector_store.embedding_profile import KnowledgeBaseEmbeddingProfile
from app.vector_store.postgres.readiness import PostgreSQLSchemaReadinessChecker

_PROMPT_TOO_LONG = "assembled prompt exceeds the configured character limit"


class PostgreSQLRAGOrchestrationLimits(BaseModel):
    """Explicit immutable operational bounds for one orchestration call."""

    model_config = ConfigDict(frozen=True)

    max_top_k: int
    max_user_input_characters: int
    max_prompt_characters: int

    @field_validator(
        "max_top_k",
        "max_user_input_characters",
        "max_prompt_characters",
        mode="before",
    )
    @classmethod
    def validate_positive_integer(cls, value: object, info: object) -> int:
        if type(value) is not int:
            raise ValueError(
                f"{getattr(info, 'field_name', 'value')} must be an integer"
            )
        if value <= 0:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be positive")
        return value


class _PromptBoundVLLMGateway:
    def __init__(
        self,
        *,
        settings: VLLMOpenAICompatibleSettings,
        max_prompt_characters: int,
        transport: httpx.BaseTransport | None,
        structured_output_json_schema: Mapping[str, object] | None = None,
    ) -> None:
        self._settings = settings
        self._max_prompt_characters = max_prompt_characters
        self._transport = transport
        self._structured_output_json_schema = structured_output_json_schema
        self._gateway: LLMGateway | None = None
        self._lock = Lock()

    def generate(self, request: LLMRequest) -> LLMResponse:
        if len(request.input_text) > self._max_prompt_characters:
            raise ValueError(_PROMPT_TOO_LONG)
        gateway = self._gateway
        if gateway is None:
            with self._lock:
                gateway = self._gateway
                if gateway is None:
                    gateway = VLLMOpenAICompatibleGateway(
                        self._settings,
                        transport=self._transport,
                        structured_output_json_schema=(
                            self._structured_output_json_schema
                        ),
                    )
                    self._gateway = gateway
        return gateway.generate(request)


def orchestrate_profile_bound_postgres_rag(
    *,
    postgres_settings: PostgreSQLVectorStoreSettings,
    knowledge_base_settings: KnowledgeBaseRAGProviderSettings,
    vllm_settings: VLLMOpenAICompatibleSettings,
    request: OrchestrationRequest,
    limits: PostgreSQLRAGOrchestrationLimits,
    psycopg_connect: PsycopgConnect,
    embedding_backend_factory: BackendFactory | None = None,
    vllm_transport: httpx.BaseTransport | None = None,
    structured_output_json_schema: Mapping[str, object] | None = None,
) -> OrchestrationResult | None:
    """Verify readiness/profile identity, then run RAG orchestration once."""
    if not isinstance(postgres_settings, PostgreSQLVectorStoreSettings):
        raise ValueError("postgres_settings must be PostgreSQLVectorStoreSettings")
    if not isinstance(knowledge_base_settings, KnowledgeBaseRAGProviderSettings):
        raise ValueError(
            "knowledge_base_settings must be KnowledgeBaseRAGProviderSettings"
        )
    if not isinstance(vllm_settings, VLLMOpenAICompatibleSettings):
        raise ValueError("vllm_settings must be VLLMOpenAICompatibleSettings")
    if not isinstance(request, OrchestrationRequest):
        raise ValueError("request must be OrchestrationRequest")
    if not isinstance(limits, PostgreSQLRAGOrchestrationLimits):
        raise ValueError("limits must be PostgreSQLRAGOrchestrationLimits")
    if not callable(psycopg_connect):
        raise ValueError("psycopg_connect must be callable")
    if embedding_backend_factory is not None and not callable(
        embedding_backend_factory
    ):
        raise ValueError("embedding_backend_factory must be callable")
    if vllm_transport is not None and not isinstance(
        vllm_transport, httpx.BaseTransport
    ):
        raise ValueError("vllm_transport must be an httpx BaseTransport")
    if structured_output_json_schema is not None and not isinstance(
        structured_output_json_schema, Mapping
    ):
        raise ValueError("structured_output_json_schema must be a mapping")
    if request.tenant_id != knowledge_base_settings.tenant_id:
        raise ValueError("request tenant_id does not match provider scope")
    if request.knowledge_base_id != knowledge_base_settings.knowledge_base_id:
        raise ValueError("request knowledge_base_id does not match provider scope")
    if request.top_k > limits.max_top_k:
        raise ValueError("request top_k exceeds the configured limit")
    if len(request.user_input) > limits.max_user_input_characters:
        raise ValueError("request user_input exceeds the configured character limit")

    def llm_gateway_factory() -> LLMGateway:
        return _PromptBoundVLLMGateway(
            settings=vllm_settings,
            max_prompt_characters=limits.max_prompt_characters,
            transport=vllm_transport,
            structured_output_json_schema=structured_output_json_schema,
        )

    composition = compose_profile_bound_postgres_rag_orchestration(
        postgres_settings=postgres_settings,
        knowledge_base_settings=knowledge_base_settings,
        psycopg_connect=psycopg_connect,
        llm_gateway_factory=llm_gateway_factory,
        embedding_backend_factory=embedding_backend_factory,
    )

    def readiness_connection_factory() -> Connection[Any]:
        return psycopg_connect(
            conninfo=postgres_settings.dsn.get_secret_value(),
            connect_timeout=postgres_settings.connect_timeout_seconds,
            sslmode=postgres_settings.ssl_mode,
            application_name=postgres_settings.application_name,
            autocommit=False,
        )

    PostgreSQLSchemaReadinessChecker(
        connection_factory=readiness_connection_factory
    ).verify()
    registered = composition.postgres_rag.profile_repository.get_profile(
        tenant_id=request.tenant_id,
        knowledge_base_id=request.knowledge_base_id,
    )
    if registered is None:
        raise ValueError("embedding profile is not registered")
    if not isinstance(registered, KnowledgeBaseEmbeddingProfile):
        raise ValueError("profile repository returned an invalid embedding profile")
    if registered != composition.postgres_rag.profile:
        raise ValueError("profile repository returned a conflicting embedding profile")
    return composition.orchestrator.run(request)
