"""Call-scoped, optional PostgreSQL/vLLM RAG dashboard activation."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, cast
from uuid import uuid4

from app.composition.postgres_rag import KnowledgeBaseRAGProviderSettings
from app.integration import RAGCoachingIntegrationDependencies
from app.integration.policy import RAGCoachingIntegrationPolicy
from app.tenancy.models import TenantConfig
from live_dashboard.runtime_wiring import (
    DashboardExecutionResource,
    DashboardExecutionResourceRegistry,
)

_ACTIVATION_ENVIRONMENT_KEYS = (
    "CALLMETRIC_DASHBOARD_RAG_PROVIDER_SETTINGS_PATH",
    "CALLMETRIC_DASHBOARD_RAG_INTEGRATION_POLICY_PATH",
    "CALLMETRIC_DASHBOARD_RAG_MAX_WORKERS",
    "CALLMETRIC_DASHBOARD_RAG_CAPACITY",
)
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
_POLICY_KEYS = frozenset(
    {
        "rag_llm_enabled_labels",
        "title",
        "action",
        "priority",
        "label_id",
        "expires_after_seconds",
    }
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
    }
)
_MAX_CONFIGURATION_BYTES = 65_536
_MAX_WORKERS = 8
_MAX_CAPACITY = 32


class DashboardRAGRuntimeStatus(str, Enum):
    """Fixed, non-sensitive optional RAG activation status."""

    DISABLED = "DISABLED"
    UNAVAILABLE = "UNAVAILABLE"
    READY = "READY"


class DashboardRAGRuntimeController:
    """Construct and retain at most one activation per exact call scope."""

    def __init__(
        self,
        *,
        registry: DashboardExecutionResourceRegistry,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(registry, DashboardExecutionResourceRegistry):
            raise ValueError("registry must be DashboardExecutionResourceRegistry")
        if environment is not None and not isinstance(environment, Mapping):
            raise ValueError("environment must be a mapping")
        self._registry = registry
        self._environment = os.environ if environment is None else environment
        self._statuses: dict[str, DashboardRAGRuntimeStatus] = {}
        self._lock = Lock()

    def activate(
        self,
        *,
        tenant_config: TenantConfig,
        call_id: str,
    ) -> tuple[DashboardRAGRuntimeStatus, DashboardExecutionResource]:
        if not isinstance(tenant_config, TenantConfig):
            raise ValueError("tenant_config must be TenantConfig")
        tenant_id = tenant_config.context.tenant_id
        with self._lock:
            existing = self._registry.find(
                tenant_id=tenant_id,
                call_id=call_id,
            )
            if existing is not None:
                return self._statuses[existing.opaque_key], existing

            status, integration = _activate_optional_integration(
                tenant_config=tenant_config,
                environment=self._environment,
            )
            try:
                resource = self._registry.acquire(
                    tenant_id=tenant_id,
                    call_id=call_id,
                    integration=integration,
                )
            except BaseException:
                if integration is not None:
                    integration.background_manager.close(wait=False)
                raise
            self._statuses[resource.opaque_key] = status
            return status, resource

    def close_and_remove(self, opaque_key: str) -> None:
        with self._lock:
            self._registry.close_and_remove(opaque_key)
            self._statuses.pop(opaque_key, None)


def _activate_optional_integration(
    *,
    tenant_config: TenantConfig,
    environment: Mapping[str, str],
) -> tuple[DashboardRAGRuntimeStatus, RAGCoachingIntegrationDependencies | None]:
    if not tenant_config.rag.enabled or not tenant_config.coaching.enable_llm:
        return DashboardRAGRuntimeStatus.DISABLED, None
    configured = tuple(key in environment for key in _ACTIVATION_ENVIRONMENT_KEYS)
    if not any(configured):
        return DashboardRAGRuntimeStatus.DISABLED, None
    if not all(configured):
        return DashboardRAGRuntimeStatus.UNAVAILABLE, None

    manager = None
    try:
        provider_payload = _load_exact_json(
            environment[_ACTIVATION_ENVIRONMENT_KEYS[0]],
            expected_keys=_PROVIDER_KEYS,
        )
        policy_payload = _load_exact_json(
            environment[_ACTIVATION_ENVIRONMENT_KEYS[1]],
            expected_keys=_POLICY_KEYS,
        )
        max_workers = _strict_bounded_integer(
            environment[_ACTIVATION_ENVIRONMENT_KEYS[2]],
            field_name="max_workers",
            maximum=_MAX_WORKERS,
        )
        capacity = _strict_bounded_integer(
            environment[_ACTIVATION_ENVIRONMENT_KEYS[3]],
            field_name="capacity",
            maximum=_MAX_CAPACITY,
        )
        if capacity < max_workers:
            raise ValueError("capacity must be greater than or equal to max_workers")

        from app.composition.postgres_rag import (
            KnowledgeBaseRAGProviderSettings,
            PostgreSQLVectorStoreSettings,
        )

        provider_settings = KnowledgeBaseRAGProviderSettings.model_validate(
            provider_payload
        )
        policy = RAGCoachingIntegrationPolicy.model_validate(policy_payload)
        _validate_local_scope_and_policy(
            tenant_config=tenant_config,
            provider_settings=provider_settings,
            policy=policy,
        )
        postgres_settings_factory = cast(
            Callable[[], PostgreSQLVectorStoreSettings],
            PostgreSQLVectorStoreSettings,
        )
        postgres_settings = postgres_settings_factory()

        from app.composition.postgres_rag_background import (
            BoundedPostgreSQLRAGManager,
        )
        from app.composition.postgres_rag_orchestration import (
            compose_profile_bound_postgres_rag_orchestration,
        )
        from app.composition.postgres_rag_runtime import (
            ProfileVerifiedPostgreSQLRAGRunner,
        )
        from app.llm.vllm_openai_compatible import (
            VLLMOpenAICompatibleGateway,
            VLLMOpenAICompatibleSettings,
        )
        from app.vector_store.postgres.readiness import (
            PostgreSQLSchemaReadinessChecker,
        )
        from psycopg import connect as psycopg_connect

        vllm_settings_factory = cast(
            Callable[[], VLLMOpenAICompatibleSettings],
            VLLMOpenAICompatibleSettings,
        )
        vllm_settings = vllm_settings_factory()

        def llm_gateway_factory() -> VLLMOpenAICompatibleGateway:
            return VLLMOpenAICompatibleGateway(vllm_settings)

        composition = compose_profile_bound_postgres_rag_orchestration(
            postgres_settings=postgres_settings,
            knowledge_base_settings=provider_settings,
            psycopg_connect=psycopg_connect,
            llm_gateway_factory=llm_gateway_factory,
        )

        def readiness_connection_factory() -> Any:
            return psycopg_connect(
                conninfo=postgres_settings.dsn.get_secret_value(),
                connect_timeout=postgres_settings.connect_timeout_seconds,
                sslmode=postgres_settings.ssl_mode,
                application_name=postgres_settings.application_name,
                autocommit=False,
            )

        readiness_checker = PostgreSQLSchemaReadinessChecker(
            connection_factory=readiness_connection_factory
        )
        runner = ProfileVerifiedPostgreSQLRAGRunner(
            composition,
            readiness_checker,
        )
        runner.prepare()
        manager = BoundedPostgreSQLRAGManager(
            runner=runner,
            max_workers=max_workers,
            capacity=capacity,
        )
        manager.start()
        integration = RAGCoachingIntegrationDependencies(
            background_manager=manager,
            policy=policy,
            suggestion_id_factory=lambda: uuid4().hex,
            utc_datetime_factory=lambda: datetime.now(UTC),
        )
    except Exception:
        if manager is not None:
            manager.close(wait=False)
        return DashboardRAGRuntimeStatus.UNAVAILABLE, None
    return DashboardRAGRuntimeStatus.READY, integration


def validated_dashboard_rag_provider_settings(
    *,
    tenant_config: TenantConfig,
    environment: Mapping[str, str],
) -> KnowledgeBaseRAGProviderSettings:
    """Load provider settings and enforce the dashboard's trusted tenant/KB scope."""
    if not isinstance(tenant_config, TenantConfig):
        raise ValueError("tenant_config must be TenantConfig")
    raw_path = environment.get(_ACTIVATION_ENVIRONMENT_KEYS[0])
    if raw_path is None:
        raise ValueError("dashboard RAG provider settings are missing")
    provider_settings = KnowledgeBaseRAGProviderSettings.model_validate(
        _load_exact_json(raw_path, expected_keys=_PROVIDER_KEYS)
    )
    if provider_settings.tenant_id != tenant_config.context.tenant_id:
        raise ValueError("provider tenant scope does not match dashboard scope")
    if provider_settings.knowledge_base_id != tenant_config.rag.knowledge_base_id:
        raise ValueError("provider knowledge-base scope does not match dashboard scope")
    return provider_settings


def _load_exact_json(
    raw_path: str,
    *,
    expected_keys: frozenset[str],
) -> dict[str, object]:
    path = _validated_configuration_path(raw_path)
    content_bytes = path.read_bytes()
    if (
        not content_bytes
        or len(content_bytes) > _MAX_CONFIGURATION_BYTES
        or content_bytes.startswith(b"\xef\xbb\xbf")
        or b"\0" in content_bytes
    ):
        raise ValueError("dashboard RAG configuration is invalid")
    content = content_bytes.decode("utf-8", errors="strict")
    payload = json.loads(content, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("dashboard RAG configuration is invalid")
    actual_keys = set(payload)
    if actual_keys != expected_keys or actual_keys & _SECRET_KEYS:
        raise ValueError("dashboard RAG configuration is invalid")
    return payload


def _validated_configuration_path(raw_path: str) -> Path:
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or raw_path != raw_path.strip()
        or "\0" in raw_path
    ):
        raise ValueError("dashboard RAG configuration path is invalid")
    path = Path(raw_path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("dashboard RAG configuration path is invalid")
    if path.is_symlink() or not path.is_file():
        raise ValueError("dashboard RAG configuration path is invalid")
    return path


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("dashboard RAG configuration is invalid")
        payload[key] = value
    return payload


def _strict_bounded_integer(
    value: object,
    *,
    field_name: str,
    maximum: int,
) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError(f"{field_name} must be an integer")
    numeric = int(value)
    if not 1 <= numeric <= maximum:
        raise ValueError(f"{field_name} is outside the allowed range")
    return numeric


def _validate_local_scope_and_policy(
    *,
    tenant_config: TenantConfig,
    provider_settings: Any,
    policy: RAGCoachingIntegrationPolicy,
) -> None:
    if provider_settings.tenant_id != tenant_config.context.tenant_id:
        raise ValueError("provider tenant scope does not match dashboard scope")
    if provider_settings.knowledge_base_id != tenant_config.rag.knowledge_base_id:
        raise ValueError("provider knowledge-base scope does not match dashboard scope")
    if policy.action.value not in tenant_config.coaching.allowed_actions:
        raise ValueError("RAG coaching action is not tenant-allowed")
