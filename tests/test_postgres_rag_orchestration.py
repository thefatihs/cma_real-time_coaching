"""Tests for the explicit PostgreSQL RAG orchestration operation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError

import app.deployment.postgres_orchestration as deployment
from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
)
from app.llm.vllm_openai_compatible import VLLMOpenAICompatibleSettings
from app.orchestration.models import OrchestrationRequest, OrchestrationResult
from app.vector_store.embedding_profile import (
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
)


def _postgres() -> PostgreSQLVectorStoreSettings:
    return PostgreSQLVectorStoreSettings.model_validate(
        {
            "dsn": "postgresql://synthetic:synthetic@db.invalid/synthetic",
            "connect_timeout_seconds": 7,
            "ssl_mode": "verify-full",
            "application_name": "callmetric-orchestration",
        }
    )


def _provider() -> KnowledgeBaseRAGProviderSettings:
    return KnowledgeBaseRAGProviderSettings(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        model_id="embedding-synthetic",
        model_name_or_path="local/synthetic",
        vector_dimension=3,
        normalize_embeddings=True,
        device="cpu",
        local_files_only=True,
    )


def _vllm() -> VLLMOpenAICompatibleSettings:
    return VLLMOpenAICompatibleSettings(
        base_url="https://vllm.invalid/v1",
        model_id="llm-synthetic",
        api_token=None,
        connect_timeout_seconds=5,
        read_timeout_seconds=20,
        max_output_tokens=20,
        temperature=0,
        verify_tls=True,
    )


def _request(**changes: object) -> OrchestrationRequest:
    values: dict[str, object] = {
        "tenant_id": "tenant-synthetic",
        "call_id": "call-synthetic",
        "transcript_revision": 4,
        "knowledge_base_id": "kb-synthetic",
        "user_input": "Synthetic question",
        "top_k": 2,
        "minimum_score": 0.25,
    }
    values.update(changes)
    return OrchestrationRequest.model_validate(values)


def _limits(**changes: object) -> deployment.PostgreSQLRAGOrchestrationLimits:
    values: dict[str, object] = {
        "max_top_k": 5,
        "max_user_input_characters": 100,
        "max_prompt_characters": 500,
    }
    values.update(changes)
    return deployment.PostgreSQLRAGOrchestrationLimits.model_validate(values)


def _profile() -> KnowledgeBaseEmbeddingProfile:
    return KnowledgeBaseEmbeddingProfile(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        model_id="embedding-synthetic",
        vector_dimension=3,
        normalize_embeddings=True,
        distance_metric=EmbeddingDistanceMetric.COSINE,
    )


@pytest.mark.parametrize(
    "field",
    ["max_top_k", "max_user_input_characters", "max_prompt_characters"],
)
@pytest.mark.parametrize("value", [True, 1.0, "1", 0, -1])
def test_limits_are_required_strict_positive_and_frozen(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        _limits(**{field: value})
    with pytest.raises(ValidationError):
        deployment.PostgreSQLRAGOrchestrationLimits.model_validate(
            {
                key: value
                for key, value in _limits().model_dump().items()
                if key != field
            }
        )
    with pytest.raises(ValidationError):
        _limits().max_top_k = 9


class _Connect:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __call__(self, **_kwargs: object) -> Any:
        self.events.append("connect")
        return object()


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: OrchestrationResult | None = None,
    registered: object | None = None,
) -> tuple[list[str], object, OrchestrationRequest]:
    events: list[str] = []
    request = _request()
    profile = _profile()
    repository = SimpleNamespace()

    def get_profile(**kwargs: object) -> object:
        events.append("profile")
        assert kwargs == {
            "tenant_id": request.tenant_id,
            "knowledge_base_id": request.knowledge_base_id,
        }
        return profile if registered is None else registered

    repository.get_profile = get_profile

    class Orchestrator:
        def run(self, actual: OrchestrationRequest) -> OrchestrationResult | None:
            events.append("run")
            assert actual is request
            return result

    composition = SimpleNamespace(
        postgres_rag=SimpleNamespace(
            profile=profile,
            profile_repository=repository,
        ),
        orchestrator=Orchestrator(),
    )

    def compose(**kwargs: object) -> object:
        events.append("compose")
        assert callable(kwargs["llm_gateway_factory"])
        return composition

    class Checker:
        def __init__(self, **_kwargs: object) -> None:
            events.append("checker")

        def verify(self) -> None:
            events.append("readiness")

    monkeypatch.setattr(
        deployment, "compose_profile_bound_postgres_rag_orchestration", compose
    )
    monkeypatch.setattr(deployment, "PostgreSQLSchemaReadinessChecker", Checker)
    return events, composition, request


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("postgres_settings", object()),
        ("knowledge_base_settings", object()),
        ("vllm_settings", object()),
        ("request", object()),
        ("limits", object()),
        ("psycopg_connect", object()),
        ("embedding_backend_factory", object()),
        ("vllm_transport", object()),
    ],
)
def test_invalid_collaborators_fail_before_composition(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    monkeypatch.setattr(
        deployment,
        "compose_profile_bound_postgres_rag_orchestration",
        lambda **_kwargs: pytest.fail("composition must not run"),
    )
    arguments: dict[str, object] = {
        "postgres_settings": _postgres(),
        "knowledge_base_settings": _provider(),
        "vllm_settings": _vllm(),
        "request": _request(),
        "limits": _limits(),
        "psycopg_connect": lambda **_kwargs: object(),
        "embedding_backend_factory": None,
        "vllm_transport": None,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        deployment.orchestrate_profile_bound_postgres_rag(**cast(Any, arguments))


@pytest.mark.parametrize(
    "orchestration_request",
    [
        _request(tenant_id="tenant-other"),
        _request(knowledge_base_id="kb-other"),
        _request(top_k=6),
        _request(user_input="x" * 101),
    ],
)
def test_scope_and_limits_fail_before_io(
    monkeypatch: pytest.MonkeyPatch,
    orchestration_request: OrchestrationRequest,
) -> None:
    monkeypatch.setattr(
        deployment,
        "compose_profile_bound_postgres_rag_orchestration",
        lambda **_kwargs: pytest.fail("composition must not run"),
    )
    with pytest.raises(ValueError):
        deployment.orchestrate_profile_bound_postgres_rag(
            postgres_settings=_postgres(),
            knowledge_base_settings=_provider(),
            vllm_settings=_vllm(),
            request=orchestration_request,
            limits=_limits(),
            psycopg_connect=cast(Any, lambda **_kwargs: object()),
        )


def test_lifecycle_and_exact_result_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    result = OrchestrationResult.model_validate(
        {
            "tenant_id": "tenant-synthetic",
            "call_id": "call-synthetic",
            "transcript_revision": 4,
            "generated_text": "Synthetic answer",
            "citations": ({"document_id": "document-a", "chunk_id": "chunk-1"},),
        }
    )
    events, _composition, request = _install(monkeypatch, result=result)
    returned = deployment.orchestrate_profile_bound_postgres_rag(
        postgres_settings=_postgres(),
        knowledge_base_settings=_provider(),
        vllm_settings=_vllm(),
        request=request,
        limits=_limits(),
        psycopg_connect=_Connect(events),
    )
    assert returned is result
    assert events == ["compose", "checker", "readiness", "profile", "run"]


@pytest.mark.parametrize(
    "registered",
    [object(), _profile().model_copy(update={"model_id": "other"})],
)
def test_invalid_profile_stops_before_run(
    monkeypatch: pytest.MonkeyPatch, registered: object
) -> None:
    events, _composition, request = _install(monkeypatch, registered=registered)
    with pytest.raises(ValueError):
        deployment.orchestrate_profile_bound_postgres_rag(
            postgres_settings=_postgres(),
            knowledge_base_settings=_provider(),
            vllm_settings=_vllm(),
            request=request,
            limits=_limits(),
            psycopg_connect=_Connect(events),
        )
    assert "run" not in events


def test_prompt_bound_precedes_gateway_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions = 0

    class Gateway:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            nonlocal constructions
            constructions += 1

    monkeypatch.setattr(deployment, "VLLMOpenAICompatibleGateway", Gateway)
    gateway = deployment._PromptBoundVLLMGateway(
        settings=_vllm(),
        max_prompt_characters=3,
        transport=None,
    )
    with pytest.raises(ValueError, match="assembled prompt"):
        gateway.generate(
            deployment.LLMRequest(
                tenant_id="tenant-synthetic",
                call_id="call-synthetic",
                input_text="too long",
            )
        )
    assert constructions == 0


def test_gateway_uses_exact_mock_transport_and_preserves_exception() -> None:
    expected = RuntimeError("synthetic transport failure")

    def handler(_request: httpx.Request) -> httpx.Response:
        raise expected

    gateway = deployment._PromptBoundVLLMGateway(
        settings=_vllm(),
        max_prompt_characters=100,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError) as raised:
        gateway.generate(
            deployment.LLMRequest(
                tenant_id="tenant-synthetic",
                call_id="call-synthetic",
                input_text="Synthetic prompt",
            )
        )
    assert raised.value is expected
