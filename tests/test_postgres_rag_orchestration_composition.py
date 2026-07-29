"""Tests for side-effect-free PostgreSQL RAG orchestration composition."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, cast

import pytest
from psycopg import Connection
from pydantic import SecretStr

import app.composition as composition_exports
import app.composition.postgres_rag_orchestration as composition
from app.composition import (
    LLMGatewayFactory,
    PostgreSQLRAGOrchestrationComposition,
    compose_profile_bound_postgres_rag_orchestration,
)
from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLRAGComposition,
    PostgreSQLVectorStoreSettings,
)
from app.embeddings.sentence_transformers import BackendFactory
from app.llm.models import LLMRequest, LLMResponse
from app.orchestration.models import OrchestrationRequest
from app.prompting.builder import DeterministicPromptBuilder
from app.retrieval.models import RetrievalDocument, RetrievalResult

_DSN = "postgresql://synthetic_user:synthetic_password@db.invalid/synthetic"


def _postgres_settings() -> PostgreSQLVectorStoreSettings:
    return PostgreSQLVectorStoreSettings(
        dsn=SecretStr(_DSN),
        connect_timeout_seconds=7,
        ssl_mode="verify-full",
        application_name="callmetric-orchestration",
    )


def _knowledge_base_settings() -> KnowledgeBaseRAGProviderSettings:
    return KnowledgeBaseRAGProviderSettings(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        model_id="model-synthetic",
        model_name_or_path="local/synthetic-model",
        vector_dimension=3,
        normalize_embeddings=True,
        device="cpu",
        local_files_only=True,
    )


def _orchestration_request() -> OrchestrationRequest:
    return OrchestrationRequest(
        tenant_id="tenant-synthetic",
        call_id="call-synthetic",
        transcript_revision=4,
        knowledge_base_id="kb-synthetic",
        user_input="Synthetic current question.",
        top_k=2,
        minimum_score=0.35,
    )


def _documents() -> tuple[RetrievalDocument, ...]:
    return (
        RetrievalDocument(
            tenant_id="tenant-synthetic",
            knowledge_base_id="kb-synthetic",
            document_id="document-b",
            chunk_id="chunk-2",
            text="Synthetic context B.",
            score=0.9,
        ),
        RetrievalDocument(
            tenant_id="tenant-synthetic",
            knowledge_base_id="kb-synthetic",
            document_id="document-a",
            chunk_id="chunk-1",
            text="Synthetic context A.",
            score=0.8,
        ),
    )


class FakeRetriever:
    def __init__(
        self,
        documents: tuple[RetrievalDocument, ...] = (),
    ) -> None:
        self.documents = documents
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        query: str,
        top_k: int,
        minimum_score: float = 0.0,
    ) -> RetrievalResult:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "knowledge_base_id": knowledge_base_id,
                "query": query,
                "top_k": top_k,
                "minimum_score": minimum_score,
            }
        )
        return RetrievalResult(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            documents=self.documents,
        )


class FakeGateway:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.calls: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return LLMResponse(
            tenant_id=request.tenant_id,
            call_id=request.call_id,
            text="Synthetic generated response.",
        )


class FalseyGateway(FakeGateway):
    def __bool__(self) -> bool:
        return False


class GatewayFactory:
    def __init__(
        self,
        gateway: FakeGateway,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.gateway = gateway
        self.error = error
        self.calls = 0

    def __call__(self) -> FakeGateway:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.gateway


class FalseyGatewayFactory(GatewayFactory):
    def __bool__(self) -> bool:
        return False


class DeferredCallable:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **_kwargs: object) -> Connection[Any]:
        self.calls += 1
        raise AssertionError("connection must remain deferred")


class DeferredEmbeddingFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _config: object) -> object:
        self.calls += 1
        raise AssertionError("embedding backend must remain deferred")


def _fake_postgres_rag(retriever: FakeRetriever) -> PostgreSQLRAGComposition:
    return cast(
        PostgreSQLRAGComposition,
        SimpleNamespace(
            retriever=retriever,
            profile=object(),
            profile_repository=object(),
            vector_store=object(),
            embedder=object(),
            ingestion_service=object(),
        ),
    )


def _install_postgres_composition(
    monkeypatch: pytest.MonkeyPatch,
    *,
    retriever: FakeRetriever,
) -> tuple[PostgreSQLRAGComposition, list[dict[str, object]]]:
    postgres_rag = _fake_postgres_rag(retriever)
    calls: list[dict[str, object]] = []

    def compose(**kwargs: object) -> PostgreSQLRAGComposition:
        calls.append(kwargs)
        return postgres_rag

    monkeypatch.setattr(
        composition,
        "compose_profile_bound_postgres_rag",
        compose,
    )
    return postgres_rag, calls


def _compose(
    monkeypatch: pytest.MonkeyPatch,
    *,
    retriever: FakeRetriever,
    gateway_factory: LLMGatewayFactory,
    connect: DeferredCallable | None = None,
    embedding_factory: DeferredEmbeddingFactory | None = None,
) -> tuple[
    PostgreSQLRAGOrchestrationComposition,
    PostgreSQLRAGComposition,
    list[dict[str, object]],
]:
    postgres_rag, calls = _install_postgres_composition(
        monkeypatch,
        retriever=retriever,
    )
    selected_connect = connect if connect is not None else DeferredCallable()
    result = compose_profile_bound_postgres_rag_orchestration(
        postgres_settings=_postgres_settings(),
        knowledge_base_settings=_knowledge_base_settings(),
        psycopg_connect=selected_connect,
        llm_gateway_factory=gateway_factory,
        embedding_backend_factory=cast(BackendFactory, embedding_factory),
    )
    return result, postgres_rag, calls


def _accept_factory(factory: LLMGatewayFactory) -> LLMGatewayFactory:
    return factory


def test_factory_is_structurally_compatible_and_public() -> None:
    factory = GatewayFactory(FakeGateway())

    assert _accept_factory(factory) is factory
    assert composition_exports.LLMGatewayFactory is LLMGatewayFactory


def test_composition_is_frozen_slotted_and_exports_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _postgres_rag, _calls = _compose(
        monkeypatch,
        retriever=FakeRetriever(),
        gateway_factory=GatewayFactory(FakeGateway()),
    )

    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(result, "prompt_builder", DeterministicPromptBuilder())
    assert composition_exports.__all__ == [
        "BoundedPostgreSQLRAGManager",
        "KnowledgeBaseRAGProviderSettings",
        "LLMGatewayFactory",
        "ProfileVerifiedPostgreSQLRAGRunner",
        "PostgreSQLRAGOrchestrationComposition",
        "PostgreSQLRAGComposition",
        "PostgreSQLVectorStoreSettings",
        "RAGOrchestrationCompletion",
        "RAGOrchestrationCompletionStatus",
        "RAGOrchestrationIdentity",
        "RAGOrchestrationSubmission",
        "RAGOrchestrationSubmissionStatus",
        "compose_profile_bound_postgres_rag",
        "compose_profile_bound_postgres_rag_orchestration",
    ]
    assert not hasattr(composition_exports, "_DeferredLLMGateway")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("postgres_settings", object()),
        ("knowledge_base_settings", object()),
        ("psycopg_connect", object()),
        ("llm_gateway_factory", object()),
        ("embedding_backend_factory", object()),
    ],
)
def test_constructor_validation_precedes_postgres_composition(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    compose_calls = 0

    def compose(**_kwargs: object) -> PostgreSQLRAGComposition:
        nonlocal compose_calls
        compose_calls += 1
        raise AssertionError("PostgreSQL composition must not run")

    monkeypatch.setattr(
        composition,
        "compose_profile_bound_postgres_rag",
        compose,
    )
    arguments: dict[str, object] = {
        "postgres_settings": _postgres_settings(),
        "knowledge_base_settings": _knowledge_base_settings(),
        "psycopg_connect": DeferredCallable(),
        "llm_gateway_factory": GatewayFactory(FakeGateway()),
        "embedding_backend_factory": None,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        compose_profile_bound_postgres_rag_orchestration(**cast(Any, arguments))

    assert compose_calls == 0


def test_construction_has_zero_side_effects_and_reuses_composition_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect = DeferredCallable()
    embedding_factory = DeferredEmbeddingFactory()
    gateway_factory = FalseyGatewayFactory(FalseyGateway())
    retriever = FakeRetriever()

    result, postgres_rag, calls = _compose(
        monkeypatch,
        retriever=retriever,
        gateway_factory=gateway_factory,
        connect=connect,
        embedding_factory=embedding_factory,
    )

    assert len(calls) == 1
    assert calls[0] == {
        "postgres_settings": _postgres_settings(),
        "knowledge_base_settings": _knowledge_base_settings(),
        "psycopg_connect": connect,
        "embedding_backend_factory": embedding_factory,
    }
    assert result.postgres_rag is postgres_rag
    assert connect.calls == 0
    assert embedding_factory.calls == 0
    assert gateway_factory.calls == 0


def test_exact_orchestrator_collaborator_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = FakeRetriever()
    result, postgres_rag, _calls = _compose(
        monkeypatch,
        retriever=retriever,
        gateway_factory=GatewayFactory(FakeGateway()),
    )

    assert isinstance(result.prompt_builder, DeterministicPromptBuilder)
    assert result.orchestrator._retriever is postgres_rag.retriever  # noqa: SLF001
    assert (  # noqa: SLF001
        result.orchestrator._prompt_builder is result.prompt_builder
    )
    assert result.orchestrator._llm_gateway is result.llm_gateway  # noqa: SLF001


def test_empty_retrieval_short_circuits_without_gateway_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = FakeRetriever()
    factory = GatewayFactory(FakeGateway())
    result, _postgres_rag, _calls = _compose(
        monkeypatch,
        retriever=retriever,
        gateway_factory=factory,
    )

    orchestration_result = result.orchestrator.run(_orchestration_request())

    assert orchestration_result is None
    assert factory.calls == 0
    assert retriever.calls == [
        {
            "tenant_id": "tenant-synthetic",
            "knowledge_base_id": "kb-synthetic",
            "query": "Synthetic current question.",
            "top_k": 2,
            "minimum_score": 0.35,
        }
    ]


def test_first_generation_creates_once_and_repeated_calls_reuse_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = FakeRetriever(_documents())
    gateway = FalseyGateway()
    factory = FalseyGatewayFactory(gateway)
    result, _postgres_rag, _calls = _compose(
        monkeypatch,
        retriever=retriever,
        gateway_factory=factory,
    )
    request = _orchestration_request()

    first = result.orchestrator.run(request)
    second = result.orchestrator.run(request)

    assert first == second
    assert factory.calls == 1
    assert len(gateway.calls) == 2
    assert gateway.calls[0].tenant_id == "tenant-synthetic"
    assert gateway.calls[0].call_id == "call-synthetic"
    assert "Synthetic current question." in gateway.calls[0].input_text
    assert first is not None
    assert tuple(
        (citation.document_id, citation.chunk_id) for citation in first.citations
    ) == (
        ("document-b", "chunk-2"),
        ("document-a", "chunk-1"),
    )
    assert first.tenant_id == "tenant-synthetic"
    assert first.call_id == "call-synthetic"
    assert first.transcript_revision == 4


def test_concurrent_first_generation_creates_gateway_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = FakeGateway()
    factory = GatewayFactory(gateway)
    result, _postgres_rag, _calls = _compose(
        monkeypatch,
        retriever=FakeRetriever(_documents()),
        gateway_factory=factory,
    )
    request = _orchestration_request()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(result.orchestrator.run, (request,) * 24))

    assert factory.calls == 1
    assert len(gateway.calls) == 24
    assert all(item == results[0] for item in results)


def test_factory_exception_identity_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = RuntimeError("synthetic factory failure")
    factory = GatewayFactory(FakeGateway(), error=expected)
    result, _postgres_rag, _calls = _compose(
        monkeypatch,
        retriever=FakeRetriever(_documents()),
        gateway_factory=factory,
    )

    with pytest.raises(RuntimeError) as raised:
        result.orchestrator.run(_orchestration_request())

    assert raised.value is expected
    assert factory.calls == 1


def test_provider_exception_identity_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = RuntimeError("synthetic provider failure")
    gateway = FakeGateway(error=expected)
    result, _postgres_rag, _calls = _compose(
        monkeypatch,
        retriever=FakeRetriever(_documents()),
        gateway_factory=GatewayFactory(gateway),
    )

    with pytest.raises(RuntimeError) as raised:
        result.orchestrator.run(_orchestration_request())

    assert raised.value is expected


def test_invalid_factory_result_fails_closed_on_first_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def invalid_factory() -> Any:
        nonlocal calls
        calls += 1
        return object()

    result, _postgres_rag, _composition_calls = _compose(
        monkeypatch,
        retriever=FakeRetriever(_documents()),
        gateway_factory=invalid_factory,
    )

    with pytest.raises(ValueError, match="must return an LLMGateway"):
        result.orchestrator.run(_orchestration_request())

    assert calls == 1


def test_composition_has_no_readiness_or_mutation_dependencies() -> None:
    assert not hasattr(composition, "PostgreSQLSchemaReadinessChecker")
    assert not hasattr(composition, "register_profile")
    assert not hasattr(composition, "apply_postgres_vector_migrations")
