"""Side-effect-free PostgreSQL RAG orchestration composition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

from app.composition.postgres_rag import (
    BoundedRAGDiagnosticObserver,
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLRAGComposition,
    PostgreSQLVectorStoreSettings,
    PsycopgConnect,
    RAGDiagnosticStage,
    compose_profile_bound_postgres_rag,
    observe_rag_stage,
)
from app.embeddings.sentence_transformers import BackendFactory
from app.llm.models import LLMRequest, LLMResponse
from app.llm.protocols import LLMGateway
from app.orchestration.retrieval import RetrievalOrchestrator
from app.prompting.builder import DeterministicPromptBuilder
from app.prompting.models import PromptBuildRequest, PromptBuildResult
from app.prompting.protocols import PromptBuilder

LLMGatewayFactory = Callable[[], LLMGateway]


@dataclass(frozen=True, slots=True)
class PostgreSQLRAGOrchestrationComposition:
    """Provider-neutral PostgreSQL retrieval and LLM orchestration bundle."""

    postgres_rag: PostgreSQLRAGComposition
    prompt_builder: PromptBuilder
    llm_gateway: LLMGateway
    orchestrator: RetrievalOrchestrator


class _ObservedPromptBuilder:
    def __init__(
        self,
        delegate: DeterministicPromptBuilder,
        observer: BoundedRAGDiagnosticObserver,
    ) -> None:
        self._delegate = delegate
        self._observer = observer

    def build(self, request: PromptBuildRequest) -> PromptBuildResult:
        return observe_rag_stage(
            self._observer,
            RAGDiagnosticStage.PROMPT,
            lambda: self._delegate.build(request),
        )


class _ObservedGateway:
    def __init__(
        self, delegate: LLMGateway, observer: BoundedRAGDiagnosticObserver
    ) -> None:
        self._delegate = delegate
        self._observer = observer

    def generate(self, request: LLMRequest) -> LLMResponse:
        return observe_rag_stage(
            self._observer,
            RAGDiagnosticStage.HTTP,
            lambda: self._delegate.generate(request),
        )


class _DeferredLLMGateway:
    def __init__(
        self,
        factory: LLMGatewayFactory,
        observer: BoundedRAGDiagnosticObserver | None = None,
    ) -> None:
        self._factory = factory
        self._observer = observer
        self._gateway: LLMGateway | None = None
        self._lock = Lock()

    def generate(self, request: LLMRequest) -> LLMResponse:
        gateway = self._gateway
        if gateway is None:
            with self._lock:
                gateway = self._gateway
                if gateway is None:
                    candidate = observe_rag_stage(
                        self._observer,
                        RAGDiagnosticStage.GATEWAY_FACTORY,
                        self._factory,
                    )
                    if not callable(getattr(candidate, "generate", None)):
                        raise ValueError(
                            "llm_gateway_factory must return an LLMGateway"
                        )
                    gateway = (
                        candidate
                        if self._observer is None
                        else _ObservedGateway(candidate, self._observer)
                    )
                    self._gateway = gateway
        return gateway.generate(request)


def compose_profile_bound_postgres_rag_orchestration(
    *,
    postgres_settings: PostgreSQLVectorStoreSettings,
    knowledge_base_settings: KnowledgeBaseRAGProviderSettings,
    psycopg_connect: PsycopgConnect,
    llm_gateway_factory: LLMGatewayFactory,
    embedding_backend_factory: BackendFactory | None = None,
    diagnostic_observer: BoundedRAGDiagnosticObserver | None = None,
) -> PostgreSQLRAGOrchestrationComposition:
    """Compose orchestration without invoking provider collaborators."""
    if not isinstance(postgres_settings, PostgreSQLVectorStoreSettings):
        raise ValueError(
            "postgres_settings must be PostgreSQLVectorStoreSettings",
        )
    if not isinstance(knowledge_base_settings, KnowledgeBaseRAGProviderSettings):
        raise ValueError(
            "knowledge_base_settings must be KnowledgeBaseRAGProviderSettings",
        )
    if not callable(psycopg_connect):
        raise ValueError("psycopg_connect must be callable")
    if not callable(llm_gateway_factory):
        raise ValueError("llm_gateway_factory must be callable")
    if embedding_backend_factory is not None and not callable(
        embedding_backend_factory
    ):
        raise ValueError("embedding_backend_factory must be callable")
    if diagnostic_observer is not None and not isinstance(
        diagnostic_observer, BoundedRAGDiagnosticObserver
    ):
        raise ValueError("diagnostic_observer is invalid")

    if diagnostic_observer is None:
        postgres_rag = compose_profile_bound_postgres_rag(
            postgres_settings=postgres_settings,
            knowledge_base_settings=knowledge_base_settings,
            psycopg_connect=psycopg_connect,
            embedding_backend_factory=embedding_backend_factory,
        )
    else:
        postgres_rag = compose_profile_bound_postgres_rag(
            postgres_settings=postgres_settings,
            knowledge_base_settings=knowledge_base_settings,
            psycopg_connect=psycopg_connect,
            embedding_backend_factory=embedding_backend_factory,
            diagnostic_observer=diagnostic_observer,
        )
    base_prompt_builder = DeterministicPromptBuilder()
    prompt_builder: PromptBuilder = base_prompt_builder
    if diagnostic_observer is not None:
        prompt_builder = _ObservedPromptBuilder(
            base_prompt_builder, diagnostic_observer
        )
    llm_gateway = _DeferredLLMGateway(llm_gateway_factory, diagnostic_observer)
    orchestrator = RetrievalOrchestrator(
        postgres_rag.retriever,
        prompt_builder,
        llm_gateway,
    )
    return PostgreSQLRAGOrchestrationComposition(
        postgres_rag=postgres_rag,
        prompt_builder=prompt_builder,
        llm_gateway=llm_gateway,
        orchestrator=orchestrator,
    )
