from dataclasses import dataclass, field

from app.llm import LLMRequest, LLMResponse
from app.orchestration import (
    OrchestrationRequest,
    RetrievalOrchestrator,
)
from app.prompting import (
    PromptBuildRequest,
    PromptBuildResult,
)
from app.retrieval import RetrievalDocument, RetrievalResult


@dataclass
class FakeRetriever:
    calls: list[str]
    documents: tuple[RetrievalDocument, ...] = ()
    requests: list[tuple[str, str, int, float]] = field(default_factory=list)

    def retrieve(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        top_k: int,
        minimum_score: float = 0.0,
    ) -> RetrievalResult:
        self.calls.append("retriever")
        self.requests.append((tenant_id, knowledge_base_id, top_k, minimum_score))
        return RetrievalResult(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            documents=self.documents,
        )


@dataclass
class FakePromptBuilder:
    calls: list[str]
    requests: list[PromptBuildRequest] = field(default_factory=list)

    def build(self, request: PromptBuildRequest) -> PromptBuildResult:
        self.calls.append("prompt_builder")
        self.requests.append(request)
        return PromptBuildResult(
            tenant_id=request.tenant_id,
            call_id=request.call_id,
            system_prompt="Synthetic system prompt.",
            user_prompt="Synthetic user prompt.",
        )


@dataclass
class FakeLLMGateway:
    calls: list[str]
    requests: list[LLMRequest] = field(default_factory=list)

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append("llm_gateway")
        self.requests.append(request)
        return LLMResponse(
            tenant_id=request.tenant_id,
            call_id=request.call_id,
            text="Synthetic generated response.",
        )


def orchestration_request() -> OrchestrationRequest:
    return OrchestrationRequest(
        tenant_id="tenant_alpha",
        call_id="call_001",
        knowledge_base_id="kb_support",
        user_input="Synthetic user input.",
        top_k=3,
    )


def document() -> RetrievalDocument:
    return RetrievalDocument(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        document_id="guide",
        chunk_id="chunk_1",
        text="Synthetic retrieved context.",
        score=0.9,
    )


def dependencies(
    documents: tuple[RetrievalDocument, ...] = (),
) -> tuple[
    RetrievalOrchestrator,
    FakeRetriever,
    FakePromptBuilder,
    FakeLLMGateway,
    list[str],
]:
    calls: list[str] = []
    retriever = FakeRetriever(calls, documents)
    prompt_builder = FakePromptBuilder(calls)
    llm_gateway = FakeLLMGateway(calls)
    return (
        RetrievalOrchestrator(retriever, prompt_builder, llm_gateway),
        retriever,
        prompt_builder,
        llm_gateway,
        calls,
    )


def test_injected_dependencies_are_invoked_in_order() -> None:
    orchestrator, _, _, _, calls = dependencies((document(),))

    orchestrator.run(orchestration_request())

    assert calls == ["retriever", "prompt_builder", "llm_gateway"]


def test_scope_and_retrieval_arguments_are_propagated() -> None:
    orchestrator, retriever, prompt_builder, llm_gateway, _ = dependencies(
        (document(),)
    )

    result = orchestrator.run(orchestration_request())

    assert retriever.requests == [("tenant_alpha", "kb_support", 3, 0.0)]
    assert prompt_builder.requests[0].tenant_id == "tenant_alpha"
    assert prompt_builder.requests[0].call_id == "call_001"
    assert llm_gateway.requests[0].tenant_id == "tenant_alpha"
    assert llm_gateway.requests[0].call_id == "call_001"
    assert result.tenant_id == "tenant_alpha"
    assert result.call_id == "call_001"


def test_retrieval_documents_are_translated_to_prompt_context() -> None:
    orchestrator, _, prompt_builder, _, _ = dependencies((document(),))

    orchestrator.run(orchestration_request())

    context = prompt_builder.requests[0].retrieved_context
    assert len(context) == 1
    assert context[0].document_id == "guide"
    assert context[0].chunk_id == "chunk_1"
    assert context[0].text == "Synthetic retrieved context."
    assert context[0].score == 0.9


def test_empty_retrieval_result_is_forwarded_safely() -> None:
    orchestrator, _, prompt_builder, _, calls = dependencies()

    result = orchestrator.run(orchestration_request())

    assert prompt_builder.requests[0].retrieved_context == ()
    assert result.generated_text == "Synthetic generated response."
    assert calls == ["retriever", "prompt_builder", "llm_gateway"]


def test_prompt_output_is_translated_to_llm_request() -> None:
    orchestrator, _, _, llm_gateway, _ = dependencies()

    orchestrator.run(orchestration_request())

    assert llm_gateway.requests[0].input_text == (
        "Synthetic system prompt.\n\nSynthetic user prompt."
    )


def test_repeated_identical_requests_are_deterministic() -> None:
    orchestrator, _, _, _, _ = dependencies((document(),))
    request = orchestration_request()

    first = orchestrator.run(request)
    second = orchestrator.run(request)

    assert first == second
