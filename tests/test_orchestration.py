from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from app.llm import LLMRequest, LLMResponse
from app.orchestration import (
    OrchestrationCitationReference,
    OrchestrationRequest,
    OrchestrationResult,
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
    requests: list[tuple[str, str, str, int, float]] = field(default_factory=list)

    def retrieve(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        query: str,
        top_k: int,
        minimum_score: float = 0.0,
    ) -> RetrievalResult:
        self.calls.append("retriever")
        self.requests.append(
            (tenant_id, knowledge_base_id, query, top_k, minimum_score)
        )
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
    response_tenant_id: str | None = None
    response_call_id: str | None = None

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append("llm_gateway")
        self.requests.append(request)
        return LLMResponse(
            tenant_id=self.response_tenant_id or request.tenant_id,
            call_id=self.response_call_id or request.call_id,
            text="Synthetic generated response.",
        )


def orchestration_request(**changes: object) -> OrchestrationRequest:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "transcript_revision": 7,
        "knowledge_base_id": "kb_support",
        "user_input": "Synthetic user input.",
        "top_k": 3,
    }
    values.update(changes)
    return OrchestrationRequest.model_validate(values)


def document() -> RetrievalDocument:
    return RetrievalDocument(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        document_id="guide",
        chunk_id="chunk_1",
        text="Synthetic retrieved context.",
        score=0.9,
    )


def second_document() -> RetrievalDocument:
    return RetrievalDocument(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        document_id="faq",
        chunk_id="chunk_2",
        text="Second synthetic retrieved context.",
        score=0.8,
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

    assert result is not None
    assert retriever.requests == [
        (
            "tenant_alpha",
            "kb_support",
            "Synthetic user input.",
            3,
            0.0,
        )
    ]
    assert prompt_builder.requests[0].tenant_id == "tenant_alpha"
    assert prompt_builder.requests[0].call_id == "call_001"
    assert prompt_builder.requests[0].transcript_revision == 7
    assert llm_gateway.requests[0].tenant_id == "tenant_alpha"
    assert llm_gateway.requests[0].call_id == "call_001"
    assert result.tenant_id == "tenant_alpha"
    assert result.call_id == "call_001"
    assert result.transcript_revision == 7


def test_retrieval_documents_are_translated_to_prompt_context() -> None:
    orchestrator, _, prompt_builder, _, _ = dependencies((document(),))

    orchestrator.run(orchestration_request())

    context = prompt_builder.requests[0].retrieved_context
    assert len(context) == 1
    assert context[0].document_id == "guide"
    assert context[0].chunk_id == "chunk_1"
    assert context[0].text == "Synthetic retrieved context."
    assert context[0].score == 0.9


def test_non_zero_minimum_score_is_forwarded_exactly() -> None:
    orchestrator, retriever, _, _, _ = dependencies()

    orchestrator.run(orchestration_request(minimum_score=0.65))

    assert retriever.requests[0][-1] == 0.65


def test_zero_minimum_score_is_preserved() -> None:
    orchestrator, retriever, _, _, _ = dependencies()

    orchestrator.run(orchestration_request(minimum_score=0.0))

    assert retriever.requests[0][-1] == 0.0


def test_empty_retrieval_result_short_circuits_generation() -> None:
    orchestrator, retriever, prompt_builder, llm_gateway, calls = dependencies()
    request = orchestration_request(minimum_score=0.65)

    result = orchestrator.run(request)

    assert result is None
    assert calls == ["retriever"]
    assert retriever.requests == [
        (
            "tenant_alpha",
            "kb_support",
            "Synthetic user input.",
            3,
            0.65,
        )
    ]
    assert prompt_builder.requests == []
    assert llm_gateway.requests == []


def test_prompt_output_is_translated_to_llm_request() -> None:
    orchestrator, _, _, llm_gateway, _ = dependencies((document(),))

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


def test_repeated_identical_empty_requests_deterministically_return_none() -> None:
    orchestrator, retriever, prompt_builder, llm_gateway, calls = dependencies()
    request = orchestration_request()

    first = orchestrator.run(request)
    second = orchestrator.run(request)

    assert first is second is None
    assert len(retriever.requests) == 2
    assert calls == ["retriever", "retriever"]
    assert prompt_builder.requests == []
    assert llm_gateway.requests == []


@pytest.mark.parametrize("transcript_revision", [-1, -5])
def test_request_rejects_negative_transcript_revision(
    transcript_revision: int,
) -> None:
    with pytest.raises(ValidationError, match="transcript_revision cannot be negative"):
        orchestration_request(transcript_revision=transcript_revision)


def test_citations_follow_exact_retrieval_order_without_content_fields() -> None:
    orchestrator, _, _, _, _ = dependencies((second_document(), document()))

    result = orchestrator.run(orchestration_request())

    assert result is not None
    assert result.citations == (
        OrchestrationCitationReference(document_id="faq", chunk_id="chunk_2"),
        OrchestrationCitationReference(document_id="guide", chunk_id="chunk_1"),
    )
    assert set(OrchestrationResult.model_fields) == {
        "tenant_id",
        "call_id",
        "transcript_revision",
        "generated_text",
        "citations",
    }
    assert set(OrchestrationCitationReference.model_fields) == {
        "document_id",
        "chunk_id",
    }


@pytest.mark.parametrize(
    ("response_tenant_id", "response_call_id", "message"),
    [
        ("tenant_other", None, "tenant_id"),
        (None, "call_other", "call_id"),
    ],
)
def test_mismatched_llm_scope_fails_closed(
    response_tenant_id: str | None,
    response_call_id: str | None,
    message: str,
) -> None:
    orchestrator, _, _, llm_gateway, _ = dependencies((document(),))
    llm_gateway.response_tenant_id = response_tenant_id
    llm_gateway.response_call_id = response_call_id

    with pytest.raises(ValueError, match=message):
        orchestrator.run(orchestration_request())


def test_duplicate_retrieval_citation_identity_fails_before_generation() -> None:
    duplicate = document().model_copy(
        update={"text": "Different synthetic context.", "score": 0.7}
    )
    orchestrator, _, prompt_builder, llm_gateway, calls = dependencies(
        (document(), duplicate)
    )

    with pytest.raises(ValueError, match="duplicate citation identities"):
        orchestrator.run(orchestration_request())

    assert calls == ["retriever"]
    assert prompt_builder.requests == []
    assert llm_gateway.requests == []


def test_result_rejects_duplicate_citation_identity() -> None:
    citation = OrchestrationCitationReference(
        document_id="guide",
        chunk_id="chunk_1",
    )

    with pytest.raises(ValidationError, match="citation identities must be unique"):
        OrchestrationResult(
            tenant_id="tenant_alpha",
            call_id="call_001",
            transcript_revision=7,
            generated_text="Synthetic generated response.",
            citations=(citation, citation),
        )
