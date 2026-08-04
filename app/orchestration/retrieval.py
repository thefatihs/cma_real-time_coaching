"""Synchronous composition of retrieval, prompting, and LLM generation."""

from app.llm.models import LLMRequest
from app.llm.protocols import LLMGateway
from app.orchestration.models import (
    OrchestrationCitationReference,
    OrchestrationRequest,
    OrchestrationResult,
)
from app.prompting.models import PromptBuildRequest, PromptContextItem
from app.prompting.protocols import PromptBuilder
from app.retrieval.protocols import Retriever


class RetrievalOrchestrator:
    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_gateway: LLMGateway,
    ) -> None:
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm_gateway = llm_gateway

    def run(self, request: OrchestrationRequest) -> OrchestrationResult | None:
        retrieval = self._retriever.retrieve(
            tenant_id=request.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
            query=request.user_input,
            top_k=request.top_k,
            minimum_score=request.minimum_score,
        )
        if not retrieval.documents:
            return None
        citations = tuple(
            OrchestrationCitationReference(
                document_id=document.document_id,
                chunk_id=document.chunk_id,
            )
            for document in retrieval.documents
        )
        citation_identities = tuple(
            (citation.document_id, citation.chunk_id) for citation in citations
        )
        if len(citation_identities) != len(set(citation_identities)):
            raise ValueError(
                "retrieval documents contain duplicate citation identities"
            )
        prompt = self._prompt_builder.build(
            PromptBuildRequest(
                tenant_id=request.tenant_id,
                call_id=request.call_id,
                transcript_revision=request.transcript_revision,
                user_input=request.user_input,
                retrieved_context=tuple(
                    PromptContextItem(
                        document_id=document.document_id,
                        chunk_id=document.chunk_id,
                        text=document.text,
                        score=document.score,
                    )
                    for document in retrieval.documents
                ),
            )
        )
        response = self._llm_gateway.generate(
            LLMRequest(
                tenant_id=request.tenant_id,
                call_id=request.call_id,
                input_text=f"{prompt.system_prompt}\n\n{prompt.user_prompt}",
            )
        )
        if response.tenant_id != request.tenant_id:
            raise ValueError("LLM response tenant_id does not match request")
        if response.call_id != request.call_id:
            raise ValueError("LLM response call_id does not match request")
        return OrchestrationResult(
            tenant_id=request.tenant_id,
            call_id=request.call_id,
            transcript_revision=request.transcript_revision,
            generated_text=response.text,
            citations=citations,
        )
