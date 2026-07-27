"""Deterministic provider-neutral prompt builder."""

from app.prompting.models import PromptBuildRequest, PromptBuildResult

_SYSTEM_PROMPT = (
    "You are a call-center coaching assistant. "
    "Answer the user input using the retrieved context when available. "
    "Do not invent information that is absent from the context."
)


class DeterministicPromptBuilder:
    def build(self, request: PromptBuildRequest) -> PromptBuildResult:
        context = sorted(
            request.retrieved_context,
            key=lambda item: (-item.score, item.document_id, item.chunk_id),
        )
        context_lines = (
            [
                (
                    f"{index}. document_id={item.document_id}; "
                    f"chunk_id={item.chunk_id}; score={item.score:.6f}\n"
                    f"{item.text}"
                )
                for index, item in enumerate(context, start=1)
            ]
            if context
            else ["No retrieved context."]
        )
        user_prompt = (
            f"User input:\n{request.user_input}\n\n"
            f"Retrieved context:\n{'\n\n'.join(context_lines)}"
        )
        return PromptBuildResult(
            tenant_id=request.tenant_id,
            call_id=request.call_id,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
