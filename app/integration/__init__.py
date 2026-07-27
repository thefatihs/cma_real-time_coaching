"""Provider-neutral application integration boundaries."""

from app.integration.composition import (
    RAGCoachingIntegrationDependencies,
    compose_rag_coaching_processor,
)
from app.integration.llm_suggestion_factory import (
    DeterministicLLMCoachingSuggestionFactory,
)
from app.integration.policy import RAGCoachingIntegrationPolicy
from app.integration.rag_coaching import (
    CoachingSuggestionFactory,
    OrchestrationRunner,
    RAGCoachingProcessorDecorator,
)

__all__ = [
    "CoachingSuggestionFactory",
    "DeterministicLLMCoachingSuggestionFactory",
    "OrchestrationRunner",
    "RAGCoachingIntegrationDependencies",
    "RAGCoachingIntegrationPolicy",
    "RAGCoachingProcessorDecorator",
    "compose_rag_coaching_processor",
]
