"""All-or-nothing construction of provider-neutral RAG coaching."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from app.coaching.coordinator import CoachingCoordinator
from app.coaching.safe_processor import SafeCoachingProcessorAdapter
from app.composition.postgres_rag_background import (
    BoundedPostgreSQLRAGManager,
)
from app.integration.llm_suggestion_factory import (
    DeterministicLLMCoachingSuggestionFactory,
)
from app.integration.citation_projection import SafeCoachingCitationProjector
from app.integration.policy import RAGCoachingIntegrationPolicy
from app.integration.rag_coaching import (
    RAGCoachingProcessorDecorator,
)
from app.tenancy.models import TenantConfig


@dataclass(frozen=True, slots=True)
class RAGCoachingIntegrationDependencies:
    background_manager: BoundedPostgreSQLRAGManager
    policy: RAGCoachingIntegrationPolicy
    suggestion_id_factory: Callable[[], str]
    utc_datetime_factory: Callable[[], datetime]
    citation_projector: SafeCoachingCitationProjector | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.background_manager,
            BoundedPostgreSQLRAGManager,
        ):
            raise ValueError("background_manager must be BoundedPostgreSQLRAGManager")
        if not callable(self.suggestion_id_factory):
            raise ValueError("suggestion_id_factory must be callable")
        if not callable(self.utc_datetime_factory):
            raise ValueError("utc_datetime_factory must be callable")


def compose_rag_coaching_processor(
    *,
    coordinator: CoachingCoordinator,
    tenant_config: TenantConfig,
    integration: RAGCoachingIntegrationDependencies | None,
) -> SafeCoachingProcessorAdapter | RAGCoachingProcessorDecorator:
    if (
        integration is None
        or not tenant_config.rag.enabled
        or not tenant_config.coaching.enable_llm
    ):
        return SafeCoachingProcessorAdapter(coordinator)

    if coordinator.call_state.tenant_id != tenant_config.context.tenant_id:
        raise ValueError("coordinator tenant_id does not match tenant config")
    if tenant_config.rag.knowledge_base_id is None:
        raise ValueError("knowledge_base_id is required for RAG coaching")
    if integration.policy.action.value not in tenant_config.coaching.allowed_actions:
        raise ValueError("RAG coaching action is not allowed by tenant config")

    suggestion_factory = DeterministicLLMCoachingSuggestionFactory(
        title=integration.policy.title,
        action=integration.policy.action,
        priority=integration.policy.priority,
        label_id=integration.policy.label_id,
        expires_after_seconds=integration.policy.expires_after_seconds,
        suggestion_id_factory=integration.suggestion_id_factory,
        utc_datetime_factory=integration.utc_datetime_factory,
    )
    return RAGCoachingProcessorDecorator(
        coordinator=coordinator,
        tenant_config=tenant_config,
        background_manager=integration.background_manager,
        suggestion_factory=suggestion_factory,
        rag_llm_enabled_labels=integration.policy.rag_llm_enabled_labels,
        citation_projector=integration.citation_projector,
    )
