"""Deterministic composition of safe coaching and trusted RAG orchestration."""

from collections.abc import Collection
from typing import Protocol

from app.coaching.coordinator import (
    CoachingCoordinator,
    CoachingCoordinatorResult,
    CoachingProcessingStatus,
    StableCoachingOutcome,
)
from app.coaching.llm_decision_gate import (
    LLMCoachingDecision,
    LLMCoachingDecisionGate,
)
from app.coaching.safe_processor import SafeCoachingProcessorAdapter
from app.events.models import (
    ClassificationResultEvent,
    CoachingSuggestionEvent,
    CoachingSuggestionSource,
    TranscriptEvent,
    TranscriptKind,
)
from app.orchestration import OrchestrationRequest, OrchestrationResult
from app.tenancy.models import TenantConfig


class OrchestrationRunner(Protocol):
    def run(
        self,
        request: OrchestrationRequest,
    ) -> OrchestrationResult | None: ...


class CoachingSuggestionFactory(Protocol):
    def create(
        self,
        *,
        event: TranscriptEvent,
        orchestration_result: OrchestrationResult,
        current_seconds: float,
    ) -> CoachingSuggestionEvent | None: ...


class RAGCoachingProcessorDecorator:
    def __init__(
        self,
        coordinator: CoachingCoordinator,
        tenant_config: TenantConfig,
        orchestration_runner: OrchestrationRunner,
        suggestion_factory: CoachingSuggestionFactory,
        rag_llm_enabled_labels: tuple[str, ...],
    ) -> None:
        if coordinator.call_state.tenant_id != tenant_config.context.tenant_id:
            raise ValueError("coordinator tenant_id does not match tenant config")
        self._coordinator = coordinator
        self._base_processor = SafeCoachingProcessorAdapter(coordinator)
        self._tenant_config = tenant_config
        self._orchestration_runner = orchestration_runner
        self._suggestion_factory = suggestion_factory
        self._rag_llm_enabled_labels = _validated_labels(rag_llm_enabled_labels)
        self._decision_gate = LLMCoachingDecisionGate()

    def process_safely(
        self,
        event: TranscriptEvent,
        current_seconds: float,
        *,
        classification_event: ClassificationResultEvent | None = None,
        active_labels: tuple[str, ...] | None = None,
    ) -> StableCoachingOutcome:
        base_outcome = self._base_processor.process_safely(
            event,
            current_seconds,
            classification_event=classification_event,
            active_labels=active_labels,
        )
        base_result = base_outcome.result
        if (
            base_outcome.status is not CoachingProcessingStatus.PROCESSED
            or base_result is None
            or event.kind not in {TranscriptKind.STABLE, TranscriptKind.FINAL}
            or base_result.transcript_revision != event.revision
        ):
            return base_outcome

        rag_config = self._tenant_config.rag
        if (
            not rag_config.enabled
            or not self._tenant_config.coaching.enable_llm
            or rag_config.knowledge_base_id is None
        ):
            return base_outcome

        diagnostic = next(
            (
                item
                for item in reversed(
                    self._coordinator.call_state.label_revision_timeline
                )
                if item.transcript_revision == event.revision
            ),
            None,
        )
        if diagnostic is None:
            return base_outcome
        decision = self._decision_gate.decide(
            tenant_id=event.tenant_id,
            call_id=event.call_id,
            revision=event.revision,
            current_labels=diagnostic.current_labels,
            newly_detected_labels=diagnostic.newly_accumulated_labels,
            rag_llm_enabled_labels=self._rag_llm_enabled_labels,
            rag_enabled=rag_config.enabled,
            llm_enabled=self._tenant_config.coaching.enable_llm,
        )
        if decision.decision is not LLMCoachingDecision.REQUEST_RAG_LLM:
            return base_outcome

        request = OrchestrationRequest(
            tenant_id=event.tenant_id,
            call_id=event.call_id,
            transcript_revision=event.revision,
            knowledge_base_id=rag_config.knowledge_base_id,
            user_input=event.text,
            top_k=rag_config.top_k,
            minimum_score=rag_config.minimum_score,
        )
        try:
            orchestration_result = self._orchestration_runner.run(request)
        except ValueError:
            return base_outcome
        if orchestration_result is None or not _trusted_result_matches(
            orchestration_result,
            event,
        ):
            return base_outcome

        suggestion = self._suggestion_factory.create(
            event=event,
            orchestration_result=orchestration_result,
            current_seconds=current_seconds,
        )
        if suggestion is None or not _suggestion_scope_matches(suggestion, event):
            return base_outcome

        snapshot = self._coordinator.snapshot_coaching_state()
        try:
            external_result = self._coordinator.process_external_suggestion(
                event,
                suggestion,
                current_seconds,
            )
        except Exception:
            self._coordinator.restore_coaching_state(snapshot)
            raise
        return _combined_outcome(base_outcome, base_result, external_result)


def _validated_labels(labels: Collection[str]) -> tuple[str, ...]:
    cleaned = tuple(label.strip() for label in labels)
    if any(not label for label in cleaned):
        raise ValueError("rag_llm_enabled_labels cannot contain empty labels")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("rag_llm_enabled_labels must be unique")
    return cleaned


def _trusted_result_matches(
    result: OrchestrationResult,
    event: TranscriptEvent,
) -> bool:
    return (
        result.tenant_id == event.tenant_id
        and result.call_id == event.call_id
        and result.transcript_revision == event.revision
        and bool(result.generated_text.strip())
    )


def _suggestion_scope_matches(
    suggestion: CoachingSuggestionEvent,
    event: TranscriptEvent,
) -> bool:
    return (
        suggestion.tenant_id == event.tenant_id
        and suggestion.call_id == event.call_id
        and suggestion.source_transcript_event_id == event.event_id
        and suggestion.source is CoachingSuggestionSource.LLM
    )


def _combined_outcome(
    base_outcome: StableCoachingOutcome,
    base_result: CoachingCoordinatorResult,
    external_result: CoachingCoordinatorResult,
) -> StableCoachingOutcome:
    return StableCoachingOutcome(
        status=base_outcome.status,
        transcript_revision=base_outcome.transcript_revision,
        result=CoachingCoordinatorResult(
            classification_event=base_result.classification_event,
            displayed_suggestions=(
                *base_result.displayed_suggestions,
                *external_result.displayed_suggestions,
            ),
            suppressed_suggestions=(
                *base_result.suppressed_suggestions,
                *external_result.suppressed_suggestions,
            ),
            matched_rule_ids=base_result.matched_rule_ids,
            suppression_reasons=(
                *base_result.suppression_reasons,
                *external_result.suppression_reasons,
            ),
            transcript_revision=base_result.transcript_revision,
            current_revision_labels=base_result.current_revision_labels,
            replaced_suggestion_ids=(
                *base_result.replaced_suggestion_ids,
                *external_result.replaced_suggestion_ids,
            ),
            suggestion_decisions=(
                *base_result.suggestion_decisions,
                *external_result.suggestion_decisions,
            ),
        ),
        error_type=base_outcome.error_type,
        error_code=base_outcome.error_code,
    )
