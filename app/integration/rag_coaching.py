"""Deterministic composition of safe coaching and trusted RAG orchestration."""

from collections.abc import Collection
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.coaching.coordinator import (
    CoachingCoordinator,
    CoachingProcessingStatus,
    StableCoachingOutcome,
)
from app.composition.postgres_rag_background import (
    BoundedPostgreSQLRAGManager,
    RAGOrchestrationCompletionStatus,
    RAGOrchestrationIdentity,
    RAGOrchestrationSubmissionStatus,
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


@runtime_checkable
class CoachingCompletionPumpProtocol(Protocol):
    def drain_completed(
        self,
        *,
        current_seconds: float,
    ) -> tuple[StableCoachingOutcome, ...]: ...


class CoachingSuggestionFactory(Protocol):
    def create(
        self,
        *,
        event: TranscriptEvent,
        orchestration_result: OrchestrationResult,
        current_seconds: float,
    ) -> CoachingSuggestionEvent | None: ...


@dataclass(frozen=True, slots=True)
class _PendingContext:
    event: TranscriptEvent
    current_seconds: float


class RAGCoachingProcessorDecorator:
    def __init__(
        self,
        coordinator: CoachingCoordinator,
        tenant_config: TenantConfig,
        background_manager: BoundedPostgreSQLRAGManager,
        suggestion_factory: CoachingSuggestionFactory,
        rag_llm_enabled_labels: tuple[str, ...],
    ) -> None:
        if coordinator.call_state.tenant_id != tenant_config.context.tenant_id:
            raise ValueError("coordinator tenant_id does not match tenant config")
        self._coordinator = coordinator
        self._base_processor = SafeCoachingProcessorAdapter(coordinator)
        self._tenant_config = tenant_config
        if not isinstance(background_manager, BoundedPostgreSQLRAGManager):
            raise ValueError("background_manager must be BoundedPostgreSQLRAGManager")
        self._background_manager = background_manager
        self._suggestion_factory = suggestion_factory
        self._rag_llm_enabled_labels = _validated_labels(rag_llm_enabled_labels)
        self._decision_gate = LLMCoachingDecisionGate()
        self._pending_contexts: dict[
            RAGOrchestrationIdentity,
            _PendingContext,
        ] = {}

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

        identity = RAGOrchestrationIdentity(
            tenant_id=event.tenant_id,
            call_id=event.call_id,
            transcript_revision=event.revision,
        )
        try:
            self._background_manager.announce_current_revision(
                tenant_id=identity.tenant_id,
                call_id=identity.call_id,
                transcript_revision=identity.transcript_revision,
            )
        except (RuntimeError, ValueError):
            return base_outcome
        self._discard_superseded_contexts(identity)

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
            submission = self._background_manager.submit(request)
        except (RuntimeError, ValueError):
            return base_outcome
        if submission.status is RAGOrchestrationSubmissionStatus.ACCEPTED:
            self._pending_contexts[submission.identity] = _PendingContext(
                event=event,
                current_seconds=current_seconds,
            )
        return base_outcome

    def drain_completed(
        self,
        *,
        current_seconds: float,
    ) -> tuple[StableCoachingOutcome, ...]:
        if current_seconds < 0:
            raise ValueError("current_seconds cannot be negative")
        call_state = self._coordinator.call_state
        current_identity = RAGOrchestrationIdentity(
            tenant_id=call_state.tenant_id,
            call_id=call_state.call_id,
            transcript_revision=call_state.transcript_revision,
        )
        try:
            self._background_manager.announce_current_revision(
                tenant_id=current_identity.tenant_id,
                call_id=current_identity.call_id,
                transcript_revision=current_identity.transcript_revision,
            )
        except (RuntimeError, ValueError):
            self._pending_contexts.clear()
            return ()
        self._discard_superseded_contexts(current_identity)

        outcomes: list[StableCoachingOutcome] = []
        for identity in sorted(
            self._pending_contexts,
            key=lambda item: (
                item.tenant_id,
                item.call_id,
                item.transcript_revision,
            ),
        ):
            completion = self._background_manager.poll(identity)
            if completion is None:
                continue
            context = self._pending_contexts.pop(identity)
            if completion.status is RAGOrchestrationCompletionStatus.EMPTY:
                continue
            if completion.status is RAGOrchestrationCompletionStatus.FAILED:
                outcomes.append(
                    StableCoachingOutcome(
                        status=CoachingProcessingStatus.FAILED,
                        transcript_revision=identity.transcript_revision,
                        error_type="rag_orchestration",
                        error_code="background_failure",
                    )
                )
                continue
            orchestration_result = completion.result
            if orchestration_result is None or not _trusted_result_matches(
                orchestration_result,
                context.event,
            ):
                continue
            suggestion = self._suggestion_factory.create(
                event=context.event,
                orchestration_result=orchestration_result,
                current_seconds=context.current_seconds,
            )
            if suggestion is None or not _suggestion_scope_matches(
                suggestion,
                context.event,
            ):
                continue
            snapshot = self._coordinator.snapshot_coaching_state()
            try:
                external_result = self._coordinator.process_external_suggestion(
                    context.event,
                    suggestion,
                    context.current_seconds,
                )
            except Exception:
                self._coordinator.restore_coaching_state(snapshot)
                raise
            outcomes.append(
                StableCoachingOutcome(
                    status=CoachingProcessingStatus.PROCESSED,
                    transcript_revision=identity.transcript_revision,
                    result=external_result,
                )
            )
        return tuple(outcomes)

    def _discard_superseded_contexts(
        self,
        current_identity: RAGOrchestrationIdentity,
    ) -> None:
        self._pending_contexts = {
            identity: context
            for identity, context in self._pending_contexts.items()
            if identity == current_identity
        }


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
