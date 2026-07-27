from dataclasses import dataclass, field
from datetime import UTC, datetime
import inspect

import pytest

from app.calls.models import CallState
from app.coaching.coordinator import (
    CoachingCoordinator,
    CoachingCoordinatorResult,
    CoachingProcessingStatus,
    CoachingStateSnapshot,
    StableCoachingOutcome,
)
from app.coaching.rule_engine import RuleBasedCoachingEngine
from app.events.models import (
    ClassificationLabel,
    ClassificationResultEvent,
    CoachingAction,
    CoachingSuggestionEvent,
    CoachingSuggestionSource,
    SuggestionPriority,
    TranscriptEvent,
    TranscriptKind,
)
from app.integration import (
    CoachingSuggestionFactory,
    OrchestrationRunner,
    RAGCoachingProcessorDecorator,
)
from app.orchestration import (
    OrchestrationCitationReference,
    OrchestrationRequest,
    OrchestrationResult,
)
from app.streaming.pipeline import CoachingProcessorProtocol
from app.tenancy.models import (
    TenantASRConfig,
    TenantClassificationConfig,
    TenantCoachingConfig,
    TenantConfig,
    TenantContext,
    TenantRAGConfig,
)

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def tenant_config(
    *,
    rag_enabled: bool = True,
    llm_enabled: bool = True,
    knowledge_base_id: str | None = "kb_support",
    allowed_actions: list[str] | None = None,
    maximum: int = 3,
) -> TenantConfig:
    return TenantConfig(
        context=TenantContext(tenant_id="tenant_alpha", tenant_name="Synthetic"),
        asr=TenantASRConfig(),
        classification=TenantClassificationConfig(
            model_id="synthetic-model",
            labels=["product_information", "complaint"],
        ),
        rag=TenantRAGConfig(
            enabled=rag_enabled,
            knowledge_base_id=knowledge_base_id,
            top_k=3,
            minimum_score=0.7,
        ),
        coaching=TenantCoachingConfig(
            enable_llm=llm_enabled,
            cooldown_seconds=0,
            max_active_suggestions=maximum,
            allowed_actions=allowed_actions
            or [action.value for action in CoachingAction],
        ),
    )


def transcript(
    *,
    revision: int = 1,
    kind: TranscriptKind = TranscriptKind.STABLE,
) -> TranscriptEvent:
    return TranscriptEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        event_id=f"transcript_{revision}",
        kind=kind,
        text="Synthetic product question.",
        start_seconds=0,
        end_seconds=float(revision),
        revision=revision,
        created_at_utc=NOW,
    )


def classification(event: TranscriptEvent) -> ClassificationResultEvent:
    return ClassificationResultEvent(
        tenant_id=event.tenant_id,
        call_id=event.call_id,
        transcript_event_id=event.event_id,
        labels=[ClassificationLabel(name="product_information", score=0.9)],
        action=CoachingAction.RAG_ACTION,
        model_id="synthetic-model",
        created_at_utc=NOW,
    )


def orchestration_result(
    event: TranscriptEvent,
    **changes: object,
) -> OrchestrationResult:
    values: dict[str, object] = {
        "tenant_id": event.tenant_id,
        "call_id": event.call_id,
        "transcript_revision": event.revision,
        "generated_text": "Synthetic generated coaching.",
        "citations": (
            OrchestrationCitationReference(
                document_id="document_1",
                chunk_id="chunk_1",
            ),
        ),
    }
    values.update(changes)
    return OrchestrationResult.model_validate(values)


def external_suggestion(
    event: TranscriptEvent,
    **changes: object,
) -> CoachingSuggestionEvent:
    values: dict[str, object] = {
        "tenant_id": event.tenant_id,
        "call_id": event.call_id,
        "suggestion_id": f"external_{event.revision}",
        "source_transcript_event_id": event.event_id,
        "action": CoachingAction.TEMPLATE_ACTION,
        "priority": SuggestionPriority.HIGH,
        "source": CoachingSuggestionSource.LLM,
        "title": "Synthetic external title",
        "suggestion": "Synthetic external suggestion.",
        "evidence_ids": ["document_1:chunk_1"],
        "created_at_utc": NOW,
    }
    values.update(changes)
    return CoachingSuggestionEvent.model_validate(values)


@dataclass
class FakeOrchestrationRunner:
    result: OrchestrationResult | None
    error: Exception | None = None
    requests: list[OrchestrationRequest] = field(default_factory=list)

    def run(self, request: OrchestrationRequest) -> OrchestrationResult | None:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result


@dataclass
class FakeSuggestionFactory:
    result: CoachingSuggestionEvent | None
    error: Exception | None = None
    calls: list[tuple[TranscriptEvent, OrchestrationResult, float]] = field(
        default_factory=list
    )

    def create(
        self,
        *,
        event: TranscriptEvent,
        orchestration_result: OrchestrationResult,
        current_seconds: float,
    ) -> CoachingSuggestionEvent | None:
        self.calls.append((event, orchestration_result, current_seconds))
        if self.error is not None:
            raise self.error
        return self.result


class AdmissionFailureCoordinator(CoachingCoordinator):
    snapshot_before_admission: CoachingStateSnapshot | None = None

    def process_external_suggestion(
        self,
        event: TranscriptEvent,
        suggestion: CoachingSuggestionEvent,
        current_seconds: float,
    ) -> CoachingCoordinatorResult:
        self.snapshot_before_admission = self.snapshot_coaching_state()
        super().process_external_suggestion(event, suggestion, current_seconds)
        raise RuntimeError("synthetic admission failure")


class TrackingCoordinator(CoachingCoordinator):
    last_base_outcome: StableCoachingOutcome | None = None
    snapshot_after_base: CoachingStateSnapshot | None = None

    def process_safely(
        self,
        event: TranscriptEvent,
        current_seconds: float,
        *,
        classification_event: ClassificationResultEvent | None = None,
        active_labels: tuple[str, ...] | None = None,
    ) -> StableCoachingOutcome:
        outcome = super().process_safely(
            event,
            current_seconds,
            classification_event=classification_event,
            active_labels=active_labels,
        )
        self.last_base_outcome = outcome
        self.snapshot_after_base = self.snapshot_coaching_state()
        return outcome


def prepared_state(event: TranscriptEvent) -> CallState:
    state = CallState(tenant_id=event.tenant_id, call_id=event.call_id)
    state.apply_transcript(event)
    state.apply_classification(
        classification(event),
        transcript_revision=event.revision,
        source_sequence=None,
    )
    return state


def coordinator(
    config: TenantConfig,
    state: CallState,
    coordinator_type: type[CoachingCoordinator] = CoachingCoordinator,
) -> CoachingCoordinator:
    return coordinator_type(
        config,
        state,
        RuleBasedCoachingEngine(
            config,
            (),
            event_id_factory=lambda: "base_suggestion",
            utc_datetime_factory=lambda: NOW,
        ),
    )


def decorator_dependencies(
    *,
    config: TenantConfig | None = None,
    event: TranscriptEvent | None = None,
    runner_result: OrchestrationResult | None = None,
    suggestion_result: CoachingSuggestionEvent | None = None,
    coordinator_type: type[CoachingCoordinator] = CoachingCoordinator,
) -> tuple[
    RAGCoachingProcessorDecorator,
    CoachingCoordinator,
    FakeOrchestrationRunner,
    FakeSuggestionFactory,
    TranscriptEvent,
]:
    actual_config = config or tenant_config()
    actual_event = event or transcript()
    subject_coordinator = coordinator(
        actual_config,
        prepared_state(actual_event),
        coordinator_type,
    )
    runner = FakeOrchestrationRunner(
        runner_result
        if runner_result is not None
        else orchestration_result(actual_event)
    )
    factory = FakeSuggestionFactory(
        suggestion_result
        if suggestion_result is not None
        else external_suggestion(actual_event)
    )
    return (
        RAGCoachingProcessorDecorator(
            subject_coordinator,
            actual_config,
            runner,
            factory,
            ("product_information",),
        ),
        subject_coordinator,
        runner,
        factory,
        actual_event,
    )


def run(
    decorator: RAGCoachingProcessorDecorator,
    event: TranscriptEvent,
) -> StableCoachingOutcome:
    return decorator.process_safely(
        event,
        event.end_seconds,
        classification_event=classification(event),
        active_labels=("product_information",),
    )


def test_protocols_and_decorator_are_structurally_compatible() -> None:
    decorator, _, runner, factory, _ = decorator_dependencies()
    processor: CoachingProcessorProtocol = decorator
    orchestration: OrchestrationRunner = runner
    suggestion_factory: CoachingSuggestionFactory = factory

    assert processor is decorator
    assert orchestration is runner
    assert suggestion_factory is factory


def test_constructor_accepts_only_one_coordinator_and_internal_adapter_shares_it() -> (
    None
):
    decorator, subject_coordinator, _, _, _ = decorator_dependencies()
    parameters = inspect.signature(RAGCoachingProcessorDecorator).parameters

    assert tuple(parameters) == (
        "coordinator",
        "tenant_config",
        "orchestration_runner",
        "suggestion_factory",
        "rag_llm_enabled_labels",
    )
    assert decorator._coordinator is subject_coordinator  # noqa: SLF001
    assert decorator._base_processor._coordinator is subject_coordinator  # noqa: SLF001


@pytest.mark.parametrize(
    "labels",
    [("",), ("product_information", "product_information")],
)
def test_constructor_rejects_invalid_enabled_labels(labels: tuple[str, ...]) -> None:
    config = tenant_config()
    event = transcript()
    subject_coordinator = coordinator(config, prepared_state(event))

    with pytest.raises(ValueError, match="rag_llm_enabled_labels"):
        RAGCoachingProcessorDecorator(
            subject_coordinator,
            config,
            FakeOrchestrationRunner(orchestration_result(event)),
            FakeSuggestionFactory(external_suggestion(event)),
            labels,
        )


@pytest.mark.parametrize(
    "config",
    [
        tenant_config(rag_enabled=False, knowledge_base_id=None),
        tenant_config(llm_enabled=False),
    ],
)
def test_config_skip_returns_exact_base_object(config: TenantConfig) -> None:
    event = transcript()
    state = prepared_state(event)
    subject_coordinator = coordinator(config, state, TrackingCoordinator)
    runner = FakeOrchestrationRunner(orchestration_result(event))
    decorator = RAGCoachingProcessorDecorator(
        subject_coordinator,
        config,
        runner,
        FakeSuggestionFactory(external_suggestion(event)),
        ("product_information",),
    )
    actual = run(decorator, event)

    tracking = subject_coordinator
    assert isinstance(tracking, TrackingCoordinator)
    assert actual is tracking.last_base_outcome
    assert runner.requests == []


def test_missing_knowledge_base_defensively_skips() -> None:
    config = tenant_config().model_copy(
        update={
            "rag": tenant_config().rag.model_copy(update={"knowledge_base_id": None})
        }
    )
    decorator, _, runner, _, event = decorator_dependencies(config=config)

    run(decorator, event)

    assert runner.requests == []


def test_missing_or_ineligible_revision_labels_skip_orchestration() -> None:
    config = tenant_config()
    event = transcript()
    state = CallState(tenant_id=event.tenant_id, call_id=event.call_id)
    state.apply_transcript(event)
    subject_coordinator = coordinator(config, state)
    runner = FakeOrchestrationRunner(orchestration_result(event))
    decorator = RAGCoachingProcessorDecorator(
        subject_coordinator,
        config,
        runner,
        FakeSuggestionFactory(external_suggestion(event)),
        ("technical_issue",),
    )

    outcome = subject_coordinator.snapshot_coaching_state()
    result = run(decorator, event)

    assert result.status is CoachingProcessingStatus.PROCESSED
    assert runner.requests == []
    assert subject_coordinator.snapshot_coaching_state() != outcome


def test_partial_and_duplicate_base_outcomes_skip_enrichment() -> None:
    partial_event = transcript(kind=TranscriptKind.PARTIAL)
    config = tenant_config()
    partial_state = CallState(
        tenant_id=partial_event.tenant_id,
        call_id=partial_event.call_id,
        transcript_revision=partial_event.revision,
    )
    partial_coordinator = coordinator(config, partial_state, TrackingCoordinator)
    partial_runner = FakeOrchestrationRunner(orchestration_result(partial_event))
    partial_decorator = RAGCoachingProcessorDecorator(
        partial_coordinator,
        config,
        partial_runner,
        FakeSuggestionFactory(external_suggestion(partial_event)),
        ("product_information",),
    )

    partial = run(partial_decorator, partial_event)

    tracking = partial_coordinator
    assert isinstance(tracking, TrackingCoordinator)
    assert partial is tracking.last_base_outcome
    assert partial.status is CoachingProcessingStatus.PARTIAL_SKIPPED
    assert partial_runner.requests == []

    duplicate_decorator, duplicate_coordinator, duplicate_runner, _, event = (
        decorator_dependencies(coordinator_type=TrackingCoordinator)
    )
    run(duplicate_decorator, event)
    duplicate = run(duplicate_decorator, event)

    duplicate_tracking = duplicate_coordinator
    assert isinstance(duplicate_tracking, TrackingCoordinator)
    assert duplicate is duplicate_tracking.last_base_outcome
    assert duplicate.status is CoachingProcessingStatus.DUPLICATE_REVISION_SKIPPED
    assert len(duplicate_runner.requests) == 1


def test_exact_orchestration_request_and_factory_inputs_are_propagated() -> None:
    decorator, _, runner, factory, event = decorator_dependencies()

    run(decorator, event)

    assert runner.requests == [
        OrchestrationRequest(
            tenant_id="tenant_alpha",
            call_id="call_001",
            transcript_revision=1,
            knowledge_base_id="kb_support",
            user_input="Synthetic product question.",
            top_k=3,
            minimum_score=0.7,
        )
    ]
    assert factory.calls == [(event, orchestration_result(event), event.end_seconds)]
    assert factory.calls[0][1].citations == orchestration_result(event).citations


@pytest.mark.parametrize(
    "result",
    [
        None,
        orchestration_result(transcript(), tenant_id="tenant_other"),
        orchestration_result(transcript(), call_id="call_other"),
        orchestration_result(transcript(), transcript_revision=2),
        OrchestrationResult.model_construct(
            tenant_id="tenant_alpha",
            call_id="call_001",
            transcript_revision=1,
            generated_text=" ",
            citations=(),
        ),
    ],
)
def test_untrusted_or_empty_orchestration_result_skips_factory(
    result: OrchestrationResult | None,
) -> None:
    decorator, _, runner, factory, event = decorator_dependencies()
    runner.result = result

    outcome = run(decorator, event)

    assert outcome.status is CoachingProcessingStatus.PROCESSED
    assert factory.calls == []


def test_orchestration_value_error_is_isolated_but_runtime_error_propagates() -> None:
    isolated, _, isolated_runner, isolated_factory, event = decorator_dependencies()
    isolated_runner.error = ValueError("synthetic trusted validation failure")

    base = run(isolated, event)

    assert base.status is CoachingProcessingStatus.PROCESSED
    assert isolated_factory.calls == []

    propagated, _, runner, _, event = decorator_dependencies()
    runner.error = RuntimeError("synthetic programming failure")
    with pytest.raises(RuntimeError, match="programming failure"):
        run(propagated, event)


@pytest.mark.parametrize(
    "changes",
    [
        {"tenant_id": "tenant_other"},
        {"call_id": "call_other"},
        {"source_transcript_event_id": "transcript_other"},
    ],
)
def test_factory_none_and_scope_mismatch_skip_external_admission(
    changes: dict[str, object],
) -> None:
    none_decorator, none_coordinator, _, none_factory, event = decorator_dependencies()
    none_factory.result = None
    before_none = none_coordinator.snapshot_coaching_state()
    run(none_decorator, event)
    after_none = none_coordinator.snapshot_coaching_state()

    mismatch_decorator, mismatch_coordinator, _, mismatch_factory, event = (
        decorator_dependencies()
    )
    mismatch_factory.result = external_suggestion(event, **changes)
    run(mismatch_decorator, event)

    assert after_none.processed_external_revisions == (
        before_none.processed_external_revisions
    )
    assert (
        mismatch_coordinator.snapshot_coaching_state().processed_external_revisions
        == (frozenset())
    )


def test_factory_exception_propagates() -> None:
    decorator, _, _, factory, event = decorator_dependencies()
    factory.error = RuntimeError("synthetic factory defect")

    with pytest.raises(RuntimeError, match="factory defect"):
        run(decorator, event)


@pytest.mark.parametrize(
    "source",
    [
        CoachingSuggestionSource.RULE,
        CoachingSuggestionSource.CLASSIFICATION,
        CoachingSuggestionSource.BOTH,
    ],
)
def test_non_llm_factory_source_preserves_exact_base_without_admission(
    source: CoachingSuggestionSource,
) -> None:
    decorator, subject_coordinator, _, factory, event = decorator_dependencies(
        coordinator_type=TrackingCoordinator
    )
    factory.result = external_suggestion(event, source=source)

    outcome = run(decorator, event)

    tracking = subject_coordinator
    assert isinstance(tracking, TrackingCoordinator)
    assert outcome is tracking.last_base_outcome
    assert tracking.snapshot_after_base is not None
    assert tracking.snapshot_coaching_state() == tracking.snapshot_after_base
    assert tracking.snapshot_after_base.processed_external_revisions == frozenset()


def test_external_admission_combines_base_and_external_results_in_order() -> None:
    decorator, coordinator, _, _, event = decorator_dependencies()

    outcome = run(decorator, event)

    assert outcome.status is CoachingProcessingStatus.PROCESSED
    assert outcome.result is not None
    assert [item.source for item in outcome.result.displayed_suggestions] == [
        CoachingSuggestionSource.CLASSIFICATION,
        CoachingSuggestionSource.LLM,
    ]
    assert outcome.result.classification_event == classification(event)
    assert outcome.result.current_revision_labels == ("product_information",)
    assert outcome.result.matched_rule_ids == ()
    assert coordinator.call_state.active_coaching_suggestions


def test_normal_external_suppression_is_combined_after_base_metadata() -> None:
    config = tenant_config(
        allowed_actions=[CoachingAction.RAG_ACTION.value],
    )
    decorator, _, _, factory, event = decorator_dependencies(config=config)
    factory.result = external_suggestion(event, action=CoachingAction.TEMPLATE_ACTION)

    outcome = run(decorator, event)

    assert outcome.result is not None
    assert outcome.result.displayed_suggestions
    assert outcome.result.suppressed_suggestions == (factory.result,)
    assert outcome.result.suppression_reasons == ("action_not_allowed",)
    assert outcome.result.suggestion_decisions[-1].reason == "action_not_allowed"


def test_external_replacement_metadata_follows_base_decisions() -> None:
    config = tenant_config(maximum=1)
    decorator, _, _, _, event = decorator_dependencies(config=config)

    outcome = run(decorator, event)

    assert outcome.result is not None
    assert outcome.result.replaced_suggestion_ids == ("base_suggestion",)
    assert [decision.reason for decision in outcome.result.suggestion_decisions] == [
        "admitted",
        "admitted",
        "replaced_by_newer_priority",
    ]


def test_admission_exception_restores_post_base_snapshot_and_reraises() -> None:
    decorator, subject_coordinator, _, _, event = decorator_dependencies(
        coordinator_type=AdmissionFailureCoordinator
    )

    with pytest.raises(RuntimeError, match="admission failure"):
        run(decorator, event)

    failing = subject_coordinator
    assert isinstance(failing, AdmissionFailureCoordinator)
    assert failing.snapshot_before_admission is not None
    assert failing.snapshot_coaching_state() == failing.snapshot_before_admission


def test_repeated_fresh_runs_are_deterministic() -> None:
    first, _, _, _, first_event = decorator_dependencies()
    second, _, _, _, second_event = decorator_dependencies()

    assert run(first, first_event) == run(second, second_event)
