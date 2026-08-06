from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from app.calls.models import CallState
from app.classification.runtime import RuntimeSetFitClassifier
from app.coaching.coordinator import CoachingCoordinator
from app.coaching.safe_processor import SafeCoachingProcessorAdapter
from app.composition import (
    BoundedPostgreSQLRAGManager,
    RAGOrchestrationCompletion,
    RAGOrchestrationIdentity,
    RAGOrchestrationSubmission,
)
from app.events.models import (
    CoachingAction,
    SuggestionPriority,
    TranscriptEvent,
    TranscriptKind,
)
from app.integration import (
    DeterministicLLMCoachingSuggestionFactory,
    RAGCoachingIntegrationDependencies,
    RAGCoachingIntegrationPolicy,
    RAGCoachingProcessorDecorator,
)
from app.orchestration import OrchestrationRequest
from app.streaming.audio_window import ASRAudioWindow
from app.streaming.pipeline import CoachingProcessorProtocol, WindowTranscriberProtocol
from app.streaming.window_transcriber import WindowTranscriptionResult
from app.tenancy.models import TenantCoachingConfig, TenantRAGConfig
from live_dashboard.demo_data import TenantDemo, tenant_demos
from live_dashboard.runtime_wiring import (
    ArtifactAvailability,
    DashboardExecutionIdentity,
    DashboardExecutionResource,
    DashboardServiceSelection,
    build_live_pipeline,
    default_service_selection,
    inspect_default_artifacts,
)
from live_dashboard.view_models import create_local_execution, dashboard_tabs


class FakeTranscriber:
    def transcribe(self, window: object) -> object:
        raise AssertionError("transcription must not run in wiring tests")


class FakeClassifier:
    pass


class TypedFakeTranscriber:
    def transcribe(self, window: ASRAudioWindow) -> WindowTranscriptionResult:
        raise AssertionError("transcription must not run in wiring tests")


class FakeBackgroundManager(BoundedPostgreSQLRAGManager):
    def __init__(self) -> None:
        self.calls = 0

    def announce_current_revision(
        self,
        *,
        tenant_id: str,
        call_id: str,
        transcript_revision: int,
    ) -> None:
        self.calls += 1
        raise AssertionError("manager must not execute during construction")

    def submit(self, request: OrchestrationRequest) -> RAGOrchestrationSubmission:
        self.calls += 1
        raise AssertionError("manager must not execute during construction")

    def poll(
        self,
        identity: RAGOrchestrationIdentity,
    ) -> RAGOrchestrationCompletion | None:
        self.calls += 1
        raise AssertionError("manager must not execute during construction")


@dataclass
class CallbackTracker:
    id_calls: int = 0
    clock_calls: int = 0

    def suggestion_id(self) -> str:
        self.id_calls += 1
        return "synthetic-llm-suggestion"

    def now(self) -> datetime:
        self.clock_calls += 1
        return datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def integration_dependencies() -> tuple[
    RAGCoachingIntegrationDependencies,
    FakeBackgroundManager,
    CallbackTracker,
]:
    manager = FakeBackgroundManager()
    callbacks = CallbackTracker()
    policy = RAGCoachingIntegrationPolicy(
        rag_llm_enabled_labels=("product_information",),
        title="Synthetic guidance",
        action=CoachingAction.RAG_ACTION,
        priority=SuggestionPriority.HIGH,
        label_id="product_information",
        expires_after_seconds=30.0,
    )
    return (
        RAGCoachingIntegrationDependencies(
            background_manager=manager,
            policy=policy,
            suggestion_id_factory=callbacks.suggestion_id,
            utc_datetime_factory=callbacks.now,
        ),
        manager,
        callbacks,
    )


def enabled_tenant() -> TenantDemo:
    base = tenant_demos()["tenant_alpha"]
    config = base.config.model_copy(
        update={
            "rag": TenantRAGConfig(
                enabled=True,
                knowledge_base_id="kb_synthetic",
                top_k=3,
                minimum_score=0.7,
            ),
            "coaching": TenantCoachingConfig(
                cooldown_seconds=8.0,
                max_active_suggestions=2,
                enable_llm=True,
                allowed_actions=[action.value for action in CoachingAction],
            ),
        }
    )
    return TenantDemo(config=config, rules=base.rules, scenarios=base.scenarios)


def runtime():
    return create_local_execution(
        tenant_demos()["tenant_alpha"], "synthetic-call"
    ).runtime


def test_default_auto_enable_follows_artifacts_and_rules() -> None:
    available = default_service_selection(
        ArtifactAvailability(True),
        deterministic_rules_available=True,
    )
    missing = default_service_selection(
        ArtifactAvailability(False, "safe"),
        deterministic_rules_available=True,
    )
    assert available == DashboardServiceSelection(True, True)
    assert missing == DashboardServiceSelection(False, True)


def test_missing_artifacts_are_safe_and_keep_rule_coaching() -> None:
    availability = inspect_default_artifacts(
        model_dir=Path("missing-model"),
        threshold_profile=Path("missing-profile.json"),
        taxonomy_path=Path("missing-taxonomy.json"),
    )
    subject = runtime()
    provider_calls = 0

    def provider() -> RuntimeSetFitClassifier:
        nonlocal provider_calls
        provider_calls += 1
        return cast(RuntimeSetFitClassifier, FakeClassifier())

    pipeline = build_live_pipeline(
        subject,
        cast(WindowTranscriberProtocol, FakeTranscriber()),
        selection=DashboardServiceSelection(True, True),
        availability=availability,
        classifier_provider=provider,
    )
    assert provider_calls == 0
    assert subject.classification_failure
    assert subject.coaching_enabled
    assert subject.rule_engine_enabled
    assert pipeline._coaching_coordinator_factory is not None  # noqa: SLF001
    assert (  # noqa: SLF001
        pipeline._classification_stage._rule_only_partial_classifier is not None
    )
    coordinator = pipeline._coaching_coordinator_factory(  # noqa: SLF001
        CallState(tenant_id="tenant_alpha", call_id="synthetic-call")
    )
    assert isinstance(coordinator, SafeCoachingProcessorAdapter)
    call_state = coordinator._call_state  # noqa: SLF001
    stable_event = TranscriptEvent(
        tenant_id="tenant_alpha",
        call_id="synthetic-call",
        event_id="stable-1",
        kind=TranscriptKind.STABLE,
        text="Aboneliğimi iptal etmek istiyorum.",
        start_seconds=0,
        end_seconds=1,
        revision=1,
        created_at_utc=tenant_demos()["tenant_alpha"]
        .scenarios[0]
        .events[0]
        .created_at_utc,
    )
    call_state.apply_transcript(stable_event)
    rule_only = coordinator.process_safely(
        stable_event,
        1,
        active_labels=(),
    )
    assert isinstance(coordinator._coordinator, CoachingCoordinator)  # noqa: SLF001
    assert rule_only.result is not None
    assert rule_only.result.displayed_suggestions
    tabs = dashboard_tabs(subject)
    assert ("SetFit", "failed") in tabs.technical.pipeline_statuses
    assert ("Coaching", "active") in tabs.technical.pipeline_statuses
    assert ("Rule Engine", "active") in tabs.technical.pipeline_statuses
    assert availability.safe_message in tabs.representative.safe_messages
    assert "missing-model" not in repr(tabs)


def test_setfit_and_coaching_toggles_control_pipeline_services() -> None:
    classifier = cast(RuntimeSetFitClassifier, FakeClassifier())
    calls = 0

    def provider() -> RuntimeSetFitClassifier:
        nonlocal calls
        calls += 1
        return classifier

    enabled_runtime = runtime()
    enabled = build_live_pipeline(
        enabled_runtime,
        cast(WindowTranscriberProtocol, FakeTranscriber()),
        selection=DashboardServiceSelection(True, True),
        availability=ArtifactAvailability(True),
        classifier_provider=provider,
    )
    assert enabled._classification_stage._classifier is classifier  # noqa: SLF001
    assert enabled._coaching_coordinator_factory is not None  # noqa: SLF001

    disabled_runtime = runtime()
    disabled = build_live_pipeline(
        disabled_runtime,
        cast(WindowTranscriberProtocol, FakeTranscriber()),
        selection=DashboardServiceSelection(False, False),
        availability=ArtifactAvailability(True),
        classifier_provider=provider,
    )
    assert disabled._classification_stage._classifier is None  # noqa: SLF001
    assert (  # noqa: SLF001
        disabled._classification_stage._rule_only_partial_classifier is None
    )
    assert disabled._coaching_coordinator_factory is None  # noqa: SLF001
    assert calls == 1
    assert ("SetFit", "disabled") in dashboard_tabs(
        disabled_runtime
    ).technical.pipeline_statuses
    assert ("Coaching", "disabled") in dashboard_tabs(
        disabled_runtime
    ).technical.pipeline_statuses
    assert ("Rule Engine", "disabled") in dashboard_tabs(
        disabled_runtime
    ).technical.pipeline_statuses


def test_dashboard_reruns_reuse_cached_classifier_instance() -> None:
    classifier = cast(RuntimeSetFitClassifier, FakeClassifier())

    def cached_provider() -> RuntimeSetFitClassifier:
        return classifier

    first = build_live_pipeline(
        runtime(),
        cast(WindowTranscriberProtocol, FakeTranscriber()),
        selection=DashboardServiceSelection(True, True),
        availability=ArtifactAvailability(True),
        classifier_provider=cached_provider,
    )
    second = build_live_pipeline(
        runtime(),
        cast(WindowTranscriberProtocol, FakeTranscriber()),
        selection=DashboardServiceSelection(True, True),
        availability=ArtifactAvailability(True),
        classifier_provider=cached_provider,
    )
    assert first._classification_stage._classifier is (  # noqa: SLF001
        second._classification_stage._classifier  # noqa: SLF001
    )


def test_explicit_none_preserves_base_processor() -> None:
    subject = runtime()
    pipeline = build_live_pipeline(
        subject,
        TypedFakeTranscriber(),
        selection=DashboardServiceSelection(False, True),
        availability=ArtifactAvailability(True),
        classifier_provider=RuntimeSetFitClassifier,
        integration=None,
    )
    factory = pipeline._coaching_coordinator_factory  # noqa: SLF001
    assert factory is not None
    call_state = CallState(tenant_id="tenant_alpha", call_id="synthetic-call")

    processor = factory(call_state)

    assert isinstance(processor, SafeCoachingProcessorAdapter)
    assert processor._coordinator.call_state is call_state  # noqa: SLF001


def test_disabled_demo_with_bundle_remains_base_only() -> None:
    subject = runtime()
    integration, runner, callbacks = integration_dependencies()
    pipeline = build_live_pipeline(
        subject,
        TypedFakeTranscriber(),
        selection=DashboardServiceSelection(False, True),
        availability=ArtifactAvailability(True),
        classifier_provider=RuntimeSetFitClassifier,
        integration=integration,
    )
    factory = pipeline._coaching_coordinator_factory  # noqa: SLF001
    assert factory is not None

    processor = factory(CallState(tenant_id="tenant_alpha", call_id="synthetic-call"))

    assert isinstance(processor, SafeCoachingProcessorAdapter)
    assert runner.calls == callbacks.id_calls == callbacks.clock_calls == 0


def test_enabled_bundle_maps_exact_dependencies_without_invocation() -> None:
    tenant = enabled_tenant()
    subject = create_local_execution(tenant, "synthetic-call").runtime
    integration, runner, callbacks = integration_dependencies()
    pipeline = build_live_pipeline(
        subject,
        TypedFakeTranscriber(),
        selection=DashboardServiceSelection(False, True),
        availability=ArtifactAvailability(True),
        classifier_provider=RuntimeSetFitClassifier,
        integration=integration,
    )
    factory = pipeline._coaching_coordinator_factory  # noqa: SLF001
    assert factory is not None
    call_state = CallState(tenant_id="tenant_alpha", call_id="call-one")

    processor = factory(call_state)

    assert isinstance(processor, RAGCoachingProcessorDecorator)
    coordinator = processor._coordinator  # noqa: SLF001
    suggestion_factory = processor._suggestion_factory  # noqa: SLF001
    assert isinstance(
        suggestion_factory,
        DeterministicLLMCoachingSuggestionFactory,
    )
    assert coordinator.call_state is call_state
    assert processor._base_processor._coordinator is coordinator  # noqa: SLF001
    assert processor._tenant_config is tenant.config  # noqa: SLF001
    assert processor._background_manager is runner  # noqa: SLF001
    assert processor._rag_llm_enabled_labels == (  # noqa: SLF001
        integration.policy.rag_llm_enabled_labels
    )
    assert suggestion_factory._title == integration.policy.title  # noqa: SLF001
    assert suggestion_factory._action is integration.policy.action  # noqa: SLF001
    assert suggestion_factory._priority is integration.policy.priority  # noqa: SLF001
    assert suggestion_factory._label_id == integration.policy.label_id  # noqa: SLF001
    assert suggestion_factory._expires_after_seconds == (  # noqa: SLF001
        integration.policy.expires_after_seconds
    )
    assert suggestion_factory._suggestion_id_factory == (  # noqa: SLF001
        integration.suggestion_id_factory
    )
    assert suggestion_factory._utc_datetime_factory == (  # noqa: SLF001
        integration.utc_datetime_factory
    )
    assert runner.calls == callbacks.id_calls == callbacks.clock_calls == 0


def test_factory_creates_isolated_structurally_compatible_processors() -> None:
    tenant = enabled_tenant()
    subject = create_local_execution(tenant, "synthetic-call").runtime
    integration, runner, callbacks = integration_dependencies()
    pipeline = build_live_pipeline(
        subject,
        TypedFakeTranscriber(),
        selection=DashboardServiceSelection(False, True),
        availability=ArtifactAvailability(True),
        classifier_provider=RuntimeSetFitClassifier,
        integration=integration,
    )
    factory = pipeline._coaching_coordinator_factory  # noqa: SLF001
    assert factory is not None
    first_state = CallState(tenant_id="tenant_alpha", call_id="call-one")
    second_state = CallState(tenant_id="tenant_alpha", call_id="call-two")

    first: CoachingProcessorProtocol = factory(first_state)
    second: CoachingProcessorProtocol = factory(second_state)

    assert isinstance(first, RAGCoachingProcessorDecorator)
    assert isinstance(second, RAGCoachingProcessorDecorator)
    assert first._coordinator is not second._coordinator  # noqa: SLF001
    assert first._coordinator.call_state is first_state  # noqa: SLF001
    assert second._coordinator.call_state is second_state  # noqa: SLF001
    assert runner.calls == callbacks.id_calls == callbacks.clock_calls == 0


def test_execution_resource_retains_exact_pipeline_and_completion_pump() -> None:
    tenant = enabled_tenant()
    subject = create_local_execution(tenant, "call-one").runtime
    integration, manager, callbacks = integration_dependencies()
    resource = DashboardExecutionResource(
        DashboardExecutionIdentity("tenant_alpha", "call-one"),
        integration=integration,
    )
    pipeline = build_live_pipeline(
        subject,
        TypedFakeTranscriber(),
        selection=DashboardServiceSelection(False, True),
        availability=ArtifactAvailability(True),
        classifier_provider=RuntimeSetFitClassifier,
        integration=integration,
        execution_resource=resource,
    )
    factory = pipeline._coaching_coordinator_factory  # noqa: SLF001
    assert factory is not None

    processor = factory(CallState(tenant_id="tenant_alpha", call_id="call-one"))

    assert isinstance(processor, RAGCoachingProcessorDecorator)
    assert resource._pipeline is pipeline  # noqa: SLF001
    assert resource._completion_pump is processor  # noqa: SLF001
    assert manager.calls == callbacks.id_calls == callbacks.clock_calls == 0


def test_execution_resource_scope_mismatch_fails_before_construction() -> None:
    subject = runtime()
    resource = DashboardExecutionResource(
        DashboardExecutionIdentity("tenant_alpha", "call-other"),
        integration=None,
    )

    with pytest.raises(ValueError, match="scope"):
        build_live_pipeline(
            subject,
            TypedFakeTranscriber(),
            selection=DashboardServiceSelection(False, True),
            availability=ArtifactAvailability(True),
            classifier_provider=RuntimeSetFitClassifier,
            execution_resource=resource,
        )
