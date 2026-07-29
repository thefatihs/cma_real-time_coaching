from collections.abc import Callable
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
import inspect
from typing import cast

import pytest

from app.calls.models import CallState
from app.coaching.coordinator import CoachingCoordinator
from app.coaching.rule_engine import RuleBasedCoachingEngine
from app.coaching.safe_processor import SafeCoachingProcessorAdapter
from app.composition import (
    BoundedPostgreSQLRAGManager,
    RAGOrchestrationCompletion,
    RAGOrchestrationIdentity,
    RAGOrchestrationSubmission,
)
from app.events.models import CoachingAction, SuggestionPriority
from app.integration import (
    CoachingSuggestionFactory,
    DeterministicLLMCoachingSuggestionFactory,
    OrchestrationRunner,
    RAGCoachingIntegrationDependencies,
    RAGCoachingIntegrationPolicy,
    RAGCoachingProcessorDecorator,
    compose_rag_coaching_processor,
)
from app.orchestration import OrchestrationRequest
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
        raise AssertionError("manager must not execute during composition")

    def submit(self, request: OrchestrationRequest) -> RAGOrchestrationSubmission:
        self.calls += 1
        raise AssertionError("manager must not execute during composition")

    def poll(
        self,
        identity: RAGOrchestrationIdentity,
    ) -> RAGOrchestrationCompletion | None:
        self.calls += 1
        raise AssertionError("manager must not execute during composition")


class Callback:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        return self.value


def tenant_config(
    *,
    tenant_id: str = "tenant_alpha",
    rag_enabled: bool = True,
    llm_enabled: bool = True,
    allowed_actions: list[str] | None = None,
) -> TenantConfig:
    return TenantConfig(
        context=TenantContext(tenant_id=tenant_id, tenant_name="Synthetic"),
        asr=TenantASRConfig(),
        classification=TenantClassificationConfig(
            model_id="synthetic-model",
            labels=["product_information"],
        ),
        rag=TenantRAGConfig(
            enabled=rag_enabled,
            knowledge_base_id="kb_synthetic" if rag_enabled else None,
            top_k=3,
            minimum_score=0.7,
        ),
        coaching=TenantCoachingConfig(
            enable_llm=llm_enabled,
            allowed_actions=allowed_actions or [CoachingAction.RAG_ACTION.value],
        ),
    )


def coordinator(
    config: TenantConfig,
) -> CoachingCoordinator:
    state = CallState(
        tenant_id=config.context.tenant_id,
        call_id="call_synthetic",
    )
    return CoachingCoordinator(
        config,
        state,
        RuleBasedCoachingEngine(
            config,
            (),
            event_id_factory=lambda: "base_suggestion",
            utc_datetime_factory=lambda: NOW,
        ),
    )


def policy(
    *,
    action: CoachingAction = CoachingAction.RAG_ACTION,
) -> RAGCoachingIntegrationPolicy:
    return RAGCoachingIntegrationPolicy(
        rag_llm_enabled_labels=("product_information",),
        title="Synthetic guidance",
        action=action,
        priority=SuggestionPriority.HIGH,
        label_id="product_information",
        expires_after_seconds=30.0,
    )


def dependencies(
    *,
    selected_policy: RAGCoachingIntegrationPolicy | None = None,
) -> tuple[
    RAGCoachingIntegrationDependencies,
    FakeBackgroundManager,
    Callback,
    Callback,
]:
    manager = FakeBackgroundManager()
    id_callback = Callback("llm_suggestion")
    clock_callback = Callback(NOW)
    subject = RAGCoachingIntegrationDependencies(
        background_manager=manager,
        policy=selected_policy or policy(),
        suggestion_id_factory=cast(Callable[[], str], id_callback),
        utc_datetime_factory=cast(Callable[[], datetime], clock_callback),
    )
    return subject, manager, id_callback, clock_callback


def test_bundle_is_frozen_and_slotted() -> None:
    subject, _, _, _ = dependencies()

    assert subject.__slots__ == (
        "background_manager",
        "policy",
        "suggestion_id_factory",
        "utc_datetime_factory",
    )
    with pytest.raises(FrozenInstanceError):
        subject.policy = policy()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("background_manager", object(), "background_manager"),
        ("suggestion_id_factory", object(), "suggestion_id_factory"),
        ("utc_datetime_factory", object(), "utc_datetime_factory"),
    ],
)
def test_bundle_rejects_non_callable_collaborators(
    field_name: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "background_manager": FakeBackgroundManager(),
        "policy": policy(),
        "suggestion_id_factory": lambda: "llm_suggestion",
        "utc_datetime_factory": lambda: NOW,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=message):
        RAGCoachingIntegrationDependencies(
            background_manager=cast(
                BoundedPostgreSQLRAGManager, values["background_manager"]
            ),
            policy=cast(RAGCoachingIntegrationPolicy, values["policy"]),
            suggestion_id_factory=cast(
                Callable[[], str], values["suggestion_id_factory"]
            ),
            utc_datetime_factory=cast(
                Callable[[], datetime], values["utc_datetime_factory"]
            ),
        )


def test_bundle_construction_does_not_invoke_collaborators() -> None:
    _, runner, id_callback, clock_callback = dependencies()

    assert runner.calls == id_callback.calls == clock_callback.calls == 0


def test_none_integration_returns_base_adapter() -> None:
    config = tenant_config()
    subject_coordinator = coordinator(config)

    result = compose_rag_coaching_processor(
        coordinator=subject_coordinator,
        tenant_config=config,
        integration=None,
    )

    assert isinstance(result, SafeCoachingProcessorAdapter)
    assert result._coordinator is subject_coordinator  # noqa: SLF001


@pytest.mark.parametrize(
    ("rag_enabled", "llm_enabled"),
    [(False, True), (True, False)],
)
def test_disabled_feature_returns_base_and_ignores_disallowed_action(
    rag_enabled: bool,
    llm_enabled: bool,
) -> None:
    config = tenant_config(
        rag_enabled=rag_enabled,
        llm_enabled=llm_enabled,
        allowed_actions=[CoachingAction.NO_ACTION.value],
    )
    subject_coordinator = coordinator(config)
    bundle, runner, id_callback, clock_callback = dependencies()

    result = compose_rag_coaching_processor(
        coordinator=subject_coordinator,
        tenant_config=config,
        integration=bundle,
    )

    assert isinstance(result, SafeCoachingProcessorAdapter)
    assert result._coordinator is subject_coordinator  # noqa: SLF001
    assert runner.calls == id_callback.calls == clock_callback.calls == 0


def test_enabled_composition_maps_exact_objects_and_policy() -> None:
    config = tenant_config()
    subject_coordinator = coordinator(config)
    bundle, runner, id_callback, clock_callback = dependencies()

    first = compose_rag_coaching_processor(
        coordinator=subject_coordinator,
        tenant_config=config,
        integration=bundle,
    )
    second = compose_rag_coaching_processor(
        coordinator=subject_coordinator,
        tenant_config=config,
        integration=bundle,
    )

    assert isinstance(first, RAGCoachingProcessorDecorator)
    assert isinstance(second, RAGCoachingProcessorDecorator)
    assert first._coordinator is subject_coordinator  # noqa: SLF001
    assert first._base_processor._coordinator is subject_coordinator  # noqa: SLF001
    assert first._tenant_config is config  # noqa: SLF001
    assert first._background_manager is runner  # noqa: SLF001
    assert first._rag_llm_enabled_labels == bundle.policy.rag_llm_enabled_labels  # noqa: SLF001
    factory = first._suggestion_factory  # noqa: SLF001
    assert isinstance(factory, DeterministicLLMCoachingSuggestionFactory)
    assert factory._title == bundle.policy.title  # noqa: SLF001
    assert factory._action is bundle.policy.action  # noqa: SLF001
    assert factory._priority is bundle.policy.priority  # noqa: SLF001
    assert factory._label_id == bundle.policy.label_id  # noqa: SLF001
    assert factory._expires_after_seconds == (  # noqa: SLF001
        bundle.policy.expires_after_seconds
    )
    assert factory._suggestion_id_factory is bundle.suggestion_id_factory  # noqa: SLF001
    assert factory._utc_datetime_factory is bundle.utc_datetime_factory  # noqa: SLF001
    assert runner.calls == id_callback.calls == clock_callback.calls == 0


def test_enabled_tenant_mismatch_fails_before_callbacks() -> None:
    config = tenant_config()
    mismatched = coordinator(tenant_config(tenant_id="tenant_beta"))
    bundle, runner, id_callback, clock_callback = dependencies()

    with pytest.raises(ValueError, match="tenant_id"):
        compose_rag_coaching_processor(
            coordinator=mismatched,
            tenant_config=config,
            integration=bundle,
        )

    assert runner.calls == id_callback.calls == clock_callback.calls == 0


def test_enabled_disallowed_action_fails_before_callbacks() -> None:
    config = tenant_config(allowed_actions=[CoachingAction.NO_ACTION.value])
    bundle, runner, id_callback, clock_callback = dependencies()

    with pytest.raises(ValueError, match="not allowed"):
        compose_rag_coaching_processor(
            coordinator=coordinator(config),
            tenant_config=config,
            integration=bundle,
        )

    assert runner.calls == id_callback.calls == clock_callback.calls == 0


def test_enabled_missing_knowledge_base_fails_defensively() -> None:
    config = tenant_config()
    invalid_rag = TenantRAGConfig.model_construct(
        enabled=True,
        knowledge_base_id=None,
        top_k=3,
        minimum_score=0.7,
    )
    config = config.model_copy(update={"rag": invalid_rag})
    bundle, runner, id_callback, clock_callback = dependencies()

    with pytest.raises(ValueError, match="knowledge_base_id"):
        compose_rag_coaching_processor(
            coordinator=coordinator(config),
            tenant_config=config,
            integration=bundle,
        )

    assert runner.calls == id_callback.calls == clock_callback.calls == 0


def test_helper_has_no_separate_adapter_or_factory_parameters() -> None:
    assert tuple(inspect.signature(compose_rag_coaching_processor).parameters) == (
        "coordinator",
        "tenant_config",
        "integration",
    )


def test_return_types_are_structurally_streaming_compatible() -> None:
    def accepts_processor(processor: CoachingProcessorProtocol) -> None:
        assert processor is not None

    config = tenant_config()
    subject_coordinator = coordinator(config)
    bundle, _, _, _ = dependencies()
    accepts_processor(
        compose_rag_coaching_processor(
            coordinator=subject_coordinator,
            tenant_config=config,
            integration=None,
        )
    )
    accepts_processor(
        compose_rag_coaching_processor(
            coordinator=subject_coordinator,
            tenant_config=config,
            integration=bundle,
        )
    )


def test_existing_integration_exports_remain_available() -> None:
    assert CoachingSuggestionFactory is not None
    assert DeterministicLLMCoachingSuggestionFactory is not None
    assert OrchestrationRunner is not None
    assert RAGCoachingIntegrationPolicy is not None
    assert RAGCoachingProcessorDecorator is not None
