from pathlib import Path
from typing import cast

from app.calls.models import CallState
from app.classification.runtime import RuntimeSetFitClassifier
from app.events.models import TranscriptEvent, TranscriptKind
from app.streaming.pipeline import WindowTranscriberProtocol
from live_dashboard.demo_data import tenant_demos
from live_dashboard.runtime_wiring import (
    ArtifactAvailability,
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
    coordinator = pipeline._coaching_coordinator_factory(  # noqa: SLF001
        CallState(tenant_id="tenant_alpha", call_id="synthetic-call")
    )
    rule_only = coordinator.process(
        TranscriptEvent(
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
        ),
        1,
        active_labels=(),
    )
    assert rule_only.displayed_suggestions
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
