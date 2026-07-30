from datetime import UTC, datetime
import logging
from typing import cast

import pytest

from app.calls.models import CallState
from app.classification.streaming import (
    ClassificationProcessingStatus,
    StableClassificationOutcome,
)
from app.coaching.coordinator import (
    CoachingCoordinator,
    CoachingCoordinatorResult,
    CoachingProcessingStatus,
    StableCoachingOutcome,
)
from app.coaching.rule_engine import CoachingRule, RuleBasedCoachingEngine
from app.coaching.safe_processor import SafeCoachingProcessorAdapter
from app.events.models import (
    ClassificationLabel,
    ClassificationResultEvent,
    CoachingAction,
    CoachingSuggestionEvent,
    CoachingSuggestionLifecycle,
    CoachingSuggestionSource,
    SuggestionPriority,
    TranscriptEvent,
    TranscriptKind,
)
from app.tenancy.models import (
    TenantASRConfig,
    TenantClassificationConfig,
    TenantCoachingConfig,
    TenantConfig,
    TenantContext,
    TenantRAGConfig,
)
from app.streaming.pipeline import StreamingASRPipeline

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)
PRIVATE_SENTINEL = "PRIVATE_SYNTHETIC_SENTINEL"
PATH_SENTINEL = "PRIVATE_PATH_SENTINEL"


def tenant_config() -> TenantConfig:
    return TenantConfig(
        context=TenantContext(tenant_id="tenant_alpha", tenant_name="Synthetic"),
        asr=TenantASRConfig(),
        classification=TenantClassificationConfig(
            model_id="synthetic-model",
            labels=["complaint"],
        ),
        rag=TenantRAGConfig(enabled=False),
        coaching=TenantCoachingConfig(
            cooldown_seconds=20,
            max_active_suggestions=2,
            allowed_actions=[action.value for action in CoachingAction],
        ),
    )


def transcript(
    revision: int = 1,
    *,
    tenant_id: str = "tenant_alpha",
    call_id: str = "call_001",
    kind: TranscriptKind = TranscriptKind.STABLE,
) -> TranscriptEvent:
    return TranscriptEvent(
        tenant_id=tenant_id,
        call_id=call_id,
        event_id=f"transcript_{revision}",
        kind=kind,
        text=f"Synthetic complaint {revision}.",
        start_seconds=float(revision - 1),
        end_seconds=float(revision),
        revision=revision,
        created_at_utc=NOW,
    )


def rule() -> CoachingRule:
    return CoachingRule(
        rule_id="complaint_rule",
        label="complaint",
        include_any=("complaint",),
        action=CoachingAction.TEMPLATE_ACTION,
        priority=SuggestionPriority.HIGH,
        title="Synthetic guidance",
        suggestion="Use synthetic guidance.",
    )


def coordinator(
    state: CallState,
    coordinator_type: type[CoachingCoordinator] = CoachingCoordinator,
) -> CoachingCoordinator:
    config = tenant_config()
    return coordinator_type(
        config,
        state,
        RuleBasedCoachingEngine(
            config,
            (rule(),),
            event_id_factory=lambda: f"suggestion_{state.transcript_revision}",
            utc_datetime_factory=lambda: NOW,
        ),
    )


def apply_current(state: CallState, event: TranscriptEvent) -> None:
    state.apply_transcript(event)


def classification(
    event: TranscriptEvent,
    *,
    tenant_id: str | None = None,
    call_id: str | None = None,
) -> ClassificationResultEvent:
    return ClassificationResultEvent(
        tenant_id=tenant_id or event.tenant_id,
        call_id=call_id or event.call_id,
        transcript_event_id=event.event_id,
        labels=[ClassificationLabel(name="complaint", score=0.8)],
        action=CoachingAction.TEMPLATE_ACTION,
        model_id="synthetic-model",
        provisional=event.kind is TranscriptKind.PARTIAL,
        created_at_utc=NOW,
    )


def suggestion(
    event: TranscriptEvent,
    *,
    tenant_id: str | None = None,
    call_id: str | None = None,
) -> CoachingSuggestionEvent:
    return CoachingSuggestionEvent(
        tenant_id=tenant_id or event.tenant_id,
        call_id=call_id or event.call_id,
        suggestion_id="synthetic_suggestion",
        source_transcript_event_id=event.event_id,
        action=CoachingAction.TEMPLATE_ACTION,
        priority=SuggestionPriority.HIGH,
        source=CoachingSuggestionSource.RULE,
        label_id="complaint",
        title="Synthetic guidance",
        suggestion="Use synthetic guidance.",
        created_at_utc=NOW,
    )


def processed_outcome(
    event: TranscriptEvent,
    *,
    classification_event: ClassificationResultEvent | None = None,
    displayed: tuple[CoachingSuggestionEvent, ...] = (),
    revision: int | None = None,
) -> StableCoachingOutcome:
    actual_revision = event.revision if revision is None else revision
    return StableCoachingOutcome(
        status=CoachingProcessingStatus.PROCESSED,
        transcript_revision=actual_revision,
        result=CoachingCoordinatorResult(
            classification_event=classification_event,
            displayed_suggestions=displayed,
            suppressed_suggestions=(),
            matched_rule_ids=(),
            suppression_reasons=(),
            transcript_revision=actual_revision,
        ),
    )


class RaiseAfterMutationCoordinator(CoachingCoordinator):
    def process_safely(
        self,
        event: TranscriptEvent,
        current_seconds: float,
        *,
        classification_event: ClassificationResultEvent | None = None,
        active_labels: tuple[str, ...] | None = None,
    ) -> StableCoachingOutcome:
        super().process_safely(
            event,
            current_seconds,
            classification_event=classification_event,
            active_labels=active_labels,
        )
        raise RuntimeError(f"{PRIVATE_SENTINEL} {PATH_SENTINEL}")


def test_provisional_failure_rolls_back_separate_lifecycle_state() -> None:
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    event = transcript(kind=TranscriptKind.PARTIAL)
    apply_current(state, event)
    subject = coordinator(state, RaiseAfterMutationCoordinator)
    before = subject.snapshot_coaching_state()

    outcome = SafeCoachingProcessorAdapter(subject).process_safely(
        event,
        event.end_seconds,
        classification_event=classification(event),
        active_labels=("complaint",),
    )

    assert outcome.status is CoachingProcessingStatus.FAILED
    assert subject.snapshot_coaching_state() == before
    assert state.active_coaching_suggestions == []
    assert not any(
        item.lifecycle is CoachingSuggestionLifecycle.PROVISIONAL
        for item in state.coaching_suggestions
    )


class ReturningCoordinator(CoachingCoordinator):
    returned: object = None

    def process_safely(
        self,
        event: TranscriptEvent,
        current_seconds: float,
        *,
        classification_event: ClassificationResultEvent | None = None,
        active_labels: tuple[str, ...] | None = None,
    ) -> StableCoachingOutcome:
        del event, current_seconds, classification_event, active_labels
        return cast(StableCoachingOutcome, self.returned)


def test_valid_coordinator_result_is_returned_unchanged() -> None:
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    event = transcript()
    apply_current(state, event)
    adapter = SafeCoachingProcessorAdapter(coordinator(state))

    outcome = adapter.process_safely(event, event.end_seconds)

    assert outcome.status is CoachingProcessingStatus.PROCESSED
    assert outcome.result is not None
    assert outcome.result.displayed_suggestions
    assert state.active_coaching_suggestions


def test_delegate_exception_rolls_back_all_coaching_bookkeeping(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    first = transcript()
    apply_current(state, first)
    subject = coordinator(state, RaiseAfterMutationCoordinator)
    before = subject.snapshot_coaching_state()
    adapter = SafeCoachingProcessorAdapter(subject)

    with caplog.at_level(logging.ERROR):
        outcome = adapter.process_safely(first, first.end_seconds)

    assert outcome == StableCoachingOutcome(
        status=CoachingProcessingStatus.FAILED,
        transcript_revision=1,
        error_type="SafeCoachingProcessorFailure",
        error_code="delegate_exception",
    )
    assert subject.snapshot_coaching_state() == before
    assert state.stable_transcript == first.text
    logged = caplog.text
    assert PRIVATE_SENTINEL not in logged
    assert PATH_SENTINEL not in logged
    assert first.text not in logged


def test_existing_active_and_history_cards_survive_failed_mutation() -> None:
    class ToggleFailureCoordinator(CoachingCoordinator):
        should_fail = False

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
            if self.should_fail:
                raise RuntimeError("synthetic failure")
            return outcome

    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    subject = coordinator(state, ToggleFailureCoordinator)
    adapter = SafeCoachingProcessorAdapter(subject)
    first = transcript()
    apply_current(state, first)
    seeded = adapter.process_safely(first, 1.0)
    assert seeded.result is not None
    active = state.active_coaching_suggestions[0]
    state.coaching_suggestion_history = [
        active.model_copy(update={"suggestion_id": "historical_suggestion"})
    ]
    before = subject.snapshot_coaching_state()

    cast(ToggleFailureCoordinator, subject).should_fail = True
    second = transcript(2)
    apply_current(state, second)
    failed = adapter.process_safely(second, 2.0)

    assert failed.status is CoachingProcessingStatus.FAILED
    assert subject.snapshot_coaching_state() == before
    assert state.transcript_revision == 2


@pytest.mark.parametrize("returned", [None, {}, object()])
def test_invalid_result_types_are_rejected(returned: object) -> None:
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    event = transcript()
    apply_current(state, event)
    subject = coordinator(state, ReturningCoordinator)
    cast(ReturningCoordinator, subject).returned = returned
    before = subject.snapshot_coaching_state()

    outcome = SafeCoachingProcessorAdapter(subject).process_safely(
        event,
        event.end_seconds,
    )

    assert outcome.status is CoachingProcessingStatus.FAILED
    assert outcome.error_code == "invalid_result_type"
    assert subject.snapshot_coaching_state() == before


@pytest.mark.parametrize(
    "returned",
    [
        processed_outcome(
            transcript(),
            classification_event=classification(
                transcript(),
                tenant_id="tenant_beta",
            ),
        ),
        processed_outcome(
            transcript(),
            classification_event=classification(
                transcript(),
                call_id="call_002",
            ),
        ),
        processed_outcome(transcript(), revision=2),
        processed_outcome(
            transcript(),
            displayed=(suggestion(transcript(), tenant_id="tenant_beta"),),
        ),
        processed_outcome(
            transcript(),
            displayed=(suggestion(transcript(), call_id="call_002"),),
        ),
    ],
)
def test_scope_and_revision_mismatches_fail_closed(
    returned: StableCoachingOutcome,
) -> None:
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    event = transcript()
    apply_current(state, event)
    subject = coordinator(state, ReturningCoordinator)
    cast(ReturningCoordinator, subject).returned = returned
    before = subject.snapshot_coaching_state()

    outcome = SafeCoachingProcessorAdapter(subject).process_safely(
        event,
        event.end_seconds,
    )

    assert outcome.status is CoachingProcessingStatus.FAILED
    assert outcome.error_code in {"scope_mismatch", "result_validation_failed"}
    assert subject.snapshot_coaching_state() == before


def test_input_scope_is_checked_against_trusted_call_state() -> None:
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    event = transcript(tenant_id="tenant_beta")
    subject = coordinator(state, ReturningCoordinator)

    outcome = SafeCoachingProcessorAdapter(subject).process_safely(event, 1.0)

    assert outcome.status is CoachingProcessingStatus.FAILED
    assert outcome.error_code == "scope_mismatch"


def test_failed_attempt_does_not_consume_revision_or_break_later_success() -> None:
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    event = transcript()
    apply_current(state, event)
    failing = coordinator(state, RaiseAfterMutationCoordinator)
    failed = SafeCoachingProcessorAdapter(failing).process_safely(event, 1.0)
    assert failed.status is CoachingProcessingStatus.FAILED

    healthy = coordinator(state)
    succeeded = SafeCoachingProcessorAdapter(healthy).process_safely(event, 1.0)

    assert succeeded.status is CoachingProcessingStatus.PROCESSED
    assert succeeded.result is not None
    assert succeeded.result.displayed_suggestions


def test_pipeline_coaching_path_continues_with_next_event_after_failure() -> None:
    class FailOnceCoordinator(CoachingCoordinator):
        failed = False

        def process_safely(
            self,
            event: TranscriptEvent,
            current_seconds: float,
            *,
            classification_event: ClassificationResultEvent | None = None,
            active_labels: tuple[str, ...] | None = None,
        ) -> StableCoachingOutcome:
            if not self.failed:
                self.failed = True
                raise RuntimeError("synthetic failure")
            return super().process_safely(
                event,
                current_seconds,
                classification_event=classification_event,
                active_labels=active_labels,
            )

    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    adapter = SafeCoachingProcessorAdapter(coordinator(state, FailOnceCoordinator))
    outcomes: list[StableCoachingOutcome] = []
    for revision in (1, 2):
        event = transcript(revision)
        apply_current(state, event)
        outcome = StreamingASRPipeline._process_coaching(
            coordinator=adapter,
            event=event,
            stable_changed=True,
            classification_outcome=StableClassificationOutcome(
                ClassificationProcessingStatus.DISABLED,
                revision,
                event.source_chunk_sequence,
            ),
        )
        assert outcome is not None
        outcomes.append(outcome)

    assert [outcome.status for outcome in outcomes] == [
        CoachingProcessingStatus.FAILED,
        CoachingProcessingStatus.PROCESSED,
    ]
    assert outcomes[1].result is not None
    assert outcomes[1].result.displayed_suggestions


def test_base_exceptions_are_not_swallowed() -> None:
    class InterruptingCoordinator(CoachingCoordinator):
        def process_safely(
            self,
            event: TranscriptEvent,
            current_seconds: float,
            *,
            classification_event: ClassificationResultEvent | None = None,
            active_labels: tuple[str, ...] | None = None,
        ) -> StableCoachingOutcome:
            del event, current_seconds, classification_event, active_labels
            raise KeyboardInterrupt

    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    event = transcript()
    apply_current(state, event)
    adapter = SafeCoachingProcessorAdapter(coordinator(state, InterruptingCoordinator))

    with pytest.raises(KeyboardInterrupt):
        adapter.process_safely(event, 1.0)
