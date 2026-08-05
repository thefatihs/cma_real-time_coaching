from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, get_ident

import av
import pytest

from app.audio_ingress.local_microphone import (
    LOCAL_MIC_CHUNK_BYTES,
    LOCAL_MIC_GATE_ENVIRONMENT_KEY,
    LocalMicrophoneASRReadiness,
    LocalMicrophoneIngressSession,
    _NormalizedAudio,
    create_local_mic_test_capability,
)
from app.classification.streaming import (
    ClassificationProcessingStatus,
    ProvisionalClassificationPolicy,
    RuntimeClassifierProtocol,
    StableClassificationOutcome,
    StableTranscriptClassificationStage,
)
from app.calls.models import CallState
from app.coaching.coordinator import (
    CoachingCoordinator,
    CoachingCoordinatorResult,
    CoachingProcessingStatus,
    StableCoachingOutcome,
)
from app.coaching.rule_engine import CoachingRule, RuleBasedCoachingEngine
from app.events.models import (
    AudioChunkEvent,
    ClassificationLabel,
    ClassificationResultEvent,
    CoachingAction,
    CoachingSuggestionLifecycle,
    CoachingSuggestionSource,
    SuggestionPriority,
    TranscriptEvent,
    TranscriptKind,
)
from app.events.labels import ClassificationViewSource
from app.diarization.models import SpeakerRole
from app.diarization.role_resolver import RoleEvidenceCode
from app.diarization.routing import (
    CustomerProjectionReason,
    CustomerProjectionStatus,
    CustomerSpeechProjection,
    RoleTaggedWord,
)
from app.streaming.customer_routing import (
    CustomerProjectionProviderProtocol,
    CustomerRoutingStatus,
)
from app.streaming.audio_window import ASRAudioWindow
from app.streaming.pipeline import (
    CoachingCoordinatorFactory,
    CoachingProcessorProtocol,
    StreamingASRPipeline,
    StreamingASRPlan,
    StreamingASRStep,
)
from app.streaming.window_transcriber import (
    WindowTranscriptionResult,
    WindowTranscriptionSegment,
)
from app.tenancy.models import (
    TenantASRConfig,
    TenantClassificationConfig,
    TenantCoachingConfig,
    TenantConfig,
    TenantContext,
    TenantRAGConfig,
)


NOW = datetime(2026, 7, 22, tzinfo=UTC)


def chunk(
    sequence: int, start: float, duration: float, sample_rate_hz: int = 10
) -> AudioChunkEvent:
    return AudioChunkEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        sequence_number=sequence,
        received_at_utc=NOW,
        chunk_start_seconds=start,
        chunk_duration_seconds=duration,
        sample_rate_hz=sample_rate_hz,
        channel_count=1,
        codec_name="pcm_s16le",
        audio_bytes=bytes([sequence + 1, 0]) * round(duration * sample_rate_hz),
    )


def transcription(
    window: ASRAudioWindow,
    *segments: tuple[str, float, float],
    tenant_id: str | None = None,
    call_id: str | None = None,
) -> WindowTranscriptionResult:
    result_segments = tuple(
        WindowTranscriptionSegment(
            text, start - window.start_seconds, end - window.start_seconds, start, end
        )
        for text, start, end in segments
    )
    return WindowTranscriptionResult(
        tenant_id=tenant_id or window.tenant_id,
        call_id=call_id or window.call_id,
        first_sequence=window.first_sequence,
        last_sequence=window.last_sequence,
        window_start_seconds=window.start_seconds,
        window_end_seconds=window.end_seconds,
        window_duration_seconds=window.duration_seconds,
        text=" ".join(item[0] for item in segments),
        detected_language="tr",
        language_probability=0.9,
        processing_time_seconds=0.05,
        segments=result_segments,
    )


class FakeTranscriber:
    def __init__(self, responses: list[list[tuple[str, float, float]]]) -> None:
        self.responses = responses
        self.windows: list[ASRAudioWindow] = []
        self.scope_override: dict[str, str] = {}

    def transcribe(self, window: ASRAudioWindow) -> WindowTranscriptionResult:
        self.windows.append(window)
        response = self.responses[len(self.windows) - 1]
        return transcription(window, *response, **self.scope_override)


def context() -> TenantContext:
    return TenantContext(tenant_id="tenant_alpha", tenant_name="Alpha")


def config() -> TenantASRConfig:
    return TenantASRConfig(
        chunk_duration_seconds=2.0,
        rolling_window_seconds=6.0,
        stable_region_seconds=2.0,
    )


def generator_for(source: list[AudioChunkEvent], calls: list[object] | None = None):
    def generate(path: Path, tenant: str, call: str, duration: float):
        if calls is not None:
            calls.append((path, tenant, call, duration))
        return iter(source)

    return generate


def pipeline(
    source: list[AudioChunkEvent],
    transcriber: FakeTranscriber,
    calls: list[object] | None = None,
    runtime_classifier: RuntimeClassifierProtocol | None = None,
    coaching_factory: CoachingCoordinatorFactory | None = None,
    customer_only_classification_enabled: bool = False,
    customer_projection_provider: CustomerProjectionProviderProtocol | None = None,
) -> StreamingASRPipeline:
    return StreamingASRPipeline(
        context(),
        config(),
        transcriber,
        chunk_generator=generator_for(source, calls),
        runtime_classifier=runtime_classifier,
        coaching_coordinator_factory=coaching_factory,
        customer_only_classification_enabled=customer_only_classification_enabled,
        customer_projection_provider=customer_projection_provider,
    )


def coaching_factory(
    *,
    phrase: str = "first",
    label: str = "complaint",
) -> CoachingCoordinatorFactory:
    tenant = TenantConfig(
        context=context(),
        asr=config(),
        classification=TenantClassificationConfig(
            model_id="common_turkish_setfit_v2",
            labels=sorted({"complaint", label}),
        ),
        rag=TenantRAGConfig(enabled=False),
        coaching=TenantCoachingConfig(
            cooldown_seconds=0,
            max_active_suggestions=2,
            allowed_actions=[action.value for action in CoachingAction],
        ),
    )
    rule = CoachingRule(
        rule_id="safe_rule",
        label=label,
        include_any=(phrase,),
        action=CoachingAction.TEMPLATE_ACTION,
        priority=SuggestionPriority.HIGH,
        title="Synthetic guidance",
        suggestion="Apply the synthetic guidance.",
    )

    def create(state: CallState) -> CoachingCoordinator:
        return CoachingCoordinator(
            tenant,
            state,
            RuleBasedCoachingEngine(
                tenant,
                (rule,),
                event_id_factory=lambda: f"suggestion-{state.transcript_revision}",
                utc_datetime_factory=lambda: NOW,
            ),
        )

    return create


class FakeCoachingProcessor:
    def __init__(self) -> None:
        self.calls: list[
            tuple[
                TranscriptEvent,
                float,
                ClassificationResultEvent | None,
                tuple[str, ...] | None,
            ]
        ] = []
        self.outcomes: list[StableCoachingOutcome] = []

    def process_safely(
        self,
        event: TranscriptEvent,
        current_seconds: float,
        *,
        classification_event: ClassificationResultEvent | None = None,
        active_labels: tuple[str, ...] | None = None,
    ) -> StableCoachingOutcome:
        self.calls.append((event, current_seconds, classification_event, active_labels))
        outcome = StableCoachingOutcome(
            status=CoachingProcessingStatus.PROCESSED,
            transcript_revision=event.revision,
            result=CoachingCoordinatorResult(
                classification_event=classification_event,
                displayed_suggestions=(),
                suppressed_suggestions=(),
                matched_rule_ids=(),
                suppression_reasons=(),
                transcript_revision=event.revision,
                current_revision_labels=active_labels or (),
            ),
        )
        self.outcomes.append(outcome)
        return outcome


class FakeCompletionPump(FakeCoachingProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.drain_calls: list[tuple[float, int]] = []

    def drain_completed(
        self,
        *,
        current_seconds: float,
    ) -> tuple[StableCoachingOutcome, ...]:
        self.drain_calls.append((current_seconds, get_ident()))
        outcome = StableCoachingOutcome(
            status=CoachingProcessingStatus.PROCESSED,
            transcript_revision=len(self.drain_calls),
        )
        return (outcome,)


class FakeRuntimeClassifier:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def classify(
        self,
        *,
        tenant_id: str,
        call_id: str,
        text: str,
        transcript_event_id: str | None = None,
        revision: int | None = None,
        sequence_number: int | None = None,
    ) -> ClassificationResultEvent:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "call_id": call_id,
                "text": text,
                "transcript_event_id": transcript_event_id,
                "revision": revision,
                "sequence_number": sequence_number,
            }
        )
        if self.fail:
            raise RuntimeError("synthetic classifier failure")
        return ClassificationResultEvent(
            tenant_id=tenant_id,
            call_id=call_id,
            transcript_event_id=transcript_event_id or "runtime",
            labels=[ClassificationLabel(name="complaint", score=0.8)],
            action=CoachingAction.TEMPLATE_ACTION,
            model_id="common_turkish_setfit_v2",
            threshold_profile_id="common_turkish_setfit_v2:calibrated:v1",
            probabilities={"complaint": 0.8},
            thresholds={"complaint": 0.45},
            processing_time_ms=4.0,
            created_at_utc=NOW,
        )


def partial_event(
    text: str,
    *,
    revision: int = 1,
    sequence: int = 0,
) -> TranscriptEvent:
    return TranscriptEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        event_id=f"partial-{revision}",
        kind=TranscriptKind.PARTIAL,
        text=text,
        start_seconds=0.0,
        end_seconds=1.0,
        revision=revision,
        created_at_utc=NOW,
        source_chunk_sequence=sequence,
    )


def test_provisional_classification_is_opt_in_and_does_not_commit_revision() -> None:
    classifier = FakeRuntimeClassifier()
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    event = partial_event("synthetic complaint now")
    state.apply_transcript(event)
    disabled = StableTranscriptClassificationStage(classifier)

    skipped = disabled.process(
        event,
        cumulative_stable_transcript="",
        stable_changed=False,
        call_state=state,
    )

    assert skipped.status is ClassificationProcessingStatus.PARTIAL_SKIPPED
    assert classifier.calls == []
    assert state.classification_transcript_revision is None


def test_meaningful_partial_uses_stricter_threshold_without_committing() -> None:
    classifier = FixedLabelClassifier("complaint")
    clock_values = iter((10.0, 10.5, 11.1))
    stage = StableTranscriptClassificationStage(
        classifier,
        provisional_policy=ProvisionalClassificationPolicy(enabled=True),
        monotonic_clock=lambda: next(clock_values),
    )
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    first = partial_event("synthetic complaint now", revision=1, sequence=1)
    state.apply_transcript(first)

    accepted = stage.process(
        first,
        cumulative_stable_transcript="",
        stable_changed=False,
        call_state=state,
    )
    duplicate_chunk = stage.process(
        first.model_copy(update={"event_id": "partial-same-chunk"}),
        cumulative_stable_transcript="",
        stable_changed=False,
        call_state=state,
    )
    too_frequent = stage.process(
        partial_event(
            "synthetic complaint now please",
            revision=2,
            sequence=2,
        ),
        cumulative_stable_transcript="",
        stable_changed=False,
        call_state=state,
    )
    accepted_later = stage.process(
        partial_event(
            "synthetic complaint now please help",
            revision=3,
            sequence=3,
        ),
        cumulative_stable_transcript="",
        stable_changed=False,
        call_state=state,
    )

    assert accepted.status is ClassificationProcessingStatus.PROVISIONAL_CLASSIFIED
    assert accepted.provisional
    assert accepted.classification_event is not None
    assert [label.name for label in accepted.classification_event.labels] == [
        "complaint"
    ]
    assert duplicate_chunk.status is ClassificationProcessingStatus.PARTIAL_SKIPPED
    assert too_frequent.status is ClassificationProcessingStatus.PARTIAL_SKIPPED
    assert (
        accepted_later.status is ClassificationProcessingStatus.PROVISIONAL_CLASSIFIED
    )
    assert state.classification_transcript_revision is None


def test_media_progress_cadence_ignores_accelerated_wall_clock() -> None:
    classifier = FixedLabelClassifier("complaint")
    stage = StableTranscriptClassificationStage(
        classifier,
        provisional_policy=ProvisionalClassificationPolicy(enabled=True),
        monotonic_clock=lambda: 0.0,
    )
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")

    outcomes = []
    for revision, media_progress in enumerate((2.0, 4.0, 6.0), start=1):
        event = partial_event(
            f"synthetic complaint now revision {revision}",
            revision=revision,
            sequence=revision,
        )
        state.apply_transcript(event)
        outcomes.append(
            stage.process(
                event,
                cumulative_stable_transcript="",
                stable_changed=False,
                call_state=state,
                media_progress_seconds=media_progress,
            )
        )

    assert all(
        outcome.status is ClassificationProcessingStatus.PROVISIONAL_CLASSIFIED
        for outcome in outcomes
    )
    assert len(classifier.calls) == 6


@pytest.mark.parametrize(
    "invalid_progress",
    [float("nan"), float("inf"), float("-inf"), -1.0],
)
def test_invalid_media_progress_fails_closed(invalid_progress: float) -> None:
    classifier = FixedLabelClassifier("complaint")
    stage = StableTranscriptClassificationStage(
        classifier,
        provisional_policy=ProvisionalClassificationPolicy(enabled=True),
    )
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    event = partial_event("synthetic complaint now")

    outcome = stage.process(
        event,
        cumulative_stable_transcript="",
        stable_changed=False,
        call_state=state,
        media_progress_seconds=invalid_progress,
    )

    assert outcome.status is ClassificationProcessingStatus.PARTIAL_SKIPPED
    assert classifier.calls == []


def test_repeated_and_regressing_media_progress_fail_closed() -> None:
    classifier = FixedLabelClassifier("complaint")
    stage = StableTranscriptClassificationStage(
        classifier,
        provisional_policy=ProvisionalClassificationPolicy(enabled=True),
    )
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")

    statuses = []
    for revision, media_progress in enumerate(
        (4.0, 4.0, 3.0, 4.5, 5.0),
        start=1,
    ):
        event = partial_event(
            f"synthetic complaint now revision {revision}",
            revision=revision,
            sequence=revision,
        )
        state.apply_transcript(event)
        statuses.append(
            stage.process(
                event,
                cumulative_stable_transcript="",
                stable_changed=False,
                call_state=state,
                media_progress_seconds=media_progress,
            ).status
        )

    assert statuses == [
        ClassificationProcessingStatus.PROVISIONAL_CLASSIFIED,
        ClassificationProcessingStatus.PARTIAL_SKIPPED,
        ClassificationProcessingStatus.PARTIAL_SKIPPED,
        ClassificationProcessingStatus.PARTIAL_SKIPPED,
        ClassificationProcessingStatus.PROVISIONAL_CLASSIFIED,
    ]
    assert len(classifier.calls) == 4


@pytest.mark.parametrize("text", ["one", "one two"])
def test_short_partial_is_skipped(text: str) -> None:
    classifier = FakeRuntimeClassifier()
    stage = StableTranscriptClassificationStage(
        classifier,
        provisional_policy=ProvisionalClassificationPolicy(enabled=True),
    )
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    event = partial_event(text)

    outcome = stage.process(
        event,
        cumulative_stable_transcript="",
        stable_changed=False,
        call_state=state,
    )

    assert outcome.status is ClassificationProcessingStatus.PARTIAL_SKIPPED
    assert classifier.calls == []


def test_provisional_threshold_rejects_committed_only_confidence() -> None:
    classifier = FakeRuntimeClassifier()
    stage = StableTranscriptClassificationStage(
        classifier,
        provisional_policy=ProvisionalClassificationPolicy(enabled=True),
    )
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    event = partial_event("synthetic complaint now")

    outcome = stage.process(
        event,
        cumulative_stable_transcript="",
        stable_changed=False,
        call_state=state,
    )

    assert outcome.classification_event is not None
    assert outcome.classification_event.labels == []
    assert outcome.classification_event.action is CoachingAction.NO_ACTION


def test_customer_only_scope_keeps_provisional_classification_disabled() -> None:
    classifier = FixedLabelClassifier("complaint")
    stage = StableTranscriptClassificationStage(
        classifier,
        provisional_policy=ProvisionalClassificationPolicy(enabled=True),
    )
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    event = partial_event("synthetic complaint now")

    outcome = stage.process(
        event,
        cumulative_stable_transcript="",
        stable_changed=False,
        call_state=state,
        allow_provisional=False,
    )

    assert outcome.status is ClassificationProcessingStatus.PARTIAL_SKIPPED
    assert classifier.calls == []


class StreamingProjectionProvider:
    def __init__(
        self,
        text: str,
        *,
        fail: bool = False,
    ) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[tuple[str, str, int]] = []

    def get_projection(
        self,
        *,
        tenant_id: str,
        call_id: str,
        transcript_revision: int,
    ) -> CustomerSpeechProjection:
        self.calls.append((tenant_id, call_id, transcript_revision))
        if self.fail:
            raise RuntimeError("private projection failure")
        if not self.text:
            return CustomerSpeechProjection(
                tenant_id=tenant_id,
                call_id=call_id,
                transcript_revision=transcript_revision,
                customer_words=(),
                customer_text="",
                excluded_agent_word_count=1,
                excluded_unknown_word_count=0,
                excluded_overlap_word_count=0,
                excluded_below_confidence_word_count=0,
                status=CustomerProjectionStatus.EMPTY,
                reason=CustomerProjectionReason.NO_TRUSTED_CUSTOMER_SPEECH,
            )
        words = tuple(
            RoleTaggedWord(
                tenant_id=tenant_id,
                call_id=call_id,
                transcript_revision=transcript_revision,
                start_seconds=float(index),
                end_seconds=float(index + 1),
                text=word,
                local_speaker_ids=("customer-local",),
                global_speaker_id="CALL_SPEAKER_0002",
                global_speaker_ids=("CALL_SPEAKER_0002",),
                role=SpeakerRole.CUSTOMER,
                role_confidence=1.0,
                role_evidence=RoleEvidenceCode.STRONG_CUSTOMER,
            )
            for index, word in enumerate(self.text.split())
        )
        return CustomerSpeechProjection(
            tenant_id=tenant_id,
            call_id=call_id,
            transcript_revision=transcript_revision,
            customer_words=words,
            customer_text=self.text,
            customer_start_seconds=words[0].start_seconds,
            customer_end_seconds=words[-1].end_seconds,
            excluded_agent_word_count=1,
            excluded_unknown_word_count=0,
            excluded_overlap_word_count=0,
            excluded_below_confidence_word_count=0,
            status=CustomerProjectionStatus.READY,
            reason=CustomerProjectionReason.TRUSTED_CUSTOMER_SPEECH,
        )


class FixedLabelClassifier(FakeRuntimeClassifier):
    def __init__(self, *labels: str) -> None:
        super().__init__()
        self.labels = labels

    def classify(self, **kwargs: object) -> ClassificationResultEvent:
        base = super().classify(**kwargs)  # type: ignore[arg-type]
        return base.model_copy(
            update={
                "labels": [
                    ClassificationLabel(name=label, score=0.9) for label in self.labels
                ],
                "probabilities": {label: 0.9 for label in self.labels},
                "thresholds": {label: 0.5 for label in self.labels},
            }
        )


class SequencedViewClassifier(FakeRuntimeClassifier):
    def __init__(self, *responses: tuple[str, ...]) -> None:
        super().__init__()
        self.responses = responses

    def classify(self, **kwargs: object) -> ClassificationResultEvent:
        base = super().classify(**kwargs)  # type: ignore[arg-type]
        labels = self.responses[len(self.calls) - 1]
        action = (
            CoachingAction.NO_ACTION
            if not labels or labels == ("no_action",)
            else CoachingAction.TEMPLATE_ACTION
        )
        return base.model_copy(
            update={
                "labels": [
                    ClassificationLabel(name=label, score=0.9) for label in labels
                ],
                "action": action,
                "probabilities": {label: 0.9 for label in labels},
                "thresholds": {label: 0.5 for label in labels},
                "processing_time_ms": float(len(self.calls)),
            }
        )


def dual_view_outcome(
    delta_labels: tuple[str, ...],
    context_labels: tuple[str, ...],
    *,
    delta: str,
    preceding: str,
) -> tuple[StableClassificationOutcome, CallState, SequencedViewClassifier]:
    classifier = SequencedViewClassifier(delta_labels, context_labels)
    stage = StableTranscriptClassificationStage(classifier)
    state = CallState(
        tenant_id="tenant_alpha",
        call_id="call_001",
        stable_transcript=preceding,
    )
    event = TranscriptEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        event_id="stable-dual",
        kind=TranscriptKind.STABLE,
        text=delta,
        start_seconds=1.0,
        end_seconds=2.0,
        revision=1,
        created_at_utc=NOW,
        source_chunk_sequence=1,
    )
    state.apply_transcript(event)
    outcome = stage.process(
        event,
        cumulative_stable_transcript=state.stable_transcript,
        stable_changed=True,
        call_state=state,
        stable_delta=delta,
        preceding_stable_transcript=preceding,
    )
    return outcome, state, classifier


def test_ordered_execution_one_transcription_per_chunk_and_short_final_chunk() -> None:
    source = [chunk(0, 0.0, 2.0), chunk(1, 2.0, 2.0), chunk(2, 4.0, 0.5)]
    calls: list[object] = []
    fake = FakeTranscriber([[], [], []])
    result = pipeline(source, fake, calls).run(Path("synthetic.wav"), "call_001")
    assert calls == [
        (Path("synthetic.wav"), "tenant_alpha", "call_001", 2.0),
        (Path("synthetic.wav"), "tenant_alpha", "call_001", 2.0),
    ]
    assert [window.last_sequence for window in fake.windows] == [0, 1, 2]
    assert [step.sequence_number for step in result.steps] == [0, 1, 2]
    assert [step.window_duration_seconds for step in result.steps] == [2.0, 4.0, 4.5]
    assert len(fake.windows) == result.total_chunks == 3
    assert result.audio_duration_seconds == 4.5


def test_stable_partial_snapshots_and_final_event_application() -> None:
    source = [chunk(0, 0.0, 2.0), chunk(1, 2.0, 2.0), chunk(2, 4.0, 2.0)]
    fake = FakeTranscriber(
        [
            [("first", 0.0, 1.0), ("draft", 1.5, 2.0)],
            [("first", 0.0, 1.0), ("second", 1.0, 2.0), ("new draft", 3.0, 4.0)],
            [("first second", 0.0, 2.0), ("ending", 5.0, 6.0)],
        ]
    )
    result = pipeline(source, fake).run(Path("synthetic.wav"), "call_001")
    assert result.steps[0].partial_transcript == "first draft"
    assert result.steps[1].stable_transcript == "first second"
    assert result.steps[1].partial_transcript == "new draft"
    assert [event.kind for event in result.steps[1].transcript_events] == [
        TranscriptKind.STABLE,
        TranscriptKind.PARTIAL,
    ]
    assert result.final_event is not None
    assert (result.final_event.kind, result.final_event.text) == (
        TranscriptKind.FINAL,
        "ending",
    )
    assert result.stable_transcript == "first second ending"
    assert result.partial_transcript == ""


def test_no_speech_and_short_audio_below_window() -> None:
    fake = FakeTranscriber([[]])
    result = pipeline([chunk(0, 0.0, 0.5)], fake).run(Path("short.wav"), "call_001")
    assert result.steps[0].transcript_events == ()
    assert result.steps[0].raw_window_text == ""
    assert result.final_event is None
    assert result.stable_transcript == result.partial_transcript == ""


def test_tenant_call_preservation_and_source_objects_unchanged() -> None:
    tenant = context()
    settings = config()
    source = [chunk(0, 0.0, 1.0)]
    originals = (tenant.model_dump(), settings.model_dump(), source[0].model_dump())
    result = StreamingASRPipeline(
        tenant, settings, FakeTranscriber([[]]), chunk_generator=generator_for(source)
    ).run(Path("synthetic.wav"), "call_001")
    assert (result.tenant_id, result.call_id) == ("tenant_alpha", "call_001")
    assert all(
        (step.tenant_id, step.call_id) == ("tenant_alpha", "call_001")
        for step in result.steps
    )
    assert originals == (
        tenant.model_dump(),
        settings.model_dump(),
        source[0].model_dump(),
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [({"tenant_id": "tenant_beta"}, "tenant_id"), ({"call_id": "call_002"}, "call_id")],
)
def test_transcriber_scope_mismatch_is_rejected(
    override: dict[str, str], message: str
) -> None:
    fake = FakeTranscriber([[]])
    fake.scope_override = override
    with pytest.raises(ValueError, match=message):
        pipeline([chunk(0, 0.0, 1.0)], fake).run(Path("synthetic.wav"), "call_001")


def test_no_generated_chunks_is_rejected() -> None:
    with pytest.raises(ValueError, match="no audio chunks"):
        pipeline([], FakeTranscriber([])).run(Path("empty.wav"), "call_001")


def test_component_exceptions_are_not_silenced() -> None:
    class BrokenTranscriber:
        def transcribe(self, window: ASRAudioWindow) -> WindowTranscriptionResult:
            raise RuntimeError("synthetic failure")

    subject = StreamingASRPipeline(
        context(),
        config(),
        BrokenTranscriber(),
        chunk_generator=generator_for([chunk(0, 0.0, 1.0)]),
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        subject.run(Path("synthetic.wav"), "call_001")


def test_binary_audio_is_absent_from_repr_and_steps_are_immutable() -> None:
    result = pipeline([chunk(0, 0.0, 1.0)], FakeTranscriber([[]])).run(
        Path("synthetic.wav"), "call_001"
    )
    assert "audio_bytes" not in repr(result)
    assert bytes([1, 0]) not in repr(result).encode()
    with pytest.raises(FrozenInstanceError):
        result.steps[0].raw_window_text = "changed"  # type: ignore[misc]


def test_optional_step_callback_receives_each_safe_step() -> None:
    source = [chunk(0, 0.0, 1.0), chunk(1, 1.0, 1.0)]
    received: list[object] = []
    result = pipeline(source, FakeTranscriber([[], []])).run(
        Path("synthetic.wav"), "call_001", step_callback=received.append
    )
    assert received == list(result.steps)
    assert "audio_bytes" not in repr(received)


def test_history_can_be_disabled_without_changing_step_callbacks() -> None:
    source = [chunk(0, 0.0, 1.0), chunk(1, 1.0, 1.0)]
    subject = pipeline(source, FakeTranscriber([[], []]))
    received: list[object] = []

    result = subject.run(
        Path("synthetic.wav"),
        "call_001",
        step_callback=received.append,
        retain_history=False,
    )

    assert [step.sequence_number for step in received] == [0, 1]  # type: ignore[union-attr]
    assert result.steps == ()
    assert result.total_chunks == 2


def test_history_retention_remains_enabled_by_default() -> None:
    subject = pipeline(
        [chunk(0, 0.0, 1.0), chunk(1, 1.0, 1.0)],
        FakeTranscriber([[], []]),
    )

    result = subject.run(Path("synthetic.wav"), "call_001")

    assert len(result.steps) == result.total_chunks == 2


def test_plan_has_exact_total_and_short_final_chunk_before_inference() -> None:
    source = [chunk(index, index * 2.0, 2.0, 100) for index in range(6)]
    source.append(chunk(6, 12.0, 0.03, 100))
    fake = FakeTranscriber([[] for _ in source])
    observed: list[tuple[StreamingASRPlan, int]] = []
    result = pipeline(source, fake).run(
        Path("synthetic.wav"),
        "call_001",
        plan_callback=lambda plan: observed.append((plan, len(fake.windows))),
    )
    plan, inference_count_at_plan = observed[0]
    assert inference_count_at_plan == 0
    assert plan.total_chunks == result.total_chunks == 7
    assert plan.audio_duration_seconds == result.audio_duration_seconds == 12.03
    assert result.steps[-1].chunk_end_seconds - result.steps[
        -1
    ].chunk_start_seconds == (pytest.approx(0.03))


def test_only_stable_changes_are_classified_with_bounded_context() -> None:
    source = [chunk(0, 0.0, 2.0), chunk(1, 2.0, 2.0), chunk(2, 4.0, 2.0)]
    transcriber = FakeTranscriber(
        [
            [("first", 0.0, 1.0), ("draft", 1.5, 2.0)],
            [
                ("first", 0.0, 1.0),
                ("second", 1.0, 2.0),
                ("new draft", 3.0, 4.0),
            ],
            [("first second", 0.0, 2.0), ("ending", 5.0, 6.0)],
        ]
    )
    classifier = FakeRuntimeClassifier()
    result = pipeline(source, transcriber, runtime_classifier=classifier).run(
        Path("synthetic.wav"), "call_001"
    )
    assert [call["text"] for call in classifier.calls] == [
        "first second",
        "first second",
        "ending",
        "first second ending",
    ]
    assert all(
        (call["tenant_id"], call["call_id"]) == ("tenant_alpha", "call_001")
        for call in classifier.calls
    )
    partial_outcomes = [
        outcome
        for outcome in result.classification_outcomes
        if outcome.status is ClassificationProcessingStatus.PARTIAL_SKIPPED
    ]
    assert partial_outcomes
    assert result.classification_metadata.active_labels == ("complaint",)
    assert result.classification_metadata.model_id == "common_turkish_setfit_v2"
    assert result.classification_metadata.inference_time_ms == 8.0
    assert result.classification_metadata.delta_inference_time_ms == 4.0
    assert result.classification_metadata.context_inference_time_ms == 4.0
    assert result.classification_metadata.context_sentence_count == 2
    assert result.classification_metadata.preceding_sentence_count == 1
    assert not hasattr(result.classification_metadata, "probabilities")


def test_classification_failure_does_not_stop_streaming_or_log_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "PRIVATE_SYNTHETIC_TRANSCRIPT"
    caplog.set_level("ERROR")
    classifier = FakeRuntimeClassifier(fail=True)
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[(secret, 0.0, 1.0)]]),
        runtime_classifier=classifier,
    ).run(Path("synthetic.wav"), "call_001")
    assert result.final_event is not None
    assert result.classification_outcomes[-1].status is (
        ClassificationProcessingStatus.FAILED
    )
    assert result.classification_outcomes[-1].error is not None
    assert result.classification_outcomes[-1].error.error_type == "RuntimeError"
    assert secret not in caplog.text
    assert result.classification_metadata.active_labels == ()


def test_classifier_disabled_preserves_asr_only_behavior() -> None:
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[("final words", 0.0, 1.0)]]),
    ).run(Path("synthetic.wav"), "call_001")
    assert result.stable_transcript == "final words"
    assert result.classification_outcomes[-1].status is (
        ClassificationProcessingStatus.DISABLED
    )
    assert result.classification_metadata.active_labels == ()
    assert result.classification_metadata.model_id is None


def test_duplicate_revision_skipped_and_newer_revision_classified() -> None:
    classifier = FakeRuntimeClassifier()
    stage = StableTranscriptClassificationStage(classifier)
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    first = TranscriptEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        event_id="stable-1",
        kind=TranscriptKind.STABLE,
        text="first stable",
        start_seconds=0.0,
        end_seconds=1.0,
        revision=1,
        created_at_utc=NOW,
        source_chunk_sequence=1,
    )
    state.apply_transcript(first)
    first_outcome = stage.process(
        first,
        cumulative_stable_transcript=state.stable_transcript,
        stable_changed=True,
        call_state=state,
    )
    duplicate_outcome = stage.process(
        first,
        cumulative_stable_transcript=state.stable_transcript,
        stable_changed=True,
        call_state=state,
    )
    second = first.model_copy(
        update={
            "event_id": "stable-2",
            "text": "second stable",
            "revision": 2,
            "source_chunk_sequence": 2,
        }
    )
    state.apply_transcript(second)
    second_outcome = stage.process(
        second,
        cumulative_stable_transcript=state.stable_transcript,
        stable_changed=True,
        call_state=state,
        stable_delta=second.text,
        preceding_stable_transcript="first stable",
    )
    assert first_outcome.status is ClassificationProcessingStatus.CLASSIFIED
    assert duplicate_outcome.status is (
        ClassificationProcessingStatus.DUPLICATE_REVISION_SKIPPED
    )
    assert second_outcome.status is ClassificationProcessingStatus.CLASSIFIED
    assert [call["text"] for call in classifier.calls] == [
        "first stable",
        "first stable",
        "second stable",
        "first stable second stable",
    ]
    assert second_outcome.preceding_sentence_count == 1


def test_empty_cumulative_stable_transcript_is_skipped() -> None:
    classifier = FakeRuntimeClassifier()
    stage = StableTranscriptClassificationStage(classifier)
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    event = TranscriptEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        event_id="stable-empty",
        kind=TranscriptKind.STABLE,
        text="synthetic",
        start_seconds=0.0,
        end_seconds=1.0,
        revision=1,
        created_at_utc=NOW,
    )
    outcome = stage.process(
        event,
        cumulative_stable_transcript=" \t ",
        stable_changed=True,
        call_state=state,
    )
    assert outcome.status is ClassificationProcessingStatus.EMPTY_SKIPPED
    assert classifier.calls == []


def test_stable_pipeline_combines_rule_and_classification_coaching() -> None:
    classifier = FakeRuntimeClassifier()
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[("first complaint", 0.0, 1.0)]]),
        runtime_classifier=classifier,
        coaching_factory=coaching_factory(phrase="first"),
    ).run(Path("synthetic.wav"), "call_001")
    assert len(result.coaching_outcomes) == 1
    outcome = result.coaching_outcomes[0]
    assert outcome.status is CoachingProcessingStatus.PROCESSED
    assert outcome.result is not None
    suggestion = outcome.result.displayed_suggestions[0]
    assert suggestion.source is CoachingSuggestionSource.BOTH
    assert (suggestion.tenant_id, suggestion.call_id) == (
        "tenant_alpha",
        "call_001",
    )
    assert outcome.transcript_revision == result.final_event.revision  # type: ignore[union-attr]


def test_customer_routing_disabled_preserves_legacy_rule_path() -> None:
    provider = StreamingProjectionProvider("devam etmek istiyorum", fail=True)
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[("iptal etmek istiyorum", 0.0, 1.0)]]),
        coaching_factory=coaching_factory(
            phrase="iptal etmek istiyorum",
            label="cancellation_request",
        ),
        customer_projection_provider=provider,
    ).run(Path("synthetic.wav"), "call_001")

    assert provider.calls == []
    assert result.customer_routing_outcomes[-1].status is (
        CustomerRoutingStatus.LEGACY_PATH
    )
    assert result.coaching_outcomes[-1].result is not None
    assert result.coaching_outcomes[-1].result.displayed_suggestions


def test_agent_cancellation_is_excluded_from_enabled_customer_routing() -> None:
    provider = StreamingProjectionProvider("devam etmek istiyorum")
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[("temsilci iptal etmek istiyorum", 0.0, 1.0)]]),
        coaching_factory=coaching_factory(
            phrase="iptal etmek istiyorum",
            label="cancellation_request",
        ),
        customer_only_classification_enabled=True,
        customer_projection_provider=provider,
    ).run(Path("synthetic.wav"), "call_001")

    assert result.customer_routing_outcomes[-1].status is (
        CustomerRoutingStatus.CUSTOMER_PROCESSED
    )
    assert result.coaching_outcomes
    assert result.coaching_outcomes[-1].result is not None
    assert result.coaching_outcomes[-1].result.displayed_suggestions == ()


def test_enabled_pipeline_sends_only_customer_text_to_classifier() -> None:
    classifier = FakeRuntimeClassifier()
    customer_text = "devam etmek istiyorum"
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[("temsilci iptal etmek istiyorum", 0.0, 1.0)]]),
        runtime_classifier=classifier,
        customer_only_classification_enabled=True,
        customer_projection_provider=StreamingProjectionProvider(customer_text),
    ).run(Path("synthetic.wav"), "call_001")

    assert result.customer_routing_outcomes[-1].status is (
        CustomerRoutingStatus.CUSTOMER_PROCESSED
    )
    assert len(classifier.calls) == 2
    assert {call["text"] for call in classifier.calls} == {customer_text}
    assert all("iptal" not in str(call["text"]) for call in classifier.calls)


def test_customer_cancellation_uses_existing_coaching_lifecycle() -> None:
    provider = StreamingProjectionProvider("iptal etmek istiyorum")
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[("temsilci devam müşteri talebi", 0.0, 1.0)]]),
        coaching_factory=coaching_factory(
            phrase="iptal etmek istiyorum",
            label="cancellation_request",
        ),
        customer_only_classification_enabled=True,
        customer_projection_provider=provider,
    ).run(Path("synthetic.wav"), "call_001")

    coaching = result.coaching_outcomes[-1]
    assert coaching.result is not None
    suggestion = coaching.result.displayed_suggestions[0]
    assert suggestion.source is CoachingSuggestionSource.RULE
    assert suggestion.label_id == "cancellation_request"


def test_empty_customer_projection_skips_classifier_and_coaching() -> None:
    classifier = FakeRuntimeClassifier()
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[("önceki karışık iptal metni", 0.0, 1.0)]]),
        runtime_classifier=classifier,
        coaching_factory=coaching_factory(phrase="iptal"),
        customer_only_classification_enabled=True,
        customer_projection_provider=StreamingProjectionProvider(""),
    ).run(Path("synthetic.wav"), "call_001")

    assert classifier.calls == []
    assert result.coaching_outcomes == ()
    assert result.classification_outcomes[-1].classification_event is None
    assert result.customer_routing_outcomes[-1].status is (
        CustomerRoutingStatus.NO_CUSTOMER_SPEECH
    )


def test_structural_coaching_processor_is_injected_with_exact_arguments() -> None:
    processor: CoachingProcessorProtocol = FakeCoachingProcessor()
    factory_states: list[CallState] = []

    def factory(state: CallState) -> CoachingProcessorProtocol:
        factory_states.append(state)
        return processor

    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[("synthetic stable text", 0.0, 1.0)]]),
        coaching_factory=factory,
    ).run(Path("synthetic.wav"), "call_001")

    assert len(factory_states) == 1
    assert (factory_states[0].tenant_id, factory_states[0].call_id) == (
        "tenant_alpha",
        "call_001",
    )
    assert isinstance(processor, FakeCoachingProcessor)
    assert len(processor.calls) == 1
    event, current_seconds, classification_event, active_labels = processor.calls[0]
    assert event is result.final_event
    assert current_seconds == event.end_seconds
    assert classification_event is None
    assert active_labels == ()
    assert result.coaching_outcomes == (processor.outcomes[0],)
    assert result.coaching_outcomes[0] is processor.outcomes[0]


def test_completion_pump_runs_after_each_chunk_and_final_reconciliation() -> None:
    processor = FakeCompletionPump()
    caller_thread = get_ident()

    result = pipeline(
        [chunk(0, 0.0, 1.0), chunk(1, 1.0, 1.0)],
        FakeTranscriber(
            [
                [("synthetic partial", 0.0, 1.0)],
                [("synthetic stable text", 0.0, 2.0)],
            ]
        ),
        coaching_factory=lambda state: processor,
    ).run(Path("synthetic.wav"), "call_001")

    assert processor.drain_calls == [
        (1.0, caller_thread),
        (2.0, caller_thread),
        (2.0, caller_thread),
    ]
    assert result.steps[0].coaching_outcomes[0].transcript_revision == 1
    assert result.steps[1].coaching_outcomes[0].transcript_revision == 2
    assert result.coaching_outcomes[-1].transcript_revision == 3


@pytest.mark.parametrize(
    ("kind", "stable_changed"),
    [
        (TranscriptKind.PARTIAL, True),
        (TranscriptKind.STABLE, False),
    ],
)
def test_suppressed_transcript_does_not_invoke_structural_coaching_processor(
    kind: TranscriptKind,
    stable_changed: bool,
) -> None:
    processor = FakeCoachingProcessor()
    event = TranscriptEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        event_id="synthetic-event",
        kind=kind,
        text="synthetic text",
        start_seconds=0.0,
        end_seconds=1.0,
        revision=1,
        created_at_utc=NOW,
    )

    outcome = StreamingASRPipeline._process_coaching(
        coordinator=processor,
        event=event,
        stable_changed=stable_changed,
        classification_outcome=StableClassificationOutcome(
            ClassificationProcessingStatus.DISABLED,
            event.revision,
            event.source_chunk_sequence,
        ),
    )

    assert outcome is None
    assert processor.calls == []


def test_concrete_coaching_coordinator_satisfies_processor_protocol() -> None:
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    processor: CoachingProcessorProtocol = coaching_factory()(state)

    assert isinstance(processor, CoachingCoordinator)


def test_classifier_failure_still_allows_rule_only_coaching() -> None:
    secret = "PRIVATE_SYNTHETIC first"
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[(secret, 0.0, 1.0)]]),
        runtime_classifier=FakeRuntimeClassifier(fail=True),
        coaching_factory=coaching_factory(phrase="first"),
    ).run(Path("synthetic.wav"), "call_001")
    assert result.classification_outcomes[-1].status is (
        ClassificationProcessingStatus.FAILED
    )
    coaching = result.coaching_outcomes[-1]
    assert coaching.result is not None
    assert coaching.result.displayed_suggestions[0].source is (
        CoachingSuggestionSource.RULE
    )


def test_coaching_disabled_preserves_classification_only_operation() -> None:
    classifier = FakeRuntimeClassifier()
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[("final words", 0.0, 1.0)]]),
        runtime_classifier=classifier,
    ).run(Path("synthetic.wav"), "call_001")
    assert result.classification_outcomes[-1].status is (
        ClassificationProcessingStatus.CLASSIFIED
    )
    assert result.coaching_outcomes == ()


def test_coaching_failure_is_safe_and_does_not_log_transcript(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "PRIVATE_COACHING_TRANSCRIPT"
    tenant = TenantConfig(
        context=context(),
        asr=config(),
        classification=TenantClassificationConfig(
            model_id="rules", labels=["complaint"]
        ),
        rag=TenantRAGConfig(enabled=False),
        coaching=TenantCoachingConfig(
            allowed_actions=[action.value for action in CoachingAction]
        ),
    )

    class BrokenEngine:
        tenant_id = "tenant_alpha"

        def evaluate(
            self,
            event: TranscriptEvent,
            classification_labels: tuple[str, ...] = (),
        ) -> object:
            raise RuntimeError("synthetic coaching failure")

    def factory(state: CallState) -> CoachingCoordinator:
        return CoachingCoordinator(tenant, state, BrokenEngine())  # type: ignore[arg-type]

    caplog.set_level("ERROR")
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[(secret, 0.0, 1.0)]]),
        coaching_factory=factory,
    ).run(Path("synthetic.wav"), "call_001")
    assert result.stable_transcript == secret
    assert result.coaching_outcomes[-1].status is CoachingProcessingStatus.FAILED
    assert secret not in caplog.text


def test_price_query_guard_prevents_price_objection_coaching_card(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class PriceClassifier(FakeRuntimeClassifier):
        def classify(self, **kwargs: object) -> ClassificationResultEvent:
            base = super().classify(**kwargs)  # type: ignore[arg-type]
            return base.model_copy(
                update={
                    "labels": [
                        ClassificationLabel(name="product_information", score=0.949),
                        ClassificationLabel(name="price_objection", score=0.940),
                    ],
                    "probabilities": {
                        "product_information": 0.949,
                        "price_objection": 0.940,
                    },
                    "thresholds": {
                        "product_information": 0.90,
                        "price_objection": 0.35,
                    },
                }
            )

    transcript = "Paketin aylık fiyatına kadar ücret seçeneklerini öğrenmek istiyorum."
    caplog.set_level("INFO")
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[(transcript, 0.0, 1.0)]]),
        runtime_classifier=PriceClassifier(),
        coaching_factory=coaching_factory(phrase="eşleşmeyen"),
    ).run(Path("synthetic.wav"), "call_001")
    classification = result.classification_outcomes[-1].classification_event
    assert classification is not None
    assert [label.name for label in classification.labels] == ["product_information"]
    assert classification.probabilities["price_objection"] == 0.940
    coaching = result.coaching_outcomes[-1].result
    assert coaching is not None
    assert len(coaching.displayed_suggestions) == 1
    assert coaching.displayed_suggestions[0].label_id == "product_information"
    assert transcript not in caplog.text


def test_short_explicit_cancellation_preserves_label_coaching_and_both_source() -> None:
    transcript = (
        "Aboneliğimi bugün iptal ettirmek istiyorum. Lütfen iptal işlemini başlatın."
    )
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[(transcript, 0.0, 1.0)]]),
        runtime_classifier=FixedLabelClassifier("cancellation_request"),
        coaching_factory=coaching_factory(phrase="eşleşmeyen"),
    ).run(Path("synthetic.wav"), "call_001")

    classification = result.classification_outcomes[-1].classification_event
    coaching = result.coaching_outcomes[-1].result
    assert classification is not None
    assert [label.name for label in classification.labels] == ["cancellation_request"]
    assert coaching is not None
    assert len(coaching.displayed_suggestions) == 1
    assert coaching.displayed_suggestions[0].label_id == "cancellation_request"
    assert coaching.displayed_suggestions[0].source is CoachingSuggestionSource.BOTH


def test_short_negated_cancellation_does_not_trigger_rule_or_coaching() -> None:
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[("İptal etmek istemiyorum.", 0.0, 1.0)]]),
        runtime_classifier=FixedLabelClassifier(),
        coaching_factory=coaching_factory(phrase="eşleşmeyen"),
    ).run(Path("synthetic.wav"), "call_001")

    classification = result.classification_outcomes[-1].classification_event
    coaching = result.coaching_outcomes[-1].result
    assert classification is not None
    assert classification.labels == []
    assert coaching is not None
    assert coaching.displayed_suggestions == ()
    assert coaching.matched_rule_ids == ()


def test_short_price_information_question_preserves_product_information_only() -> None:
    transcript = "Paketin aylık fiyatı ne kadar, ücret seçeneklerini öğrenebilir miyim?"
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[(transcript, 0.0, 1.0)]]),
        runtime_classifier=FixedLabelClassifier(
            "product_information", "price_objection"
        ),
        coaching_factory=coaching_factory(phrase="eşleşmeyen"),
    ).run(Path("synthetic.wav"), "call_001")

    classification = result.classification_outcomes[-1].classification_event
    assert classification is not None
    assert [label.name for label in classification.labels] == ["product_information"]


def test_short_true_price_objection_preserves_price_objection() -> None:
    transcript = "Bu ücret çok pahalı, bütçemi aşıyor."
    result = pipeline(
        [chunk(0, 0.0, 1.0)],
        FakeTranscriber([[(transcript, 0.0, 1.0)]]),
        runtime_classifier=FixedLabelClassifier("price_objection"),
        coaching_factory=coaching_factory(phrase="eşleşmeyen"),
    ).run(Path("synthetic.wav"), "call_001")

    classification = result.classification_outcomes[-1].classification_event
    coaching = result.coaching_outcomes[-1].result
    assert classification is not None
    assert [label.name for label in classification.labels] == ["price_objection"]
    assert coaching is not None
    assert coaching.displayed_suggestions[0].label_id == "price_objection"
    assert coaching.displayed_suggestions[0].source is (
        CoachingSuggestionSource.CLASSIFICATION
    )


def test_synthetic_long_call_classifies_deltas_and_accumulates_all_labels() -> None:
    label_sentences = (
        ("product_information", "Ürün özelliklerini öğrenmek istiyorum."),
        ("price_objection", "Bu ücret çok pahalı."),
        ("technical_issue", "Uygulama açılmıyor."),
        ("complaint", "Bu durumdan şikayetçiyim."),
        ("churn_risk", "Böyle devam ederse başka firmaya geçeceğim."),
        ("cancellation_request", "Aboneliğimi iptal etmek istiyorum."),
    )

    class LongCallClassifier(FakeRuntimeClassifier):
        def classify(self, **kwargs: object) -> ClassificationResultEvent:
            base = super().classify(**kwargs)  # type: ignore[arg-type]
            text = str(kwargs["text"])
            label = next(
                label
                for label, sentence in reversed(label_sentences)
                if sentence in text
            )
            return base.model_copy(
                update={
                    "labels": [ClassificationLabel(name=label, score=0.9)],
                    "probabilities": {label: 0.9},
                    "thresholds": {label: 0.5},
                }
            )

    classifier = LongCallClassifier()
    stage = StableTranscriptClassificationStage(classifier)
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    outcomes = []
    for revision, (_, sentence) in enumerate(label_sentences, start=1):
        previous = state.stable_transcript
        event = TranscriptEvent(
            tenant_id="tenant_alpha",
            call_id="call_001",
            event_id=f"stable-{revision}",
            kind=TranscriptKind.STABLE,
            text=sentence,
            start_seconds=float(revision - 1),
            end_seconds=float(revision),
            revision=revision,
            created_at_utc=NOW,
            source_chunk_sequence=revision,
        )
        state.apply_transcript(event)
        outcomes.append(
            stage.process(
                event,
                cumulative_stable_transcript=state.stable_transcript,
                stable_changed=True,
                call_state=state,
                stable_delta=event.text,
                preceding_stable_transcript=previous,
            )
        )

    assert len(classifier.calls) == 12
    assert all(outcome.context_sentence_count <= 3 for outcome in outcomes)
    assert all(outcome.preceding_sentence_count <= 2 for outcome in outcomes)
    assert label_sentences[0][1] not in str(classifier.calls[-1]["text"])
    assert label_sentences[-1][1] in str(classifier.calls[-1]["text"])
    assert [item.label for item in state.detected_labels] == [
        label for label, _ in label_sentences
    ]
    assert state.active_labels == ["cancellation_request"]
    metadata = state.classification_metadata()
    assert not hasattr(metadata, "transcript_text")
    assert not hasattr(metadata, "probabilities")


def test_delta_view_recovers_product_information() -> None:
    outcome, state, classifier = dual_view_outcome(
        ("product_information",),
        (),
        delta="Ürün özelliklerini öğrenmek istiyorum.",
        preceding="Merhaba.",
    )
    assert [label.name for label in outcome.classification_event.labels] == [  # type: ignore[union-attr]
        "product_information"
    ]
    assert outcome.delta_labels == ("product_information",)
    assert outcome.context_labels == ()
    assert outcome.label_view_sources == (
        ("product_information", ClassificationViewSource.DELTA),
    )
    assert [call["text"] for call in classifier.calls] == [
        "Ürün özelliklerini öğrenmek istiyorum.",
        "Merhaba. Ürün özelliklerini öğrenmek istiyorum.",
    ]
    assert state.classification_metadata().delta_inference_ran
    assert (
        state.label_revision_timeline[0].evidence[0].classification_view
        is ClassificationViewSource.DELTA
    )


def test_delta_view_recovers_technical_issue_while_context_keeps_price() -> None:
    outcome, state, _ = dual_view_outcome(
        ("technical_issue",),
        ("price_objection",),
        delta="Uygulama açılmıyor ve bağlantı kurulamıyor.",
        preceding="Bu ücret çok pahalı.",
    )
    classification = outcome.classification_event
    assert classification is not None
    assert {label.name for label in classification.labels} == {
        "technical_issue",
        "price_objection",
    }
    assert dict(outcome.label_view_sources) == {
        "technical_issue": ClassificationViewSource.DELTA,
        "price_objection": ClassificationViewSource.BOUNDED_CONTEXT,
    }
    assert set(state.active_labels) == {"technical_issue", "price_objection"}


def test_context_view_recovers_context_dependent_churn_risk() -> None:
    outcome, _, _ = dual_view_outcome(
        (),
        ("churn_risk",),
        delta="Böyle devam ederse geçiş yapacağım.",
        preceding="Sorunlar haftalardır çözülmedi.",
    )
    classification = outcome.classification_event
    assert classification is not None
    assert [label.name for label in classification.labels] == ["churn_risk"]
    assert outcome.label_view_sources == (
        ("churn_risk", ClassificationViewSource.BOUNDED_CONTEXT),
    )


def test_dual_view_merge_preserves_multilabel_and_no_action_exclusivity() -> None:
    outcome, _, _ = dual_view_outcome(
        ("product_information", "no_action"),
        ("technical_issue", "no_action"),
        delta="Özellik bilgisini verir misiniz, uygulama da açılmıyor.",
        preceding="Destek rica ediyorum.",
    )
    classification = outcome.classification_event
    assert classification is not None
    assert {label.name for label in classification.labels} == {
        "product_information",
        "technical_issue",
    }
    assert "no_action" not in dict(outcome.label_view_sources)


def test_three_partial_chunks_finalize_stable_transcript_without_failure() -> None:
    sentence = (
        "Program Windows ve MacBook bilgisayarlarda çalışıyor mu? "
        "Sistem gereksinimlerini öğrenmek istiyorum."
    )
    source = [
        chunk(0, 0.0, 2.0),
        chunk(1, 2.0, 2.0),
        chunk(2, 4.0, 2.0),
    ]
    transcriber = FakeTranscriber(
        [
            [("Program Windows", 0.0, 2.0)],
            [("Program Windows ve MacBook bilgisayarlarda çalışıyor mu?", 0.0, 4.0)],
            [(sentence, 0.0, 6.0)],
        ]
    )
    result = pipeline(source, transcriber).run(Path("synthetic.wav"), "call_001")
    assert result.total_chunks == 3
    assert all(
        all(event.kind is TranscriptKind.PARTIAL for event in step.transcript_events)
        for step in result.steps
    )
    assert result.final_event is not None
    assert result.final_event.kind is TranscriptKind.FINAL
    assert result.stable_transcript == sentence


def test_pipeline_publishes_provisional_card_before_finalization() -> None:
    subject = pipeline(
        [chunk(0, 0.0, 2.0)],
        FakeTranscriber([[("synthetic complaint now", 0.0, 2.0)]]),
        runtime_classifier=FixedLabelClassifier("complaint"),
        coaching_factory=coaching_factory(label="complaint"),
    )
    subject.configure_provisional_coaching(
        ProvisionalClassificationPolicy(enabled=True),
        monotonic_clock=lambda: 1.0,
    )
    published = []

    result = subject.run(
        Path("synthetic.wav"),
        "call_001",
        step_callback=published.append,
    )

    assert len(published) == 1
    provisional = published[0].coaching_outcomes[0].result
    assert provisional is not None
    assert provisional.lifecycle is CoachingSuggestionLifecycle.PROVISIONAL
    assert (
        provisional.displayed_suggestions[0].lifecycle
        is CoachingSuggestionLifecycle.PROVISIONAL
    )
    assert result.final_event is not None
    confirmed = result.coaching_outcomes[-1].result
    assert confirmed is not None
    assert (
        confirmed.displayed_suggestions[0].suggestion_id
        == provisional.displayed_suggestions[0].suggestion_id
    )
    assert (
        confirmed.displayed_suggestions[0].lifecycle
        is CoachingSuggestionLifecycle.CONFIRMED
    )


def test_accelerated_upload_admits_distinct_risks_before_end_with_bounded_cards() -> (
    None
):
    class RegionClassifier(FakeRuntimeClassifier):
        def classify(self, **kwargs: object) -> ClassificationResultEvent:
            base = super().classify(**kwargs)  # type: ignore[arg-type]
            text = str(kwargs["text"])
            if "product" in text:
                label = "product_information"
            elif "renewal" in text:
                label = "renewal_interest"
            else:
                label = "cancellation_request"
            return base.model_copy(
                update={
                    "labels": [ClassificationLabel(name=label, score=0.95)],
                    "probabilities": {label: 0.95},
                    "thresholds": {label: 0.5},
                }
            )

    tenant = TenantConfig(
        context=context(),
        asr=config(),
        classification=TenantClassificationConfig(
            model_id="common_turkish_setfit_v2",
            labels=[
                "product_information",
                "renewal_interest",
                "cancellation_request",
            ],
        ),
        rag=TenantRAGConfig(enabled=False),
        coaching=TenantCoachingConfig(
            cooldown_seconds=8,
            max_active_suggestions=2,
            allowed_actions=[action.value for action in CoachingAction],
        ),
    )
    states: list[CallState] = []

    def create_coordinator(state: CallState) -> CoachingCoordinator:
        states.append(state)
        return CoachingCoordinator(
            tenant,
            state,
            RuleBasedCoachingEngine(
                tenant,
                (),
                event_id_factory=lambda: f"suggestion-{state.transcript_revision}",
                utc_datetime_factory=lambda: NOW,
            ),
        )

    subject = pipeline(
        [
            chunk(0, 0.0, 2.0),
            chunk(1, 2.0, 2.0),
            chunk(2, 4.0, 2.0),
        ],
        FakeTranscriber(
            [
                [("product details please", 0.0, 2.0)],
                [("renewal interest now", 2.0, 4.0)],
                [("cancel service now", 4.0, 6.0)],
            ]
        ),
        runtime_classifier=RegionClassifier(),
        coaching_factory=create_coordinator,
    )
    subject.configure_provisional_coaching(
        ProvisionalClassificationPolicy(enabled=True),
        monotonic_clock=lambda: 0.0,
    )
    published: list[StreamingASRStep] = []
    active_counts: list[int] = []
    active_label_sets: list[set[str | None]] = []

    def publish(step: StreamingASRStep) -> None:
        published.append(step)
        active_counts.append(len(states[0].active_coaching_suggestions))
        active_label_sets.append(
            {item.label_id for item in states[0].active_coaching_suggestions}
        )

    result = subject.run(
        Path("synthetic.wav"),
        "call_001",
        step_callback=publish,
    )

    displayed_before_end = [
        suggestion.label_id
        for step in published
        for outcome in step.coaching_outcomes
        if outcome.result is not None
        for suggestion in outcome.result.displayed_suggestions
    ]
    assert displayed_before_end == [
        "product_information",
        "renewal_interest",
        "cancellation_request",
    ]
    assert active_counts == [1, 2, 2]
    assert active_label_sets[-1] == {"renewal_interest", "cancellation_request"}
    assert {item.label_id for item in states[0].active_coaching_suggestions} == {
        "cancellation_request"
    }
    assert {item.label_id for item in states[0].coaching_suggestion_history} == {
        "product_information",
        "renewal_interest",
    }
    assert result.final_event is not None
    final = result.coaching_outcomes[-1].result
    assert final is not None
    assert len(final.displayed_suggestions) == 1
    last_provisional = published[-1].coaching_outcomes[0].result
    assert last_provisional is not None
    assert (
        final.displayed_suggestions[0].suggestion_id
        == last_provisional.displayed_suggestions[0].suggestion_id
    )


def test_accelerated_upload_same_label_repetition_remains_single_card() -> None:
    subject = pipeline(
        [
            chunk(0, 0.0, 2.0),
            chunk(1, 2.0, 2.0),
        ],
        FakeTranscriber(
            [
                [("synthetic complaint now", 0.0, 2.0)],
                [("synthetic complaint repeated", 2.0, 4.0)],
            ]
        ),
        runtime_classifier=FixedLabelClassifier("complaint"),
        coaching_factory=coaching_factory(label="complaint"),
    )
    subject.configure_provisional_coaching(
        ProvisionalClassificationPolicy(enabled=True),
        monotonic_clock=lambda: 0.0,
    )
    published: list[StreamingASRStep] = []

    subject.run(
        Path("synthetic.wav"),
        "call_001",
        step_callback=published.append,
    )

    displayed = [
        suggestion
        for step in published
        for outcome in step.coaching_outcomes
        if outcome.result is not None
        for suggestion in outcome.result.displayed_suggestions
    ]
    assert len(displayed) == 1
    assert displayed[0].label_id == "complaint"


def test_new_uploaded_source_resets_media_cadence_state() -> None:
    class RepeatingTranscriber(FakeTranscriber):
        def transcribe(self, window: ASRAudioWindow) -> WindowTranscriptionResult:
            self.windows.append(window)
            return transcription(window, ("synthetic complaint now", 0.0, 2.0))

    subject = pipeline(
        [chunk(0, 0.0, 2.0)],
        RepeatingTranscriber([]),
        runtime_classifier=FixedLabelClassifier("complaint"),
        coaching_factory=coaching_factory(label="complaint"),
    )
    subject.configure_provisional_coaching(
        ProvisionalClassificationPolicy(enabled=True),
        monotonic_clock=lambda: 0.0,
    )
    published_first: list[StreamingASRStep] = []
    published_second: list[StreamingASRStep] = []

    subject.run(
        Path("first-synthetic.wav"),
        "call_001",
        step_callback=published_first.append,
    )
    subject.run(
        Path("replacement-synthetic.wav"),
        "call_001",
        step_callback=published_second.append,
    )

    assert (
        published_first[0].classification_outcomes[0].status
        is ClassificationProcessingStatus.PROVISIONAL_CLASSIFIED
    )
    assert (
        published_second[0].classification_outcomes[0].status
        is ClassificationProcessingStatus.PROVISIONAL_CLASSIFIED
    )


def test_live_microphone_keeps_wall_clock_partial_cadence() -> None:
    subject = pipeline(
        [],
        FakeTranscriber(
            [
                [("synthetic complaint now", 0.0, 2.0)],
                [("synthetic complaint now repeated", 2.0, 4.0)],
            ]
        ),
        runtime_classifier=FixedLabelClassifier("complaint"),
        coaching_factory=coaching_factory(label="complaint"),
    )
    clock_values = iter((10.0, 10.5))
    subject.configure_provisional_coaching(
        ProvisionalClassificationPolicy(enabled=True),
        monotonic_clock=lambda: next(clock_values),
    )
    resource = object()
    capability = create_local_mic_test_capability(
        tenant_id="tenant_alpha",
        call_id="call_001",
        resource=resource,
        server_address="127.0.0.1",
        environment={LOCAL_MIC_GATE_ENVIRONMENT_KEY: "1"},
    )
    published: list[StreamingASRStep] = []

    subject.run_live(
        iter((chunk(0, 0.0, 2.0), chunk(1, 2.0, 2.0))),
        "call_001",
        capability=capability,
        execution_resource=resource,
        cancellation=Event(),
        step_callback=published.append,
    )

    assert (
        published[0].classification_outcomes[0].status
        is ClassificationProcessingStatus.PROVISIONAL_CLASSIFIED
    )
    assert (
        published[1].classification_outcomes[0].status
        is ClassificationProcessingStatus.PARTIAL_SKIPPED
    )


def test_live_microphone_publishes_provisional_coaching_before_end() -> None:
    subject = pipeline(
        [],
        FakeTranscriber([[("synthetic complaint now", 0.0, 2.0)]]),
        runtime_classifier=FixedLabelClassifier("complaint"),
        coaching_factory=coaching_factory(label="complaint"),
    )
    subject.configure_provisional_coaching(
        ProvisionalClassificationPolicy(enabled=True),
        monotonic_clock=lambda: 1.0,
    )
    resource = object()
    capability = create_local_mic_test_capability(
        tenant_id="tenant_alpha",
        call_id="call_001",
        resource=resource,
        server_address="127.0.0.1",
        environment={LOCAL_MIC_GATE_ENVIRONMENT_KEY: "1"},
    )
    published = []

    def live_chunks():
        yield chunk(0, 0.0, 2.0)
        assert published
        result = published[0].coaching_outcomes[0].result
        assert result is not None
        assert result.lifecycle is CoachingSuggestionLifecycle.PROVISIONAL

    result = subject.run_live(
        live_chunks(),
        "call_001",
        capability=capability,
        execution_resource=resource,
        cancellation=Event(),
        step_callback=published.append,
    )

    assert result.total_chunks == 1
    assert result.final_event is not None


def test_live_microphone_pause_resume_keeps_one_pipeline_and_monotonic_transcript() -> (
    None
):
    class OneChunkNormalizer:
        def normalize(self, frame: av.AudioFrame) -> tuple[_NormalizedAudio, ...]:
            del frame
            return (_NormalizedAudio(b"\0" * LOCAL_MIC_CHUNK_BYTES, None),)

        def flush(self) -> tuple[_NormalizedAudio, ...]:
            return ()

    subject = pipeline(
        [],
        FakeTranscriber(
            [
                [("synthetic first", 0.0, 2.0)],
                [
                    ("synthetic first", 0.0, 2.0),
                    ("synthetic second", 2.0, 4.0),
                ],
            ]
        ),
    )
    resource = object()
    initial_capability = create_local_mic_test_capability(
        tenant_id="tenant_alpha",
        call_id="call_001",
        resource=resource,
        server_address="127.0.0.1",
        environment={LOCAL_MIC_GATE_ENVIRONMENT_KEY: "1"},
    )
    session = LocalMicrophoneIngressSession(
        capability=initial_capability,
        resource=resource,
        normalizer=OneChunkNormalizer(),
    )
    session.set_asr_readiness(
        LocalMicrophoneASRReadiness.WARMING_UP,
        resource=resource,
    )
    session.set_asr_readiness(
        LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
        resource=resource,
    )
    cancellation = Event()
    published: list[StreamingASRStep] = []

    def live_chunks() -> Iterator[AudioChunkEvent]:
        chunks = iter(session.iter_audio_chunks(cancellation=cancellation))
        session.accept_frame(av.AudioFrame(), arrived_at_utc=NOW)
        yield next(chunks)
        assert session.pause_capture(resource=resource, arrived_at_utc=NOW)
        assert not initial_capability.active
        resumed_capability = create_local_mic_test_capability(
            tenant_id="tenant_alpha",
            call_id="call_001",
            resource=resource,
            server_address="127.0.0.1",
            environment={LOCAL_MIC_GATE_ENVIRONMENT_KEY: "1"},
        )
        assert session.resume_capture(
            resumed_capability,
            resource=resource,
            normalizer=OneChunkNormalizer(),
        )
        session.accept_frame(av.AudioFrame(), arrived_at_utc=NOW)
        yield next(chunks)
        assert session.finish_call(resource=resource, arrived_at_utc=NOW)
        assert tuple(chunks) == ()

    def publish(step: StreamingASRStep) -> None:
        session.acknowledge_processed_chunk(resource=resource)
        published.append(step)

    result = subject.run_live(
        live_chunks(),
        "call_001",
        capability=initial_capability,
        execution_resource=resource,
        cancellation=cancellation,
        step_callback=publish,
    )

    assert [step.sequence_number for step in published] == [1, 2]
    assert [step.asr_segment_count for step in published] == [1, 2]
    revisions = [
        event.revision for step in published for event in step.transcript_events
    ]
    assert revisions == sorted(revisions)
    assert len(set(revisions)) == len(revisions)
    assert result.total_chunks == 2
    assert result.audio_duration_seconds == pytest.approx(4.0)
    assert result.stable_transcript
    assert session.diagnostics.end_emitted


def test_live_microphone_capability_mismatch_and_revocation_fail_closed() -> None:
    subject = pipeline([], FakeTranscriber([]))
    resource = object()
    capability = create_local_mic_test_capability(
        tenant_id="tenant_alpha",
        call_id="call_001",
        resource=resource,
        server_address="localhost",
        environment={LOCAL_MIC_GATE_ENVIRONMENT_KEY: "1"},
    )

    with pytest.raises(PermissionError, match="invalid_local_microphone"):
        subject.run_live(
            (),
            "call_001",
            capability=capability,
            execution_resource=object(),
            cancellation=Event(),
        )

    capability.revoke()
    with pytest.raises(PermissionError, match="invalid_local_microphone"):
        subject.run_live(
            (),
            "call_001",
            capability=capability,
            execution_resource=resource,
            cancellation=Event(),
        )
