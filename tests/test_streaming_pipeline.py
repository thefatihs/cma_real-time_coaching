from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.classification.streaming import (
    ClassificationProcessingStatus,
    RuntimeClassifierProtocol,
    StableTranscriptClassificationStage,
)
from app.calls.models import CallState
from app.events.models import (
    AudioChunkEvent,
    ClassificationLabel,
    ClassificationResultEvent,
    CoachingAction,
    TranscriptEvent,
    TranscriptKind,
)
from app.streaming.audio_window import ASRAudioWindow
from app.streaming.pipeline import StreamingASRPipeline, StreamingASRPlan
from app.streaming.window_transcriber import (
    WindowTranscriptionResult,
    WindowTranscriptionSegment,
)
from app.tenancy.models import TenantASRConfig, TenantContext


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
) -> StreamingASRPipeline:
    return StreamingASRPipeline(
        context(),
        config(),
        transcriber,
        chunk_generator=generator_for(source, calls),
        runtime_classifier=runtime_classifier,
    )


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


def test_only_stable_changes_are_classified_with_cumulative_context() -> None:
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
    assert result.classification_metadata.inference_time_ms == 4.0
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
    )
    assert first_outcome.status is ClassificationProcessingStatus.CLASSIFIED
    assert duplicate_outcome.status is (
        ClassificationProcessingStatus.DUPLICATE_REVISION_SKIPPED
    )
    assert second_outcome.status is ClassificationProcessingStatus.CLASSIFIED
    assert [call["text"] for call in classifier.calls] == [
        "first stable",
        "first stable second stable",
    ]


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
