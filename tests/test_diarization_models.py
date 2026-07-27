from math import inf, nan

import pytest
from pydantic import ValidationError

from app.diarization import (
    DiarizationRequest,
    DiarizationResult,
    DiarizationTurn,
    DiarizedTranscriptEvent,
    DiarizedWord,
    SpeakerRole,
)


def turn(**changes: object) -> DiarizationTurn:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "start_seconds": 1.0,
        "end_seconds": 2.0,
        "local_speaker_ids": ("speaker_0",),
        "global_speaker_id": None,
        "role": SpeakerRole.UNKNOWN,
        "speaker_confidence": 0.8,
        "role_confidence": None,
    }
    values.update(changes)
    return DiarizationTurn.model_validate(values)


def word(**changes: object) -> DiarizedWord:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "transcript_revision": 3,
        "start_seconds": 1.1,
        "end_seconds": 1.4,
        "text": "sentetik",
        "local_speaker_ids": ("speaker_0",),
        "global_speaker_id": "call_speaker_0",
        "role": SpeakerRole.AGENT,
        "speaker_confidence": 0.9,
        "role_confidence": 0.7,
    }
    values.update(changes)
    return DiarizedWord.model_validate(values)


def event(**changes: object) -> DiarizedTranscriptEvent:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "transcript_event_id": "transcript_3",
        "transcript_revision": 3,
        "start_seconds": 1.0,
        "end_seconds": 2.0,
        "turns": (turn(),),
        "words": (word(),),
    }
    values.update(changes)
    return DiarizedTranscriptEvent.model_validate(values)


def request(**changes: object) -> DiarizationRequest:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "window_start_seconds": 1.0,
        "window_end_seconds": 2.0,
        "sample_rate_hz": 16_000,
        "mono_audio": (0.0, 0.25, -0.25),
    }
    values.update(changes)
    return DiarizationRequest.model_validate(values)


def test_valid_single_speaker_turn() -> None:
    subject = turn(
        global_speaker_id="call_speaker_0",
        role=SpeakerRole.CUSTOMER,
        role_confidence=0.75,
    )

    assert subject.local_speaker_ids == ("speaker_0",)
    assert subject.global_speaker_id == "call_speaker_0"
    assert subject.role is SpeakerRole.CUSTOMER


def test_valid_overlap_requires_multiple_real_local_speaker_ids() -> None:
    subject = turn(
        local_speaker_ids=("speaker_0", "speaker_1"),
        global_speaker_id=None,
        role=SpeakerRole.OVERLAP,
    )

    assert subject.local_speaker_ids == ("speaker_0", "speaker_1")
    assert subject.role is SpeakerRole.OVERLAP


@pytest.mark.parametrize(
    ("start_seconds", "end_seconds"),
    [(1.0, 1.0), (2.0, 1.0), (-1.0, 1.0)],
)
def test_invalid_timestamp_ranges_are_rejected(
    start_seconds: float,
    end_seconds: float,
) -> None:
    with pytest.raises(ValidationError):
        turn(start_seconds=start_seconds, end_seconds=end_seconds)


@pytest.mark.parametrize("value", [nan, inf, -inf])
@pytest.mark.parametrize("field", ["start_seconds", "end_seconds"])
def test_non_finite_timestamps_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        turn(**{field: value})


@pytest.mark.parametrize("value", [-0.1, 1.1, nan, inf])
@pytest.mark.parametrize("field", ["speaker_confidence", "role_confidence"])
def test_invalid_confidence_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match="confidence"):
        turn(**{field: value})


@pytest.mark.parametrize(
    ("role", "speaker_ids"),
    [
        (SpeakerRole.OVERLAP, ("speaker_0",)),
        (SpeakerRole.AGENT, ("speaker_0", "speaker_1")),
        (SpeakerRole.UNKNOWN, ()),
        (SpeakerRole.CUSTOMER, ("speaker_0", "speaker_0")),
    ],
)
def test_speaker_cardinality_is_fail_closed(
    role: SpeakerRole,
    speaker_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError):
        turn(role=role, local_speaker_ids=speaker_ids)


def test_models_are_immutable() -> None:
    source_turn = turn()
    source_word = word()
    source_event = event()
    source_request = request()
    source_result = DiarizationResult(
        tenant_id="tenant_alpha",
        call_id="call_001",
        window_start_seconds=1.0,
        window_end_seconds=2.0,
        turns=(source_turn,),
    )

    for model, field, value in (
        (source_turn, "role", SpeakerRole.AGENT),
        (source_word, "text", "changed"),
        (source_event, "transcript_revision", 4),
        (source_request, "sample_rate_hz", 8_000),
        (source_result, "turns", ()),
    ):
        with pytest.raises(ValidationError):
            setattr(model, field, value)


def test_children_must_use_deterministic_timestamp_order() -> None:
    later = turn(start_seconds=1.5, end_seconds=1.9)
    earlier = turn(start_seconds=1.0, end_seconds=1.4)

    with pytest.raises(ValidationError, match="deterministic timestamp order"):
        event(turns=(later, earlier), words=())

    result = DiarizationResult(
        tenant_id="tenant_alpha",
        call_id="call_001",
        window_start_seconds=1.0,
        window_end_seconds=2.0,
        turns=(earlier, later),
    )
    assert result.turns == (earlier, later)


@pytest.mark.parametrize(
    "changes",
    [
        {"tenant_id": "tenant_beta"},
        {"call_id": "call_002"},
    ],
)
def test_child_scope_must_match_parent(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="scope"):
        event(turns=(turn(**changes),), words=())


@pytest.mark.parametrize("revision", [-1, -5])
def test_revision_validation(revision: int) -> None:
    with pytest.raises(ValidationError, match="revision"):
        word(transcript_revision=revision)
    with pytest.raises(ValidationError, match="revision"):
        event(transcript_revision=revision, turns=(), words=())


def test_word_revision_must_match_parent() -> None:
    with pytest.raises(ValidationError, match="revision"):
        event(words=(word(transcript_revision=2),))


@pytest.mark.parametrize(
    ("field", "child"),
    [
        ("turns", turn(start_seconds=0.5, end_seconds=1.5)),
        ("turns", turn(start_seconds=1.5, end_seconds=2.5)),
        ("words", word(start_seconds=0.5, end_seconds=1.2)),
        ("words", word(start_seconds=1.8, end_seconds=2.5)),
    ],
)
def test_children_outside_parent_range_are_rejected(
    field: str,
    child: DiarizationTurn | DiarizedWord,
) -> None:
    values: dict[str, object] = {"turns": (), "words": ()}
    values[field] = (child,)

    with pytest.raises(ValidationError, match="inside parent"):
        event(**values)


def test_request_preserves_trusted_mono_window_without_exposing_audio() -> None:
    subject = request(mono_audio=(0.123456789, -0.5))

    assert subject.tenant_id == "tenant_alpha"
    assert subject.call_id == "call_001"
    assert subject.window_start_seconds == 1.0
    assert subject.window_end_seconds == 2.0
    assert subject.sample_rate_hz == 16_000
    assert subject.mono_audio == (0.123456789, -0.5)
    assert "0.123456789" not in repr(subject)


@pytest.mark.parametrize("value", [nan, inf, -inf, 1.1, -1.1])
def test_request_rejects_unsafe_audio_samples(value: float) -> None:
    with pytest.raises(ValidationError, match="normalized samples") as error:
        request(mono_audio=(value,))

    assert "input_value" not in str(error.value)
