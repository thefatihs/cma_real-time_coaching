from datetime import UTC, datetime

import pytest

from app.events.models import AudioChunkEvent
from app.streaming.rolling_buffer import RollingAudioBuffer


NOW = datetime(2026, 7, 22, tzinfo=UTC)


def audio_event(sequence: int, **changes: object) -> AudioChunkEvent:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "sequence_number": sequence,
        "received_at_utc": NOW,
        "chunk_start_seconds": sequence * 2.0,
        "chunk_duration_seconds": 2.0,
        "sample_rate_hz": 8_000,
        "channel_count": 1,
        "codec_name": "pcm_s16le",
        "audio_bytes": b"synthetic",
    }
    values.update(changes)
    return AudioChunkEvent.model_validate(values)


def test_valid_sequential_append_and_properties() -> None:
    buffer = RollingAudioBuffer()
    first = audio_event(0)
    second = audio_event(1)

    buffer.append(first)
    buffer.append(second)

    assert buffer.events() == (first, second)
    assert not buffer.is_empty
    assert buffer.chunk_count == 2
    assert buffer.start_seconds == 0.0
    assert buffer.end_seconds == 4.0
    assert buffer.duration_seconds == 4.0
    assert buffer.first_sequence == 0
    assert buffer.last_sequence == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tenant_id": "tenant_beta"}, "tenant_id"),
        ({"call_id": "call_002"}, "call_id"),
        ({"sample_rate_hz": 16_000}, "sample_rate_hz"),
        ({"channel_count": 2}, "channel_count"),
        ({"codec_name": "pcm_f32le"}, "codec_name"),
    ],
)
def test_binding_mismatches_are_rejected(
    changes: dict[str, object], message: str
) -> None:
    buffer = RollingAudioBuffer()
    buffer.append(audio_event(0))

    with pytest.raises(ValueError, match=message):
        buffer.append(audio_event(1, **changes))


@pytest.mark.parametrize(
    ("initial_sequence", "rejected_sequence"), [(0, 0), (2, 1), (0, 2)]
)
def test_duplicate_backward_and_gap_sequences_are_rejected(
    initial_sequence: int, rejected_sequence: int
) -> None:
    buffer = RollingAudioBuffer()
    buffer.append(audio_event(initial_sequence))

    with pytest.raises(ValueError, match="contiguous"):
        buffer.append(
            audio_event(
                rejected_sequence,
                chunk_start_seconds=(initial_sequence + 1) * 2.0,
            )
        )


@pytest.mark.parametrize("start_seconds", [1.0, 0.0])
def test_overlapping_and_backward_timestamps_are_rejected(
    start_seconds: float,
) -> None:
    buffer = RollingAudioBuffer()
    buffer.append(audio_event(0))

    with pytest.raises(ValueError, match="overlap"):
        buffer.append(audio_event(1, chunk_start_seconds=start_seconds))


def test_rolling_window_evicts_completely_old_chunks() -> None:
    buffer = RollingAudioBuffer(window_seconds=5.0)
    for sequence in range(4):
        buffer.append(audio_event(sequence))

    assert [event.sequence_number for event in buffer.events()] == [1, 2, 3]
    assert buffer.start_seconds == 2.0
    assert buffer.duration_seconds == 6.0


def test_final_short_chunk_is_retained() -> None:
    buffer = RollingAudioBuffer()
    buffer.append(audio_event(0))
    final = audio_event(1, chunk_duration_seconds=0.25)

    buffer.append(final)

    assert buffer.events()[-1] is final
    assert buffer.end_seconds == 2.25
    assert buffer.duration_seconds == 2.25


def test_events_returns_an_immutable_tuple() -> None:
    buffer = RollingAudioBuffer()
    event = audio_event(0)
    buffer.append(event)

    snapshot = buffer.events()
    assert isinstance(snapshot, tuple)
    assert snapshot == (event,)


def test_clear_resets_binding_for_reuse() -> None:
    buffer = RollingAudioBuffer()
    buffer.append(audio_event(7, chunk_start_seconds=10.0))

    buffer.clear()
    replacement = audio_event(
        0,
        tenant_id="tenant_beta",
        call_id="call_002",
        sample_rate_hz=16_000,
        channel_count=2,
        codec_name="pcm_f32le",
        chunk_start_seconds=0.0,
    )
    buffer.append(replacement)

    assert buffer.events() == (replacement,)


@pytest.mark.parametrize("window", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_window_duration_is_rejected(window: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        RollingAudioBuffer(window)


def test_empty_buffer_properties() -> None:
    buffer = RollingAudioBuffer()

    assert buffer.is_empty
    assert buffer.chunk_count == 0
    assert buffer.events() == ()
    assert buffer.start_seconds is None
    assert buffer.end_seconds is None
    assert buffer.duration_seconds is None
    assert buffer.first_sequence is None
    assert buffer.last_sequence is None
