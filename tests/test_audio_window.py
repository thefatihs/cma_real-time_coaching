from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.events.models import AudioChunkEvent
from app.streaming.audio_window import AudioWindowBuilder
from app.streaming.rolling_buffer import RollingAudioBuffer


NOW = datetime(2026, 7, 22, tzinfo=UTC)


def pcm_event(
    sequence: int,
    start: float,
    duration: float,
    *,
    sample_rate: int = 10,
    channels: int = 1,
    codec: str = "pcm_s16le",
    marker: int | None = None,
) -> AudioChunkEvent:
    frame = bytes([marker if marker is not None else sequence, 0]) * channels
    return AudioChunkEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        sequence_number=sequence,
        received_at_utc=NOW,
        chunk_start_seconds=start,
        chunk_duration_seconds=duration,
        sample_rate_hz=sample_rate,
        channel_count=channels,
        codec_name=codec,
        audio_bytes=frame * round(duration * sample_rate),
    )


def build(events: list[AudioChunkEvent], window: float = 20.0):
    buffer = RollingAudioBuffer(window)
    for event in events:
        buffer.append(event)
    return AudioWindowBuilder().build(buffer)


def test_buffer_shorter_than_window() -> None:
    window = build([pcm_event(0, 0.0, 2.0), pcm_event(1, 2.0, 1.0)])
    assert window.duration_seconds == 3.0
    assert window.start_seconds == 0.0
    assert len(window.pcm_bytes) == 60


def test_exact_twenty_second_buffer() -> None:
    window = build([pcm_event(i, i * 2.0, 2.0) for i in range(10)])
    assert (window.first_sequence, window.last_sequence) == (0, 9)
    assert window.duration_seconds == 20.0
    assert len(window.pcm_bytes) == 400


def test_chunk_aligned_eviction() -> None:
    window = build([pcm_event(i, i * 5.0, 5.0) for i in range(5)])
    assert (window.first_sequence, window.last_sequence) == (1, 4)
    assert window.pcm_bytes[:2] == bytes([1, 0])


def test_partial_first_chunk_is_trimmed() -> None:
    window = build([pcm_event(i, i * 7.0, 7.0) for i in range(3)])
    assert window.duration_seconds == 20.0
    assert window.start_seconds == 1.0
    assert window.pcm_bytes[:2] == bytes([0, 0])
    assert len(window.pcm_bytes) == 400


def test_stereo_trimming_preserves_channel_frame_alignment() -> None:
    window = build(
        [pcm_event(i, i * 1.5, 1.5, channels=2) for i in range(3)], window=4.0
    )
    assert len(window.pcm_bytes) == 160
    assert len(window.pcm_bytes) % 4 == 0
    assert window.start_seconds == 0.5


def test_final_short_chunk() -> None:
    window = build([pcm_event(0, 0.0, 2.0), pcm_event(1, 2.0, 0.3)])
    assert window.end_seconds == 2.3
    assert window.duration_seconds == 2.3


def test_empty_buffer_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty buffer"):
        AudioWindowBuilder().build(RollingAudioBuffer())


def test_unsupported_codec_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported audio codec.*pcm_f32le"):
        build([pcm_event(0, 0.0, 1.0, codec="pcm_f32le")])


def test_pcm_bytes_are_hidden_from_repr_and_metadata() -> None:
    window = build([pcm_event(0, 0.0, 1.0, marker=123)])
    assert "pcm_bytes" not in repr(window)
    assert "pcm_bytes" not in window.metadata_summary()
    assert bytes([123, 0]) not in repr(window).encode()


def test_window_is_immutable() -> None:
    window = build([pcm_event(0, 0.0, 1.0)])
    with pytest.raises(ValidationError):
        window.call_id = "changed"


def test_source_events_remain_unchanged() -> None:
    events = [pcm_event(i, i * 7.0, 7.0) for i in range(3)]
    originals = [event.model_dump() for event in events]
    buffer = RollingAudioBuffer()
    for event in events:
        buffer.append(event)

    AudioWindowBuilder().build(buffer)

    assert [event.model_dump() for event in events] == originals
    assert buffer.events() == tuple(events)
