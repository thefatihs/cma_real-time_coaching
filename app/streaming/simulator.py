"""Safely simulate ordered audio streaming from a local media file."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import sleep

from app.events.models import AudioChunkEvent
from app.streaming.chunk_generator import (
    DEFAULT_CHUNK_DURATION_SECONDS,
    generate_audio_chunks,
)
from app.streaming.rolling_buffer import RollingAudioBuffer


DEFAULT_WINDOW_SECONDS = 20.0
SleepFunction = Callable[[float], None]
ChunkGenerator = Callable[[Path, str, str, float], Iterator[AudioChunkEvent]]


@dataclass(frozen=True, slots=True)
class StreamStep:
    """Immutable, audio-free state produced after one chunk is buffered."""

    tenant_id: str
    call_id: str
    sequence_number: int
    chunk_start_seconds: float
    chunk_end_seconds: float
    buffer_start_seconds: float
    buffer_end_seconds: float
    buffer_duration_seconds: float
    buffer_chunk_count: int
    first_buffer_sequence: int
    last_buffer_sequence: int


def simulate_audio_stream(
    audio_path: Path,
    tenant_id: str,
    call_id: str,
    chunk_duration_seconds: float = DEFAULT_CHUNK_DURATION_SECONDS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    *,
    realtime: bool = False,
    sleep_function: SleepFunction = sleep,
    chunk_generator: ChunkGenerator = generate_audio_chunks,
) -> Iterator[StreamStep]:
    """Return safe steps for chunks decoded from ``audio_path`` in order."""
    clean_tenant_id = _required_identifier(tenant_id, "tenant_id")
    clean_call_id = _required_identifier(call_id, "call_id")
    buffer = RollingAudioBuffer(window_seconds)
    chunks = chunk_generator(
        audio_path,
        clean_tenant_id,
        clean_call_id,
        chunk_duration_seconds,
    )

    def process_chunks() -> Iterator[StreamStep]:
        for event in chunks:
            if realtime:
                sleep_function(event.chunk_duration_seconds)
            buffer.append(event)
            yield _make_step(event, buffer)

    return process_chunks()


def _make_step(event: AudioChunkEvent, buffer: RollingAudioBuffer) -> StreamStep:
    buffer_start = buffer.start_seconds
    buffer_end = buffer.end_seconds
    buffer_duration = buffer.duration_seconds
    first_sequence = buffer.first_sequence
    last_sequence = buffer.last_sequence
    assert buffer_start is not None
    assert buffer_end is not None
    assert buffer_duration is not None
    assert first_sequence is not None
    assert last_sequence is not None
    return StreamStep(
        tenant_id=event.tenant_id,
        call_id=event.call_id,
        sequence_number=event.sequence_number,
        chunk_start_seconds=event.chunk_start_seconds,
        chunk_end_seconds=event.chunk_start_seconds + event.chunk_duration_seconds,
        buffer_start_seconds=buffer_start,
        buffer_end_seconds=buffer_end,
        buffer_duration_seconds=buffer_duration,
        buffer_chunk_count=buffer.chunk_count,
        first_buffer_sequence=first_sequence,
        last_buffer_sequence=last_sequence,
    )


def _required_identifier(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
