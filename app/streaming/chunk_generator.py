"""Generate tenant-aware audio chunk events from a local media file."""

from collections.abc import Iterator
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

import av
from av.audio.fifo import AudioFifo

from app.events.models import AudioChunkEvent


DEFAULT_CHUNK_DURATION_SECONDS = 2.0


def generate_audio_chunks(
    audio_path: Path,
    tenant_id: str,
    call_id: str,
    chunk_duration_seconds: float = DEFAULT_CHUNK_DURATION_SECONDS,
) -> Iterator[AudioChunkEvent]:
    """Decode ``audio_path`` and return ordered, fixed-duration PCM events."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"Audio path must be a file: {audio_path}")
    if not isfinite(chunk_duration_seconds) or chunk_duration_seconds <= 0:
        raise ValueError("chunk_duration_seconds must be finite and positive")

    return _decode_chunks(
        audio_path,
        tenant_id,
        call_id,
        chunk_duration_seconds,
    )


def _decode_chunks(
    audio_path: Path,
    tenant_id: str,
    call_id: str,
    chunk_duration_seconds: float,
) -> Iterator[AudioChunkEvent]:
    with av.open(str(audio_path)) as container:
        audio_streams = container.streams.audio
        if not audio_streams:
            raise ValueError(f"Media file contains no audio stream: {audio_path}")

        fifo = AudioFifo()
        sample_rate_hz: int | None = None
        channel_count: int | None = None
        samples_per_chunk: int | None = None
        sequence_number = 0
        emitted_samples = 0

        for frame in container.decode(audio_streams[0]):
            frame_sample_rate = frame.sample_rate
            if frame_sample_rate is None or frame_sample_rate <= 0:
                raise ValueError("Decoded audio has no valid sample rate")

            frame_channel_count = len(frame.layout.channels)
            if frame_channel_count <= 0:
                raise ValueError("Decoded audio has no channels")

            if sample_rate_hz is None:
                sample_rate_hz = frame_sample_rate
                channel_count = frame_channel_count
                samples_per_chunk = max(
                    1, round(chunk_duration_seconds * sample_rate_hz)
                )
            elif (
                frame_sample_rate != sample_rate_hz
                or frame_channel_count != channel_count
            ):
                raise ValueError("Sample rate and channel count must remain constant")

            fifo.write(frame)
            assert samples_per_chunk is not None
            while fifo.samples >= samples_per_chunk:
                chunk = fifo.read(samples_per_chunk)
                assert chunk is not None
                yield _make_event(
                    chunk,
                    tenant_id,
                    call_id,
                    sequence_number,
                    emitted_samples,
                    sample_rate_hz,
                    frame_channel_count,
                )
                sequence_number += 1
                emitted_samples += chunk.samples

        if fifo.samples:
            chunk = fifo.read(fifo.samples, partial=True)
            assert chunk is not None
            assert sample_rate_hz is not None
            assert channel_count is not None
            yield _make_event(
                chunk,
                tenant_id,
                call_id,
                sequence_number,
                emitted_samples,
                sample_rate_hz,
                channel_count,
            )


def _make_event(
    frame: av.AudioFrame,
    tenant_id: str,
    call_id: str,
    sequence_number: int,
    emitted_samples: int,
    sample_rate_hz: int,
    channel_count: int,
) -> AudioChunkEvent:
    return AudioChunkEvent(
        tenant_id=tenant_id,
        call_id=call_id,
        sequence_number=sequence_number,
        received_at_utc=datetime.now(UTC),
        chunk_start_seconds=emitted_samples / sample_rate_hz,
        chunk_duration_seconds=frame.samples / sample_rate_hz,
        sample_rate_hz=sample_rate_hz,
        channel_count=channel_count,
        codec_name=f"pcm_{frame.format.name}",
        audio_bytes=frame.to_ndarray().tobytes(),
    )
