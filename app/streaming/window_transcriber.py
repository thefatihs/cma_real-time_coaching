"""In-memory transcription of tenant-aware ASR audio windows."""

from dataclasses import dataclass
from numbers import Real
from typing import Protocol

import av
import numpy as np
from numpy.typing import NDArray

from app.asr.models import TranscriptionResult
from app.streaming.audio_window import ASRAudioWindow


@dataclass(frozen=True, slots=True)
class WindowTranscriptionSegment:
    text: str
    relative_start_seconds: float
    relative_end_seconds: float
    absolute_start_seconds: float
    absolute_end_seconds: float


@dataclass(frozen=True, slots=True)
class WindowTranscriptionResult:
    tenant_id: str
    call_id: str
    first_sequence: int
    last_sequence: int
    window_start_seconds: float
    window_end_seconds: float
    window_duration_seconds: float
    text: str
    detected_language: str
    language_probability: float
    processing_time_seconds: float
    segments: tuple[WindowTranscriptionSegment, ...]


class InMemoryASREngine(Protocol):
    def transcribe_audio(self, audio: NDArray[np.float32]) -> TranscriptionResult: ...


class WindowTranscriber:
    """Convert a PCM window to a waveform and transcribe it exactly once."""

    _SUPPORTED_CODEC = "pcm_s16le"
    _TIMESTAMP_TOLERANCE_SECONDS = 1e-6

    def __init__(self, engine: InMemoryASREngine) -> None:
        self._engine = engine

    def transcribe(self, window: ASRAudioWindow) -> WindowTranscriptionResult:
        if (
            window.start_seconds < 0
            or window.end_seconds < window.start_seconds
            or window.duration_seconds < 0
        ):
            raise ValueError("ASR window times must be non-negative and ordered")

        waveform = prepare_whisper_waveform(window)
        engine_result = self._engine.transcribe_audio(waveform)

        segments: list[WindowTranscriptionSegment] = []
        for segment in engine_result.segments:
            start, end = _numeric_times(
                segment.start_seconds,
                segment.end_seconds,
                window.duration_seconds,
            )
            if not np.isfinite(start) or not np.isfinite(end):
                raise _invalid_times(start, end, window.duration_seconds)
            if end < start - self._TIMESTAMP_TOLERANCE_SECONDS:
                raise _invalid_times(start, end, window.duration_seconds)
            if end < start:
                end = start
            if end <= 0 or start >= window.duration_seconds:
                continue

            start = max(0.0, start)
            end = min(window.duration_seconds, end)
            absolute_start = min(window.end_seconds, window.start_seconds + start)
            absolute_end = min(window.end_seconds, window.start_seconds + end)
            segments.append(
                WindowTranscriptionSegment(
                    text=segment.text.strip(),
                    relative_start_seconds=start,
                    relative_end_seconds=end,
                    absolute_start_seconds=max(window.start_seconds, absolute_start),
                    absolute_end_seconds=max(absolute_start, absolute_end),
                )
            )

        transcript = " ".join(segment.text for segment in segments if segment.text)

        return WindowTranscriptionResult(
            tenant_id=window.tenant_id,
            call_id=window.call_id,
            first_sequence=window.first_sequence,
            last_sequence=window.last_sequence,
            window_start_seconds=window.start_seconds,
            window_end_seconds=window.end_seconds,
            window_duration_seconds=window.duration_seconds,
            text=transcript,
            detected_language=engine_result.language.strip(),
            language_probability=engine_result.language_probability,
            processing_time_seconds=max(0.0, engine_result.processing_time_seconds),
            segments=tuple(segments),
        )


WHISPER_SAMPLE_RATE_HZ = 16_000


def prepare_whisper_waveform(
    window: ASRAudioWindow,
) -> NDArray[np.float32]:
    """Convert interleaved PCM into mono 16 kHz float audio in memory."""
    if window.codec_name != WindowTranscriber._SUPPORTED_CODEC:
        raise ValueError(
            f"Unsupported audio codec {window.codec_name!r}; "
            f"only {WindowTranscriber._SUPPORTED_CODEC!r} is supported"
        )
    if not window.pcm_bytes:
        raise ValueError("PCM audio data cannot be empty")
    if window.sample_rate_hz <= 0:
        raise ValueError("PCM sample_rate_hz must be positive")
    if window.channel_count <= 0:
        raise ValueError("PCM channel_count must be positive")

    frame_width = 2 * window.channel_count
    if len(window.pcm_bytes) % frame_width:
        raise ValueError(
            "PCM audio data must contain complete interleaved channel frames"
        )

    samples = np.frombuffer(window.pcm_bytes, dtype="<i2")
    channels = samples.reshape(-1, window.channel_count).astype(np.float32)
    mono = np.ascontiguousarray(channels.mean(axis=1) / 32768.0, dtype=np.float32)
    if window.sample_rate_hz == WHISPER_SAMPLE_RATE_HZ:
        return mono

    frame = av.AudioFrame.from_ndarray(mono.reshape(1, -1), format="flt", layout="mono")
    frame.sample_rate = window.sample_rate_hz
    resampler = av.AudioResampler(
        format="flt", layout="mono", rate=WHISPER_SAMPLE_RATE_HZ
    )
    output_frames = [*resampler.resample(frame), *resampler.resample(None)]
    if not output_frames:
        raise ValueError("PCM audio could not be resampled")
    waveform = np.ascontiguousarray(
        np.concatenate([item.to_ndarray().reshape(-1) for item in output_frames]),
        dtype=np.float32,
    )
    np.clip(waveform, -1.0, 1.0, out=waveform)
    return waveform


def _numeric_times(start: object, end: object, duration: float) -> tuple[float, float]:
    if (
        isinstance(start, bool)
        or not isinstance(start, Real)
        or isinstance(end, bool)
        or not isinstance(end, Real)
    ):
        raise _invalid_times(start, end, duration)
    return float(start), float(end)


def _invalid_times(start: object, end: object, duration: float) -> ValueError:
    return ValueError(
        "ASR engine returned invalid segment times "
        f"(relative_start={_safe_time(start)}, relative_end={_safe_time(end)}, "
        f"window_duration={duration})"
    )


def _safe_time(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, Real):
        return "non-numeric"
    return str(float(value))
