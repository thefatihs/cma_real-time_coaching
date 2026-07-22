"""In-memory transcription of tenant-aware ASR audio windows."""

from dataclasses import dataclass
from typing import Protocol

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
        if window.codec_name != self._SUPPORTED_CODEC:
            raise ValueError(
                f"Unsupported audio codec {window.codec_name!r}; "
                f"only {self._SUPPORTED_CODEC!r} is supported"
            )
        if not window.pcm_bytes:
            raise ValueError("PCM audio data cannot be empty")
        if (
            window.start_seconds < 0
            or window.end_seconds < window.start_seconds
            or window.duration_seconds < 0
        ):
            raise ValueError("ASR window times must be non-negative and ordered")

        frame_width = 2 * window.channel_count
        if len(window.pcm_bytes) % frame_width:
            raise ValueError("PCM audio data must contain complete channel frames")

        samples = np.frombuffer(window.pcm_bytes, dtype="<i2")
        waveform = samples.reshape(-1, window.channel_count).astype(np.float32)
        waveform = waveform.mean(axis=1) / 32768.0
        engine_result = self._engine.transcribe_audio(waveform)

        segments: list[WindowTranscriptionSegment] = []
        for segment in engine_result.segments:
            start = float(segment.start_seconds)
            end = float(segment.end_seconds)
            if (
                not np.isfinite(start)
                or not np.isfinite(end)
                or start < 0
                or end < start
            ):
                raise ValueError("ASR engine returned invalid segment times")
            if (
                start > window.duration_seconds + self._TIMESTAMP_TOLERANCE_SECONDS
                or end > window.duration_seconds + self._TIMESTAMP_TOLERANCE_SECONDS
            ):
                raise ValueError("ASR engine returned invalid segment times")
            start = min(start, window.duration_seconds)
            end = min(end, window.duration_seconds)
            if end < start:
                raise ValueError("ASR engine returned invalid segment times")
            segments.append(
                WindowTranscriptionSegment(
                    text=segment.text.strip(),
                    relative_start_seconds=start,
                    relative_end_seconds=end,
                    absolute_start_seconds=window.start_seconds + start,
                    absolute_end_seconds=window.start_seconds + end,
                )
            )

        return WindowTranscriptionResult(
            tenant_id=window.tenant_id,
            call_id=window.call_id,
            first_sequence=window.first_sequence,
            last_sequence=window.last_sequence,
            window_start_seconds=window.start_seconds,
            window_end_seconds=window.end_seconds,
            window_duration_seconds=window.duration_seconds,
            text=engine_result.text.strip(),
            detected_language=engine_result.language.strip(),
            language_probability=engine_result.language_probability,
            processing_time_seconds=max(0.0, engine_result.processing_time_seconds),
            segments=tuple(segments),
        )
