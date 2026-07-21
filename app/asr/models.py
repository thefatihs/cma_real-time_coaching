from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration_seconds: float
    processing_time_seconds: float
    segments: list[TranscriptionSegment]
