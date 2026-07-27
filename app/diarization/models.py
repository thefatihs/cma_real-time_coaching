"""Immutable models for provider-neutral speaker diarization."""

from enum import Enum
from math import isfinite
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SpeakerRole(str, Enum):
    AGENT = "AGENT"
    CUSTOMER = "CUSTOMER"
    UNKNOWN = "UNKNOWN"
    OVERLAP = "OVERLAP"


class _SpeakerAssignment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    call_id: str
    start_seconds: float
    end_seconds: float
    local_speaker_ids: tuple[str, ...]
    global_speaker_id: str | None = None
    role: SpeakerRole = SpeakerRole.UNKNOWN
    speaker_confidence: float | None = None
    role_confidence: float | None = None

    @field_validator("tenant_id", "call_id")
    @classmethod
    def validate_scope(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "scope"))

    @field_validator("global_speaker_id")
    @classmethod
    def validate_global_speaker_id(cls, value: str | None) -> str | None:
        return None if value is None else _required_text(value, "global_speaker_id")

    @field_validator("local_speaker_ids")
    @classmethod
    def validate_local_speaker_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(_required_text(value, "local_speaker_id") for value in values)
        if not cleaned:
            raise ValueError("at least one local speaker ID is required")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("local speaker IDs must be unique")
        return cleaned

    @field_validator("start_seconds", "end_seconds")
    @classmethod
    def validate_timestamp(cls, value: float, info: object) -> float:
        return _timestamp(value, getattr(info, "field_name", "timestamp"))

    @field_validator("speaker_confidence", "role_confidence")
    @classmethod
    def validate_confidence(cls, value: float | None, info: object) -> float | None:
        if value is None:
            return None
        return _confidence(value, getattr(info, "field_name", "confidence"))

    @model_validator(mode="after")
    def validate_assignment(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        speaker_count = len(self.local_speaker_ids)
        if self.role is SpeakerRole.OVERLAP and speaker_count < 2:
            raise ValueError("OVERLAP requires at least two local speaker IDs")
        if self.role is not SpeakerRole.OVERLAP and speaker_count != 1:
            raise ValueError("non-OVERLAP roles require exactly one local speaker ID")
        return self


class DiarizationTurn(_SpeakerAssignment):
    """One absolute speaker turn within a call."""


class DiarizedWord(_SpeakerAssignment):
    """One absolute transcript word with speaker metadata."""

    transcript_revision: int
    text: str

    @field_validator("transcript_revision")
    @classmethod
    def validate_revision(cls, value: int) -> int:
        if value < 0:
            raise ValueError("transcript_revision cannot be negative")
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _required_text(value, "text")


class DiarizedTranscriptEvent(BaseModel):
    """Companion metadata for one existing transcript event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    call_id: str
    transcript_event_id: str
    transcript_revision: int
    start_seconds: float
    end_seconds: float
    turns: tuple[DiarizationTurn, ...] = ()
    words: tuple[DiarizedWord, ...] = ()

    @field_validator("tenant_id", "call_id", "transcript_event_id")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "identifier"))

    @field_validator("transcript_revision")
    @classmethod
    def validate_revision(cls, value: int) -> int:
        if value < 0:
            raise ValueError("transcript_revision cannot be negative")
        return value

    @field_validator("start_seconds", "end_seconds")
    @classmethod
    def validate_timestamp(cls, value: float, info: object) -> float:
        return _timestamp(value, getattr(info, "field_name", "timestamp"))

    @model_validator(mode="after")
    def validate_children(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        _validate_child_sequence(
            self.turns,
            tenant_id=self.tenant_id,
            call_id=self.call_id,
            start_seconds=self.start_seconds,
            end_seconds=self.end_seconds,
            child_name="turn",
        )
        _validate_child_sequence(
            self.words,
            tenant_id=self.tenant_id,
            call_id=self.call_id,
            start_seconds=self.start_seconds,
            end_seconds=self.end_seconds,
            child_name="word",
            transcript_revision=self.transcript_revision,
        )
        return self


class DiarizationRequest(BaseModel):
    """Trusted mono, in-memory audio for one absolute ASR window."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    tenant_id: str
    call_id: str
    window_start_seconds: float
    window_end_seconds: float
    sample_rate_hz: int
    mono_audio: tuple[float, ...] = Field(repr=False)

    @field_validator("tenant_id", "call_id")
    @classmethod
    def validate_scope(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "scope"))

    @field_validator("window_start_seconds", "window_end_seconds")
    @classmethod
    def validate_timestamp(cls, value: float, info: object) -> float:
        return _timestamp(value, getattr(info, "field_name", "timestamp"))

    @field_validator("sample_rate_hz")
    @classmethod
    def validate_sample_rate(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("sample_rate_hz must be positive")
        return value

    @field_validator("mono_audio")
    @classmethod
    def validate_mono_audio(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values:
            raise ValueError("mono_audio cannot be empty")
        if any(not isfinite(value) or not -1.0 <= value <= 1.0 for value in values):
            raise ValueError("mono_audio must contain finite normalized samples")
        return values

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.window_end_seconds <= self.window_start_seconds:
            raise ValueError(
                "window_end_seconds must be greater than window_start_seconds"
            )
        return self


class DiarizationResult(BaseModel):
    """Immutable diarization output for one trusted request window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    call_id: str
    window_start_seconds: float
    window_end_seconds: float
    turns: tuple[DiarizationTurn, ...]

    @field_validator("tenant_id", "call_id")
    @classmethod
    def validate_scope(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "scope"))

    @field_validator("window_start_seconds", "window_end_seconds")
    @classmethod
    def validate_timestamp(cls, value: float, info: object) -> float:
        return _timestamp(value, getattr(info, "field_name", "timestamp"))

    @model_validator(mode="after")
    def validate_turns(self) -> Self:
        if self.window_end_seconds <= self.window_start_seconds:
            raise ValueError(
                "window_end_seconds must be greater than window_start_seconds"
            )
        _validate_child_sequence(
            self.turns,
            tenant_id=self.tenant_id,
            call_id=self.call_id,
            start_seconds=self.window_start_seconds,
            end_seconds=self.window_end_seconds,
            child_name="turn",
        )
        return self


def _validate_child_sequence(
    children: tuple[DiarizationTurn, ...] | tuple[DiarizedWord, ...],
    *,
    tenant_id: str,
    call_id: str,
    start_seconds: float,
    end_seconds: float,
    child_name: str,
    transcript_revision: int | None = None,
) -> None:
    keys = [
        (
            child.start_seconds,
            child.end_seconds,
            child.local_speaker_ids,
        )
        for child in children
    ]
    if keys != sorted(keys):
        raise ValueError(f"{child_name}s must be in deterministic timestamp order")
    for child in children:
        if child.tenant_id != tenant_id or child.call_id != call_id:
            raise ValueError(f"{child_name} scope does not match parent")
        if child.start_seconds < start_seconds or child.end_seconds > end_seconds:
            raise ValueError(f"{child_name} must remain inside parent time range")
        if (
            transcript_revision is not None
            and isinstance(child, DiarizedWord)
            and child.transcript_revision != transcript_revision
        ):
            raise ValueError(f"{child_name} revision does not match parent")


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def _timestamp(value: float, field_name: str) -> float:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return value


def _confidence(value: float, field_name: str) -> float:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0.0 and 1.0")
    return value
