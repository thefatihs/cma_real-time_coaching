"""Provider-neutral, bounded contracts for mono live-audio ingress."""

from collections.abc import Iterable
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_PROVIDER_NAME_LENGTH = 64
MAX_PROVIDER_STREAM_ID_LENGTH = 128
MAX_SCOPE_ID_LENGTH = 128
MAX_CODEC_NAME_LENGTH = 64
MAX_AUDIO_PAYLOAD_BYTES = 1_048_576
MAX_CHUNK_DURATION_SECONDS = 10.0
MAX_SAMPLE_RATE_HZ = 384_000


class LiveAudioEventType(str, Enum):
    START = "START"
    AUDIO_CHUNK = "AUDIO_CHUNK"
    END = "END"


class LiveAudioEndReason(str, Enum):
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class LiveAudioIngressEvent(BaseModel):
    """Immutable provider event with hidden, bounded audio content."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    tenant_id: str = Field(max_length=MAX_SCOPE_ID_LENGTH)
    call_id: str = Field(max_length=MAX_SCOPE_ID_LENGTH)
    provider_name: str = Field(max_length=MAX_PROVIDER_NAME_LENGTH)
    provider_stream_id: str = Field(max_length=MAX_PROVIDER_STREAM_ID_LENGTH)
    event_type: LiveAudioEventType
    sequence_number: int = Field(ge=0)
    codec_name: str = Field(max_length=MAX_CODEC_NAME_LENGTH)
    sample_rate_hz: int = Field(ge=1, le=MAX_SAMPLE_RATE_HZ)
    channel_count: int = Field(ge=1, le=1)
    captured_at_utc: datetime
    arrived_at_utc: datetime
    duration_seconds: float = Field(ge=0.0, le=MAX_CHUNK_DURATION_SECONDS)
    audio_payload: bytes = Field(
        default=b"", max_length=MAX_AUDIO_PAYLOAD_BYTES, repr=False
    )
    end_reason: LiveAudioEndReason | None = None

    @field_validator(
        "tenant_id",
        "call_id",
        "provider_name",
        "provider_stream_id",
        "codec_name",
    )
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("required_text")
        return cleaned

    @field_validator("duration_seconds")
    @classmethod
    def validate_duration(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("invalid_duration")
        return value

    @field_validator("captured_at_utc", "arrived_at_utc")
    @classmethod
    def validate_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timezone_required")
        return value

    @model_validator(mode="after")
    def validate_event_shape(self) -> Self:
        if self.arrived_at_utc < self.captured_at_utc:
            raise ValueError("arrival_precedes_capture")
        if self.event_type is LiveAudioEventType.AUDIO_CHUNK:
            if not self.audio_payload or self.duration_seconds <= 0:
                raise ValueError("audio_chunk_payload_required")
            if self.end_reason is not None:
                raise ValueError("audio_chunk_end_reason_forbidden")
        elif self.audio_payload or self.duration_seconds != 0:
            raise ValueError("control_event_audio_forbidden")
        elif self.event_type is LiveAudioEventType.END:
            if self.end_reason is None:
                raise ValueError("end_reason_required")
        elif self.end_reason is not None:
            raise ValueError("start_end_reason_forbidden")
        return self


class LiveAudioProviderAdapterProtocol(Protocol):
    """Injectable transport adapter contract; no transport is selected here."""

    def events(self) -> Iterable[LiveAudioIngressEvent]: ...
