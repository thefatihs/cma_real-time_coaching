from datetime import datetime
from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class TranscriptKind(str, Enum):
    PARTIAL = "PARTIAL"
    STABLE = "STABLE"
    FINAL = "FINAL"


class CoachingAction(str, Enum):
    NO_ACTION = "NO_ACTION"
    TEMPLATE_ACTION = "TEMPLATE_ACTION"
    RAG_ACTION = "RAG_ACTION"
    ESCALATE = "ESCALATE"


class SuggestionPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AudioChunkEvent(BaseModel):
    tenant_id: str
    call_id: str
    sequence_number: int
    received_at_utc: datetime
    chunk_start_seconds: float
    chunk_duration_seconds: float
    sample_rate_hz: int
    channel_count: int
    codec_name: str
    audio_bytes: bytes = Field(repr=False)

    @field_validator("tenant_id", "call_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))

    @field_validator("codec_name")
    @classmethod
    def normalize_codec_name(cls, value: str) -> str:
        return canonical_audio_codec_name(value)

    @field_validator("sequence_number", "chunk_start_seconds")
    @classmethod
    def validate_non_negative(cls, value: int | float, info: object) -> int | float:
        if value < 0:
            raise ValueError(
                f"{getattr(info, 'field_name', 'value')} cannot be negative"
            )
        return value

    @field_validator("chunk_duration_seconds", "sample_rate_hz", "channel_count")
    @classmethod
    def validate_positive(cls, value: int | float, info: object) -> int | float:
        if value <= 0:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be positive")
        return value

    @field_validator("audio_bytes")
    @classmethod
    def validate_audio_bytes(cls, value: bytes) -> bytes:
        if not value:
            raise ValueError("audio_bytes cannot be empty")
        return value

    @field_validator("received_at_utc")
    @classmethod
    def validate_received_time(cls, value: datetime) -> datetime:
        return _aware_datetime(value, "received_at_utc")

    def metadata_summary(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "call_id": self.call_id,
            "sequence_number": self.sequence_number,
            "received_at_utc": self.received_at_utc.isoformat(),
            "chunk_start_seconds": self.chunk_start_seconds,
            "chunk_duration_seconds": self.chunk_duration_seconds,
            "sample_rate_hz": self.sample_rate_hz,
            "channel_count": self.channel_count,
            "codec_name": self.codec_name,
        }


class TranscriptEvent(BaseModel):
    tenant_id: str
    call_id: str
    event_id: str
    kind: TranscriptKind
    text: str
    start_seconds: float
    end_seconds: float
    revision: int
    created_at_utc: datetime
    source_chunk_sequence: int | None = None

    @field_validator("tenant_id", "call_id", "event_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return _required_text(value, "text")

    @field_validator("start_seconds", "revision")
    @classmethod
    def validate_non_negative(cls, value: int | float, info: object) -> int | float:
        if value < 0:
            raise ValueError(
                f"{getattr(info, 'field_name', 'value')} cannot be negative"
            )
        return value

    @field_validator("source_chunk_sequence")
    @classmethod
    def validate_source_sequence(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("source_chunk_sequence cannot be negative")
        return value

    @field_validator("created_at_utc")
    @classmethod
    def validate_created_time(cls, value: datetime) -> datetime:
        return _aware_datetime(value, "created_at_utc")

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        if self.end_seconds < self.start_seconds:
            raise ValueError(
                "end_seconds must be greater than or equal to start_seconds"
            )
        return self


class ClassificationLabel(BaseModel):
    name: str
    score: float

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _required_text(value, "name")

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        return _probability(value, "score")


class ClassificationResultEvent(BaseModel):
    tenant_id: str
    call_id: str
    transcript_event_id: str
    labels: list[ClassificationLabel]
    action: CoachingAction
    model_id: str
    processing_time_ms: float | None = None
    created_at_utc: datetime

    @field_validator("tenant_id", "call_id", "transcript_event_id", "model_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))

    @field_validator("processing_time_ms")
    @classmethod
    def validate_processing_time(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("processing_time_ms cannot be negative")
        return value

    @field_validator("created_at_utc")
    @classmethod
    def validate_created_time(cls, value: datetime) -> datetime:
        return _aware_datetime(value, "created_at_utc")

    @model_validator(mode="after")
    def normalize_labels(self) -> Self:
        names = [label.name for label in self.labels]
        if len(names) != len(set(names)):
            raise ValueError("classification label names must be unique")
        self.labels = sorted(self.labels, key=lambda label: label.score, reverse=True)
        return self


class RetrievalRequestEvent(BaseModel):
    tenant_id: str
    call_id: str
    transcript_event_id: str
    query: str
    knowledge_base_id: str
    top_k: int
    minimum_score: float
    created_at_utc: datetime

    @field_validator(
        "tenant_id", "call_id", "transcript_event_id", "query", "knowledge_base_id"
    )
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k must be positive")
        return value

    @field_validator("minimum_score")
    @classmethod
    def validate_minimum_score(cls, value: float) -> float:
        return _probability(value, "minimum_score")

    @field_validator("created_at_utc")
    @classmethod
    def validate_created_time(cls, value: datetime) -> datetime:
        return _aware_datetime(value, "created_at_utc")


class CoachingSuggestionEvent(BaseModel):
    tenant_id: str
    call_id: str
    suggestion_id: str
    source_transcript_event_id: str
    action: CoachingAction
    priority: SuggestionPriority
    title: str
    suggestion: str
    evidence_ids: list[str] = Field(default_factory=list)
    expires_after_seconds: float | None = None
    created_at_utc: datetime

    @field_validator(
        "tenant_id",
        "call_id",
        "suggestion_id",
        "source_transcript_event_id",
        "title",
        "suggestion",
    )
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))

    @field_validator("evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, values: list[str]) -> list[str]:
        return _unique_non_empty(values, "evidence ID")

    @field_validator("expires_after_seconds")
    @classmethod
    def validate_expiration(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("expires_after_seconds cannot be negative")
        return value

    @field_validator("created_at_utc")
    @classmethod
    def validate_created_time(cls, value: datetime) -> datetime:
        return _aware_datetime(value, "created_at_utc")


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def canonical_audio_codec_name(value: str) -> str:
    """Normalize the two accepted names for little-endian signed 16-bit PCM."""
    cleaned = _required_text(value, "codec_name")
    if cleaned in {"pcm_s16", "pcm_s16le"}:
        return "pcm_s16le"
    return cleaned


def _unique_non_empty(values: list[str], field_name: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _required_text(value, field_name)
        if cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _probability(value: float, field_name: str) -> float:
    if not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return value


def _aware_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value
