"""Validated domain models for scoped document registry operations."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

MAX_REGISTRY_IDENTIFIER_CHARACTERS = 255
MAX_STORAGE_OBJECT_KEY_CHARACTERS = 512
MAX_INGESTION_ATTEMPTS = 10
MAX_DOCUMENT_LIST_PAGE_SIZE = 50
MAX_CHUNK_COUNT = 2_147_483_647

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STORAGE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9/_-]*$")
_MEDIA_TYPES = frozenset({"application/pdf", "text/plain", "text/markdown"})


class DocumentIngestionState(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DocumentIngestionPhase(str, Enum):
    VALIDATION = "VALIDATION"
    STORAGE = "STORAGE"
    EXTRACTION = "EXTRACTION"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    VECTOR_WRITE = "VECTOR_WRITE"
    FINALIZE = "FINALIZE"


class DocumentReadiness(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DocumentOperationPhase(str, Enum):
    REGISTRY_CREATE = "REGISTRY_CREATE"
    JOB_CLAIM = "JOB_CLAIM"
    EMBEDDING = "EMBEDDING"
    VECTOR_WRITE = "VECTOR_WRITE"
    FINALIZE = "FINALIZE"
    RETRY = "RETRY"
    DELETE = "DELETE"
    PROGRESS = "PROGRESS"
    CANCEL = "CANCEL"
    LIST = "LIST"


_OPERATION_FAILURE_MESSAGES = {
    phase: f"Document ingestion failed during {phase.value}."
    for phase in DocumentOperationPhase
}


class DocumentRegistryError(RuntimeError):
    """Fixed secret-free registry or orchestration failure."""

    def __init__(self, phase: DocumentOperationPhase) -> None:
        self.phase = phase
        super().__init__(_OPERATION_FAILURE_MESSAGES[phase])


class DocumentRegistryCreateRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    knowledge_base_id: str
    document_id: str
    job_id: str
    original_filename: str
    media_type: str
    byte_size: int
    sha256_hex: str
    storage_object_key: str
    total_chunks: int
    initial_phase: DocumentIngestionPhase = DocumentIngestionPhase.EMBEDDING

    @field_validator("tenant_id", "knowledge_base_id", "document_id", "job_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        return _bounded_identifier(value)

    @field_validator("original_filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        return _display_filename(value)

    @field_validator("media_type")
    @classmethod
    def validate_media_type(cls, value: str) -> str:
        return _media_type(value)

    @field_validator("byte_size")
    @classmethod
    def validate_byte_size(cls, value: int) -> int:
        return _positive_byte_size(value)

    @field_validator("sha256_hex")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256_hex is invalid")
        return value

    @field_validator("storage_object_key")
    @classmethod
    def validate_storage_key(cls, value: str) -> str:
        return validate_storage_object_key(value)

    @field_validator("total_chunks")
    @classmethod
    def validate_total_chunks(cls, value: int) -> int:
        if type(value) is not int or not 0 <= value <= MAX_CHUNK_COUNT:
            raise ValueError("total_chunks is outside the allowed range")
        return value

    @field_validator("initial_phase")
    @classmethod
    def validate_initial_phase(
        cls, value: DocumentIngestionPhase
    ) -> DocumentIngestionPhase:
        if value not in {
            DocumentIngestionPhase.VALIDATION,
            DocumentIngestionPhase.STORAGE,
            DocumentIngestionPhase.EXTRACTION,
            DocumentIngestionPhase.EMBEDDING,
        }:
            raise ValueError("initial_phase is invalid")
        return value


class DocumentRegistryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    knowledge_base_id: str
    document_id: str
    original_filename: str
    media_type: str
    byte_size: int
    storage_object_key: str
    created_at_utc: datetime
    ready_at_utc: datetime | None = None

    @field_validator("tenant_id", "knowledge_base_id", "document_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        return _bounded_identifier(value)

    @field_validator("original_filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        return _display_filename(value)

    @field_validator("media_type")
    @classmethod
    def validate_media_type(cls, value: str) -> str:
        return _media_type(value)

    @field_validator("byte_size")
    @classmethod
    def validate_byte_size(cls, value: int) -> int:
        return _positive_byte_size(value)

    @field_validator("storage_object_key")
    @classmethod
    def validate_storage_key(cls, value: str) -> str:
        return validate_storage_object_key(value)

    @model_validator(mode="after")
    def validate_timestamp_order(self) -> DocumentRegistryRecord:
        _aware_datetime(self.created_at_utc)
        if self.ready_at_utc is not None:
            _aware_datetime(self.ready_at_utc)
            if self.ready_at_utc < self.created_at_utc:
                raise ValueError("ready_at_utc cannot precede created_at_utc")
        return self


class DocumentIngestionJob(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    knowledge_base_id: str
    job_id: str
    document_id: str
    state: DocumentIngestionState
    phase: DocumentIngestionPhase
    processed_chunks: int
    total_chunks: int
    attempt_count: int
    created_at_utc: datetime
    started_at_utc: datetime | None = None
    updated_at_utc: datetime
    finished_at_utc: datetime | None = None

    @field_validator("tenant_id", "knowledge_base_id", "job_id", "document_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        return _bounded_identifier(value)

    @field_validator("processed_chunks", "total_chunks", "attempt_count")
    @classmethod
    def validate_nonnegative(cls, value: int) -> int:
        if type(value) is not int or not 0 <= value <= MAX_CHUNK_COUNT:
            raise ValueError("job counters are outside the allowed range")
        return value

    @model_validator(mode="after")
    def validate_job(self) -> DocumentIngestionJob:
        if self.total_chunks and self.processed_chunks > self.total_chunks:
            raise ValueError("processed_chunks cannot exceed total_chunks")
        if self.attempt_count > MAX_INGESTION_ATTEMPTS:
            raise ValueError("attempt_count exceeds the retry bound")
        for value in (
            self.created_at_utc,
            self.started_at_utc,
            self.updated_at_utc,
            self.finished_at_utc,
        ):
            if value is not None:
                _aware_datetime(value)
        if self.updated_at_utc < self.created_at_utc:
            raise ValueError("updated_at_utc cannot precede created_at_utc")
        if self.started_at_utc is not None and (
            self.started_at_utc < self.created_at_utc
            or self.updated_at_utc < self.started_at_utc
        ):
            raise ValueError("started_at_utc is inconsistent")
        if self.finished_at_utc is not None and (
            self.finished_at_utc < (self.started_at_utc or self.created_at_utc)
            or self.finished_at_utc < self.updated_at_utc
        ):
            raise ValueError("finished_at_utc is inconsistent")
        if self.state is DocumentIngestionState.QUEUED and (
            self.started_at_utc is not None or self.finished_at_utc is not None
        ):
            raise ValueError("queued job timestamps are inconsistent")
        if self.state is DocumentIngestionState.PROCESSING and (
            self.started_at_utc is None or self.finished_at_utc is not None
        ):
            raise ValueError("processing job timestamps are inconsistent")
        if (
            self.state
            in {
                DocumentIngestionState.SUCCEEDED,
                DocumentIngestionState.FAILED,
                DocumentIngestionState.CANCELLED,
            }
            and self.finished_at_utc is None
        ):
            raise ValueError("finished job timestamp is missing")
        if self.state is DocumentIngestionState.SUCCEEDED and (
            self.phase is not DocumentIngestionPhase.FINALIZE
            or self.started_at_utc is None
            or self.total_chunks <= 0
            or self.processed_chunks != self.total_chunks
        ):
            raise ValueError("succeeded job is inconsistent")
        return self


class DocumentRegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    document: DocumentRegistryRecord
    job: DocumentIngestionJob
    readiness: DocumentReadiness

    @model_validator(mode="after")
    def validate_scope_and_readiness(self) -> DocumentRegistryEntry:
        if (
            self.document.tenant_id != self.job.tenant_id
            or self.document.knowledge_base_id != self.job.knowledge_base_id
            or self.document.document_id != self.job.document_id
        ):
            raise ValueError("document and job scope must match")
        if self.readiness is not derive_document_readiness(self.document, self.job):
            raise ValueError("document readiness is inconsistent")
        return self


class DocumentRegistryCreateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry: DocumentRegistryEntry
    created: bool


@dataclass(frozen=True, slots=True)
class DocumentListCursor:
    """Opaque repository cursor; presentation code must never serialize it."""

    _created_at_utc: datetime = field(repr=False)
    _document_id: str = field(repr=False)

    def __post_init__(self) -> None:
        _aware_datetime(self._created_at_utc)
        _bounded_identifier(self._document_id)


@dataclass(frozen=True, slots=True)
class DocumentListPage:
    entries: tuple[DocumentRegistryEntry, ...]
    continuation: DocumentListCursor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, DocumentRegistryEntry) for entry in self.entries
        ):
            raise ValueError("document list entries are invalid")
        if self.continuation is not None and not isinstance(
            self.continuation, DocumentListCursor
        ):
            raise ValueError("document list continuation is invalid")

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, index: int) -> DocumentRegistryEntry:
        return self.entries[index]


class DocumentDeletionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    storage_object_key: str

    @field_validator("storage_object_key")
    @classmethod
    def validate_storage_key(cls, value: str) -> str:
        return validate_storage_object_key(value)


def derive_document_readiness(
    document: DocumentRegistryRecord,
    job: DocumentIngestionJob,
) -> DocumentReadiness:
    if (
        job.state is DocumentIngestionState.SUCCEEDED
        and document.ready_at_utc is not None
    ):
        return DocumentReadiness.READY
    if job.state is DocumentIngestionState.FAILED:
        return DocumentReadiness.FAILED
    if job.state is DocumentIngestionState.CANCELLED:
        return DocumentReadiness.CANCELLED
    return DocumentReadiness.PENDING


def validate_storage_object_key(value: object) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value) > MAX_STORAGE_OBJECT_KEY_CHARACTERS
        or not _STORAGE_KEY_PATTERN.fullmatch(value)
        or value.endswith("/")
        or "//" in value
        or any(part == ".." for part in value.split("/"))
        or _has_control(value)
    ):
        raise ValueError("storage_object_key is invalid")
    return value


def _display_filename(value: object) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value) > MAX_REGISTRY_IDENTIFIER_CHARACTERS
        or value in {".", ".."}
        or any(character in value for character in ("/", "\\", ":"))
        or _has_control(value)
    ):
        raise ValueError("original_filename is invalid")
    return value


def _media_type(value: object) -> str:
    if not isinstance(value, str) or value not in _MEDIA_TYPES:
        raise ValueError("media_type is unsupported")
    return value


def _positive_byte_size(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("byte_size must be positive")
    return value


def _bounded_identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value) > MAX_REGISTRY_IDENTIFIER_CHARACTERS
        or _has_control(value)
    ):
        raise ValueError("identifier is invalid")
    return value


def _has_control(value: str) -> bool:
    return any(unicodedata.category(character).startswith("C") for character in value)


def _aware_datetime(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
