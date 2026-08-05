"""Secret-free Turkish projections for dashboard document management."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from app.ingestion.registry_models import (
    DocumentIngestionPhase,
    DocumentIngestionState,
    DocumentReadiness,
    DocumentRegistryEntry,
)
from app.ingestion.upload_preparation import MAX_UPLOAD_BYTES


class DocumentRuntimeStatus(str, Enum):
    READY = "READY"
    DISABLED = "DISABLED"
    UNAVAILABLE = "UNAVAILABLE"


RUNTIME_STATUS_MESSAGES = {
    DocumentRuntimeStatus.READY: "Bilgi tabanı hazır",
    DocumentRuntimeStatus.DISABLED: "Bilgi tabanı yapılandırılmadı",
    DocumentRuntimeStatus.UNAVAILABLE: (
        "Bilgi tabanı geçici olarak kullanılamıyor; temel görüşme analizi devam ediyor"
    ),
}

_PROGRESS_MESSAGES = {
    DocumentIngestionPhase.VALIDATION: "Doğrulanıyor",
    DocumentIngestionPhase.STORAGE: "Saklanıyor",
    DocumentIngestionPhase.EXTRACTION: "Metin çıkarılıyor",
    DocumentIngestionPhase.CHUNKING: "Parçalanıyor",
    DocumentIngestionPhase.EMBEDDING: "Embedding oluşturuluyor",
    DocumentIngestionPhase.VECTOR_WRITE: "Vektörler kaydediliyor",
    DocumentIngestionPhase.FINALIZE: "Tamamlanıyor",
}
_MEDIA_LABELS = {
    "application/pdf": "PDF",
    "text/plain": "TXT",
    "text/markdown": "Markdown",
}


@dataclass(frozen=True, slots=True)
class SafeUploadSelection:
    filename: str
    media_type: str
    media_label: str
    byte_size: int
    formatted_size: str


@dataclass(frozen=True, slots=True)
class DocumentProgressViewModel:
    label: str
    active: bool
    processed_chunks: int | None = None
    total_chunks: int | None = None


@dataclass(frozen=True, slots=True)
class DocumentListItemViewModel:
    filename: str
    media_label: str
    formatted_size: str
    readiness_label: str
    created_at_utc: str
    action_token: str
    active: bool


@dataclass(frozen=True, slots=True)
class DocumentSectionViewModel:
    runtime_status: DocumentRuntimeStatus
    runtime_message: str
    manager_busy: bool
    progress: DocumentProgressViewModel | None
    documents: tuple[DocumentListItemViewModel, ...]
    warning: str | None = None


def safe_upload_selection(
    *, filename: object, media_type: object, byte_size: object
) -> SafeUploadSelection:
    if (
        not isinstance(filename, str)
        or not filename
        or filename != filename.strip()
        or len(filename) > 255
        or filename in {".", ".."}
        or any(character in filename for character in ("/", "\\", ":", "\0"))
        or any(
            unicodedata.category(character).startswith("C") for character in filename
        )
    ):
        raise ValueError("Belge adı geçersiz.")
    expected = _media_type_for_filename(filename)
    if not isinstance(media_type, str) or media_type != expected:
        raise ValueError("Yalnızca PDF, TXT ve Markdown belgeleri desteklenir.")
    if type(byte_size) is not int or byte_size <= 0:
        raise ValueError("Belge boş olamaz.")
    if byte_size > MAX_UPLOAD_BYTES:
        raise ValueError("Belge boyutu en fazla 10 MiB olabilir.")
    return SafeUploadSelection(
        filename=filename,
        media_type=media_type,
        media_label=_MEDIA_LABELS[media_type],
        byte_size=byte_size,
        formatted_size=format_byte_size(byte_size),
    )


def project_progress(entry: DocumentRegistryEntry) -> DocumentProgressViewModel:
    job = entry.job
    if job.state is DocumentIngestionState.QUEUED:
        return DocumentProgressViewModel("Bekliyor", active=True)
    if job.state is DocumentIngestionState.SUCCEEDED:
        return DocumentProgressViewModel("Hazır", active=False)
    if job.state is DocumentIngestionState.FAILED:
        return DocumentProgressViewModel("Başarısız", active=False)
    if job.state is DocumentIngestionState.CANCELLED:
        return DocumentProgressViewModel("İptal edildi", active=False)
    total = job.total_chunks or None
    return DocumentProgressViewModel(
        _PROGRESS_MESSAGES[job.phase],
        active=True,
        processed_chunks=job.processed_chunks if total is not None else None,
        total_chunks=total,
    )


def project_document(
    entry: DocumentRegistryEntry, *, action_token: str
) -> DocumentListItemViewModel:
    readiness = {
        DocumentReadiness.READY: "Hazır",
        DocumentReadiness.PENDING: "İşleniyor",
        DocumentReadiness.FAILED: "Başarısız",
        DocumentReadiness.CANCELLED: "İptal edildi",
    }[entry.readiness]
    return DocumentListItemViewModel(
        filename=entry.document.original_filename,
        media_label=_MEDIA_LABELS[entry.document.media_type],
        formatted_size=format_byte_size(entry.document.byte_size),
        readiness_label=readiness,
        created_at_utc=_format_utc(entry.document.created_at_utc),
        action_token=action_token,
        active=entry.job.state
        in {DocumentIngestionState.QUEUED, DocumentIngestionState.PROCESSING},
    )


def format_byte_size(byte_size: int) -> str:
    if byte_size < 1024:
        return f"{byte_size} B"
    if byte_size < 1024 * 1024:
        return f"{byte_size / 1024:.1f} KiB"
    return f"{byte_size / (1024 * 1024):.1f} MiB"


def _media_type_for_filename(filename: str) -> str:
    lowered = filename.casefold()
    if lowered.endswith(".pdf"):
        return "application/pdf"
    if lowered.endswith(".txt"):
        return "text/plain"
    if lowered.endswith((".md", ".markdown")):
        return "text/markdown"
    raise ValueError("Yalnızca PDF, TXT ve Markdown belgeleri desteklenir.")


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
