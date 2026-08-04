"""Secure in-memory preparation of bounded dashboard RAG documents."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.ingestion.chunking import FixedCharacterDocumentChunker
from app.ingestion.document_source import TextDocumentSource
from app.ingestion.models import DocumentChunkInput, DocumentIngestionRequest

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_EXTRACTED_CHARACTERS = 1_000_000
MAX_DISPLAY_FILENAME_CHARACTERS = 255
MAX_SCOPE_IDENTIFIER_CHARACTERS = 255
DEFAULT_UPLOAD_CHUNK_CHARACTERS = 1_000

_UTF8_BOM = b"\xef\xbb\xbf"
_PDF_SIGNATURE = b"%PDF-"
_MEDIA_TYPE_BY_EXTENSION = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
}
_SUPPORTED_MEDIA_TYPES = frozenset(_MEDIA_TYPE_BY_EXTENSION.values())


class UploadDocumentFailure(str, Enum):
    UNSUPPORTED_TYPE = "unsupported_type"
    INVALID_FILENAME = "invalid_filename"
    INVALID_IDENTIFIER = "invalid_identifier"
    UPLOAD_TOO_LARGE = "upload_too_large"
    INVALID_ENCODING = "invalid_encoding"
    INVALID_PDF = "invalid_pdf"
    ENCRYPTED_PDF = "encrypted_pdf"
    UNSAFE_CONTENT = "unsafe_content"
    PAGE_LIMIT = "page_limit"
    TEXT_LIMIT = "text_limit"
    EMPTY_DOCUMENT = "empty_document"
    EXTRACTION_UNAVAILABLE = "extraction_unavailable"


_FAILURE_MESSAGES = {
    UploadDocumentFailure.UNSUPPORTED_TYPE: "Document type is not supported.",
    UploadDocumentFailure.INVALID_FILENAME: "Document filename is invalid.",
    UploadDocumentFailure.INVALID_IDENTIFIER: "Document scope is invalid.",
    UploadDocumentFailure.UPLOAD_TOO_LARGE: "Document upload exceeds the limit.",
    UploadDocumentFailure.INVALID_ENCODING: "Document encoding is invalid.",
    UploadDocumentFailure.INVALID_PDF: "PDF document is invalid.",
    UploadDocumentFailure.ENCRYPTED_PDF: "Encrypted PDF documents are not supported.",
    UploadDocumentFailure.UNSAFE_CONTENT: "Document contains unsupported active content.",
    UploadDocumentFailure.PAGE_LIMIT: "PDF document exceeds the page limit.",
    UploadDocumentFailure.TEXT_LIMIT: "Extracted document text exceeds the limit.",
    UploadDocumentFailure.EMPTY_DOCUMENT: "Document contains no usable text.",
    UploadDocumentFailure.EXTRACTION_UNAVAILABLE: "Document extraction is unavailable.",
}


class UploadDocumentPreparationError(ValueError):
    """Fixed secret-free upload preparation failure."""

    def __init__(self, code: UploadDocumentFailure) -> None:
        self.code = code
        super().__init__(_FAILURE_MESSAGES[code])


@dataclass(frozen=True, slots=True)
class PreparedUploadDocument:
    tenant_id: str
    knowledge_base_id: str
    document_id: str
    original_filename: str
    media_type: str
    byte_size: int
    sha256_hex: str
    ingestion_request: DocumentIngestionRequest

    @property
    def chunks(self) -> tuple[DocumentChunkInput, ...]:
        return self.ingestion_request.chunks


@dataclass(frozen=True, slots=True)
class ValidatedUploadEnvelope:
    """Immutable, bounded upload bytes plus safe display metadata."""

    original_filename: str
    media_type: str
    byte_size: int
    sha256_hex: str
    content: bytes


def validate_upload_envelope(
    *,
    content: bytes,
    original_filename: str,
    declared_media_type: str,
) -> ValidatedUploadEnvelope:
    """Validate upload metadata and bytes without parsing document content."""
    filename, media_type = _validated_filename_and_media_type(
        original_filename, declared_media_type
    )
    upload_bytes = _validated_content(content)
    return ValidatedUploadEnvelope(
        original_filename=filename,
        media_type=media_type,
        byte_size=len(upload_bytes),
        sha256_hex=hashlib.sha256(upload_bytes).hexdigest(),
        content=upload_bytes,
    )


def prepare_upload_document(
    *,
    content: bytes,
    original_filename: str,
    declared_media_type: str,
    document_id: str,
    tenant_id: str,
    knowledge_base_id: str,
    max_chunk_characters: int = DEFAULT_UPLOAD_CHUNK_CHARACTERS,
) -> PreparedUploadDocument:
    """Validate, extract and deterministically chunk one in-memory document."""
    envelope = validate_upload_envelope(
        content=content,
        original_filename=original_filename,
        declared_media_type=declared_media_type,
    )
    return prepare_validated_upload_document(
        envelope=envelope,
        document_id=document_id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        max_chunk_characters=max_chunk_characters,
    )


def prepare_validated_upload_document(
    *,
    envelope: ValidatedUploadEnvelope,
    document_id: str,
    tenant_id: str,
    knowledge_base_id: str,
    max_chunk_characters: int = DEFAULT_UPLOAD_CHUNK_CHARACTERS,
) -> PreparedUploadDocument:
    """Extract and chunk a previously validated immutable upload envelope."""
    if not isinstance(envelope, ValidatedUploadEnvelope):
        raise UploadDocumentPreparationError(UploadDocumentFailure.INVALID_ENCODING)
    tenant = _trusted_identifier(tenant_id)
    knowledge_base = _trusted_identifier(knowledge_base_id)
    server_document_id = _trusted_identifier(document_id)
    filename = envelope.original_filename
    media_type = envelope.media_type
    upload_bytes = envelope.content
    digest = envelope.sha256_hex
    chunk_size = _positive_chunk_limit(max_chunk_characters)

    base_metadata = (
        ("original_filename", filename),
        ("media_type", media_type),
        ("sha256", digest),
    )
    if media_type == "application/pdf":
        page_texts = _extract_pdf_pages(upload_bytes)
        chunks = _chunk_pdf_pages(
            document_id=server_document_id,
            page_texts=page_texts,
            base_metadata=base_metadata,
            max_chunk_characters=chunk_size,
        )
    else:
        text = _decode_utf8_text(upload_bytes)
        chunks = _chunk_text(
            document_id=server_document_id,
            text=text,
            metadata=base_metadata,
            max_chunk_characters=chunk_size,
        )

    request = DocumentIngestionRequest(
        tenant_id=tenant,
        knowledge_base_id=knowledge_base,
        chunks=chunks,
    )
    return PreparedUploadDocument(
        tenant_id=tenant,
        knowledge_base_id=knowledge_base,
        document_id=server_document_id,
        original_filename=filename,
        media_type=media_type,
        byte_size=envelope.byte_size,
        sha256_hex=digest,
        ingestion_request=request,
    )


def _validated_content(value: object) -> bytes:
    if type(value) is not bytes:
        raise UploadDocumentPreparationError(UploadDocumentFailure.INVALID_ENCODING)
    if not value:
        raise UploadDocumentPreparationError(UploadDocumentFailure.EMPTY_DOCUMENT)
    if len(value) > MAX_UPLOAD_BYTES:
        raise UploadDocumentPreparationError(UploadDocumentFailure.UPLOAD_TOO_LARGE)
    return value


def _validated_filename_and_media_type(
    filename: object,
    media_type: object,
) -> tuple[str, str]:
    if not isinstance(filename, str):
        raise UploadDocumentPreparationError(UploadDocumentFailure.INVALID_FILENAME)
    normalized = unicodedata.normalize("NFC", filename).strip()
    if (
        not normalized
        or len(normalized) > MAX_DISPLAY_FILENAME_CHARACTERS
        or normalized in {".", ".."}
        or any(character in normalized for character in ("/", "\\", ":"))
        or any(
            unicodedata.category(character).startswith("C") for character in normalized
        )
    ):
        raise UploadDocumentPreparationError(UploadDocumentFailure.INVALID_FILENAME)
    separator = normalized.rfind(".")
    extension = normalized[separator:].casefold() if separator > 0 else ""
    expected_media_type = _MEDIA_TYPE_BY_EXTENSION.get(extension)
    if (
        not isinstance(media_type, str)
        or media_type not in _SUPPORTED_MEDIA_TYPES
        or expected_media_type != media_type
    ):
        raise UploadDocumentPreparationError(UploadDocumentFailure.UNSUPPORTED_TYPE)
    return normalized, media_type


def _trusted_identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value) > MAX_SCOPE_IDENTIFIER_CHARACTERS
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise UploadDocumentPreparationError(UploadDocumentFailure.INVALID_IDENTIFIER)
    return value


def _positive_chunk_limit(value: object) -> int:
    if type(value) is not int or value <= 0 or value > MAX_EXTRACTED_CHARACTERS:
        raise UploadDocumentPreparationError(UploadDocumentFailure.TEXT_LIMIT)
    return value


def _decode_utf8_text(content: bytes) -> str:
    try:
        text = content.decode("utf-8-sig" if content.startswith(_UTF8_BOM) else "utf-8")
    except UnicodeDecodeError:
        raise UploadDocumentPreparationError(
            UploadDocumentFailure.INVALID_ENCODING
        ) from None
    return _validated_normalized_text(text)


def _validated_normalized_text(text: str) -> str:
    if "\0" in text:
        raise UploadDocumentPreparationError(UploadDocumentFailure.INVALID_ENCODING)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise UploadDocumentPreparationError(UploadDocumentFailure.EMPTY_DOCUMENT)
    if len(normalized) > MAX_EXTRACTED_CHARACTERS:
        raise UploadDocumentPreparationError(UploadDocumentFailure.TEXT_LIMIT)
    return normalized


def _extract_pdf_pages(content: bytes) -> tuple[tuple[int, str], ...]:
    if not content.startswith(_PDF_SIGNATURE):
        raise UploadDocumentPreparationError(UploadDocumentFailure.INVALID_PDF)
    try:
        import pymupdf
    except (ImportError, OSError):
        raise UploadDocumentPreparationError(
            UploadDocumentFailure.EXTRACTION_UNAVAILABLE
        ) from None

    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception:
        raise UploadDocumentPreparationError(
            UploadDocumentFailure.INVALID_PDF
        ) from None

    try:
        try:
            if document.needs_pass:
                raise UploadDocumentPreparationError(
                    UploadDocumentFailure.ENCRYPTED_PDF
                )
            if document.page_count > MAX_PDF_PAGES:
                raise UploadDocumentPreparationError(UploadDocumentFailure.PAGE_LIMIT)
            _reject_pdf_active_content(document)
            pages: list[tuple[int, str]] = []
            total_characters = 0
            for page_index in range(document.page_count):
                raw_text: Any = document.load_page(page_index).get_text(
                    "text", sort=True
                )
                if not isinstance(raw_text, str):
                    raise UploadDocumentPreparationError(
                        UploadDocumentFailure.EXTRACTION_UNAVAILABLE
                    )
                normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
                if "\0" in normalized:
                    raise UploadDocumentPreparationError(
                        UploadDocumentFailure.INVALID_ENCODING
                    )
                total_characters += len(normalized)
                if total_characters > MAX_EXTRACTED_CHARACTERS:
                    raise UploadDocumentPreparationError(
                        UploadDocumentFailure.TEXT_LIMIT
                    )
                if normalized:
                    pages.append((page_index + 1, normalized))
        except UploadDocumentPreparationError:
            raise
        except Exception:
            raise UploadDocumentPreparationError(
                UploadDocumentFailure.EXTRACTION_UNAVAILABLE
            ) from None
    finally:
        try:
            document.close()
        except Exception:
            pass
    if not pages:
        raise UploadDocumentPreparationError(UploadDocumentFailure.EMPTY_DOCUMENT)
    return tuple(pages)


def _reject_pdf_active_content(document: Any) -> None:
    try:
        if document.embfile_count() > 0:
            raise UploadDocumentPreparationError(UploadDocumentFailure.UNSAFE_CONTENT)
        unsafe_keys = {"AA", "JavaScript", "JS", "Launch", "OpenAction"}
        for xref in range(1, document.xref_length()):
            if unsafe_keys.intersection(document.xref_get_keys(xref)):
                raise UploadDocumentPreparationError(
                    UploadDocumentFailure.UNSAFE_CONTENT
                )
    except UploadDocumentPreparationError:
        raise
    except Exception:
        raise UploadDocumentPreparationError(
            UploadDocumentFailure.EXTRACTION_UNAVAILABLE
        ) from None


def _chunk_text(
    *,
    document_id: str,
    text: str,
    metadata: tuple[tuple[str, str], ...],
    max_chunk_characters: int,
) -> tuple[DocumentChunkInput, ...]:
    source = TextDocumentSource(
        document_id=document_id,
        text=text,
        metadata=metadata,
    )
    return FixedCharacterDocumentChunker(
        max_chunk_characters=max_chunk_characters,
        max_document_characters=MAX_EXTRACTED_CHARACTERS,
    ).chunk(source)


def _chunk_pdf_pages(
    *,
    document_id: str,
    page_texts: tuple[tuple[int, str], ...],
    base_metadata: tuple[tuple[str, str], ...],
    max_chunk_characters: int,
) -> tuple[DocumentChunkInput, ...]:
    prepared: list[DocumentChunkInput] = []
    chunker = FixedCharacterDocumentChunker(
        max_chunk_characters=max_chunk_characters,
        max_document_characters=MAX_EXTRACTED_CHARACTERS,
    )
    for page_number, text in page_texts:
        page_chunks = chunker.chunk(
            TextDocumentSource(
                document_id=document_id,
                text=text,
                metadata=(*base_metadata, ("page_number", str(page_number))),
            )
        )
        for page_chunk in page_chunks:
            prepared.append(
                DocumentChunkInput(
                    document_id=document_id,
                    chunk_id=f"chunk_{len(prepared) + 1:06d}",
                    text=page_chunk.text,
                    metadata=page_chunk.metadata,
                )
            )
    return tuple(prepared)
