"""Pure deterministic construction of fixed-character ingestion requests."""

from app.ingestion.chunking import FixedCharacterDocumentChunker
from app.ingestion.document_source import TextDocumentSource
from app.ingestion.models import DocumentIngestionRequest


def build_fixed_character_document_ingestion_request(
    *,
    tenant_id: str,
    knowledge_base_id: str,
    document: TextDocumentSource,
    max_chunk_characters: int,
    max_document_characters: int,
) -> DocumentIngestionRequest:
    """Build one immutable request without filesystem or provider activity."""
    tenant = _canonical_required_text(tenant_id, "tenant_id")
    knowledge_base = _canonical_required_text(
        knowledge_base_id,
        "knowledge_base_id",
    )
    source = _canonical_document(document)
    chunker = FixedCharacterDocumentChunker(
        max_chunk_characters=max_chunk_characters,
        max_document_characters=max_document_characters,
    )
    if len(source.text) > max_document_characters:
        raise ValueError("document text exceeds max_document_characters")
    chunks = chunker.chunk(source)
    if not chunks:
        raise ValueError("document chunking produced no chunks")
    return DocumentIngestionRequest(
        tenant_id=tenant,
        knowledge_base_id=knowledge_base,
        chunks=chunks,
    )


def _canonical_document(value: object) -> TextDocumentSource:
    if not isinstance(value, TextDocumentSource):
        raise ValueError("document must be a TextDocumentSource")
    try:
        validated = TextDocumentSource.model_validate(value.model_dump())
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("document must be canonical") from error
    if validated != value:
        raise ValueError("document must be canonical")
    return value


def _canonical_required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    if cleaned != value:
        raise ValueError(f"{field_name} must be canonical")
    return cleaned
