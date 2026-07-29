"""Provider-neutral document ingestion foundation."""

from app.ingestion.chunking import DocumentChunker, FixedCharacterDocumentChunker
from app.ingestion.document_source import TextDocumentSource
from app.ingestion.document_request_builder import (
    build_fixed_character_document_ingestion_request,
)
from app.ingestion.models import DocumentChunkInput, DocumentIngestionRequest
from app.ingestion.service import DocumentIngestionService

__all__ = [
    "DocumentChunkInput",
    "DocumentChunker",
    "DocumentIngestionRequest",
    "DocumentIngestionService",
    "FixedCharacterDocumentChunker",
    "TextDocumentSource",
    "build_fixed_character_document_ingestion_request",
]
