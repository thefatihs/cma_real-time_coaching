"""Provider-neutral deterministic document chunking."""

from typing import Protocol

from app.ingestion.document_source import TextDocumentSource
from app.ingestion.models import DocumentChunkInput


class DocumentChunker(Protocol):
    def chunk(
        self,
        document: TextDocumentSource,
    ) -> tuple[DocumentChunkInput, ...]: ...


class FixedCharacterDocumentChunker:
    def __init__(
        self,
        *,
        max_chunk_characters: int,
        max_document_characters: int,
    ) -> None:
        self._max_chunk_characters = _positive_integer(
            max_chunk_characters,
            "max_chunk_characters",
        )
        self._max_document_characters = _positive_integer(
            max_document_characters,
            "max_document_characters",
        )

    def chunk(
        self,
        document: TextDocumentSource,
    ) -> tuple[DocumentChunkInput, ...]:
        if len(document.text) > self._max_document_characters:
            raise ValueError("document text exceeds max_document_characters")

        chunks: list[DocumentChunkInput] = []
        for start in range(0, len(document.text), self._max_chunk_characters):
            window = document.text[start : start + self._max_chunk_characters]
            if not window.strip():
                continue
            chunks.append(
                DocumentChunkInput(
                    document_id=document.document_id,
                    chunk_id=f"chunk_{len(chunks) + 1:06d}",
                    text=window,
                    metadata=document.metadata,
                )
            )

        if not chunks:
            raise ValueError("document text must produce at least one chunk")
        return tuple(chunks)


def _positive_integer(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value
