"""Tests for pure fixed-character document ingestion request construction."""

from inspect import signature
from typing import Any, cast

import pytest
from pydantic import ValidationError

import app.ingestion as ingestion_exports
import app.ingestion.document_request_builder as builder
from app.ingestion import (
    DocumentChunkInput,
    DocumentIngestionRequest,
    TextDocumentSource,
    build_fixed_character_document_ingestion_request,
)


def _document(
    *,
    text: str = "Synthetic trusted document.",
) -> TextDocumentSource:
    return TextDocumentSource(
        document_id="document-v1",
        text=text,
        metadata=(("source", "synthetic"), ("version", "v1")),
    )


def _build(
    document: TextDocumentSource | None = None,
    *,
    max_chunk_characters: int = 8,
    max_document_characters: int = 100,
) -> DocumentIngestionRequest:
    return build_fixed_character_document_ingestion_request(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        document=_document() if document is None else document,
        max_chunk_characters=max_chunk_characters,
        max_document_characters=max_document_characters,
    )


def test_public_export_preserves_existing_ingestion_contracts() -> None:
    assert ingestion_exports.__all__ == [
        "DocumentChunkInput",
        "DocumentChunker",
        "DocumentIngestionRequest",
        "DocumentIngestionService",
        "FixedCharacterDocumentChunker",
        "TextDocumentSource",
        "build_fixed_character_document_ingestion_request",
    ]
    assert ingestion_exports.build_fixed_character_document_ingestion_request is (
        builder.build_fixed_character_document_ingestion_request
    )
    assert tuple(
        signature(build_fixed_character_document_ingestion_request).parameters
    ) == (
        "tenant_id",
        "knowledge_base_id",
        "document",
        "max_chunk_characters",
        "max_document_characters",
    )
    assert all(
        parameter.kind.name == "KEYWORD_ONLY"
        for parameter in signature(
            build_fixed_character_document_ingestion_request
        ).parameters.values()
    )


def test_exact_existing_chunker_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    document = _document()
    chunks = (
        DocumentChunkInput(
            document_id=document.document_id,
            chunk_id="chunk_000001",
            text="Synthetic",
            metadata=document.metadata,
        ),
    )

    class FakeChunker:
        def __init__(
            self,
            *,
            max_chunk_characters: int,
            max_document_characters: int,
        ) -> None:
            calls.append(("construct", max_chunk_characters, max_document_characters))

        def chunk(
            self,
            source: TextDocumentSource,
        ) -> tuple[DocumentChunkInput, ...]:
            calls.append(("chunk", source))
            return chunks

    monkeypatch.setattr(builder, "FixedCharacterDocumentChunker", FakeChunker)

    result = _build(
        document,
        max_chunk_characters=12,
        max_document_characters=200,
    )

    assert calls == [("construct", 12, 200), ("chunk", document)]
    assert result == DocumentIngestionRequest(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        chunks=chunks,
    )


def test_deterministic_chunk_ids_metadata_and_requests() -> None:
    document = _document(text="123456789")

    first = _build(document, max_chunk_characters=4)
    second = _build(document, max_chunk_characters=4)

    assert first == second
    assert tuple(chunk.chunk_id for chunk in first.chunks) == (
        "chunk_000001",
        "chunk_000002",
        "chunk_000003",
    )
    assert tuple(chunk.text for chunk in first.chunks) == ("1234", "5678", "9")
    assert all(chunk.document_id == "document-v1" for chunk in first.chunks)
    assert all(chunk.metadata == document.metadata for chunk in first.chunks)


def test_unicode_code_points_and_internal_newlines_are_preserved() -> None:
    request = _build(
        _document(text="A\nBÇD"),
        max_chunk_characters=4,
    )

    assert tuple(chunk.text for chunk in request.chunks) == ("A\nBÇ", "D")


def test_window_edges_are_normalized_and_blank_windows_are_skipped() -> None:
    request = _build(
        _document(text="abc   def"),
        max_chunk_characters=3,
    )

    assert tuple(chunk.text for chunk in request.chunks) == ("abc", "def")
    assert tuple(chunk.chunk_id for chunk in request.chunks) == (
        "chunk_000001",
        "chunk_000002",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", " tenant-synthetic"),
        ("tenant_id", ""),
        ("tenant_id", 1),
        ("knowledge_base_id", "kb-synthetic "),
        ("knowledge_base_id", " "),
        ("knowledge_base_id", object()),
        ("document", object()),
        ("max_chunk_characters", True),
        ("max_chunk_characters", 0),
        ("max_chunk_characters", 1.5),
        ("max_document_characters", False),
        ("max_document_characters", -1),
        ("max_document_characters", 10.0),
    ],
)
def test_invalid_inputs_and_strict_limits_are_rejected(
    field: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {
        "tenant_id": "tenant-synthetic",
        "knowledge_base_id": "kb-synthetic",
        "document": _document(),
        "max_chunk_characters": 8,
        "max_document_characters": 100,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        build_fixed_character_document_ingestion_request(
            **cast(Any, arguments),
        )


def test_noncanonical_constructed_source_is_rejected() -> None:
    malformed = TextDocumentSource.model_construct(
        document_id=" document-v1 ",
        text=" Synthetic ",
        metadata=((" source ", " synthetic "),),
    )

    with pytest.raises(ValueError, match="canonical"):
        _build(malformed)


def test_oversized_normalized_text_is_rejected_without_partial_result() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        _build(
            _document(text="123456"),
            max_chunk_characters=2,
            max_document_characters=5,
        )


def test_source_metadata_and_generated_request_are_immutable() -> None:
    document = _document()
    before = document.model_dump()

    request = _build(document)

    assert document.model_dump() == before
    assert request.chunks[0].metadata == document.metadata
    with pytest.raises(ValidationError):
        document.text = "changed"
    with pytest.raises(ValidationError):
        request.tenant_id = "changed"
    with pytest.raises(ValidationError):
        request.chunks[0].text = "changed"


def test_empty_chunker_output_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyChunker:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chunk(
            self,
            _document: TextDocumentSource,
        ) -> tuple[DocumentChunkInput, ...]:
            return ()

    monkeypatch.setattr(builder, "FixedCharacterDocumentChunker", EmptyChunker)

    with pytest.raises(ValueError, match="no chunks"):
        _build()
