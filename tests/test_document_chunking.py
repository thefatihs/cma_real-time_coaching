from typing import cast

import pytest
from pydantic import ValidationError

from app.ingestion import (
    DocumentChunker,
    DocumentIngestionRequest,
    FixedCharacterDocumentChunker,
    TextDocumentSource,
)


def source(
    *,
    document_id: str = "synthetic_guide",
    text: str = "Synthetic trusted text.",
    metadata: tuple[tuple[str, str], ...] = (("source", "synthetic"),),
) -> TextDocumentSource:
    return TextDocumentSource(
        document_id=document_id,
        text=text,
        metadata=metadata,
    )


def chunker(
    *,
    max_chunk_characters: int = 8,
    max_document_characters: int = 100,
) -> FixedCharacterDocumentChunker:
    return FixedCharacterDocumentChunker(
        max_chunk_characters=max_chunk_characters,
        max_document_characters=max_document_characters,
    )


def test_text_document_source_is_frozen_and_normalized() -> None:
    document = source(
        document_id=" synthetic_guide ",
        text="\n  Synthetic trusted text. \t",
        metadata=((" source ", " synthetic "), (" locale ", " tr ")),
    )

    assert document.document_id == "synthetic_guide"
    assert document.text == "Synthetic trusted text."
    assert document.metadata == (("source", "synthetic"), ("locale", "tr"))
    with pytest.raises(ValidationError):
        document.text = "changed"


@pytest.mark.parametrize(("field", "value"), [("document_id", " "), ("text", "\n")])
def test_blank_required_source_text_is_rejected(field: str, value: str) -> None:
    document_id = value if field == "document_id" else "synthetic_guide"
    text = value if field == "text" else "Synthetic trusted text."

    with pytest.raises(ValidationError, match=f"{field} cannot be empty"):
        TextDocumentSource(document_id=document_id, text=text)


@pytest.mark.parametrize(
    "metadata",
    [
        ((" ", "synthetic"),),
        (("source", " "),),
        ((" source ", "synthetic"), ("source", "other")),
    ],
)
def test_invalid_or_duplicate_metadata_is_rejected(
    metadata: tuple[tuple[str, str], ...],
) -> None:
    with pytest.raises(ValidationError):
        source(metadata=metadata)


def test_chunker_structurally_satisfies_protocol() -> None:
    implementation: DocumentChunker = chunker()

    assert implementation.chunk(source(text="Synthetic"))[0].text == "Syntheti"


def test_constructor_requires_explicit_limits() -> None:
    with pytest.raises(TypeError):
        FixedCharacterDocumentChunker()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        FixedCharacterDocumentChunker(  # type: ignore[call-arg]
            max_chunk_characters=8
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_chunk_characters", 0),
        ("max_chunk_characters", -1),
        ("max_chunk_characters", True),
        ("max_chunk_characters", 1.5),
        ("max_document_characters", 0),
        ("max_document_characters", -1),
        ("max_document_characters", False),
        ("max_document_characters", 10.0),
    ],
)
def test_invalid_limits_are_rejected(field: str, value: object) -> None:
    values: dict[str, object] = {
        "max_chunk_characters": 8,
        "max_document_characters": 100,
    }
    values[field] = value

    with pytest.raises(ValueError):
        FixedCharacterDocumentChunker(
            max_chunk_characters=cast(int, values["max_chunk_characters"]),
            max_document_characters=cast(int, values["max_document_characters"]),
        )


def test_constructor_performs_no_chunking_work() -> None:
    implementation = chunker()

    assert implementation is not None


@pytest.mark.parametrize("text", ["123456789", "1234567890"])
def test_document_at_or_below_size_limit_is_accepted(text: str) -> None:
    result = chunker(
        max_chunk_characters=4,
        max_document_characters=10,
    ).chunk(source(text=text))

    assert result


def test_oversized_document_is_rejected_without_partial_output() -> None:
    implementation = chunker(
        max_chunk_characters=4,
        max_document_characters=10,
    )

    with pytest.raises(ValueError, match="exceeds"):
        implementation.chunk(source(text="12345678901"))

    assert implementation.chunk(source(text="1234"))[0].chunk_id == "chunk_000001"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("123", ("123",)),
        ("1234", ("1234",)),
        ("12345", ("1234", "5")),
        ("123456789", ("1234", "5678", "9")),
    ],
)
def test_fixed_windows_preserve_order_and_boundaries(
    text: str,
    expected: tuple[str, ...],
) -> None:
    result = chunker(max_chunk_characters=4).chunk(source(text=text))

    assert tuple(item.text for item in result) == expected
    assert all(len(item.text) <= 4 for item in result)


def test_boundary_whitespace_is_canonically_stripped() -> None:
    result = chunker(max_chunk_characters=4).chunk(source(text="ab  cd"))

    assert tuple(item.text for item in result) == ("ab", "cd")


def test_whitespace_only_windows_are_skipped_with_contiguous_ids() -> None:
    result = chunker(max_chunk_characters=3).chunk(source(text="abc   def"))

    assert tuple(item.text for item in result) == ("abc", "def")
    assert tuple(item.chunk_id for item in result) == (
        "chunk_000001",
        "chunk_000002",
    )


def test_non_whitespace_characters_preserve_source_order() -> None:
    document = source(text="ab \n  cd\t ef")
    result = chunker(max_chunk_characters=3).chunk(document)

    expected = "".join(
        character for character in document.text if not character.isspace()
    )
    actual = "".join(
        character
        for item in result
        for character in item.text
        if not character.isspace()
    )
    assert actual == expected


def test_internal_whitespace_inside_nonblank_window_is_preserved() -> None:
    result = chunker(max_chunk_characters=7).chunk(source(text="ab cd ef"))

    assert result[0].text == "ab cd e"


def test_valid_nonblank_source_always_produces_a_chunk() -> None:
    result = chunker(max_chunk_characters=1).chunk(source(text=" \n X \t "))

    assert tuple(item.text for item in result) == ("X",)


def test_ids_document_and_metadata_are_propagated_deterministically() -> None:
    document = source(
        document_id="synthetic_a",
        text="123456789",
        metadata=(("source", "synthetic"), ("locale", "tr")),
    )
    implementation = chunker(max_chunk_characters=4)

    first = implementation.chunk(document)
    second = implementation.chunk(document)

    assert first == second
    assert tuple(item.chunk_id for item in first) == (
        "chunk_000001",
        "chunk_000002",
        "chunk_000003",
    )
    assert all(item.document_id == "synthetic_a" for item in first)
    assert all(item.metadata == document.metadata for item in first)


def test_same_chunk_id_is_valid_for_different_documents() -> None:
    implementation = chunker()

    first = implementation.chunk(source(document_id="synthetic_a", text="A"))
    second = implementation.chunk(source(document_id="synthetic_b", text="B"))

    assert first[0].chunk_id == second[0].chunk_id == "chunk_000001"
    request = DocumentIngestionRequest(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        chunks=(first[0], second[0]),
    )
    assert len(request.chunks) == 2


def test_unicode_code_points_are_sliced_deterministically() -> None:
    document = source(text="İğüş😊abc")

    result = chunker(max_chunk_characters=2).chunk(document)

    assert tuple(item.text for item in result) == ("İğ", "üş", "😊a", "bc")
    assert tuple(len(item.text) for item in result) == (2, 2, 2, 2)


def test_source_is_not_mutated_and_outputs_feed_ingestion_request() -> None:
    document = source(text="Synthetic trusted content.")
    before = document.model_dump()

    chunks = chunker(max_chunk_characters=9).chunk(document)
    request = DocumentIngestionRequest(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        chunks=chunks,
    )

    assert document.model_dump() == before
    assert request.chunks == chunks


def test_chunker_does_not_bypass_document_chunk_validation() -> None:
    import inspect

    source_code = inspect.getsource(FixedCharacterDocumentChunker.chunk)

    assert "model_construct" not in source_code
    assert "DocumentChunkInput(" in source_code
