"""Secure in-memory upload-document preparation tests."""

from __future__ import annotations

import builtins
import hashlib
from inspect import signature

import pymupdf
import pytest

from app.ingestion.upload_preparation import (
    MAX_EXTRACTED_CHARACTERS,
    MAX_PDF_PAGES,
    MAX_UPLOAD_BYTES,
    PreparedUploadDocument,
    UploadDocumentFailure,
    UploadDocumentPreparationError,
    prepare_upload_document,
)


def _prepare(
    content: bytes = b"Synthetic trusted text.",
    *,
    filename: str = "guide.txt",
    media_type: str = "text/plain",
    max_chunk_characters: int = 1_000,
) -> PreparedUploadDocument:
    return prepare_upload_document(
        content=content,
        original_filename=filename,
        declared_media_type=media_type,
        document_id="document-server-generated",
        tenant_id="tenant-trusted",
        knowledge_base_id="kb-trusted",
        max_chunk_characters=max_chunk_characters,
    )


def _pdf_bytes(page_texts: tuple[str, ...]) -> bytes:
    document = pymupdf.open()
    try:
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()


def _encrypted_pdf_bytes() -> bytes:
    document = pymupdf.open()
    try:
        document.new_page().insert_text((72, 72), "Synthetic encrypted text")
        return document.tobytes(
            encryption=getattr(pymupdf, "PDF_ENCRYPT_AES_256"),
            owner_pw="synthetic-owner",
            user_pw="synthetic-user",
        )
    finally:
        document.close()


def _pdf_with_embedded_file() -> bytes:
    document = pymupdf.open()
    try:
        document.new_page().insert_text((72, 72), "Synthetic visible text")
        document.embfile_add("payload.bin", b"synthetic-payload")
        return document.tobytes()
    finally:
        document.close()


def _assert_failure(
    code: UploadDocumentFailure,
    content: bytes = b"Synthetic trusted text.",
    **kwargs: object,
) -> UploadDocumentPreparationError:
    arguments: dict[str, object] = {
        "content": content,
        "original_filename": "guide.txt",
        "declared_media_type": "text/plain",
        "document_id": "document-server-generated",
        "tenant_id": "tenant-trusted",
        "knowledge_base_id": "kb-trusted",
    }
    arguments.update(kwargs)
    with pytest.raises(UploadDocumentPreparationError) as raised:
        prepare_upload_document(**arguments)  # type: ignore[arg-type]
    assert raised.value.code is code
    return raised.value


def test_public_api_is_keyword_only_and_result_retains_no_upload_bytes() -> None:
    parameters = signature(prepare_upload_document).parameters
    result = _prepare()

    assert all(item.kind.name == "KEYWORD_ONLY" for item in parameters.values())
    assert not hasattr(result, "content")
    assert not hasattr(result, "path")
    assert result.ingestion_request.tenant_id == "tenant-trusted"
    assert result.ingestion_request.knowledge_base_id == "kb-trusted"


@pytest.mark.parametrize(
    ("filename", "media_type", "text"),
    [
        ("guide.txt", "text/plain", "Merhaba dünya"),
        ("guide.md", "text/markdown", "# Başlık\n\nİçerik"),
        ("guide.markdown", "text/markdown", "## Güvenli"),
    ],
)
def test_valid_utf8_text_types_preserve_turkish_unicode(
    filename: str,
    media_type: str,
    text: str,
) -> None:
    result = _prepare(text.encode(), filename=filename, media_type=media_type)

    assert "".join(chunk.text for chunk in result.chunks) == text
    assert result.sha256_hex == hashlib.sha256(text.encode()).hexdigest()
    assert dict(result.chunks[0].metadata) == {
        "original_filename": filename,
        "media_type": media_type,
        "sha256": result.sha256_hex,
    }


def test_utf8_bom_is_removed_and_crlf_is_normalized() -> None:
    content = b"\xef\xbb\xbfBirinci\r\nIkinci\rUcuncu"

    result = _prepare(content)

    assert result.chunks[0].text == "Birinci\nIkinci\nUcuncu"
    assert result.sha256_hex == hashlib.sha256(content).hexdigest()


def test_preparation_is_deterministic_for_identical_inputs() -> None:
    content = "Bir iki üç dört".encode()

    first = _prepare(content, max_chunk_characters=5)
    second = _prepare(content, max_chunk_characters=5)

    assert first == second
    assert tuple(chunk.chunk_id for chunk in first.chunks) == (
        "chunk_000001",
        "chunk_000002",
        "chunk_000003",
    )


def test_valid_multi_page_pdf_preserves_page_order_and_metadata() -> None:
    result = _prepare(
        _pdf_bytes(("First page", "Second page")),
        filename="guide.pdf",
        media_type="application/pdf",
        max_chunk_characters=100,
    )

    assert tuple(chunk.text for chunk in result.chunks) == (
        "First page",
        "Second page",
    )
    assert tuple(dict(chunk.metadata)["page_number"] for chunk in result.chunks) == (
        "1",
        "2",
    )
    assert tuple(chunk.chunk_id for chunk in result.chunks) == (
        "chunk_000001",
        "chunk_000002",
    )


@pytest.mark.parametrize(
    ("filename", "media_type"),
    [
        ("guide.pdf", "text/plain"),
        ("guide.txt", "application/pdf"),
        ("guide.exe", "text/plain"),
        ("guide", "text/plain"),
        ("guide.txt", "application/octet-stream"),
    ],
)
def test_extension_and_declared_type_must_exactly_agree(
    filename: str,
    media_type: str,
) -> None:
    _assert_failure(
        UploadDocumentFailure.UNSUPPORTED_TYPE,
        original_filename=filename,
        declared_media_type=media_type,
    )


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "   ",
        ".",
        "..",
        "../guide.txt",
        "folder/guide.txt",
        "folder\\guide.txt",
        "C:guide.txt",
        "guide\x00.txt",
        "guide\n.txt",
        f"{'a' * 252}.txt",
    ],
)
def test_invalid_or_path_like_display_filenames_are_rejected(filename: str) -> None:
    _assert_failure(
        UploadDocumentFailure.INVALID_FILENAME,
        original_filename=filename,
    )


def test_display_filename_is_unicode_normalized_without_path_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        builtins,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("filesystem access is forbidden")
        ),
    )

    result = _prepare(b"Text", filename="Gu\u0308venli.txt")

    assert result.original_filename == "Güvenli.txt"


@pytest.mark.parametrize("content", [b"", b" \r\n\t "])
def test_empty_and_whitespace_only_documents_are_rejected(content: bytes) -> None:
    _assert_failure(UploadDocumentFailure.EMPTY_DOCUMENT, content)


@pytest.mark.parametrize("content", [b"\xff", b"valid\x00invalid"])
def test_invalid_utf8_and_nul_are_rejected(content: bytes) -> None:
    _assert_failure(UploadDocumentFailure.INVALID_ENCODING, content)


def test_non_bytes_upload_is_rejected_as_not_immutable() -> None:
    _assert_failure(UploadDocumentFailure.INVALID_ENCODING, bytearray(b"text"))  # type: ignore[arg-type]


def test_invalid_pdf_signature_and_parser_failure_are_distinctly_bounded() -> None:
    _assert_failure(
        UploadDocumentFailure.INVALID_PDF,
        b"not-a-pdf",
        original_filename="guide.pdf",
        declared_media_type="application/pdf",
    )
    _assert_failure(
        UploadDocumentFailure.INVALID_PDF,
        b"%PDF-not-parseable",
        original_filename="guide.pdf",
        declared_media_type="application/pdf",
    )


def test_encrypted_pdf_is_rejected_before_extraction() -> None:
    _assert_failure(
        UploadDocumentFailure.ENCRYPTED_PDF,
        _encrypted_pdf_bytes(),
        original_filename="guide.pdf",
        declared_media_type="application/pdf",
    )


def test_pdf_embedded_content_is_rejected_without_extraction() -> None:
    _assert_failure(
        UploadDocumentFailure.UNSAFE_CONTENT,
        _pdf_with_embedded_file(),
        original_filename="guide.pdf",
        declared_media_type="application/pdf",
    )


def test_pdf_page_limit_rejects_101_pages() -> None:
    content = _pdf_bytes(tuple("" for _ in range(MAX_PDF_PAGES + 1)))

    _assert_failure(
        UploadDocumentFailure.PAGE_LIMIT,
        content,
        original_filename="guide.pdf",
        declared_media_type="application/pdf",
    )


def test_upload_byte_limit_accepts_boundary_and_rejects_one_byte_over() -> None:
    boundary_error = _assert_failure(
        UploadDocumentFailure.EMPTY_DOCUMENT,
        b" " * MAX_UPLOAD_BYTES,
    )
    over_error = _assert_failure(
        UploadDocumentFailure.UPLOAD_TOO_LARGE,
        b" " * (MAX_UPLOAD_BYTES + 1),
    )

    assert boundary_error.code is not UploadDocumentFailure.UPLOAD_TOO_LARGE
    assert over_error.code is UploadDocumentFailure.UPLOAD_TOO_LARGE


def test_extracted_text_limit_accepts_boundary_and_rejects_one_over() -> None:
    accepted = _prepare(b"a" * MAX_EXTRACTED_CHARACTERS)

    assert sum(len(chunk.text) for chunk in accepted.chunks) == (
        MAX_EXTRACTED_CHARACTERS
    )
    _assert_failure(
        UploadDocumentFailure.TEXT_LIMIT,
        b"a" * (MAX_EXTRACTED_CHARACTERS + 1),
    )


def test_errors_never_contain_content_scope_filename_digest_or_parser_text() -> None:
    secrets = (
        "secret-tenant",
        "secret-kb",
        "secret-document",
        "secret-name.pdf",
        "secret-content",
        hashlib.sha256(b"secret-content").hexdigest(),
    )
    error = _assert_failure(
        UploadDocumentFailure.INVALID_PDF,
        b"%PDF-secret-content",
        original_filename="secret-name.pdf",
        declared_media_type="application/pdf",
        tenant_id="secret-tenant",
        knowledge_base_id="secret-kb",
        document_id="secret-document",
    )

    assert all(secret not in str(error) for secret in secrets)
    assert str(error) == "PDF document is invalid."
