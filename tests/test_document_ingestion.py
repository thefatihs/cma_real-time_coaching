from math import inf, nan
from typing import cast

import pytest
from pydantic import ValidationError

from app.embeddings import DocumentEmbedder
from app.ingestion import (
    DocumentChunkInput,
    DocumentIngestionRequest,
    DocumentIngestionService,
)
from app.vector_store import (
    AtomicVectorBatchWriter,
    InMemoryVectorStore,
    VectorBatchWriteRequest,
    VectorBatchWriteResult,
    VectorRecordIdentity,
)


class FakeDocumentEmbedder:
    def __init__(
        self,
        output: object = ((0.25, 0.75), (0.5, 0.5)),
        *,
        error: Exception | None = None,
        order: list[str] | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.order = order
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def embed_documents(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        if self.order is not None:
            self.order.append("embedder")
        self.calls.append((tenant_id, knowledge_base_id, texts))
        if self.error is not None:
            raise self.error
        return cast(tuple[tuple[float, ...], ...], self.output)


class FakeBatchWriter:
    def __init__(
        self,
        result: VectorBatchWriteResult | None = None,
        *,
        error: Exception | None = None,
        order: list[str] | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.order = order
        self.calls: list[VectorBatchWriteRequest] = []

    def admit_batch(
        self,
        request: VectorBatchWriteRequest,
    ) -> VectorBatchWriteResult:
        if self.order is not None:
            self.order.append("writer")
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        return VectorBatchWriteResult(
            tenant_id=request.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
            inserted_identities=tuple(
                sorted(
                    (
                        VectorRecordIdentity(
                            document_id=record.document_id,
                            chunk_id=record.chunk_id,
                        )
                        for record in request.records
                    ),
                    key=lambda identity: (identity.document_id, identity.chunk_id),
                )
            ),
            unchanged_identities=(),
        )


def chunk(
    document_id: str = "guide_a",
    chunk_id: str = "chunk_1",
    *,
    text: str | None = None,
    metadata: tuple[tuple[str, str], ...] = (("source", "synthetic"),),
) -> DocumentChunkInput:
    return DocumentChunkInput(
        document_id=document_id,
        chunk_id=chunk_id,
        text=text or f"Synthetic content for {document_id}/{chunk_id}.",
        metadata=metadata,
    )


def request(
    *chunks: DocumentChunkInput,
) -> DocumentIngestionRequest:
    selected = chunks or (
        chunk("guide_a", "chunk_1"),
        chunk("guide_b", "chunk_2"),
    )
    return DocumentIngestionRequest(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        chunks=selected,
    )


def identity(
    document_id: str,
    chunk_id: str,
) -> VectorRecordIdentity:
    return VectorRecordIdentity(document_id=document_id, chunk_id=chunk_id)


def malformed_result(
    *,
    tenant_id: str = "tenant_alpha",
    knowledge_base_id: str = "kb_support",
    inserted: tuple[VectorRecordIdentity, ...] = (),
    unchanged: tuple[VectorRecordIdentity, ...] = (),
) -> VectorBatchWriteResult:
    return VectorBatchWriteResult.model_construct(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        inserted_identities=inserted,
        unchanged_identities=unchanged,
    )


def test_models_are_frozen_and_normalize_required_text_and_metadata() -> None:
    source = DocumentChunkInput(
        document_id=" guide ",
        chunk_id=" chunk_1 ",
        text=" Synthetic content. ",
        metadata=((" source ", " synthetic "), ("locale", " tr ")),
    )
    ingestion = DocumentIngestionRequest(
        tenant_id=" tenant_alpha ",
        knowledge_base_id=" kb_support ",
        chunks=(source,),
    )

    assert source.document_id == "guide"
    assert source.chunk_id == "chunk_1"
    assert source.text == "Synthetic content."
    assert source.metadata == (("source", "synthetic"), ("locale", "tr"))
    assert ingestion.tenant_id == "tenant_alpha"
    assert ingestion.knowledge_base_id == "kb_support"
    with pytest.raises(ValidationError):
        source.text = "changed"
    with pytest.raises(ValidationError):
        ingestion.chunks = ()


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (chunk, "document_id"),
        (chunk, "chunk_id"),
        (chunk, "text"),
    ],
)
def test_required_chunk_text_is_rejected(factory: object, field: str) -> None:
    with pytest.raises(ValidationError, match=f"{field} cannot be empty"):
        factory(**{field: " "})  # type: ignore[operator]


def test_duplicate_normalized_metadata_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="metadata keys must be unique"):
        chunk(metadata=((" source ", "synthetic"), ("source", "other")))


def test_empty_chunks_are_rejected_before_collaborator_calls() -> None:
    embedder = FakeDocumentEmbedder()
    writer = FakeBatchWriter()
    service = DocumentIngestionService(embedder, writer)

    with pytest.raises(ValidationError, match="chunks cannot be empty"):
        service.ingest(
            DocumentIngestionRequest(
                tenant_id="tenant_alpha",
                knowledge_base_id="kb_support",
                chunks=(),
            )
        )

    assert embedder.calls == []
    assert writer.calls == []


def test_duplicate_identity_is_rejected_and_cross_document_chunk_is_allowed() -> None:
    with pytest.raises(ValidationError, match="identities must be unique"):
        request(chunk("guide", "chunk_1"), chunk("guide", "chunk_1"))

    accepted = request(
        chunk("guide_a", "chunk_1"),
        chunk("guide_b", "chunk_1"),
    )
    assert len(accepted.chunks) == 2


@pytest.mark.parametrize(
    ("embedder", "writer", "message"),
    [
        (object(), FakeBatchWriter(), "embed_documents"),
        (FakeDocumentEmbedder(), object(), "admit_batch"),
    ],
)
def test_invalid_collaborators_are_rejected_without_invocation(
    embedder: object,
    writer: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DocumentIngestionService(
            cast(DocumentEmbedder, embedder),
            cast(AtomicVectorBatchWriter, writer),
        )


def test_constructor_performs_no_embedding_or_write() -> None:
    embedder = FakeDocumentEmbedder()
    writer = FakeBatchWriter()

    service = DocumentIngestionService(embedder, writer)

    assert service is not None
    assert embedder.calls == []
    assert writer.calls == []


def test_exact_scope_text_record_and_batch_mapping() -> None:
    order: list[str] = []
    embedder = FakeDocumentEmbedder(order=order)
    writer = FakeBatchWriter(order=order)
    service = DocumentIngestionService(embedder, writer)
    source = DocumentIngestionRequest(
        tenant_id=" tenant_alpha ",
        knowledge_base_id=" kb_support ",
        chunks=(
            chunk(
                "guide_a",
                "chunk_1",
                text=" Synthetic first. ",
                metadata=((" source ", " synthetic "),),
            ),
            chunk("guide_b", "chunk_2", text="Synthetic second."),
        ),
    )

    service.ingest(source)

    assert embedder.calls == [
        (
            "tenant_alpha",
            "kb_support",
            ("Synthetic first.", "Synthetic second."),
        )
    ]
    assert order == ["embedder", "writer"]
    assert len(writer.calls) == 1
    written = writer.calls[0]
    assert written.tenant_id == "tenant_alpha"
    assert written.knowledge_base_id == "kb_support"
    assert tuple(record.document_id for record in written.records) == (
        "guide_a",
        "guide_b",
    )
    assert tuple(record.chunk_id for record in written.records) == (
        "chunk_1",
        "chunk_2",
    )
    assert tuple(record.text for record in written.records) == (
        "Synthetic first.",
        "Synthetic second.",
    )
    assert tuple(record.embedding for record in written.records) == (
        (0.25, 0.75),
        (0.5, 0.5),
    )
    assert written.records[0].metadata == (("source", "synthetic"),)


@pytest.mark.parametrize(
    "output",
    [
        (),
        ((0.25, 0.75),),
        ((0.25, 0.75), (0.5, 0.5), (0.1, 0.9)),
        0.25,
        (0.25, (0.5, 0.5)),
        ((), (0.5, 0.5)),
        (((0.25, 0.75),), ((0.5, 0.5),)),
    ],
)
def test_wrong_shape_or_empty_vectors_prevent_writer_call(output: object) -> None:
    embedder = FakeDocumentEmbedder(output)
    writer = FakeBatchWriter()

    with pytest.raises(ValueError):
        DocumentIngestionService(embedder, writer).ingest(request())

    assert writer.calls == []


@pytest.mark.parametrize(
    "value",
    [True, "0.25", object(), 1 + 2j, nan, inf, -inf],
)
def test_invalid_embedding_values_prevent_writer_call(value: object) -> None:
    embedder = FakeDocumentEmbedder(((value, 0.5), (0.25, 0.75)))
    writer = FakeBatchWriter()

    with pytest.raises(ValueError):
        DocumentIngestionService(embedder, writer).ingest(request())

    assert writer.calls == []


def test_inconsistent_dimensions_prevent_writer_call() -> None:
    embedder = FakeDocumentEmbedder(((0.2, 0.8), (0.2, 0.3, 0.5)))
    writer = FakeBatchWriter()

    with pytest.raises(ValueError, match="equal dimensions"):
        DocumentIngestionService(embedder, writer).ingest(request())

    assert writer.calls == []


def test_embedder_and_writer_exceptions_propagate_unchanged() -> None:
    embedder_error = RuntimeError("synthetic embedder failure")
    embedder = FakeDocumentEmbedder(error=embedder_error)
    writer = FakeBatchWriter()
    with pytest.raises(RuntimeError, match="embedder failure"):
        DocumentIngestionService(embedder, writer).ingest(request())
    assert writer.calls == []

    writer_error = RuntimeError("synthetic writer conflict")
    writer = FakeBatchWriter(error=writer_error)
    with pytest.raises(RuntimeError, match="writer conflict"):
        DocumentIngestionService(FakeDocumentEmbedder(), writer).ingest(request())
    assert len(writer.calls) == 1


def test_repeated_requests_preserve_writer_idempotency_and_determinism() -> None:
    store = InMemoryVectorStore()
    service = DocumentIngestionService(FakeDocumentEmbedder(), store)
    source = request()

    first = service.ingest(source)
    second = service.ingest(source)
    third = service.ingest(source)

    assert first.inserted_identities == (
        identity("guide_a", "chunk_1"),
        identity("guide_b", "chunk_2"),
    )
    assert second == third
    assert second.inserted_identities == ()
    assert second.unchanged_identities == first.inserted_identities


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            malformed_result(
                tenant_id="tenant_beta",
                inserted=(
                    identity("guide_a", "chunk_1"),
                    identity("guide_b", "chunk_2"),
                ),
            ),
            "tenant_id",
        ),
        (
            malformed_result(
                knowledge_base_id="kb_other",
                inserted=(
                    identity("guide_a", "chunk_1"),
                    identity("guide_b", "chunk_2"),
                ),
            ),
            "knowledge_base_id",
        ),
        (
            malformed_result(
                inserted=(
                    identity("guide_a", "chunk_1"),
                    identity("guide_a", "chunk_1"),
                )
            ),
            "unique",
        ),
        (
            malformed_result(
                inserted=(identity("guide_a", "chunk_1"),),
                unchanged=(identity("guide_a", "chunk_1"),),
            ),
            "disjoint",
        ),
        (
            malformed_result(inserted=(identity("guide_a", "chunk_1"),)),
            "exactly match",
        ),
        (
            malformed_result(
                inserted=(
                    identity("guide_a", "chunk_1"),
                    identity("guide_b", "chunk_2"),
                    identity("guide_c", "chunk_3"),
                )
            ),
            "exactly match",
        ),
        (
            malformed_result(
                inserted=(
                    identity("guide_b", "chunk_2"),
                    identity("guide_a", "chunk_1"),
                )
            ),
            "ordered",
        ),
    ],
    ids=[
        "wrong-tenant",
        "wrong-knowledge-base",
        "duplicate",
        "overlap",
        "missing",
        "unexpected",
        "order",
    ],
)
def test_malformed_writer_result_is_rejected(
    result: VectorBatchWriteResult,
    message: str,
) -> None:
    writer = FakeBatchWriter(result)

    with pytest.raises(ValueError, match=message):
        DocumentIngestionService(FakeDocumentEmbedder(), writer).ingest(request())


def test_valid_result_is_returned_unchanged_without_mutating_inputs() -> None:
    expected = VectorBatchWriteResult(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        inserted_identities=(identity("guide_a", "chunk_1"),),
        unchanged_identities=(identity("guide_b", "chunk_2"),),
    )
    output = ((0.25, 0.75), (0.5, 0.5))
    embedder = FakeDocumentEmbedder(output)
    writer = FakeBatchWriter(expected)
    source = request()

    actual = DocumentIngestionService(embedder, writer).ingest(source)

    assert actual is expected
    assert output == ((0.25, 0.75), (0.5, 0.5))
    assert source == request()


def test_current_protocol_fakes_are_structurally_compatible() -> None:
    embedder: DocumentEmbedder = FakeDocumentEmbedder()
    writer: AtomicVectorBatchWriter = FakeBatchWriter()
    service = DocumentIngestionService(embedder, writer)

    assert service.ingest(request()).inserted_identities == (
        identity("guide_a", "chunk_1"),
        identity("guide_b", "chunk_2"),
    )
