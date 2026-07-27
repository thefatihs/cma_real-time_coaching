"""Synchronous composition of document embedding and atomic vector admission."""

from collections.abc import Sequence
from math import isfinite
from numbers import Real

from app.embeddings import DocumentEmbedder
from app.ingestion.models import DocumentIngestionRequest
from app.vector_store import (
    AtomicVectorBatchWriter,
    VectorBatchWriteRequest,
    VectorBatchWriteResult,
    VectorRecord,
    VectorRecordIdentity,
)


class DocumentIngestionService:
    def __init__(
        self,
        document_embedder: DocumentEmbedder,
        batch_writer: AtomicVectorBatchWriter,
    ) -> None:
        if not callable(getattr(document_embedder, "embed_documents", None)):
            raise ValueError("document_embedder.embed_documents must be callable")
        if not callable(getattr(batch_writer, "admit_batch", None)):
            raise ValueError("batch_writer.admit_batch must be callable")
        self._document_embedder = document_embedder
        self._batch_writer = batch_writer

    def ingest(
        self,
        request: DocumentIngestionRequest,
    ) -> VectorBatchWriteResult:
        output: object = self._document_embedder.embed_documents(
            tenant_id=request.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
            texts=tuple(chunk.text for chunk in request.chunks),
        )
        vectors = _validated_vectors(output, expected_rows=len(request.chunks))
        records = tuple(
            VectorRecord(
                tenant_id=request.tenant_id,
                knowledge_base_id=request.knowledge_base_id,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                embedding=vector,
                metadata=chunk.metadata,
            )
            for chunk, vector in zip(request.chunks, vectors, strict=True)
        )
        result = self._batch_writer.admit_batch(
            VectorBatchWriteRequest(
                tenant_id=request.tenant_id,
                knowledge_base_id=request.knowledge_base_id,
                records=records,
            )
        )
        _validate_result(result, request)
        return result


def _validated_vectors(
    output: object,
    *,
    expected_rows: int,
) -> tuple[tuple[float, ...], ...]:
    if (
        isinstance(output, (str, bytes))
        or not isinstance(output, Sequence)
        or len(output) != expected_rows
    ):
        raise ValueError("embedding output row count does not match chunk count")

    vectors: list[tuple[float, ...]] = []
    dimension: int | None = None
    for row in output:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError("embedding output row must be a vector")
        if not row:
            raise ValueError("embedding output vector cannot be empty")
        values: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, Real):
                raise ValueError("embedding output must contain real numbers")
            value = float(item)
            if not isfinite(value):
                raise ValueError("embedding output must contain finite values")
            values.append(value)
        vector = tuple(values)
        if dimension is not None and len(vector) != dimension:
            raise ValueError("embedding output vectors must have equal dimensions")
        dimension = len(vector)
        vectors.append(vector)
    return tuple(vectors)


def _validate_result(
    result: VectorBatchWriteResult,
    request: DocumentIngestionRequest,
) -> None:
    if result.tenant_id != request.tenant_id:
        raise ValueError("batch result tenant_id does not match ingestion request")
    if result.knowledge_base_id != request.knowledge_base_id:
        raise ValueError(
            "batch result knowledge_base_id does not match ingestion request"
        )

    inserted = _identity_keys(result.inserted_identities)
    unchanged = _identity_keys(result.unchanged_identities)
    if len(inserted) != len(set(inserted)):
        raise ValueError("batch result inserted identities must be unique")
    if len(unchanged) != len(set(unchanged)):
        raise ValueError("batch result unchanged identities must be unique")
    if set(inserted) & set(unchanged):
        raise ValueError("batch result identity groups must be disjoint")
    if inserted != tuple(sorted(inserted)):
        raise ValueError("batch result inserted identities must be ordered")
    if unchanged != tuple(sorted(unchanged)):
        raise ValueError("batch result unchanged identities must be ordered")

    requested = {(chunk.document_id, chunk.chunk_id) for chunk in request.chunks}
    if set(inserted) | set(unchanged) != requested:
        raise ValueError("batch result identities must exactly match ingestion request")


def _identity_keys(
    identities: tuple[VectorRecordIdentity, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple((identity.document_id, identity.chunk_id) for identity in identities)
