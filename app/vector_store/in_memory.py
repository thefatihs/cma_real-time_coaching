"""Deterministic in-memory vector store."""

from dataclasses import dataclass
from math import fsum

from app.vector_store.models import (
    SearchRequest,
    SearchResult,
    VectorBatchWriteRequest,
    VectorBatchWriteResult,
    VectorRecord,
    VectorRecordIdentity,
    VectorSearchHit,
)

RecordKey = tuple[str, str, str, str]
ScopeKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class _StoreState:
    records: dict[RecordKey, VectorRecord]
    dimensions: dict[ScopeKey, int]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._state = _StoreState(records={}, dimensions={})

    def upsert(self, record: VectorRecord) -> None:
        scope = (record.tenant_id, record.knowledge_base_id)
        dimension = self._state.dimensions.get(scope)
        if dimension is not None and dimension != len(record.embedding):
            raise ValueError("embedding dimension does not match vector-store scope")
        records = dict(self._state.records)
        dimensions = dict(self._state.dimensions)
        dimensions[scope] = len(record.embedding)
        records[(*scope, record.document_id, record.chunk_id)] = record
        self._state = _StoreState(records=records, dimensions=dimensions)

    def admit_batch(
        self,
        request: VectorBatchWriteRequest,
    ) -> VectorBatchWriteResult:
        scope = (request.tenant_id, request.knowledge_base_id)
        batch_dimension = len(request.records[0].embedding)
        existing_dimension = self._state.dimensions.get(scope)
        if existing_dimension is not None and existing_dimension != batch_dimension:
            raise ValueError("embedding dimension does not match vector-store scope")

        candidate_records = dict(self._state.records)
        candidate_dimensions = dict(self._state.dimensions)
        inserted: list[VectorRecordIdentity] = []
        unchanged: list[VectorRecordIdentity] = []
        for record in request.records:
            key = (*scope, record.document_id, record.chunk_id)
            identity = VectorRecordIdentity(
                document_id=record.document_id,
                chunk_id=record.chunk_id,
            )
            existing = candidate_records.get(key)
            if existing is None:
                candidate_records[key] = record
                inserted.append(identity)
            elif existing == record:
                unchanged.append(identity)
            else:
                raise ValueError("existing vector record conflicts with batch record")

        candidate_dimensions[scope] = batch_dimension
        result = VectorBatchWriteResult(
            tenant_id=request.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
            inserted_identities=tuple(
                sorted(
                    inserted,
                    key=lambda identity: (identity.document_id, identity.chunk_id),
                )
            ),
            unchanged_identities=tuple(
                sorted(
                    unchanged,
                    key=lambda identity: (identity.document_id, identity.chunk_id),
                )
            ),
        )
        self._state = _StoreState(
            records=candidate_records,
            dimensions=candidate_dimensions,
        )
        return result

    def search(self, request: SearchRequest) -> SearchResult:
        scope = (request.tenant_id, request.knowledge_base_id)
        state = self._state
        dimension = state.dimensions.get(scope)
        if dimension is not None and dimension != len(request.query_embedding):
            raise ValueError(
                "query embedding dimension does not match vector-store scope"
            )

        hits: list[VectorSearchHit] = []
        for key, record in state.records.items():
            if key[:2] != scope:
                continue
            score = _dot_product(request.query_embedding, record.embedding)
            if not 0 <= score <= 1:
                raise ValueError(
                    "computed relevance score must be finite and between 0 and 1"
                )
            if score >= request.minimum_score:
                hits.append(VectorSearchHit(record=record, score=score))
        ranked = sorted(
            hits,
            key=lambda hit: (
                -hit.score,
                hit.record.document_id,
                hit.record.chunk_id,
            ),
        )
        return SearchResult(
            tenant_id=request.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
            hits=tuple(ranked[: request.top_k]),
        )


def _dot_product(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    return fsum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
