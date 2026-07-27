"""Deterministic in-memory vector store."""

from math import fsum

from app.vector_store.models import (
    SearchRequest,
    SearchResult,
    VectorRecord,
    VectorSearchHit,
)

RecordKey = tuple[str, str, str]
ScopeKey = tuple[str, str]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: dict[RecordKey, VectorRecord] = {}
        self._dimensions: dict[ScopeKey, int] = {}

    def upsert(self, record: VectorRecord) -> None:
        scope = (record.tenant_id, record.knowledge_base_id)
        dimension = self._dimensions.get(scope)
        if dimension is not None and dimension != len(record.embedding):
            raise ValueError("embedding dimension does not match vector-store scope")
        self._dimensions[scope] = len(record.embedding)
        self._records[(*scope, record.chunk_id)] = record

    def search(self, request: SearchRequest) -> SearchResult:
        scope = (request.tenant_id, request.knowledge_base_id)
        dimension = self._dimensions.get(scope)
        if dimension is not None and dimension != len(request.query_embedding):
            raise ValueError(
                "query embedding dimension does not match vector-store scope"
            )

        hits: list[VectorSearchHit] = []
        for key, record in self._records.items():
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
