"""Provider-neutral query-aware retrieval backed by vector search."""

from math import isfinite

from app.embeddings import QueryEmbedder
from app.retrieval.models import RetrievalDocument, RetrievalResult
from app.vector_store import SearchRequest, VectorStore


class VectorBackedRetriever:
    def __init__(
        self,
        query_embedder: QueryEmbedder,
        vector_store: VectorStore,
    ) -> None:
        self._query_embedder = query_embedder
        self._vector_store = vector_store

    def retrieve(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        query: str,
        top_k: int,
        minimum_score: float = 0.0,
    ) -> RetrievalResult:
        tenant = _required_text(tenant_id, "tenant_id")
        knowledge_base = _required_text(knowledge_base_id, "knowledge_base_id")
        normalized_query = _required_text(query, "query")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        _validated_score(minimum_score, "minimum_score")

        query_embedding = self._query_embedder.embed_query(
            tenant_id=tenant,
            knowledge_base_id=knowledge_base,
            text=normalized_query,
        )
        search_request = SearchRequest(
            tenant_id=tenant,
            knowledge_base_id=knowledge_base,
            query_embedding=query_embedding,
            top_k=top_k,
            minimum_score=minimum_score,
        )
        search_result = self._vector_store.search(search_request)
        if search_result.tenant_id != tenant:
            raise ValueError("vector search result tenant_id does not match request")
        if search_result.knowledge_base_id != knowledge_base:
            raise ValueError(
                "vector search result knowledge_base_id does not match request"
            )

        accepted: list[RetrievalDocument] = []
        identities: set[tuple[str, str]] = set()
        for hit in search_result.hits:
            record = hit.record
            if record.tenant_id != tenant:
                raise ValueError("vector search hit tenant_id does not match request")
            if record.knowledge_base_id != knowledge_base:
                raise ValueError(
                    "vector search hit knowledge_base_id does not match request"
                )
            document_id = _required_text(record.document_id, "document_id")
            chunk_id = _required_text(record.chunk_id, "chunk_id")
            text = _required_text(record.text, "text")
            score = _validated_score(hit.score, "score")
            if len(record.embedding) != len(search_request.query_embedding):
                raise ValueError("vector search hit embedding dimension does not match")
            identity = (document_id, chunk_id)
            if identity in identities:
                raise ValueError("vector search hit identities must be unique")
            identities.add(identity)
            if score >= minimum_score:
                accepted.append(
                    RetrievalDocument(
                        tenant_id=tenant,
                        knowledge_base_id=knowledge_base,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        text=text,
                        score=score,
                    )
                )

        ordered = sorted(
            accepted,
            key=lambda document: (
                -document.score,
                document.document_id,
                document.chunk_id,
            ),
        )
        return RetrievalResult(
            tenant_id=tenant,
            knowledge_base_id=knowledge_base,
            documents=tuple(ordered[:top_k]),
        )


def _validated_score(value: float, field_name: str) -> float:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return value


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
