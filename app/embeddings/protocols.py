"""Provider-neutral query embedding interface."""

from typing import Protocol


class QueryEmbedder(Protocol):
    def embed_query(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        text: str,
    ) -> tuple[float, ...]: ...


class DocumentEmbedder(Protocol):
    def embed_documents(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]: ...
