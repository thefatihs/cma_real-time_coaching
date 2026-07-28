"""SQL-free transaction contracts for a future PostgreSQL vector adapter."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from app.vector_store.embedding_profile import KnowledgeBaseEmbeddingProfile
from app.vector_store.models import VectorRecordIdentity


@dataclass(frozen=True, slots=True)
class PostgreSQLStoredVectorRow:
    tenant_id: str
    knowledge_base_id: str
    document_id: str
    chunk_id: str
    text: str
    embedding: tuple[float, ...]
    metadata_json: str


@dataclass(frozen=True, slots=True)
class PostgreSQLCosineSearchRow:
    tenant_id: str
    knowledge_base_id: str
    document_id: str
    chunk_id: str
    text: str
    embedding: tuple[float, ...]
    metadata_json: str
    cosine_distance: float


class PostgreSQLVectorTransaction(Protocol):
    """Domain operations executed within a runner-owned transaction.

    Implementations execute within the transaction established by
    ``PostgreSQLVectorTransactionRunner`` and must not commit, roll back, close,
    or otherwise manage the connection lifecycle.
    """

    def acquire_scope_lock(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> None: ...

    def get_profile(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        for_update: bool,
    ) -> KnowledgeBaseEmbeddingProfile | None: ...

    def insert_profile(
        self,
        profile: KnowledgeBaseEmbeddingProfile,
    ) -> None: ...

    def get_records(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        identities: tuple[VectorRecordIdentity, ...],
    ) -> tuple[PostgreSQLStoredVectorRow, ...]: ...

    def insert_records(
        self,
        rows: tuple[PostgreSQLStoredVectorRow, ...],
    ) -> None: ...

    def replace_record(
        self,
        row: PostgreSQLStoredVectorRow,
    ) -> None: ...

    def search_cosine(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        query_embedding: tuple[float, ...],
        top_k: int,
        maximum_cosine_distance: float,
    ) -> tuple[PostgreSQLCosineSearchRow, ...]: ...


T = TypeVar("T")


class PostgreSQLVectorTransactionRunner(Protocol):
    """Own one complete transaction around a callback invocation.

    Implementations acquire a connection, begin one transaction, and invoke
    ``operation`` exactly once. They commit only after a successful callback
    and return its result unchanged. If the callback raises, they roll back and
    re-raise the original exception. The connection is released in every case.
    """

    def run_in_transaction(
        self,
        operation: Callable[[PostgreSQLVectorTransaction], T],
    ) -> T: ...
