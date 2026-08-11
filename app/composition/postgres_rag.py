"""Side-effect-free production PostgreSQL RAG dependency composition."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Literal, TypeVar

from psycopg import Connection
from pydantic import (
    BaseModel,
    ConfigDict,
    SecretStr,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.embeddings.sentence_transformers import (
    BackendFactory,
    SentenceTransformerQueryEmbedder,
    SentenceTransformerQueryEmbedderConfig,
)
from app.ingestion.service import DocumentIngestionService
from app.retrieval.vector_backed import VectorBackedRetriever
from app.vector_store.embedding_profile import (
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
)
from app.vector_store.postgres.adapter import ProfileBoundPostgreSQLVectorStore
from app.vector_store.postgres.connection_factory import (
    PgvectorPsycopgConnectionFactory,
)
from app.vector_store.postgres.profile_repository import (
    PostgreSQLEmbeddingProfileRepository,
)
from app.vector_store.postgres.runner import (
    PsycopgPostgreSQLVectorTransactionRunner,
)
from app.vector_store.postgres.transaction import (
    PsycopgPostgreSQLVectorTransaction,
)
from app.vector_store.models import (
    SearchRequest,
    SearchResult,
    VectorBatchWriteRequest,
    VectorBatchWriteResult,
    VectorRecord,
)

PsycopgConnect = Callable[..., Connection[Any]]
SecureSSLMode = Literal["require", "verify-ca", "verify-full"]

_APPLICATION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_ResultT = TypeVar("_ResultT")


class RAGDiagnosticStage(str, Enum):
    RUN = "D_RUN"
    EMBED = "D_EMBED"
    VECTOR = "D_VECTOR"
    PROMPT = "D_PROMPT"
    GATEWAY_FACTORY = "D_GATEWAY_FACTORY"
    HTTP = "D_HTTP"
    CALLBACK = "D_CALLBACK"


class RAGDiagnosticStatus(str, Enum):
    ENTER = "ENTER"
    OK = "OK"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class RAGDiagnosticEvent:
    stage: RAGDiagnosticStage
    status: RAGDiagnosticStatus


@dataclass(frozen=True, slots=True)
class RAGDiagnosticSnapshot:
    events: tuple[RAGDiagnosticEvent, ...]
    worker_live: bool
    future_terminal: bool
    callback_entered: bool
    authoritative_completion_published: bool
    running_after_close: bool
    unclassified: bool


class BoundedRAGDiagnosticObserver:
    """Thread-safe, value-free diagnostics for one bounded orchestration."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._events: list[RAGDiagnosticEvent] = []
        self._worker_live = False
        self._future_terminal = False
        self._callback_entered = False
        self._completion_published = False
        self._running_after_close = False
        self._unclassified = False

    def record(self, stage: RAGDiagnosticStage, status: RAGDiagnosticStatus) -> None:
        try:
            if not isinstance(stage, RAGDiagnosticStage) or not isinstance(
                status, RAGDiagnosticStatus
            ):
                raise ValueError
            with self._lock:
                if len(self._events) >= len(RAGDiagnosticStage) * 3:
                    self._unclassified = True
                    return
                self._events.append(RAGDiagnosticEvent(stage, status))
        except BaseException:
            self._mark_unclassified()

    def update_execution(
        self,
        *,
        worker_live: bool | None = None,
        future_terminal: bool | None = None,
        callback_entered: bool | None = None,
        completion_published: bool | None = None,
        running_after_close: bool | None = None,
    ) -> None:
        values = (
            worker_live,
            future_terminal,
            callback_entered,
            completion_published,
            running_after_close,
        )
        if any(value is not None and type(value) is not bool for value in values):
            self._mark_unclassified()
            return
        try:
            with self._lock:
                if worker_live is not None:
                    self._worker_live = worker_live
                if future_terminal is not None:
                    self._future_terminal = future_terminal
                if callback_entered is not None:
                    self._callback_entered = callback_entered
                if completion_published is not None:
                    self._completion_published = completion_published
                if running_after_close is not None:
                    self._running_after_close = running_after_close
        except BaseException:
            self._mark_unclassified()

    def snapshot(self) -> RAGDiagnosticSnapshot:
        with self._lock:
            return RAGDiagnosticSnapshot(
                events=tuple(self._events),
                worker_live=self._worker_live,
                future_terminal=self._future_terminal,
                callback_entered=self._callback_entered,
                authoritative_completion_published=self._completion_published,
                running_after_close=self._running_after_close,
                unclassified=self._unclassified,
            )

    def _mark_unclassified(self) -> None:
        try:
            with self._lock:
                self._unclassified = True
        except BaseException:
            pass


def observe_rag_stage(
    observer: BoundedRAGDiagnosticObserver | None,
    stage: RAGDiagnosticStage,
    operation: Callable[[], _ResultT],
) -> _ResultT:
    """Run an unchanged operation with optional fixed-value observation."""
    if observer is None:
        return operation()
    observer.record(stage, RAGDiagnosticStatus.ENTER)
    try:
        result = operation()
    except BaseException:
        observer.record(stage, RAGDiagnosticStatus.FAILED)
        raise
    observer.record(stage, RAGDiagnosticStatus.OK)
    return result


class _ObservedEmbedder(SentenceTransformerQueryEmbedder):
    def __init__(
        self,
        delegate: SentenceTransformerQueryEmbedder,
        observer: BoundedRAGDiagnosticObserver,
    ) -> None:
        self._delegate = delegate
        self._observer = observer

    def embed_query(
        self, *, tenant_id: str, knowledge_base_id: str, text: str
    ) -> tuple[float, ...]:
        return observe_rag_stage(
            self._observer,
            RAGDiagnosticStage.EMBED,
            lambda: self._delegate.embed_query(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                text=text,
            ),
        )

    def embed_documents(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        return self._delegate.embed_documents(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            texts=texts,
        )


class _ObservedVectorStore(ProfileBoundPostgreSQLVectorStore):
    def __init__(
        self,
        delegate: ProfileBoundPostgreSQLVectorStore,
        observer: BoundedRAGDiagnosticObserver,
    ) -> None:
        self._delegate = delegate
        self._observer = observer

    def upsert(self, record: VectorRecord) -> None:
        self._delegate.upsert(record)

    def search(self, request: SearchRequest) -> SearchResult:
        return observe_rag_stage(
            self._observer,
            RAGDiagnosticStage.VECTOR,
            lambda: self._delegate.search(request),
        )

    def admit_batch(self, request: VectorBatchWriteRequest) -> VectorBatchWriteResult:
        return self._delegate.admit_batch(request)


class PostgreSQLVectorStoreSettings(BaseSettings):
    """Secret-safe PostgreSQL connection settings loaded only when instantiated."""

    model_config = SettingsConfigDict(
        env_prefix="CALLMETRIC_POSTGRES_",
        env_file=None,
        extra="forbid",
        frozen=True,
    )

    dsn: SecretStr
    connect_timeout_seconds: int
    ssl_mode: SecureSSLMode
    application_name: str

    @field_validator("dsn")
    @classmethod
    def validate_dsn(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()
        if not secret or secret != secret.strip():
            raise ValueError("PostgreSQL DSN must be canonical and nonblank")
        return value

    @field_validator("connect_timeout_seconds", mode="before")
    @classmethod
    def validate_connect_timeout(cls, value: object) -> int:
        if isinstance(value, str):
            if not value.isascii() or not value.isdigit():
                raise ValueError("connect timeout must be an integer")
            value = int(value)
        if type(value) is not int:
            raise ValueError("connect timeout must be an integer")
        if not 1 <= value <= 60:
            raise ValueError("connect timeout must be between 1 and 60 seconds")
        return value

    @field_validator("application_name")
    @classmethod
    def validate_application_name(cls, value: str) -> str:
        cleaned = _required_text(value, "application_name")
        if not _APPLICATION_NAME_PATTERN.fullmatch(cleaned):
            raise ValueError("application_name contains unsafe characters")
        return cleaned


class KnowledgeBaseRAGProviderSettings(BaseModel):
    """Explicit immutable provider configuration for one tenant/KB scope."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    knowledge_base_id: str
    model_id: str
    model_name_or_path: str
    vector_dimension: int
    normalize_embeddings: bool
    device: Literal["cpu", "cuda"]
    local_files_only: bool

    @field_validator(
        "tenant_id",
        "knowledge_base_id",
        "model_id",
        "model_name_or_path",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))

    @field_validator("vector_dimension", mode="before")
    @classmethod
    def validate_vector_dimension(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("vector_dimension must be an integer")
        if value <= 0:
            raise ValueError("vector_dimension must be positive")
        return value

    @field_validator("normalize_embeddings", mode="before")
    @classmethod
    def validate_normalization(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("normalize_embeddings must be a boolean")
        return value

    @field_validator("local_files_only", mode="before")
    @classmethod
    def validate_local_files_only(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("local_files_only must be a boolean")
        if value is not True:
            raise ValueError("local_files_only must be exactly True")
        return value


@dataclass(frozen=True, slots=True)
class PostgreSQLRAGComposition:
    """Complete profile-bound PostgreSQL embedding/retrieval dependencies."""

    profile: KnowledgeBaseEmbeddingProfile
    profile_repository: PostgreSQLEmbeddingProfileRepository
    vector_store: ProfileBoundPostgreSQLVectorStore
    embedder: SentenceTransformerQueryEmbedder
    ingestion_service: DocumentIngestionService
    retriever: VectorBackedRetriever


def compose_profile_bound_postgres_rag(
    *,
    postgres_settings: PostgreSQLVectorStoreSettings,
    knowledge_base_settings: KnowledgeBaseRAGProviderSettings,
    psycopg_connect: PsycopgConnect,
    embedding_backend_factory: BackendFactory | None = None,
    diagnostic_observer: BoundedRAGDiagnosticObserver | None = None,
) -> PostgreSQLRAGComposition:
    """Construct dependencies without opening connections or loading models."""
    if not isinstance(postgres_settings, PostgreSQLVectorStoreSettings):
        raise ValueError(
            "postgres_settings must be PostgreSQLVectorStoreSettings",
        )
    if not isinstance(knowledge_base_settings, KnowledgeBaseRAGProviderSettings):
        raise ValueError(
            "knowledge_base_settings must be KnowledgeBaseRAGProviderSettings",
        )
    if not callable(psycopg_connect):
        raise ValueError("psycopg_connect must be callable")
    if embedding_backend_factory is not None and not callable(
        embedding_backend_factory
    ):
        raise ValueError("embedding_backend_factory must be callable")
    if diagnostic_observer is not None and not isinstance(
        diagnostic_observer, BoundedRAGDiagnosticObserver
    ):
        raise ValueError("diagnostic_observer is invalid")

    def base_connection_factory() -> Connection[Any]:
        return psycopg_connect(
            conninfo=postgres_settings.dsn.get_secret_value(),
            connect_timeout=postgres_settings.connect_timeout_seconds,
            sslmode=postgres_settings.ssl_mode,
            application_name=postgres_settings.application_name,
            autocommit=False,
        )

    connection_factory = PgvectorPsycopgConnectionFactory(
        base_connection_factory=base_connection_factory,
    )
    transaction_runner = PsycopgPostgreSQLVectorTransactionRunner(
        connection_factory=connection_factory,
        transaction_factory=PsycopgPostgreSQLVectorTransaction,
    )
    profile = KnowledgeBaseEmbeddingProfile(
        tenant_id=knowledge_base_settings.tenant_id,
        knowledge_base_id=knowledge_base_settings.knowledge_base_id,
        model_id=knowledge_base_settings.model_id,
        vector_dimension=knowledge_base_settings.vector_dimension,
        normalize_embeddings=knowledge_base_settings.normalize_embeddings,
        distance_metric=EmbeddingDistanceMetric.COSINE,
    )
    profile_repository = PostgreSQLEmbeddingProfileRepository(transaction_runner)
    vector_store = ProfileBoundPostgreSQLVectorStore(
        expected_profile=profile,
        transaction_runner=transaction_runner,
    )
    embedder_config = SentenceTransformerQueryEmbedderConfig(
        expected_tenant_id=knowledge_base_settings.tenant_id,
        expected_knowledge_base_id=knowledge_base_settings.knowledge_base_id,
        model_name_or_path=knowledge_base_settings.model_name_or_path,
        device=knowledge_base_settings.device,
        normalize_embeddings=knowledge_base_settings.normalize_embeddings,
        local_files_only=knowledge_base_settings.local_files_only,
    )
    embedder = SentenceTransformerQueryEmbedder(
        embedder_config,
        backend_factory=embedding_backend_factory,
    )
    effective_embedder = embedder
    effective_vector_store = vector_store
    if diagnostic_observer is not None:
        effective_embedder = _ObservedEmbedder(embedder, diagnostic_observer)
        effective_vector_store = _ObservedVectorStore(vector_store, diagnostic_observer)
    ingestion_service = DocumentIngestionService(
        effective_embedder, effective_vector_store
    )
    retriever = VectorBackedRetriever(effective_embedder, effective_vector_store)
    return PostgreSQLRAGComposition(
        profile=profile,
        profile_repository=profile_repository,
        vector_store=effective_vector_store,
        embedder=effective_embedder,
        ingestion_service=ingestion_service,
        retriever=retriever,
    )


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
