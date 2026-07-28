"""Side-effect-free production PostgreSQL RAG dependency composition."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

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

PsycopgConnect = Callable[..., Connection[Any]]
SecureSSLMode = Literal["require", "verify-ca", "verify-full"]

_APPLICATION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")


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
    ingestion_service = DocumentIngestionService(embedder, vector_store)
    retriever = VectorBackedRetriever(embedder, vector_store)
    return PostgreSQLRAGComposition(
        profile=profile,
        profile_repository=profile_repository,
        vector_store=vector_store,
        embedder=embedder,
        ingestion_service=ingestion_service,
        retriever=retriever,
    )


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
