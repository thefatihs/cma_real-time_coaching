"""Production composition for bounded PostgreSQL/MiniLM document ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, field_validator

from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLRAGComposition,
    PostgreSQLVectorStoreSettings,
    PsycopgConnect,
    compose_profile_bound_postgres_rag,
)
from app.embeddings.sentence_transformers import BackendFactory
from app.ingestion.document_background import BoundedDocumentIngestionManager
from app.ingestion.postgres_registry import PsycopgDocumentRegistryRepository

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIMENSION = 384


class PostgreSQLDocumentIngestionSettings(BaseModel):
    """Strict non-secret settings supplied by later dashboard wiring."""

    model_config = ConfigDict(frozen=True)

    max_workers: int = 1
    capacity: int = 1

    @field_validator("max_workers", mode="before")
    @classmethod
    def validate_workers(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("max_workers must be exactly 1")
        return value

    @field_validator("capacity", mode="before")
    @classmethod
    def validate_capacity(cls, value: object) -> int:
        if type(value) is not int or not 1 <= value <= 8:
            raise ValueError("capacity must be between 1 and 8")
        return value


@dataclass(frozen=True, slots=True)
class PostgreSQLDocumentIngestionRuntime:
    """Narrow lifecycle surface intended for later dashboard ownership."""

    manager: BoundedDocumentIngestionManager
    registry: PsycopgDocumentRegistryRepository
    postgres_rag: PostgreSQLRAGComposition
    tenant_id: str
    knowledge_base_id: str

    def close(self, *, wait: bool = False) -> None:
        self.manager.close(wait=wait)


def compose_postgres_document_ingestion(
    *,
    postgres_settings: PostgreSQLVectorStoreSettings,
    knowledge_base_settings: KnowledgeBaseRAGProviderSettings,
    ingestion_settings: PostgreSQLDocumentIngestionSettings,
    psycopg_connect: PsycopgConnect,
    embedding_backend_factory: BackendFactory | None = None,
) -> PostgreSQLDocumentIngestionRuntime:
    """Compose dependencies without a connection, model load, or network action."""
    _validate_minilm_settings(knowledge_base_settings)
    postgres_rag = compose_profile_bound_postgres_rag(
        postgres_settings=postgres_settings,
        knowledge_base_settings=knowledge_base_settings,
        psycopg_connect=psycopg_connect,
        embedding_backend_factory=embedding_backend_factory,
    )

    def connection_factory():
        return psycopg_connect(
            conninfo=postgres_settings.dsn.get_secret_value(),
            connect_timeout=postgres_settings.connect_timeout_seconds,
            sslmode=postgres_settings.ssl_mode,
            application_name=postgres_settings.application_name,
            autocommit=False,
        )

    registry = PsycopgDocumentRegistryRepository(connection_factory=connection_factory)

    def verify_registered_profile() -> None:
        stored = postgres_rag.profile_repository.get_profile(
            tenant_id=knowledge_base_settings.tenant_id,
            knowledge_base_id=knowledge_base_settings.knowledge_base_id,
        )
        if stored != postgres_rag.profile:
            raise ValueError("registered embedding profile is incompatible")

    manager = BoundedDocumentIngestionManager(
        tenant_id=knowledge_base_settings.tenant_id,
        knowledge_base_id=knowledge_base_settings.knowledge_base_id,
        capacity=ingestion_settings.capacity,
        registry=registry,
        document_embedder=postgres_rag.embedder,
        vector_writer=postgres_rag.vector_store,
        expected_vector_dimension=MINILM_DIMENSION,
        availability_check=verify_registered_profile,
    )
    return PostgreSQLDocumentIngestionRuntime(
        manager=manager,
        registry=registry,
        postgres_rag=postgres_rag,
        tenant_id=knowledge_base_settings.tenant_id,
        knowledge_base_id=knowledge_base_settings.knowledge_base_id,
    )


def _validate_minilm_settings(settings: KnowledgeBaseRAGProviderSettings) -> None:
    if not isinstance(settings, KnowledgeBaseRAGProviderSettings):
        raise ValueError("knowledge_base_settings is invalid")
    if (
        settings.model_id != MINILM_MODEL
        or settings.model_name_or_path != MINILM_MODEL
        or settings.vector_dimension != MINILM_DIMENSION
        or settings.normalize_embeddings is not True
        or settings.device != "cpu"
        or settings.local_files_only is not True
    ):
        raise ValueError("knowledge base embedding profile is incompatible")
