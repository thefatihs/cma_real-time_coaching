"""Production composition for bounded PostgreSQL/MiniLM document ingestion."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

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
from app.vector_store.postgres.connection_factory import (
    PgvectorPsycopgConnectionFactory,
)

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIMENSION = 384
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_APPROVED_ARTIFACT_ROOT = _REPOSITORY_ROOT / "local_artifacts"
_MINILM_MANIFEST = "minilm-1110a243.sha256"
_MINILM_REQUIRED_FILES = frozenset(
    {
        "1_Pooling/config.json",
        "config.json",
        "config_sentence_transformers.json",
        "model.safetensors",
        "modules.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    }
)
_LOCAL_SNAPSHOT_ERROR = "local embedding snapshot is invalid"


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

    def base_connection_factory():
        return psycopg_connect(
            conninfo=postgres_settings.dsn.get_secret_value(),
            connect_timeout=postgres_settings.connect_timeout_seconds,
            sslmode=postgres_settings.ssl_mode,
            application_name=postgres_settings.application_name,
            autocommit=False,
        )

    registry = PsycopgDocumentRegistryRepository(
        connection_factory=PgvectorPsycopgConnectionFactory(
            base_connection_factory=base_connection_factory,
        )
    )

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
        or settings.vector_dimension != MINILM_DIMENSION
        or settings.normalize_embeddings is not True
        or settings.device != "cpu"
        or settings.local_files_only is not True
    ):
        raise ValueError("knowledge base embedding profile is incompatible")
    if settings.model_name_or_path != MINILM_MODEL:
        _validate_local_minilm_snapshot(settings.model_name_or_path)


def _validate_local_minilm_snapshot(value: str) -> None:
    try:
        snapshot = Path(value)
        if (
            not snapshot.is_absolute()
            or ".." in snapshot.parts
            or os.path.normpath(value) != value
            or not snapshot.exists()
            or not snapshot.is_dir()
            or snapshot.is_symlink()
        ):
            raise ValueError
        artifact_root = _APPROVED_ARTIFACT_ROOT
        if (
            not artifact_root.exists()
            or not artifact_root.is_dir()
            or artifact_root.is_symlink()
        ):
            raise ValueError
        resolved_root = artifact_root.resolve(strict=True)
        resolved_snapshot = snapshot.resolve(strict=True)
        resolved_snapshot.relative_to(resolved_root)
        if resolved_snapshot.name != MINILM_REVISION:
            raise ValueError
        manifest = artifact_root / _MINILM_MANIFEST
        if not manifest.is_file() or manifest.is_symlink():
            raise ValueError
        manifest.resolve(strict=True).relative_to(resolved_root)
        expected = _read_manifest(manifest)
        if set(expected) != _MINILM_REQUIRED_FILES:
            raise ValueError
        actual = {
            path.relative_to(snapshot).as_posix()
            for path in snapshot.rglob("*")
            if path.is_file()
        }
        if actual != _MINILM_REQUIRED_FILES:
            raise ValueError
        for relative_path, (expected_digest, expected_size) in expected.items():
            candidate = snapshot / Path(relative_path)
            if not candidate.is_file() or candidate.stat().st_size != expected_size:
                raise ValueError
            candidate.resolve(strict=True).relative_to(resolved_root)
            if _sha256_file(candidate) != expected_digest:
                raise ValueError
    except (OSError, UnicodeError, ValueError):
        raise ValueError(_LOCAL_SNAPSHOT_ERROR) from None


def _read_manifest(path: Path) -> dict[str, tuple[str, int]]:
    entries: dict[str, tuple[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ")
        if len(parts) != 3:
            raise ValueError
        digest, relative_path, raw_size = parts
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not raw_size.isascii()
            or not raw_size.isdigit()
            or relative_path in entries
        ):
            raise ValueError
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError
        entries[relative_path] = (digest, int(raw_size))
    return entries


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
