"""Side-effect-free PostgreSQL document-ingestion composition tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import SecretStr, ValidationError

import app.composition.postgres_document_ingestion as subject
from app.composition.postgres_document_ingestion import (
    MINILM_MODEL,
    PostgreSQLDocumentIngestionSettings,
    compose_postgres_document_ingestion,
)
from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
)


def _postgres() -> PostgreSQLVectorStoreSettings:
    return PostgreSQLVectorStoreSettings(
        dsn=SecretStr("postgresql://synthetic.invalid/db"),
        connect_timeout_seconds=3,
        ssl_mode="require",
        application_name="document-tests",
    )


def _provider(**updates: object) -> KnowledgeBaseRAGProviderSettings:
    values: dict[str, object] = {
        "tenant_id": "tenant-trusted",
        "knowledge_base_id": "kb-trusted",
        "model_id": MINILM_MODEL,
        "model_name_or_path": MINILM_MODEL,
        "vector_dimension": 384,
        "normalize_embeddings": True,
        "device": "cpu",
        "local_files_only": True,
    }
    values.update(updates)
    return KnowledgeBaseRAGProviderSettings(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "other"),
        ("model_name_or_path", "other"),
        ("vector_dimension", 768),
        ("normalize_embeddings", False),
        ("device", "cuda"),
    ],
)
def test_composition_rejects_non_minilm_profile(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="incompatible"):
        compose_postgres_document_ingestion(
            postgres_settings=_postgres(),
            knowledge_base_settings=_provider(**{field: value}),
            ingestion_settings=PostgreSQLDocumentIngestionSettings(),
            psycopg_connect=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("connection forbidden")
            ),
        )


def test_settings_are_strict_and_bounded() -> None:
    with pytest.raises(ValidationError):
        PostgreSQLDocumentIngestionSettings(max_workers=2)
    with pytest.raises(ValidationError):
        PostgreSQLDocumentIngestionSettings(capacity=9)


def test_composition_does_not_connect_or_load_model_before_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    fake_rag = SimpleNamespace(embedder=object(), vector_store=object())
    monkeypatch.setattr(
        subject,
        "compose_profile_bound_postgres_rag",
        lambda **kwargs: fake_rag,
    )
    fake_manager = SimpleNamespace(close=lambda **kwargs: None)
    monkeypatch.setattr(
        subject,
        "BoundedDocumentIngestionManager",
        lambda **kwargs: fake_manager,
    )

    runtime = compose_postgres_document_ingestion(
        postgres_settings=_postgres(),
        knowledge_base_settings=_provider(),
        ingestion_settings=PostgreSQLDocumentIngestionSettings(capacity=3),
        psycopg_connect=lambda **kwargs: events.append("connect"),  # type: ignore[arg-type]
        embedding_backend_factory=lambda config: events.append("model"),  # type: ignore[arg-type]
    )
    assert runtime.manager is fake_manager
    assert events == []


def test_runtime_exposes_no_persistent_storage_or_reconciliation() -> None:
    fields = PostgreSQLDocumentIngestionSettings.model_fields
    assert set(fields) == {"max_workers", "capacity"}
    assert not hasattr(subject.PostgreSQLDocumentIngestionRuntime, "reconcile_orphans")
