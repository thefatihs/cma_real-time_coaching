"""Side-effect-free PostgreSQL document-ingestion composition tests."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
from pydantic import SecretStr, ValidationError

import app.composition.postgres_document_ingestion as subject
import app.vector_store.postgres.connection_factory as connection_factory_module
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


class _Connection:
    autocommit = False

    def __init__(self, events: list[str], *, close_error: Exception | None = None):
        self._events = events
        self._close_error = close_error

    def close(self) -> None:
        self._events.append("close")
        if self._close_error is not None:
            raise self._close_error


def _capture_registry_connection_factory(
    monkeypatch: pytest.MonkeyPatch,
    *,
    psycopg_connect: Callable[..., object],
) -> tuple[Callable[[], object], dict[str, object], object, object]:
    captured: dict[str, object] = {}
    registry = object()
    vector_store = object()
    fake_rag = SimpleNamespace(
        embedder=object(),
        vector_store=vector_store,
        profile_repository=object(),
        profile=object(),
    )

    def compose(**kwargs: object) -> object:
        captured["rag_connect"] = kwargs["psycopg_connect"]
        return fake_rag

    def registry_factory(*, connection_factory: Callable[[], object]) -> object:
        captured["connection_factory"] = connection_factory
        return registry

    def manager_factory(**kwargs: object) -> object:
        captured["manager_kwargs"] = kwargs
        return SimpleNamespace(close=lambda **close_kwargs: None)

    monkeypatch.setattr(subject, "compose_profile_bound_postgres_rag", compose)
    monkeypatch.setattr(subject, "PsycopgDocumentRegistryRepository", registry_factory)
    monkeypatch.setattr(subject, "BoundedDocumentIngestionManager", manager_factory)
    runtime = compose_postgres_document_ingestion(
        postgres_settings=_postgres(),
        knowledge_base_settings=_provider(),
        ingestion_settings=PostgreSQLDocumentIngestionSettings(capacity=2),
        psycopg_connect=psycopg_connect,  # type: ignore[arg-type]
    )
    assert runtime.registry is registry
    assert captured["rag_connect"] is psycopg_connect
    manager_kwargs = captured["manager_kwargs"]
    assert isinstance(manager_kwargs, dict)
    assert manager_kwargs["registry"] is registry
    assert manager_kwargs["vector_writer"] is vector_store
    connection_factory = captured["connection_factory"]
    assert callable(connection_factory)
    return connection_factory, captured, registry, vector_store


def test_registry_connections_are_fresh_and_pgvector_registered_before_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    connections: list[_Connection] = []

    def connect(**kwargs: object) -> _Connection:
        assert kwargs == {
            "conninfo": "postgresql://synthetic.invalid/db",
            "connect_timeout": 3,
            "sslmode": "require",
            "application_name": "document-tests",
            "autocommit": False,
        }
        events.append("connect")
        connection = _Connection(events)
        connections.append(connection)
        return connection

    def register(connection: object) -> None:
        assert connection is connections[-1]
        events.append("register")

    monkeypatch.setattr(connection_factory_module, "register_vector", register)
    connection_factory, _, _, _ = _capture_registry_connection_factory(
        monkeypatch,
        psycopg_connect=connect,
    )
    assert events == []

    first = connection_factory()
    events.append("vector-sql")
    second = connection_factory()
    events.append("vector-sql")

    assert first is connections[0]
    assert second is connections[1]
    assert first is not second
    assert events == [
        "connect",
        "register",
        "vector-sql",
        "connect",
        "register",
        "vector-sql",
    ]


def test_pgvector_registration_failure_closes_without_sql_or_error_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    registration_error = RuntimeError("synthetic registration failure")
    close_error = RuntimeError("synthetic close failure")

    def connect(**kwargs: object) -> _Connection:
        events.append("connect")
        return _Connection(events, close_error=close_error)

    def register(connection: object) -> None:
        events.append("register")
        raise registration_error

    monkeypatch.setattr(connection_factory_module, "register_vector", register)
    connection_factory, _, _, _ = _capture_registry_connection_factory(
        monkeypatch,
        psycopg_connect=connect,
    )

    with pytest.raises(RuntimeError) as caught:
        connection_factory()

    assert caught.value is registration_error
    assert isinstance(caught.value.__cause__, ExceptionGroup)
    assert events == ["connect", "register", "close"]


def test_runtime_exposes_no_persistent_storage_or_reconciliation() -> None:
    fields = PostgreSQLDocumentIngestionSettings.model_fields
    assert set(fields) == {"max_workers", "capacity"}
    assert not hasattr(subject.PostgreSQLDocumentIngestionRuntime, "reconcile_orphans")
