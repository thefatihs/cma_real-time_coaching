"""Deterministic tests for production-safe PostgreSQL RAG composition."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest
from psycopg import Connection
from pydantic import SecretStr, ValidationError

import app.composition as composition_exports
from app.composition import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLRAGComposition,
    PostgreSQLVectorStoreSettings,
    compose_profile_bound_postgres_rag,
)
from app.embeddings.sentence_transformers import SentenceTransformerQueryEmbedder
from app.vector_store.embedding_profile import EmbeddingDistanceMetric
from app.vector_store.postgres.connection_factory import (
    PgvectorPsycopgConnectionFactory,
)
from app.vector_store.postgres.runner import (
    PsycopgPostgreSQLVectorTransactionRunner,
)

DSN = "postgresql://synthetic_user:synthetic_password@db.invalid/synthetic"
POSTGRES_ENVIRONMENT = {
    "CALLMETRIC_POSTGRES_DSN": DSN,
    "CALLMETRIC_POSTGRES_CONNECT_TIMEOUT_SECONDS": "5",
    "CALLMETRIC_POSTGRES_SSL_MODE": "verify-full",
    "CALLMETRIC_POSTGRES_APPLICATION_NAME": "callmetric-rag",
}


class FalseyConnect:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[dict[str, object]] = []

    def __bool__(self) -> bool:
        return False

    def __call__(self, **kwargs: object) -> Connection[Any]:
        self.calls.append(kwargs)
        raise self.error


class BackendFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, config: object) -> object:
        del config
        self.calls += 1
        raise AssertionError("backend factory must remain deferred")


def _postgres_settings() -> PostgreSQLVectorStoreSettings:
    return PostgreSQLVectorStoreSettings(
        dsn=SecretStr(DSN),
        connect_timeout_seconds=5,
        ssl_mode="verify-full",
        application_name="callmetric-rag",
    )


def _knowledge_base_settings() -> KnowledgeBaseRAGProviderSettings:
    return KnowledgeBaseRAGProviderSettings(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        model_id="model-synthetic-v1",
        model_name_or_path="local/model-synthetic-v1",
        vector_dimension=3,
        normalize_embeddings=True,
        device="cpu",
        local_files_only=True,
    )


def _compose(
    *,
    connect: Callable[..., Connection[Any]] | None = None,
    backend_factory: Callable[..., object] | None = None,
) -> PostgreSQLRAGComposition:
    connection_error = RuntimeError("connection must remain deferred")
    selected_connect = FalseyConnect(connection_error) if connect is None else connect
    return compose_profile_bound_postgres_rag(
        postgres_settings=_postgres_settings(),
        knowledge_base_settings=_knowledge_base_settings(),
        psycopg_connect=selected_connect,
        embedding_backend_factory=cast(Any, backend_factory),
    )


def test_postgres_settings_load_exact_environment_and_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in POSTGRES_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    settings_factory = cast(
        Callable[[], PostgreSQLVectorStoreSettings],
        PostgreSQLVectorStoreSettings,
    )
    settings = settings_factory()

    assert settings.dsn.get_secret_value() == DSN
    assert settings.connect_timeout_seconds == 5
    assert settings.ssl_mode == "verify-full"
    assert settings.application_name == "callmetric-rag"
    with pytest.raises(ValidationError):
        settings.connect_timeout_seconds = 6


def test_postgres_settings_do_not_load_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in POSTGRES_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(f"{name}={value}" for name, value in POSTGRES_ENVIRONMENT.items()),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError):
        cast(
            Callable[[], PostgreSQLVectorStoreSettings],
            PostgreSQLVectorStoreSettings,
        )()


@pytest.mark.parametrize(
    "missing",
    [
        "dsn",
        "connect_timeout_seconds",
        "ssl_mode",
        "application_name",
    ],
)
def test_all_postgres_settings_are_required(missing: str) -> None:
    values: dict[str, object] = {
        "dsn": DSN,
        "connect_timeout_seconds": 5,
        "ssl_mode": "require",
        "application_name": "callmetric-rag",
    }
    del values[missing]

    with pytest.raises(ValidationError):
        PostgreSQLVectorStoreSettings.model_validate(values)


def test_dsn_is_absent_from_repr_dumps_and_validation_errors() -> None:
    settings = _postgres_settings()

    representations = (
        repr(settings),
        str(settings),
        repr(settings.model_dump()),
        settings.model_dump_json(),
    )
    assert all(DSN not in representation for representation in representations)
    with pytest.raises(ValidationError) as raised:
        PostgreSQLVectorStoreSettings.model_validate(
            {
                "dsn": DSN,
                "connect_timeout_seconds": 0,
                "ssl_mode": "require",
                "application_name": "callmetric-rag",
            }
        )
    assert DSN not in str(raised.value)


@pytest.mark.parametrize(
    "value",
    [0, 61, True, 1.5, "1.5", " 5"],
)
def test_connect_timeout_is_strict_and_bounded(value: object) -> None:
    with pytest.raises(ValidationError):
        PostgreSQLVectorStoreSettings.model_validate(
            {
                "dsn": DSN,
                "connect_timeout_seconds": value,
                "ssl_mode": "require",
                "application_name": "callmetric-rag",
            }
        )


@pytest.mark.parametrize("ssl_mode", ["disable", "prefer", "allow", ""])
def test_only_secure_ssl_modes_are_accepted(ssl_mode: str) -> None:
    with pytest.raises(ValidationError):
        PostgreSQLVectorStoreSettings.model_validate(
            {
                "dsn": DSN,
                "connect_timeout_seconds": 5,
                "ssl_mode": ssl_mode,
                "application_name": "callmetric-rag",
            }
        )


@pytest.mark.parametrize(
    "application_name",
    ["", " ", "callmetric rag", "callmetric/rag", "-callmetric"],
)
def test_application_name_must_be_safe(application_name: str) -> None:
    with pytest.raises(ValidationError):
        PostgreSQLVectorStoreSettings.model_validate(
            {
                "dsn": DSN,
                "connect_timeout_seconds": 5,
                "ssl_mode": "require",
                "application_name": application_name,
            }
        )


def test_knowledge_base_settings_are_explicit_normalized_and_frozen() -> None:
    settings = KnowledgeBaseRAGProviderSettings(
        tenant_id=" tenant-synthetic ",
        knowledge_base_id=" kb-synthetic ",
        model_id=" model-synthetic-v1 ",
        model_name_or_path=" local/model-synthetic-v1 ",
        vector_dimension=3,
        normalize_embeddings=True,
        device="cuda",
        local_files_only=True,
    )

    assert settings.tenant_id == "tenant-synthetic"
    assert settings.knowledge_base_id == "kb-synthetic"
    assert settings.model_id == "model-synthetic-v1"
    assert settings.model_name_or_path == "local/model-synthetic-v1"
    assert settings.device == "cuda"
    with pytest.raises(ValidationError):
        settings.vector_dimension = 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", " "),
        ("knowledge_base_id", ""),
        ("model_id", "\n"),
        ("model_name_or_path", " "),
        ("vector_dimension", 0),
        ("vector_dimension", True),
        ("normalize_embeddings", 1),
        ("device", "mps"),
        ("local_files_only", False),
        ("local_files_only", 1),
    ],
)
def test_knowledge_base_settings_reject_invalid_values(
    field: str,
    value: object,
) -> None:
    values = _knowledge_base_settings().model_dump()
    values[field] = value

    with pytest.raises(ValidationError):
        KnowledgeBaseRAGProviderSettings(**values)


def test_composition_has_zero_io_and_preserves_falsey_callables() -> None:
    connection_error = RuntimeError("synthetic connection failure")
    connect = FalseyConnect(connection_error)
    backend_factory = BackendFactory()
    sentence_transformers_before = "sentence_transformers" in sys.modules
    torch_before = "torch" in sys.modules

    composition = _compose(
        connect=connect,
        backend_factory=backend_factory,
    )

    assert isinstance(composition, PostgreSQLRAGComposition)
    assert connect.calls == []
    assert backend_factory.calls == 0
    assert ("sentence_transformers" in sys.modules) is sentence_transformers_before
    assert ("torch" in sys.modules) is torch_before
    runner = cast(
        PsycopgPostgreSQLVectorTransactionRunner,
        composition.profile_repository._transaction_runner,  # noqa: SLF001
    )
    connection_factory = cast(
        PgvectorPsycopgConnectionFactory,
        runner._connection_factory,  # noqa: SLF001
    )
    assert connection_factory._base_connection_factory.__closure__ is not None  # noqa: SLF001


def test_composition_shares_runner_embedder_and_store_collaborators() -> None:
    composition = _compose()
    repository_runner = composition.profile_repository._transaction_runner  # noqa: SLF001
    store_runner = composition.vector_store._transaction_runner  # noqa: SLF001

    assert repository_runner is store_runner
    assert isinstance(composition.embedder, SentenceTransformerQueryEmbedder)
    assert (
        composition.ingestion_service._document_embedder  # noqa: SLF001
        is composition.embedder
    )
    assert (
        composition.ingestion_service._batch_writer  # noqa: SLF001
        is composition.vector_store
    )
    assert composition.retriever._query_embedder is composition.embedder  # noqa: SLF001
    assert composition.retriever._vector_store is composition.vector_store  # noqa: SLF001
    assert callable(composition.embedder.embed_query)
    assert callable(composition.embedder.embed_documents)
    assert not hasattr(composition, "query_embedder")
    assert not hasattr(composition, "document_embedder")


def test_composition_builds_exact_cosine_profile_without_registration() -> None:
    composition = _compose()

    assert composition.profile.tenant_id == "tenant-synthetic"
    assert composition.profile.knowledge_base_id == "kb-synthetic"
    assert composition.profile.model_id == "model-synthetic-v1"
    assert composition.profile.vector_dimension == 3
    assert composition.profile.normalize_embeddings is True
    assert composition.profile.distance_metric is EmbeddingDistanceMetric.COSINE
    assert composition.vector_store._expected_profile is composition.profile  # noqa: SLF001


def test_deferred_connect_receives_exact_secret_kwargs_and_preserves_exception() -> (
    None
):
    expected = RuntimeError("synthetic provider failure")
    connect = FalseyConnect(expected)
    composition = _compose(connect=connect)
    runner = cast(
        PsycopgPostgreSQLVectorTransactionRunner,
        composition.profile_repository._transaction_runner,  # noqa: SLF001
    )
    connection_factory = cast(
        PgvectorPsycopgConnectionFactory,
        runner._connection_factory,  # noqa: SLF001
    )

    with pytest.raises(RuntimeError) as raised:
        connection_factory()

    assert raised.value is expected
    assert connect.calls == [
        {
            "conninfo": DSN,
            "connect_timeout": 5,
            "sslmode": "verify-full",
            "application_name": "callmetric-rag",
            "autocommit": False,
        }
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("postgres_settings", object()),
        ("knowledge_base_settings", object()),
        ("psycopg_connect", object()),
        ("embedding_backend_factory", object()),
    ],
)
def test_composition_rejects_invalid_dependencies_before_side_effects(
    field: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {
        "postgres_settings": _postgres_settings(),
        "knowledge_base_settings": _knowledge_base_settings(),
        "psycopg_connect": FalseyConnect(RuntimeError("must remain deferred")),
        "embedding_backend_factory": BackendFactory(),
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        compose_profile_bound_postgres_rag(**cast(Any, arguments))


def test_composition_is_frozen_and_slotted() -> None:
    composition = _compose()

    assert not hasattr(composition, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(composition, "profile", composition.profile)


def test_public_exports_are_exact() -> None:
    assert composition_exports.__all__ == [
        "KnowledgeBaseRAGProviderSettings",
        "LLMGatewayFactory",
        "ProfileVerifiedPostgreSQLRAGRunner",
        "PostgreSQLRAGOrchestrationComposition",
        "PostgreSQLRAGComposition",
        "PostgreSQLVectorStoreSettings",
        "compose_profile_bound_postgres_rag",
        "compose_profile_bound_postgres_rag_orchestration",
    ]
    assert (
        composition_exports.PostgreSQLVectorStoreSettings
        is PostgreSQLVectorStoreSettings
    )
