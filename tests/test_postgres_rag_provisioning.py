"""Deterministic tests for explicit PostgreSQL RAG profile provisioning."""

from __future__ import annotations

from inspect import signature
from types import SimpleNamespace
from typing import Any, cast

import pytest
from psycopg import Connection
from pydantic import SecretStr

import app.deployment as deployment_exports
import app.deployment.postgres_rag as deployment
from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
)
from app.embeddings.sentence_transformers import BackendFactory
from app.vector_store.embedding_profile import (
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
)

_DSN = "postgresql://synthetic_user:synthetic_password@db.invalid/synthetic"


def _postgres_settings() -> PostgreSQLVectorStoreSettings:
    return PostgreSQLVectorStoreSettings(
        dsn=SecretStr(_DSN),
        connect_timeout_seconds=7,
        ssl_mode="verify-full",
        application_name="callmetric-provision",
    )


def _provider_settings() -> KnowledgeBaseRAGProviderSettings:
    return KnowledgeBaseRAGProviderSettings(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        model_id="model-synthetic",
        model_name_or_path="local/synthetic-model",
        vector_dimension=3,
        normalize_embeddings=True,
        device="cpu",
        local_files_only=True,
    )


def _profile() -> KnowledgeBaseEmbeddingProfile:
    return KnowledgeBaseEmbeddingProfile(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        model_id="model-synthetic",
        vector_dimension=3,
        normalize_embeddings=True,
        distance_metric=EmbeddingDistanceMetric.COSINE,
    )


class FalseyBackendFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __bool__(self) -> bool:
        return False

    def __call__(self, _config: object) -> object:
        self.calls += 1
        raise AssertionError("embedding backend must remain deferred")


class FakeConnect:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> Connection[Any]:
        self.calls.append(kwargs)
        return cast(Connection[Any], object())


class FakeRepository:
    def __init__(
        self,
        *,
        profile: KnowledgeBaseEmbeddingProfile,
        connect: FakeConnect | None = None,
        result: object | None = None,
        error: BaseException | None = None,
        calls: list[str] | None = None,
    ) -> None:
        self.profile = profile
        self.connect = connect
        self.result = profile if result is None else result
        self.error = error
        self.calls = calls if calls is not None else []
        self.register_calls: list[KnowledgeBaseEmbeddingProfile] = []

    def register_profile(
        self,
        profile: KnowledgeBaseEmbeddingProfile,
    ) -> KnowledgeBaseEmbeddingProfile:
        self.calls.append("register")
        self.register_calls.append(profile)
        if self.connect is not None:
            self.connect(
                conninfo=_DSN,
                connect_timeout=7,
                sslmode="verify-full",
                application_name="callmetric-provision",
                autocommit=False,
            )
        if self.error is not None:
            raise self.error
        return cast(KnowledgeBaseEmbeddingProfile, self.result)


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    repository: FakeRepository,
    calls: list[str],
    readiness_error: BaseException | None = None,
) -> list[dict[str, object]]:
    composition_arguments: list[dict[str, object]] = []

    def compose(**kwargs: object) -> object:
        calls.append("compose")
        composition_arguments.append(kwargs)
        return SimpleNamespace(
            profile=repository.profile,
            profile_repository=repository,
        )

    class FakeReadinessChecker:
        def __init__(self, *, connection_factory: Any) -> None:
            calls.append("checker")
            self.connection_factory = connection_factory

        def verify(self) -> None:
            calls.append("readiness")
            self.connection_factory()
            if readiness_error is not None:
                raise readiness_error

    monkeypatch.setattr(deployment, "compose_profile_bound_postgres_rag", compose)
    monkeypatch.setattr(
        deployment,
        "PostgreSQLSchemaReadinessChecker",
        FakeReadinessChecker,
    )
    return composition_arguments


def test_public_signature_and_exports_are_exact() -> None:
    parameters = signature(deployment.provision_profile_bound_postgres_rag).parameters

    assert tuple(parameters) == (
        "postgres_settings",
        "knowledge_base_settings",
        "psycopg_connect",
        "embedding_backend_factory",
    )
    assert all(
        parameter.kind.name == "KEYWORD_ONLY" for parameter in parameters.values()
    )
    assert deployment_exports.__all__ == [
        "PostgreSQLMigrationResult",
        "PostgreSQLMigrationSettings",
        "PostgreSQLRAGRetrievalRequest",
        "apply_postgres_vector_migrations",
        "ingest_profile_bound_postgres_rag",
        "provision_profile_bound_postgres_rag",
        "retrieve_profile_bound_postgres_rag",
    ]
    assert (
        deployment_exports.provision_profile_bound_postgres_rag
        is deployment.provision_profile_bound_postgres_rag
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("postgres_settings", object()),
        ("knowledge_base_settings", object()),
        ("psycopg_connect", object()),
        ("embedding_backend_factory", object()),
    ],
)
def test_argument_validation_precedes_composition_and_io(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    compose_calls = 0

    def compose(**_kwargs: object) -> object:
        nonlocal compose_calls
        compose_calls += 1
        raise AssertionError("composition must not run")

    monkeypatch.setattr(deployment, "compose_profile_bound_postgres_rag", compose)
    arguments: dict[str, object] = {
        "postgres_settings": _postgres_settings(),
        "knowledge_base_settings": _provider_settings(),
        "psycopg_connect": FakeConnect(),
        "embedding_backend_factory": None,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        deployment.provision_profile_bound_postgres_rag(**cast(Any, arguments))

    assert compose_calls == 0


def test_exact_order_kwargs_profile_and_falsey_backend_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    connect = FakeConnect()
    profile = _profile()
    repository = FakeRepository(profile=profile, connect=connect, calls=calls)
    composition_arguments = _install_fakes(
        monkeypatch,
        repository=repository,
        calls=calls,
    )
    backend = FalseyBackendFactory()

    result = deployment.provision_profile_bound_postgres_rag(
        postgres_settings=_postgres_settings(),
        knowledge_base_settings=_provider_settings(),
        psycopg_connect=connect,
        embedding_backend_factory=cast(BackendFactory, backend),
    )

    assert result is profile
    assert calls == ["compose", "checker", "readiness", "register"]
    assert repository.register_calls == [profile]
    assert connect.calls == [
        {
            "conninfo": _DSN,
            "connect_timeout": 7,
            "sslmode": "verify-full",
            "application_name": "callmetric-provision",
            "autocommit": False,
        },
        {
            "conninfo": _DSN,
            "connect_timeout": 7,
            "sslmode": "verify-full",
            "application_name": "callmetric-provision",
            "autocommit": False,
        },
    ]
    assert composition_arguments[0]["embedding_backend_factory"] is backend
    assert backend.calls == 0


def test_none_backend_is_forwarded_without_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    repository = FakeRepository(profile=_profile(), calls=calls)
    arguments = _install_fakes(monkeypatch, repository=repository, calls=calls)

    deployment.provision_profile_bound_postgres_rag(
        postgres_settings=_postgres_settings(),
        knowledge_base_settings=_provider_settings(),
        psycopg_connect=FakeConnect(),
    )

    assert arguments[0]["embedding_backend_factory"] is None


def test_readiness_failure_prevents_registration_and_preserves_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = RuntimeError("synthetic readiness failure")
    calls: list[str] = []
    repository = FakeRepository(profile=_profile(), calls=calls)
    _install_fakes(
        monkeypatch,
        repository=repository,
        calls=calls,
        readiness_error=expected,
    )

    with pytest.raises(RuntimeError) as raised:
        deployment.provision_profile_bound_postgres_rag(
            postgres_settings=_postgres_settings(),
            knowledge_base_settings=_provider_settings(),
            psycopg_connect=FakeConnect(),
        )

    assert raised.value is expected
    assert calls == ["compose", "checker", "readiness"]
    assert repository.register_calls == []


def test_profile_conflict_propagates_without_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ValueError("synthetic profile conflict")
    calls: list[str] = []
    repository = FakeRepository(profile=_profile(), error=expected, calls=calls)
    _install_fakes(monkeypatch, repository=repository, calls=calls)

    with pytest.raises(ValueError) as raised:
        deployment.provision_profile_bound_postgres_rag(
            postgres_settings=_postgres_settings(),
            knowledge_base_settings=_provider_settings(),
            psycopg_connect=FakeConnect(),
        )

    assert raised.value is expected
    assert repository.register_calls == [repository.profile]


@pytest.mark.parametrize(
    ("returned", "message"),
    [
        (object(), "invalid embedding profile"),
        (
            _profile().model_copy(update={"model_id": "different-model"}),
            "conflicting embedding profile",
        ),
    ],
)
def test_malformed_or_wrong_returned_profile_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    returned: object,
    message: str,
) -> None:
    calls: list[str] = []
    repository = FakeRepository(
        profile=_profile(),
        result=returned,
        calls=calls,
    )
    _install_fakes(monkeypatch, repository=repository, calls=calls)

    with pytest.raises(ValueError, match=message):
        deployment.provision_profile_bound_postgres_rag(
            postgres_settings=_postgres_settings(),
            knowledge_base_settings=_provider_settings(),
            psycopg_connect=FakeConnect(),
        )


def test_equal_canonical_repository_profile_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _profile()
    equal_stored = _profile()
    calls: list[str] = []
    repository = FakeRepository(
        profile=expected,
        result=equal_stored,
        calls=calls,
    )
    _install_fakes(monkeypatch, repository=repository, calls=calls)

    result = deployment.provision_profile_bound_postgres_rag(
        postgres_settings=_postgres_settings(),
        knowledge_base_settings=_provider_settings(),
        psycopg_connect=FakeConnect(),
    )

    assert result is equal_stored


def test_repeated_idempotent_provisioning_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    calls: list[str] = []
    repository = FakeRepository(profile=profile, calls=calls)
    _install_fakes(monkeypatch, repository=repository, calls=calls)
    arguments = {
        "postgres_settings": _postgres_settings(),
        "knowledge_base_settings": _provider_settings(),
        "psycopg_connect": FakeConnect(),
    }

    first = deployment.provision_profile_bound_postgres_rag(**arguments)
    second = deployment.provision_profile_bound_postgres_rag(**arguments)

    assert first is profile
    assert second is profile
    assert repository.register_calls == [profile, profile]
