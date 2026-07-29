"""Deterministic tests for explicit PostgreSQL RAG chunk ingestion."""

from __future__ import annotations

from inspect import signature
from types import SimpleNamespace
from typing import Any, cast

import pytest
from psycopg import Connection
from pydantic import SecretStr

import app.deployment as deployment_exports
import app.deployment.postgres_ingestion as deployment
from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
)
from app.embeddings.sentence_transformers import BackendFactory
from app.ingestion.models import DocumentChunkInput, DocumentIngestionRequest
from app.vector_store.embedding_profile import (
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
)
from app.vector_store.models import VectorBatchWriteResult, VectorRecordIdentity

_DSN = "postgresql://synthetic_user:synthetic_password@db.invalid/synthetic"


def _postgres_settings() -> PostgreSQLVectorStoreSettings:
    return PostgreSQLVectorStoreSettings(
        dsn=SecretStr(_DSN),
        connect_timeout_seconds=7,
        ssl_mode="verify-full",
        application_name="callmetric-ingestion",
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


def _request() -> DocumentIngestionRequest:
    return DocumentIngestionRequest(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        chunks=(
            DocumentChunkInput(
                document_id="document-a",
                chunk_id="chunk-1",
                text="Synthetic support guidance.",
                metadata=(("category", "synthetic"),),
            ),
        ),
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


def _result() -> VectorBatchWriteResult:
    return VectorBatchWriteResult(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        inserted_identities=(
            VectorRecordIdentity(document_id="document-a", chunk_id="chunk-1"),
        ),
        unchanged_identities=(),
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
    def __init__(self, calls: list[str] | None = None) -> None:
        self.calls = calls
        self.kwargs: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> Connection[Any]:
        if self.calls is not None:
            self.calls.append("readiness-connection")
        self.kwargs.append(kwargs)
        return cast(Connection[Any], object())


class FakeRepository:
    def __init__(
        self,
        returned: object,
        *,
        calls: list[str],
        error: BaseException | None = None,
    ) -> None:
        self.returned = returned
        self.calls = calls
        self.error = error
        self.lookup_calls: list[tuple[str, str]] = []

    def get_profile(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> object:
        self.calls.append("profile")
        self.lookup_calls.append((tenant_id, knowledge_base_id))
        if self.error is not None:
            raise self.error
        return self.returned


class FakeIngestionService:
    def __init__(
        self,
        result: VectorBatchWriteResult,
        *,
        calls: list[str],
        error: BaseException | None = None,
    ) -> None:
        self.result = result
        self.calls = calls
        self.error = error
        self.requests: list[DocumentIngestionRequest] = []

    def ingest(self, request: DocumentIngestionRequest) -> VectorBatchWriteResult:
        self.calls.append("ingestion")
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    repository: FakeRepository,
    ingestion_service: FakeIngestionService,
    calls: list[str],
    readiness_error: BaseException | None = None,
) -> list[dict[str, object]]:
    composition_arguments: list[dict[str, object]] = []

    def compose(**kwargs: object) -> object:
        calls.append("compose")
        composition_arguments.append(kwargs)
        return SimpleNamespace(
            profile=_profile(),
            profile_repository=repository,
            ingestion_service=ingestion_service,
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


def test_public_signature_and_export_are_exact() -> None:
    parameters = signature(deployment.ingest_profile_bound_postgres_rag).parameters

    assert tuple(parameters) == (
        "postgres_settings",
        "knowledge_base_settings",
        "request",
        "psycopg_connect",
        "embedding_backend_factory",
    )
    assert all(
        parameter.kind.name == "KEYWORD_ONLY" for parameter in parameters.values()
    )
    assert deployment_exports.ingest_profile_bound_postgres_rag is (
        deployment.ingest_profile_bound_postgres_rag
    )
    assert deployment_exports.__all__.count("ingest_profile_bound_postgres_rag") == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("postgres_settings", object()),
        ("knowledge_base_settings", object()),
        ("request", object()),
        ("psycopg_connect", object()),
        ("embedding_backend_factory", object()),
    ],
)
def test_invalid_arguments_stop_before_composition_and_io(
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
        "request": _request(),
        "psycopg_connect": FakeConnect(),
        "embedding_backend_factory": None,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        deployment.ingest_profile_bound_postgres_rag(**cast(Any, arguments))

    assert compose_calls == 0


@pytest.mark.parametrize(
    "ingestion_request",
    [
        _request().model_copy(update={"tenant_id": "tenant-other"}),
        _request().model_copy(update={"knowledge_base_id": "kb-other"}),
    ],
)
def test_scope_mismatch_stops_before_composition_and_connection(
    monkeypatch: pytest.MonkeyPatch,
    ingestion_request: DocumentIngestionRequest,
) -> None:
    monkeypatch.setattr(
        deployment,
        "compose_profile_bound_postgres_rag",
        lambda **kwargs: pytest.fail(f"unexpected composition: {tuple(kwargs)}"),
    )
    connect = FakeConnect()

    with pytest.raises(ValueError):
        deployment.ingest_profile_bound_postgres_rag(
            postgres_settings=_postgres_settings(),
            knowledge_base_settings=_provider_settings(),
            request=ingestion_request,
            psycopg_connect=connect,
        )

    assert connect.kwargs == []


def test_exact_lifecycle_scope_result_and_lazy_backend_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    result = _result()
    repository = FakeRepository(_profile(), calls=calls)
    ingestion_service = FakeIngestionService(result, calls=calls)
    composition_arguments = _install_fakes(
        monkeypatch,
        repository=repository,
        ingestion_service=ingestion_service,
        calls=calls,
    )
    connect = FakeConnect(calls)
    backend = FalseyBackendFactory()
    request = _request()

    returned = deployment.ingest_profile_bound_postgres_rag(
        postgres_settings=_postgres_settings(),
        knowledge_base_settings=_provider_settings(),
        request=request,
        psycopg_connect=connect,
        embedding_backend_factory=cast(BackendFactory, backend),
    )

    assert returned is result
    assert calls == [
        "compose",
        "checker",
        "readiness",
        "readiness-connection",
        "profile",
        "ingestion",
    ]
    assert repository.lookup_calls == [("tenant-synthetic", "kb-synthetic")]
    assert ingestion_service.requests == [request]
    assert composition_arguments[0]["embedding_backend_factory"] is backend
    assert backend.calls == 0
    assert connect.kwargs == [
        {
            "conninfo": _DSN,
            "connect_timeout": 7,
            "sslmode": "verify-full",
            "application_name": "callmetric-ingestion",
            "autocommit": False,
        }
    ]


@pytest.mark.parametrize(
    "stored_profile",
    [
        None,
        object(),
        _profile().model_copy(update={"model_id": "different-model"}),
    ],
    ids=["missing", "malformed", "conflicting"],
)
def test_profile_failure_stops_before_ingestion(
    monkeypatch: pytest.MonkeyPatch,
    stored_profile: object,
) -> None:
    calls: list[str] = []
    repository = FakeRepository(stored_profile, calls=calls)
    ingestion_service = FakeIngestionService(_result(), calls=calls)
    _install_fakes(
        monkeypatch,
        repository=repository,
        ingestion_service=ingestion_service,
        calls=calls,
    )

    with pytest.raises(ValueError):
        deployment.ingest_profile_bound_postgres_rag(
            postgres_settings=_postgres_settings(),
            knowledge_base_settings=_provider_settings(),
            request=_request(),
            psycopg_connect=FakeConnect(calls),
        )

    assert calls[-1] == "profile"
    assert ingestion_service.requests == []


@pytest.mark.parametrize("stage", ["readiness", "profile", "ingestion"])
def test_provider_and_backend_exception_identity_propagates(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    expected = RuntimeError("synthetic operational failure")
    calls: list[str] = []
    repository = FakeRepository(
        _profile(),
        calls=calls,
        error=expected if stage == "profile" else None,
    )
    ingestion_service = FakeIngestionService(
        _result(),
        calls=calls,
        error=expected if stage == "ingestion" else None,
    )
    _install_fakes(
        monkeypatch,
        repository=repository,
        ingestion_service=ingestion_service,
        calls=calls,
        readiness_error=expected if stage == "readiness" else None,
    )

    with pytest.raises(RuntimeError) as raised:
        deployment.ingest_profile_bound_postgres_rag(
            postgres_settings=_postgres_settings(),
            knowledge_base_settings=_provider_settings(),
            request=_request(),
            psycopg_connect=FakeConnect(calls),
        )

    assert raised.value is expected
    if stage != "ingestion":
        assert ingestion_service.requests == []
