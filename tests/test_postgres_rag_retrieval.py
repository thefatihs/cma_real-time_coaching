"""Deterministic tests for explicit profile-bound PostgreSQL RAG retrieval."""

from __future__ import annotations

from inspect import signature
from types import SimpleNamespace
from typing import Any, cast

import pytest
from psycopg import Connection
from pydantic import SecretStr, ValidationError

import app.deployment as deployment_exports
import app.deployment.postgres_retrieval as deployment
from app.composition.postgres_rag import (
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
)
from app.embeddings.sentence_transformers import BackendFactory
from app.retrieval.models import RetrievalDocument, RetrievalResult
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
        application_name="callmetric-retrieval",
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


def _request(**updates: object) -> deployment.PostgreSQLRAGRetrievalRequest:
    values: dict[str, object] = {
        "tenant_id": "tenant-synthetic",
        "knowledge_base_id": "kb-synthetic",
        "query": "Synthetic current question",
        "top_k": 2,
        "minimum_score": 0.25,
    }
    values.update(updates)
    return deployment.PostgreSQLRAGRetrievalRequest.model_validate(values)


def _profile() -> KnowledgeBaseEmbeddingProfile:
    return KnowledgeBaseEmbeddingProfile(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        model_id="model-synthetic",
        vector_dimension=3,
        normalize_embeddings=True,
        distance_metric=EmbeddingDistanceMetric.COSINE,
    )


def _result() -> RetrievalResult:
    return RetrievalResult(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
        documents=(
            RetrievalDocument(
                tenant_id="tenant-synthetic",
                knowledge_base_id="kb-synthetic",
                document_id="document-a",
                chunk_id="chunk-1",
                text="Synthetic guidance A",
                score=0.9,
            ),
            RetrievalDocument(
                tenant_id="tenant-synthetic",
                knowledge_base_id="kb-synthetic",
                document_id="document-b",
                chunk_id="chunk-2",
                text="Synthetic guidance B",
                score=0.7,
            ),
        ),
    )


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
        returned: object,
        calls: list[str],
        error: BaseException | None = None,
    ) -> None:
        self.returned = returned
        self.calls = calls
        self.error = error
        self.get_calls: list[dict[str, str]] = []

    def get_profile(self, *, tenant_id: str, knowledge_base_id: str) -> object:
        self.calls.append("profile")
        self.get_calls.append(
            {
                "tenant_id": tenant_id,
                "knowledge_base_id": knowledge_base_id,
            }
        )
        if self.error is not None:
            raise self.error
        return self.returned


class FakeRetriever:
    def __init__(
        self,
        *,
        returned: RetrievalResult,
        calls: list[str],
        error: BaseException | None = None,
        backend: FalseyBackendFactory | None = None,
    ) -> None:
        self.returned = returned
        self.calls = calls
        self.error = error
        self.backend = backend
        self.retrieve_calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs: object) -> RetrievalResult:
        self.calls.append("retrieve")
        self.retrieve_calls.append(kwargs)
        if self.backend is not None:
            self.backend.load()
        if self.error is not None:
            raise self.error
        return self.returned


class FalseyBackendFactory:
    def __init__(self, calls: list[str] | None = None) -> None:
        self.calls = 0
        self.events = calls if calls is not None else []

    def __bool__(self) -> bool:
        return False

    def __call__(self, _config: object) -> object:
        self.load()
        return object()

    def load(self) -> None:
        self.calls += 1
        self.events.append("backend")


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile_result: object,
    result: RetrievalResult | None = None,
    readiness_error: BaseException | None = None,
    profile_error: BaseException | None = None,
    retrieval_error: BaseException | None = None,
    backend: FalseyBackendFactory | None = None,
) -> tuple[list[str], FakeRepository, FakeRetriever, list[dict[str, object]]]:
    calls: list[str] = []
    repository = FakeRepository(
        returned=profile_result,
        calls=calls,
        error=profile_error,
    )
    retriever = FakeRetriever(
        returned=result if result is not None else _result(),
        calls=calls,
        error=retrieval_error,
        backend=backend,
    )
    composition_arguments: list[dict[str, object]] = []

    def compose(**kwargs: object) -> object:
        calls.append("compose")
        composition_arguments.append(kwargs)
        return SimpleNamespace(
            profile=_profile(),
            profile_repository=repository,
            retriever=retriever,
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
    return calls, repository, retriever, composition_arguments


def test_request_is_frozen_normalized_and_deterministic() -> None:
    request = _request(
        tenant_id=" tenant-synthetic ",
        knowledge_base_id=" kb-synthetic ",
        query=" Synthetic current question ",
        minimum_score=0,
    )

    assert request.tenant_id == "tenant-synthetic"
    assert request.knowledge_base_id == "kb-synthetic"
    assert request.query == "Synthetic current question"
    assert request.minimum_score == 0.0
    assert request == _request(minimum_score=0.0)
    assert request.model_dump() == {
        "tenant_id": "tenant-synthetic",
        "knowledge_base_id": "kb-synthetic",
        "query": "Synthetic current question",
        "top_k": 2,
        "minimum_score": 0.0,
    }
    with pytest.raises(ValidationError):
        request.top_k = 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", ""),
        ("knowledge_base_id", "   "),
        ("query", "\t"),
        ("top_k", 0),
        ("top_k", -1),
        ("top_k", True),
        ("top_k", 1.0),
        ("minimum_score", True),
        ("minimum_score", "0.5"),
        ("minimum_score", -0.1),
        ("minimum_score", 1.1),
        ("minimum_score", float("nan")),
        ("minimum_score", float("inf")),
    ],
)
def test_request_strict_validation(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _request(**{field: value})


def test_request_all_fields_are_required() -> None:
    for field in (
        "tenant_id",
        "knowledge_base_id",
        "query",
        "top_k",
        "minimum_score",
    ):
        values = _request().model_dump()
        del values[field]
        with pytest.raises(ValidationError):
            deployment.PostgreSQLRAGRetrievalRequest.model_validate(values)


def test_public_signature_and_exports_are_exact() -> None:
    parameters = signature(deployment.retrieve_profile_bound_postgres_rag).parameters

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
    assert (
        deployment_exports.PostgreSQLRAGRetrievalRequest
        is deployment.PostgreSQLRAGRetrievalRequest
    )
    assert (
        deployment_exports.retrieve_profile_bound_postgres_rag
        is deployment.retrieve_profile_bound_postgres_rag
    )
    assert deployment_exports.__all__.count("PostgreSQLRAGRetrievalRequest") == 1
    assert deployment_exports.__all__.count("retrieve_profile_bound_postgres_rag") == 1


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
        "request": _request(),
        "psycopg_connect": FakeConnect(),
        "embedding_backend_factory": None,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        deployment.retrieve_profile_bound_postgres_rag(**cast(Any, arguments))

    assert compose_calls == 0


@pytest.mark.parametrize(
    "retrieval_request",
    [
        _request(tenant_id="tenant-other"),
        _request(knowledge_base_id="kb-other"),
    ],
)
def test_scope_mismatch_precedes_composition_and_io(
    monkeypatch: pytest.MonkeyPatch,
    retrieval_request: deployment.PostgreSQLRAGRetrievalRequest,
) -> None:
    compose_calls = 0

    def compose(**_kwargs: object) -> object:
        nonlocal compose_calls
        compose_calls += 1
        raise AssertionError("composition must not run")

    monkeypatch.setattr(deployment, "compose_profile_bound_postgres_rag", compose)

    with pytest.raises(ValueError, match="does not match provider scope"):
        deployment.retrieve_profile_bound_postgres_rag(
            postgres_settings=_postgres_settings(),
            knowledge_base_settings=_provider_settings(),
            request=retrieval_request,
            psycopg_connect=FakeConnect(),
        )

    assert compose_calls == 0


def test_exact_lifecycle_arguments_result_identity_and_lazy_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result()
    backend = FalseyBackendFactory()
    calls, repository, retriever, composition_arguments = _install_fakes(
        monkeypatch,
        profile_result=_profile(),
        result=result,
        backend=backend,
    )
    backend.events = calls
    connect = FakeConnect()
    request = _request()

    returned = deployment.retrieve_profile_bound_postgres_rag(
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
        "profile",
        "retrieve",
        "backend",
    ]
    assert repository.get_calls == [
        {
            "tenant_id": "tenant-synthetic",
            "knowledge_base_id": "kb-synthetic",
        }
    ]
    assert retriever.retrieve_calls == [
        {
            "tenant_id": "tenant-synthetic",
            "knowledge_base_id": "kb-synthetic",
            "query": "Synthetic current question",
            "top_k": 2,
            "minimum_score": 0.25,
        }
    ]
    assert composition_arguments[0]["embedding_backend_factory"] is backend
    assert backend.calls == 1
    assert connect.calls == [
        {
            "conninfo": _DSN,
            "connect_timeout": 7,
            "sslmode": "verify-full",
            "application_name": "callmetric-retrieval",
            "autocommit": False,
        }
    ]


@pytest.mark.parametrize(
    ("profile_result", "message"),
    [
        (None, "not registered"),
        (object(), "invalid embedding profile"),
        (
            _profile().model_copy(update={"model_id": "different-model"}),
            "conflicting embedding profile",
        ),
    ],
)
def test_profile_failure_prevents_retrieval_and_backend_loading(
    monkeypatch: pytest.MonkeyPatch,
    profile_result: object,
    message: str,
) -> None:
    backend = FalseyBackendFactory()
    calls, _repository, retriever, _arguments = _install_fakes(
        monkeypatch,
        profile_result=profile_result,
        backend=backend,
    )

    with pytest.raises(ValueError, match=message):
        deployment.retrieve_profile_bound_postgres_rag(
            postgres_settings=_postgres_settings(),
            knowledge_base_settings=_provider_settings(),
            request=_request(),
            psycopg_connect=FakeConnect(),
            embedding_backend_factory=cast(BackendFactory, backend),
        )

    assert calls[-1] == "profile"
    assert retriever.retrieve_calls == []
    assert backend.calls == 0


@pytest.mark.parametrize("stage", ["readiness", "profile", "retrieval"])
def test_provider_exception_identity_propagates(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    expected = RuntimeError("synthetic provider failure")
    _install_fakes(
        monkeypatch,
        profile_result=_profile(),
        readiness_error=expected if stage == "readiness" else None,
        profile_error=expected if stage == "profile" else None,
        retrieval_error=expected if stage == "retrieval" else None,
    )

    with pytest.raises(RuntimeError) as raised:
        deployment.retrieve_profile_bound_postgres_rag(
            postgres_settings=_postgres_settings(),
            knowledge_base_settings=_provider_settings(),
            request=_request(),
            psycopg_connect=FakeConnect(),
        )

    assert raised.value is expected


def test_empty_and_repeated_results_preserve_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = RetrievalResult(
        tenant_id="tenant-synthetic",
        knowledge_base_id="kb-synthetic",
    )
    _calls, _repository, _retriever, _arguments = _install_fakes(
        monkeypatch,
        profile_result=_profile(),
        result=result,
    )
    arguments = {
        "postgres_settings": _postgres_settings(),
        "knowledge_base_settings": _provider_settings(),
        "request": _request(),
        "psycopg_connect": FakeConnect(),
    }

    first = deployment.retrieve_profile_bound_postgres_rag(**arguments)
    second = deployment.retrieve_profile_bound_postgres_rag(**arguments)

    assert first is result
    assert second is result
    assert first.documents == ()
