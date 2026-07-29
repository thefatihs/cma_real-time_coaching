"""Tests for the reusable readiness-verified PostgreSQL RAG runtime."""

from concurrent.futures import ThreadPoolExecutor
import inspect
from threading import Event
from typing import Any, cast

import pytest

import app.composition as composition_exports
import app.composition.postgres_rag_runtime as runtime_module
from app.composition.postgres_rag import PostgreSQLRAGComposition
from app.composition.postgres_rag_orchestration import (
    PostgreSQLRAGOrchestrationComposition,
)
from app.composition.postgres_rag_runtime import (
    ProfileVerifiedPostgreSQLRAGRunner,
)
from app.orchestration.models import (
    OrchestrationCitationReference,
    OrchestrationRequest,
    OrchestrationResult,
)
from app.vector_store.embedding_profile import (
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
)
from app.vector_store.postgres.readiness import PostgreSQLSchemaReadinessChecker


def _profile(**changes: object) -> KnowledgeBaseEmbeddingProfile:
    values: dict[str, object] = {
        "tenant_id": "tenant-synthetic",
        "knowledge_base_id": "kb-synthetic",
        "model_id": "embedding-synthetic",
        "vector_dimension": 3,
        "normalize_embeddings": True,
        "distance_metric": EmbeddingDistanceMetric.COSINE,
    }
    values.update(changes)
    return KnowledgeBaseEmbeddingProfile.model_validate(values)


def _request(**changes: object) -> OrchestrationRequest:
    values: dict[str, object] = {
        "tenant_id": "tenant-synthetic",
        "call_id": "call-synthetic",
        "transcript_revision": 4,
        "knowledge_base_id": "kb-synthetic",
        "user_input": "Synthetic question",
        "top_k": 2,
        "minimum_score": 0.25,
    }
    values.update(changes)
    return OrchestrationRequest.model_validate(values)


def _result() -> OrchestrationResult:
    return OrchestrationResult(
        tenant_id="tenant-synthetic",
        call_id="call-synthetic",
        transcript_revision=4,
        generated_text="Synthetic answer",
        citations=(
            OrchestrationCitationReference(
                document_id="document-b",
                chunk_id="chunk-2",
            ),
            OrchestrationCitationReference(
                document_id="document-a",
                chunk_id="chunk-1",
            ),
        ),
    )


class FakeRepository:
    def __init__(
        self,
        profile: object,
        events: list[str],
        *,
        error: BaseException | None = None,
    ) -> None:
        self.profile = profile
        self.events = events
        self.error = error
        self.calls: list[dict[str, str]] = []

    def get_profile(self, *, tenant_id: str, knowledge_base_id: str) -> object:
        self.events.append("profile")
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "knowledge_base_id": knowledge_base_id,
            }
        )
        if self.error is not None:
            raise self.error
        return self.profile


class FakeOrchestrator:
    def __init__(
        self,
        result: OrchestrationResult | None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[OrchestrationRequest] = []

    def run(self, request: OrchestrationRequest) -> OrchestrationResult | None:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.result


class TrackingReadinessChecker(PostgreSQLSchemaReadinessChecker):
    def __init__(
        self,
        events: list[str],
        *,
        errors: list[BaseException] | None = None,
        entered: Event | None = None,
        release: Event | None = None,
    ) -> None:
        super().__init__(
            connection_factory=lambda: pytest.fail(
                "connection factory must not be invoked by the fake checker"
            )
        )
        self.events = events
        self.errors = [] if errors is None else errors
        self.entered = entered
        self.release = release
        self.calls = 0

    def verify(self) -> None:
        self.calls += 1
        self.events.append("readiness")
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=5)
        if self.errors:
            raise self.errors.pop(0)


def _composition(
    repository: FakeRepository,
    orchestrator: FakeOrchestrator,
) -> PostgreSQLRAGOrchestrationComposition:
    postgres_rag = PostgreSQLRAGComposition(
        profile=_profile(),
        profile_repository=cast(Any, repository),
        vector_store=cast(Any, object()),
        embedder=cast(Any, object()),
        ingestion_service=cast(Any, object()),
        retriever=cast(Any, object()),
    )
    return PostgreSQLRAGOrchestrationComposition(
        postgres_rag=postgres_rag,
        prompt_builder=cast(Any, object()),
        llm_gateway=cast(Any, object()),
        orchestrator=cast(Any, orchestrator),
    )


def _subject(
    *,
    registered: object | None = None,
    result: OrchestrationResult | None = None,
    readiness: TrackingReadinessChecker | None = None,
    repository_error: BaseException | None = None,
    orchestration_error: BaseException | None = None,
) -> tuple[
    ProfileVerifiedPostgreSQLRAGRunner,
    PostgreSQLRAGOrchestrationComposition,
    TrackingReadinessChecker,
    FakeRepository,
    FakeOrchestrator,
]:
    events: list[str] = [] if readiness is None else readiness.events
    checker = TrackingReadinessChecker(events) if readiness is None else readiness
    repository = FakeRepository(
        _profile() if registered is None else registered,
        events,
        error=repository_error,
    )
    orchestrator = FakeOrchestrator(
        result,
        error=orchestration_error,
    )
    composition = _composition(repository, orchestrator)
    return (
        ProfileVerifiedPostgreSQLRAGRunner(composition, checker),
        composition,
        checker,
        repository,
        orchestrator,
    )


def test_public_api_signature_and_export_are_exact() -> None:
    assert tuple(inspect.signature(ProfileVerifiedPostgreSQLRAGRunner).parameters) == (
        "composition",
        "readiness_checker",
    )
    assert tuple(
        inspect.signature(ProfileVerifiedPostgreSQLRAGRunner.prepare).parameters
    ) == ("self",)
    assert tuple(
        inspect.signature(ProfileVerifiedPostgreSQLRAGRunner.run).parameters
    ) == ("self", "request")
    assert (
        composition_exports.ProfileVerifiedPostgreSQLRAGRunner
        is ProfileVerifiedPostgreSQLRAGRunner
    )
    assert composition_exports.__all__.count("ProfileVerifiedPostgreSQLRAGRunner") == 1


@pytest.mark.parametrize(
    ("composition", "checker"),
    [
        (object(), object()),
        (object(), TrackingReadinessChecker([])),
    ],
)
def test_invalid_collaborators_are_rejected(
    composition: object,
    checker: object,
) -> None:
    with pytest.raises(ValueError):
        ProfileVerifiedPostgreSQLRAGRunner(
            cast(Any, composition),
            cast(Any, checker),
        )


def test_construction_is_side_effect_free_and_retains_exact_identity() -> None:
    subject, composition, checker, repository, orchestrator = _subject()

    assert subject._composition is composition  # noqa: SLF001
    assert subject._readiness_checker is checker  # noqa: SLF001
    assert checker.calls == 0
    assert repository.calls == []
    assert orchestrator.calls == []


def test_unprepared_invalid_type_and_wrong_scope_stop_before_orchestration() -> None:
    subject, _composition_value, _checker, _repository, orchestrator = _subject()

    with pytest.raises(ValueError, match="request must be OrchestrationRequest"):
        subject.run(cast(Any, object()))
    with pytest.raises(ValueError, match="tenant_id"):
        subject.run(_request(tenant_id="tenant-other"))
    with pytest.raises(ValueError, match="knowledge_base_id"):
        subject.run(_request(knowledge_base_id="kb-other"))
    with pytest.raises(RuntimeError, match="runtime is not prepared"):
        subject.run(_request())

    assert orchestrator.calls == []


def test_prepare_order_scope_and_successful_idempotency() -> None:
    subject, _composition_value, checker, repository, _orchestrator = _subject()

    subject.prepare()
    subject.prepare()

    assert checker.events == ["readiness", "profile"]
    assert checker.calls == 1
    assert repository.calls == [
        {
            "tenant_id": "tenant-synthetic",
            "knowledge_base_id": "kb-synthetic",
        }
    ]


@pytest.mark.parametrize(
    ("registered", "message"),
    [
        (False, "not registered"),
        (object(), "invalid embedding profile"),
        (_profile(model_id="other"), "conflicting embedding profile"),
    ],
)
def test_missing_malformed_and_conflicting_profiles_fail_closed(
    registered: object,
    message: str,
) -> None:
    value = None if registered is False else registered
    events: list[str] = []
    checker = TrackingReadinessChecker(events)
    repository = FakeRepository(value, events)
    orchestrator = FakeOrchestrator(None)
    composition = _composition(repository, orchestrator)
    subject = ProfileVerifiedPostgreSQLRAGRunner(composition, checker)

    with pytest.raises(ValueError, match=message):
        subject.prepare()
    with pytest.raises(RuntimeError, match="not prepared"):
        subject.run(_request())

    assert events == ["readiness", "profile"]
    assert orchestrator.calls == []


def test_readiness_and_repository_exception_identity_and_retry() -> None:
    readiness_error = RuntimeError("synthetic readiness failure")
    events: list[str] = []
    checker = TrackingReadinessChecker(events, errors=[readiness_error])
    subject, _composition_value, _checker, repository, _orchestrator = _subject(
        readiness=checker
    )

    with pytest.raises(RuntimeError) as raised:
        subject.prepare()
    assert raised.value is readiness_error
    subject.prepare()

    assert events == ["readiness", "readiness", "profile"]
    assert repository.calls

    repository_error = RuntimeError("synthetic repository failure")
    failed, _composition_value, _checker, _repository, _orchestrator = _subject(
        repository_error=repository_error
    )
    with pytest.raises(RuntimeError) as repository_raised:
        failed.prepare()
    assert repository_raised.value is repository_error


def test_concurrent_prepare_performs_one_successful_sequence() -> None:
    subject, _composition_value, checker, repository, _orchestrator = _subject()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(lambda _item: subject.prepare(), range(24)))

    assert results == (None,) * 24
    assert checker.calls == 1
    assert len(repository.calls) == 1
    assert checker.events == ["readiness", "profile"]


def test_run_waits_for_incomplete_preparation() -> None:
    events: list[str] = []
    entered = Event()
    release = Event()
    checker = TrackingReadinessChecker(
        events,
        entered=entered,
        release=release,
    )
    expected = _result()
    subject, _composition_value, _checker, _repository, orchestrator = _subject(
        result=expected,
        readiness=checker,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        preparation = executor.submit(subject.prepare)
        assert entered.wait(timeout=5)
        execution = executor.submit(subject.run, _request())
        assert not execution.done()
        release.set()
        assert preparation.result(timeout=5) is None
        assert execution.result(timeout=5) is expected

    assert len(orchestrator.calls) == 1


def test_run_delegates_exact_request_and_preserves_result_and_citations() -> None:
    expected = _result()
    subject, _composition_value, checker, repository, orchestrator = _subject(
        result=expected
    )
    request = _request()
    subject.prepare()

    returned = subject.run(request)
    repeated = subject.run(request)

    assert returned is expected
    assert repeated is expected
    assert returned is not None
    assert returned.citations == expected.citations
    assert orchestrator.calls == [request, request]
    assert checker.calls == 1
    assert len(repository.calls) == 1


def test_empty_result_and_orchestration_exception_identity_are_preserved() -> None:
    empty, _composition_value, _checker, _repository, empty_orchestrator = _subject()
    empty.prepare()
    assert empty.run(_request()) is None
    assert len(empty_orchestrator.calls) == 1

    expected = RuntimeError("synthetic orchestration failure")
    failed, _composition_value, _checker, _repository, failed_orchestrator = _subject(
        orchestration_error=expected
    )
    failed.prepare()
    with pytest.raises(RuntimeError) as raised:
        failed.run(_request())
    assert raised.value is expected
    assert len(failed_orchestrator.calls) == 1


def test_runtime_module_has_no_deferred_operational_features() -> None:
    source = inspect.getsource(runtime_module)

    assert "register_profile" not in source
    assert "migration" not in source.casefold()
    assert "Executor" not in source
    assert "pool" not in source.casefold()
    assert "logging" not in source
