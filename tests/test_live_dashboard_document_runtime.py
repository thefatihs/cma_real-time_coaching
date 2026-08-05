"""Document runtime activation, ownership, and exact-action tests."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import live_dashboard.document_runtime as subject
from app.ingestion.registry_models import (
    DocumentDeletionResult,
    DocumentIngestionJob,
    DocumentIngestionPhase,
    DocumentIngestionState,
    DocumentReadiness,
    DocumentRegistryEntry,
    DocumentRegistryRecord,
)
from app.tenancy.models import TenantRAGConfig
from live_dashboard.demo_data import tenant_demos
from live_dashboard.document_runtime import (
    DashboardDocumentResource,
    DashboardDocumentResourceRegistry,
    DashboardDocumentRuntimeController,
    DocumentResourceIdentity,
)
from live_dashboard.document_view_models import DocumentRuntimeStatus


def _tenant(*, tenant_id: str = "tenant_alpha"):
    config = tenant_demos()["tenant_alpha"].config
    return config.model_copy(
        update={
            "context": config.context.model_copy(update={"tenant_id": tenant_id}),
            "rag": TenantRAGConfig(enabled=True, knowledge_base_id="kb-trusted"),
        }
    )


def _entry(state: DocumentIngestionState) -> DocumentRegistryEntry:
    now = datetime.now(UTC)
    terminal = state in {
        DocumentIngestionState.SUCCEEDED,
        DocumentIngestionState.FAILED,
        DocumentIngestionState.CANCELLED,
    }
    return DocumentRegistryEntry(
        document=DocumentRegistryRecord(
            tenant_id="tenant_alpha",
            knowledge_base_id="kb-trusted",
            document_id="doc-internal",
            original_filename="rehber.txt",
            media_type="text/plain",
            byte_size=10,
            storage_object_key="obj_" + "a" * 64,
            created_at_utc=now,
            ready_at_utc=now if state is DocumentIngestionState.SUCCEEDED else None,
        ),
        job=DocumentIngestionJob(
            tenant_id="tenant_alpha",
            knowledge_base_id="kb-trusted",
            job_id="job-internal",
            document_id="doc-internal",
            state=state,
            phase=DocumentIngestionPhase.FINALIZE
            if state is DocumentIngestionState.SUCCEEDED
            else DocumentIngestionPhase.EXTRACTION,
            processed_chunks=1 if state is DocumentIngestionState.SUCCEEDED else 0,
            total_chunks=1,
            attempt_count=0 if state is DocumentIngestionState.QUEUED else 1,
            created_at_utc=now,
            started_at_utc=None if state is DocumentIngestionState.QUEUED else now,
            updated_at_utc=now,
            finished_at_utc=now if terminal else None,
        ),
        readiness={
            DocumentIngestionState.SUCCEEDED: DocumentReadiness.READY,
            DocumentIngestionState.FAILED: DocumentReadiness.FAILED,
            DocumentIngestionState.CANCELLED: DocumentReadiness.CANCELLED,
        }.get(state, DocumentReadiness.PENDING),
    )


class _Registry:
    def __init__(self, entry: DocumentRegistryEntry) -> None:
        self.entry = entry
        self.deletions: list[tuple[str, str, str]] = []

    def list_document_page(self, **scope: object):
        return SimpleNamespace(entries=(self.entry,))

    def get_entry(self, **scope: str) -> DocumentRegistryEntry | None:
        return self.entry

    def delete_document(
        self, *, tenant_id: str, knowledge_base_id: str, document_id: str
    ) -> DocumentDeletionResult:
        self.deletions.append((tenant_id, knowledge_base_id, document_id))
        return DocumentDeletionResult(storage_object_key="obj_" + "a" * 64)


class _Runtime:
    def __init__(self, entry: DocumentRegistryEntry) -> None:
        self.registry = _Registry(entry)
        self.manager = SimpleNamespace(cancel=lambda **kwargs: True)
        self.closed = 0

    def close(self, *, wait: bool) -> None:
        self.closed += 1


def test_disabled_activation_does_not_compose_or_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "compose_postgres_document_ingestion",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("forbidden")),
    )
    status, runtime = subject._activate_optional_document_runtime(
        tenant_config=tenant_demos()["tenant_alpha"].config,
        environment={},
    )
    assert (status, runtime) == (DocumentRuntimeStatus.DISABLED, None)


@pytest.mark.parametrize(
    "environment",
    [
        {"CALLMETRIC_DASHBOARD_DOCUMENT_CAPACITY": "2"},
        {
            "CALLMETRIC_DASHBOARD_DOCUMENT_MAX_WORKERS": "2",
            "CALLMETRIC_DASHBOARD_DOCUMENT_CAPACITY": "9",
        },
    ],
)
def test_partial_or_invalid_configuration_is_unavailable(
    environment: dict[str, str],
) -> None:
    status, runtime = subject._activate_optional_document_runtime(
        tenant_config=_tenant(), environment=environment
    )
    assert (status, runtime) == (DocumentRuntimeStatus.UNAVAILABLE, None)


def test_complete_valid_configuration_composes_ready_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runtime = SimpleNamespace()
    calls: list[str] = []
    monkeypatch.setattr(
        subject,
        "validated_dashboard_rag_provider_settings",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(subject, "PostgreSQLVectorStoreSettings", lambda: object())
    monkeypatch.setattr(
        subject,
        "compose_postgres_document_ingestion",
        lambda **kwargs: (calls.append("compose"), fake_runtime)[1],
    )
    environment = {
        "CALLMETRIC_DASHBOARD_DOCUMENT_MAX_WORKERS": "1",
        "CALLMETRIC_DASHBOARD_DOCUMENT_CAPACITY": "2",
    }
    status, runtime = subject._activate_optional_document_runtime(
        tenant_config=_tenant(), environment=environment
    )
    assert (status, runtime) == (DocumentRuntimeStatus.READY, fake_runtime)
    assert calls == ["compose"]


def test_registry_is_idempotent_per_session_and_separates_sessions() -> None:
    registry = DashboardDocumentResourceRegistry()
    created: list[DashboardDocumentResource] = []

    def acquire(owner: str) -> DashboardDocumentResource:
        identity = DocumentResourceIdentity(owner, "tenant_alpha", "kb-trusted")

        def factory() -> DashboardDocumentResource:
            resource = DashboardDocumentResource(
                identity=identity,
                status=DocumentRuntimeStatus.DISABLED,
                runtime=None,
            )
            created.append(resource)
            return resource

        return registry.acquire(identity=identity, factory=factory)

    assert acquire("session-a") is acquire("session-a")
    assert acquire("session-b") is not acquire("session-a")
    assert len(created) == 2
    registry.close_all()
    registry.close_all()
    assert all(resource.closed for resource in created)


def test_controller_returns_fixed_unavailable_without_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "_activate_optional_document_runtime",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("PRIVATE DSN C:/secret")),
    )
    resource = DashboardDocumentRuntimeController(
        registry=DashboardDocumentResourceRegistry(), environment={}
    ).activate(tenant_config=_tenant(), owner_token="session-a")
    assert resource.status is DocumentRuntimeStatus.UNAVAILABLE
    assert "PRIVATE" not in repr(resource.view())


def test_two_step_delete_is_scope_bound_one_use_and_exact() -> None:
    runtime = _Runtime(_entry(DocumentIngestionState.SUCCEEDED))
    resource = DashboardDocumentResource(
        identity=DocumentResourceIdentity("session-a", "tenant_alpha", "kb-trusted"),
        status=DocumentRuntimeStatus.READY,
        runtime=runtime,  # type: ignore[arg-type]
    )
    item = resource.view().documents[0]
    confirmation = resource.begin_delete(action_token=item.action_token)
    assert confirmation is not None
    result = resource.confirm_delete(confirmation_token=confirmation.token)
    assert result.succeeded
    assert runtime.registry.deletions == [
        ("tenant_alpha", "kb-trusted", "doc-internal")
    ]
    assert not resource.confirm_delete(confirmation_token=confirmation.token).succeeded


def test_active_document_delete_is_refused() -> None:
    resource = DashboardDocumentResource(
        identity=DocumentResourceIdentity("session-a", "tenant_alpha", "kb-trusted"),
        status=DocumentRuntimeStatus.READY,
        runtime=_Runtime(_entry(DocumentIngestionState.PROCESSING)),  # type: ignore[arg-type]
    )
    item = resource.view().documents[0]
    assert resource.begin_delete(action_token=item.action_token) is None


def test_delete_confirmation_expires_without_delegation() -> None:
    now = [100.0]
    runtime = _Runtime(_entry(DocumentIngestionState.SUCCEEDED))
    resource = DashboardDocumentResource(
        identity=DocumentResourceIdentity("session-a", "tenant_alpha", "kb-trusted"),
        status=DocumentRuntimeStatus.READY,
        runtime=runtime,  # type: ignore[arg-type]
        clock=lambda: now[0],
    )
    confirmation = resource.begin_delete(
        action_token=resource.view().documents[0].action_token
    )
    assert confirmation is not None
    now[0] += 301
    assert not resource.confirm_delete(confirmation_token=confirmation.token).succeeded
    assert runtime.registry.deletions == []
