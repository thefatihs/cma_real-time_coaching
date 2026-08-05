"""Tenant-scoped process ownership for dashboard document management."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock
from time import monotonic
from typing import cast
from uuid import uuid4

from app.composition.postgres_document_ingestion import (
    PostgreSQLDocumentIngestionRuntime,
    PostgreSQLDocumentIngestionSettings,
    compose_postgres_document_ingestion,
)
from app.composition.postgres_rag import PostgreSQLVectorStoreSettings
from app.ingestion.document_background import (
    DocumentSubmissionResult,
    DocumentSubmissionStatus,
)
from app.ingestion.registry_models import (
    DocumentIngestionState,
    DocumentRegistryEntry,
)
from app.tenancy.models import TenantConfig
from live_dashboard.document_view_models import (
    DocumentProgressViewModel,
    DocumentRuntimeStatus,
    DocumentSectionViewModel,
    RUNTIME_STATUS_MESSAGES,
    project_document,
    project_progress,
)
from live_dashboard.rag_runtime import validated_dashboard_rag_provider_settings

_DOCUMENT_ENVIRONMENT_KEYS = (
    "CALLMETRIC_DASHBOARD_DOCUMENT_MAX_WORKERS",
    "CALLMETRIC_DASHBOARD_DOCUMENT_CAPACITY",
)
_CONFIRMATION_TTL_SECONDS = 300.0
_MAX_DOCUMENTS = 50


@dataclass(frozen=True, slots=True)
class DocumentResourceIdentity:
    owner_token: str
    tenant_id: str
    knowledge_base_id: str

    @property
    def opaque_key(self) -> str:
        payload = (
            f"{self.owner_token}\0{self.tenant_id}\0{self.knowledge_base_id}".encode()
        )
        return sha256(payload).hexdigest()


def document_resource_identity(
    *, tenant_config: TenantConfig, owner_token: str
) -> DocumentResourceIdentity:
    knowledge_base_id = tenant_config.rag.knowledge_base_id
    if tenant_config.rag.enabled and knowledge_base_id is None:
        raise ValueError("enabled document scope requires a knowledge base")
    return DocumentResourceIdentity(
        owner_token=owner_token,
        tenant_id=tenant_config.context.tenant_id,
        knowledge_base_id=knowledge_base_id or "disabled",
    )


@dataclass(frozen=True, slots=True)
class DeleteConfirmation:
    token: str
    filename: str


@dataclass(frozen=True, slots=True)
class DocumentActionResult:
    succeeded: bool


@dataclass(frozen=True, slots=True)
class _ConfirmationRecord:
    document_id: str
    expires_at: float


class DashboardDocumentResource:
    """Own one optional runtime and all internal UI-action identity mappings."""

    def __init__(
        self,
        *,
        identity: DocumentResourceIdentity,
        status: DocumentRuntimeStatus,
        runtime: PostgreSQLDocumentIngestionRuntime | None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.identity = identity
        self.status = status
        self._runtime = runtime
        self._clock = clock
        self._action_documents: dict[str, str] = {}
        self._document_actions: dict[str, str] = {}
        self._confirmations: dict[str, _ConfirmationRecord] = {}
        self._active_submission_token: str | None = None
        self._closed = False
        self._lock = Lock()

    @property
    def opaque_key(self) -> str:
        return self.identity.opaque_key

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def submit_upload(
        self,
        *,
        submission_token: str,
        content: bytes,
        original_filename: str,
        media_type: str,
    ) -> DocumentSubmissionResult:
        runtime = self._require_runtime()
        result = runtime.manager.submit(
            submission_token=submission_token,
            content=content,
            original_filename=original_filename,
            declared_media_type=media_type,
        )
        if result.status is DocumentSubmissionStatus.ACCEPTED:
            with self._lock:
                self._active_submission_token = submission_token
        return result

    def view(self) -> DocumentSectionViewModel:
        if self.status is not DocumentRuntimeStatus.READY or self._runtime is None:
            return DocumentSectionViewModel(
                runtime_status=self.status,
                runtime_message=RUNTIME_STATUS_MESSAGES[self.status],
                manager_busy=False,
                progress=None,
                documents=(),
            )
        try:
            page = self._runtime.registry.list_document_page(
                tenant_id=self.identity.tenant_id,
                knowledge_base_id=self.identity.knowledge_base_id,
                page_size=_MAX_DOCUMENTS,
            )
            entries = page.entries
            progress = _active_progress(entries)
            visible_ids = frozenset(entry.document.document_id for entry in entries)
            with self._lock:
                for document_id in tuple(self._document_actions):
                    if document_id not in visible_ids:
                        token = self._document_actions.pop(document_id)
                        self._action_documents.pop(token, None)
            documents = tuple(
                project_document(entry, action_token=self._action_token(entry))
                for entry in entries
            )
            return DocumentSectionViewModel(
                runtime_status=self.status,
                runtime_message=RUNTIME_STATUS_MESSAGES[self.status],
                manager_busy=progress is not None and progress.active,
                progress=progress,
                documents=documents,
            )
        except Exception:
            return DocumentSectionViewModel(
                runtime_status=DocumentRuntimeStatus.UNAVAILABLE,
                runtime_message=RUNTIME_STATUS_MESSAGES[
                    DocumentRuntimeStatus.UNAVAILABLE
                ],
                manager_busy=False,
                progress=None,
                documents=(),
            )

    def begin_delete(self, *, action_token: str) -> DeleteConfirmation | None:
        runtime = self._require_runtime()
        document_id = self._resolve_action(action_token)
        entry = runtime.registry.get_entry(
            tenant_id=self.identity.tenant_id,
            knowledge_base_id=self.identity.knowledge_base_id,
            document_id=document_id,
        )
        if entry is None or entry.job.state in {
            DocumentIngestionState.QUEUED,
            DocumentIngestionState.PROCESSING,
        }:
            return None
        token = uuid4().hex
        with self._lock:
            now = self._clock()
            self._confirmations = {
                key: value
                for key, value in self._confirmations.items()
                if value.expires_at >= now and value.document_id != document_id
            }
            self._confirmations[token] = _ConfirmationRecord(
                document_id=document_id,
                expires_at=now + _CONFIRMATION_TTL_SECONDS,
            )
        return DeleteConfirmation(
            token=token, filename=entry.document.original_filename
        )

    def confirm_delete(self, *, confirmation_token: str) -> DocumentActionResult:
        runtime = self._require_runtime()
        with self._lock:
            confirmation = self._confirmations.pop(confirmation_token, None)
        if confirmation is None or confirmation.expires_at < self._clock():
            return DocumentActionResult(False)
        entry = runtime.registry.get_entry(
            tenant_id=self.identity.tenant_id,
            knowledge_base_id=self.identity.knowledge_base_id,
            document_id=confirmation.document_id,
        )
        if entry is None or entry.job.state in {
            DocumentIngestionState.QUEUED,
            DocumentIngestionState.PROCESSING,
        }:
            return DocumentActionResult(False)
        try:
            deleted = runtime.registry.delete_document(
                tenant_id=self.identity.tenant_id,
                knowledge_base_id=self.identity.knowledge_base_id,
                document_id=confirmation.document_id,
            )
        except Exception:
            return DocumentActionResult(False)
        if deleted is None:
            return DocumentActionResult(False)
        return DocumentActionResult(True)

    def confirmation_filename(self, *, confirmation_token: str) -> str | None:
        runtime = self._require_runtime()
        with self._lock:
            confirmation = self._confirmations.get(confirmation_token)
        if confirmation is None or confirmation.expires_at < self._clock():
            return None
        entry = runtime.registry.get_entry(
            tenant_id=self.identity.tenant_id,
            knowledge_base_id=self.identity.knowledge_base_id,
            document_id=confirmation.document_id,
        )
        return None if entry is None else entry.document.original_filename

    def cancel_active_submission(self) -> bool:
        runtime = self._require_runtime()
        with self._lock:
            token = self._active_submission_token
        if token is None:
            return False
        return runtime.manager.cancel(submission_token=token)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            runtime = self._runtime
            self._confirmations.clear()
            self._action_documents.clear()
            self._document_actions.clear()
        if runtime is not None:
            runtime.close(wait=True)

    def _require_runtime(self) -> PostgreSQLDocumentIngestionRuntime:
        with self._lock:
            if self._closed or self._runtime is None:
                raise RuntimeError("document runtime is unavailable")
            return self._runtime

    def _action_token(self, entry: DocumentRegistryEntry) -> str:
        document_id = entry.document.document_id
        with self._lock:
            token = self._document_actions.get(document_id)
            if token is None:
                token = uuid4().hex
                self._document_actions[document_id] = token
                self._action_documents[token] = document_id
            return token

    def _resolve_action(self, token: str) -> str:
        with self._lock:
            document_id = self._action_documents.get(token)
        if document_id is None:
            raise ValueError("document action is invalid")
        return document_id


class DashboardDocumentResourceRegistry:
    def __init__(self, *, capacity: int = 32) -> None:
        self._capacity = capacity
        self._resources: dict[str, DashboardDocumentResource] = {}
        self._lock = Lock()

    def acquire(
        self,
        *,
        identity: DocumentResourceIdentity,
        factory: Callable[[], DashboardDocumentResource],
    ) -> DashboardDocumentResource:
        key = identity.opaque_key
        with self._lock:
            existing = self._resources.get(key)
            if existing is not None:
                return existing
            if len(self._resources) >= self._capacity:
                raise RuntimeError("document resource capacity is exhausted")
            resource = factory()
            self._resources[key] = resource
            return resource

    def lookup(self, opaque_key: str) -> DashboardDocumentResource:
        with self._lock:
            resource = self._resources.get(opaque_key)
        if resource is None or resource.closed:
            raise ValueError("document resource is unavailable")
        return resource

    def close_and_remove(self, opaque_key: str) -> None:
        with self._lock:
            resource = self._resources.pop(opaque_key, None)
        if resource is not None:
            resource.close()

    def close_all(self) -> None:
        with self._lock:
            resources = tuple(self._resources.values())
            self._resources.clear()
        for resource in resources:
            resource.close()


class DashboardDocumentRuntimeController:
    def __init__(
        self,
        *,
        registry: DashboardDocumentResourceRegistry,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._registry = registry
        self._environment = os.environ if environment is None else environment

    def activate(
        self, *, tenant_config: TenantConfig, owner_token: str
    ) -> DashboardDocumentResource:
        identity = document_resource_identity(
            tenant_config=tenant_config, owner_token=owner_token
        )

        def factory() -> DashboardDocumentResource:
            status, runtime = _activate_optional_document_runtime(
                tenant_config=tenant_config,
                environment=self._environment,
            )
            return DashboardDocumentResource(
                identity=identity, status=status, runtime=runtime
            )

        try:
            return self._registry.acquire(identity=identity, factory=factory)
        except Exception:
            return DashboardDocumentResource(
                identity=identity,
                status=DocumentRuntimeStatus.UNAVAILABLE,
                runtime=None,
            )

    def lookup(self, opaque_key: str) -> DashboardDocumentResource:
        return self._registry.lookup(opaque_key)

    def close_and_remove(self, opaque_key: str) -> None:
        self._registry.close_and_remove(opaque_key)

    def close_all(self) -> None:
        self._registry.close_all()


def _activate_optional_document_runtime(
    *, tenant_config: TenantConfig, environment: Mapping[str, str]
) -> tuple[DocumentRuntimeStatus, PostgreSQLDocumentIngestionRuntime | None]:
    configured = tuple(key in environment for key in _DOCUMENT_ENVIRONMENT_KEYS)
    if not tenant_config.rag.enabled or not any(configured):
        return DocumentRuntimeStatus.DISABLED, None
    if not all(configured):
        return DocumentRuntimeStatus.UNAVAILABLE, None
    try:
        provider = validated_dashboard_rag_provider_settings(
            tenant_config=tenant_config, environment=environment
        )
        settings = PostgreSQLDocumentIngestionSettings(
            max_workers=_strict_integer(environment[_DOCUMENT_ENVIRONMENT_KEYS[0]]),
            capacity=_strict_integer(environment[_DOCUMENT_ENVIRONMENT_KEYS[1]]),
        )
        postgres_factory = cast(
            Callable[[], PostgreSQLVectorStoreSettings], PostgreSQLVectorStoreSettings
        )
        postgres_settings = postgres_factory()
        from psycopg import connect as psycopg_connect

        runtime = compose_postgres_document_ingestion(
            postgres_settings=postgres_settings,
            knowledge_base_settings=provider,
            ingestion_settings=settings,
            psycopg_connect=psycopg_connect,
        )
    except Exception:
        return DocumentRuntimeStatus.UNAVAILABLE, None
    return DocumentRuntimeStatus.READY, runtime


def _active_progress(
    entries: tuple[DocumentRegistryEntry, ...],
) -> DocumentProgressViewModel | None:
    for entry in entries:
        if entry.job.state in {
            DocumentIngestionState.QUEUED,
            DocumentIngestionState.PROCESSING,
        }:
            return project_progress(entry)
    return None


def _strict_integer(value: object) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError("document setting must be an integer")
    return int(value)
