"""Secret-safe Windows document-backed dashboard RAG/vLLM E2E controller."""

from __future__ import annotations

import json
import math
import os
import queue
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from ctypes import POINTER, WinDLL, byref, c_int, c_void_p, wintypes
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Protocol, TypeVar
from urllib.parse import urlsplit

from app.composition.postgres_document_ingestion import (
    MINILM_DIMENSION,
    MINILM_MODEL,
    validate_local_minilm_snapshot,
)
from app.composition.postgres_rag import (
    BoundedRAGDiagnosticObserver,
    KnowledgeBaseRAGProviderSettings,
    PostgreSQLVectorStoreSettings,
    RAGDiagnosticSnapshot,
    RAGDiagnosticStage,
    RAGDiagnosticStatus,
    RAGDiagnosticFutureState,
    RAGDiagnosticSubmissionState,
)
from app.composition.postgres_document_ingestion import (
    PostgreSQLDocumentIngestionRuntime,
)
from app.composition.postgres_rag_background import BoundedPostgreSQLRAGManager
from app.coaching.coordinator import CoachingProcessingStatus, StableCoachingOutcome
from app.ingestion.registry_models import DocumentRegistryEntry
from app.integration.rag_coaching import RAGCoachingProcessorDecorator
from app.integration.policy import RAGCoachingIntegrationPolicy
from app.llm.vllm_openai_compatible import VLLMOpenAICompatibleSettings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BRANCH_ENV = "CALLMETRIC_DASHBOARD_RAG_E2E_EXPECTED_BRANCH"
HEAD_ENV = "CALLMETRIC_DASHBOARD_RAG_E2E_EXPECTED_HEAD"
BASELINE_ENV = "CALLMETRIC_DASHBOARD_RAG_E2E_EXPECTED_BASELINE"
HANDOFF_ROOT_ENV = "CALLMETRIC_POSTGRES_TLS_SERVICE_HANDOFF_ROOT"
PROVIDER_ENV = "CALLMETRIC_DASHBOARD_RAG_PROVIDER_SETTINGS_PATH"
POLICY_ENV = "CALLMETRIC_DASHBOARD_RAG_INTEGRATION_POLICY_PATH"
TOKEN_ENV = "CALLMETRIC_VLLM_API_TOKEN"
CA_ENV = "CALLMETRIC_VLLM_CA_CERTIFICATE_PATH"
TTL_ENV = "CALLMETRIC_DASHBOARD_RAG_E2E_POSTGRES_TTL_SECONDS"
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
BRANCH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
MINIMUM_TTL_SECONDS = 300
MAXIMUM_TTL_SECONDS = 7_200
FULL_E2E_TTL_SECONDS = MAXIMUM_TTL_SECONDS
DOCUMENT_POLL_TIMEOUT_SECONDS = 300.0
POLL_INTERVAL_SECONDS = 0.2
ORCHESTRATION_MARGIN_SECONDS = 60.0
MINIMUM_E2E_OUTPUT_TOKENS = 256
OWNER_MARKER_PATTERN = re.compile(r"^callmetric-owner-[0-9a-f]{32}$")
_MAX_GIT_OUTPUT_BYTES = 65_536
_MAX_WMI_OUTPUT_BYTES = 4_194_304
_MAX_RESOURCE_OUTPUT_BYTES = 65_536

PREFLIGHT_OK = "PREFLIGHT_OK"
E2E_OK = "E2E_OK"
POSTGRES_STARTUP_OK = "POSTGRES_STARTUP_OK"
E_CLEANUP_PROCESS_ACTION = "E_CLEANUP_PROCESS_ACTION"
E_CLEANUP_PROJECT_ACTION = "E_CLEANUP_PROJECT_ACTION"
E_CLEANUP_HANDOFF_ACTION = "E_CLEANUP_HANDOFF_ACTION"
E_CLEANUP_PROCESS_VERIFY = "E_CLEANUP_PROCESS_VERIFY"
E_CLEANUP_PROJECT_VERIFY = "E_CLEANUP_PROJECT_VERIFY"
E_CLEANUP_HANDOFF_VERIFY = "E_CLEANUP_HANDOFF_VERIFY"
E_CLEANUP_PROTECTED_VERIFY = "E_CLEANUP_PROTECTED_VERIFY"
E_CLEANUP_UNVERIFIABLE = "E_CLEANUP_UNVERIFIABLE"
CLEANUP_FAILURE_PHASES = frozenset(
    {
        E_CLEANUP_PROCESS_ACTION,
        E_CLEANUP_PROJECT_ACTION,
        E_CLEANUP_HANDOFF_ACTION,
        E_CLEANUP_PROCESS_VERIFY,
        E_CLEANUP_PROJECT_VERIFY,
        E_CLEANUP_HANDOFF_VERIFY,
        E_CLEANUP_PROTECTED_VERIFY,
        E_CLEANUP_UNVERIFIABLE,
    }
)
E_DUPLICATE_RUNTIME = "E_DUPLICATE_RUNTIME"
E_DUPLICATE_BEFORE_LIST = "E_DUPLICATE_BEFORE_LIST"
E_DUPLICATE_SUBMIT = "E_DUPLICATE_SUBMIT"
E_DUPLICATE_STATUS = "E_DUPLICATE_STATUS"
E_DUPLICATE_AFTER_LIST = "E_DUPLICATE_AFTER_LIST"
E_DUPLICATE_TARGET_LOOKUP = "E_DUPLICATE_TARGET_LOOKUP"
E_DUPLICATE_OTHER_LOOKUP = "E_DUPLICATE_OTHER_LOOKUP"
E_DUPLICATE_VECTOR_BEFORE_SETTINGS = "E_DUPLICATE_VECTOR_BEFORE_SETTINGS"
E_DUPLICATE_VECTOR_BEFORE_CONNECT = "E_DUPLICATE_VECTOR_BEFORE_CONNECT"
E_DUPLICATE_VECTOR_BEFORE_CURSOR = "E_DUPLICATE_VECTOR_BEFORE_CURSOR"
E_DUPLICATE_VECTOR_BEFORE_EXECUTE = "E_DUPLICATE_VECTOR_BEFORE_EXECUTE"
E_DUPLICATE_VECTOR_BEFORE_FETCH = "E_DUPLICATE_VECTOR_BEFORE_FETCH"
E_DUPLICATE_VECTOR_BEFORE_RESULT_SHAPE = "E_DUPLICATE_VECTOR_BEFORE_RESULT_SHAPE"
E_DUPLICATE_VECTOR_BEFORE_CLOSE = "E_DUPLICATE_VECTOR_BEFORE_CLOSE"
E_DUPLICATE_VECTOR_BEFORE_UNCLASSIFIED = "E_DUPLICATE_VECTOR_BEFORE_UNCLASSIFIED"
E_DUPLICATE_VECTOR_AFTER_SETTINGS = "E_DUPLICATE_VECTOR_AFTER_SETTINGS"
E_DUPLICATE_VECTOR_AFTER_CONNECT = "E_DUPLICATE_VECTOR_AFTER_CONNECT"
E_DUPLICATE_VECTOR_AFTER_CURSOR = "E_DUPLICATE_VECTOR_AFTER_CURSOR"
E_DUPLICATE_VECTOR_AFTER_EXECUTE = "E_DUPLICATE_VECTOR_AFTER_EXECUTE"
E_DUPLICATE_VECTOR_AFTER_FETCH = "E_DUPLICATE_VECTOR_AFTER_FETCH"
E_DUPLICATE_VECTOR_AFTER_RESULT_SHAPE = "E_DUPLICATE_VECTOR_AFTER_RESULT_SHAPE"
E_DUPLICATE_VECTOR_AFTER_CLOSE = "E_DUPLICATE_VECTOR_AFTER_CLOSE"
E_DUPLICATE_VECTOR_AFTER_UNCLASSIFIED = "E_DUPLICATE_VECTOR_AFTER_UNCLASSIFIED"
E_DUPLICATE_RESULT_SHAPE = "E_DUPLICATE_RESULT_SHAPE"
E_DUPLICATE_DOCUMENT_IDENTITY = "E_DUPLICATE_DOCUMENT_IDENTITY"
E_DUPLICATE_JOB_IDENTITY = "E_DUPLICATE_JOB_IDENTITY"
E_DUPLICATE_READINESS = "E_DUPLICATE_READINESS"
E_DUPLICATE_REGISTRY_CARDINALITY = "E_DUPLICATE_REGISTRY_CARDINALITY"
E_DUPLICATE_VECTOR_CARDINALITY = "E_DUPLICATE_VECTOR_CARDINALITY"
E_DUPLICATE_OTHER_DOCUMENT = "E_DUPLICATE_OTHER_DOCUMENT"
E_DUPLICATE_SOURCE_KEY = "E_DUPLICATE_SOURCE_KEY"
E_DUPLICATE_UNCLASSIFIED = "E_DUPLICATE_UNCLASSIFIED"
DUPLICATE_FAILURE_PHASES = frozenset(
    {
        E_DUPLICATE_RUNTIME,
        E_DUPLICATE_BEFORE_LIST,
        E_DUPLICATE_SUBMIT,
        E_DUPLICATE_STATUS,
        E_DUPLICATE_AFTER_LIST,
        E_DUPLICATE_TARGET_LOOKUP,
        E_DUPLICATE_OTHER_LOOKUP,
        E_DUPLICATE_VECTOR_BEFORE_SETTINGS,
        E_DUPLICATE_VECTOR_BEFORE_CONNECT,
        E_DUPLICATE_VECTOR_BEFORE_CURSOR,
        E_DUPLICATE_VECTOR_BEFORE_EXECUTE,
        E_DUPLICATE_VECTOR_BEFORE_FETCH,
        E_DUPLICATE_VECTOR_BEFORE_RESULT_SHAPE,
        E_DUPLICATE_VECTOR_BEFORE_CLOSE,
        E_DUPLICATE_VECTOR_BEFORE_UNCLASSIFIED,
        E_DUPLICATE_VECTOR_AFTER_SETTINGS,
        E_DUPLICATE_VECTOR_AFTER_CONNECT,
        E_DUPLICATE_VECTOR_AFTER_CURSOR,
        E_DUPLICATE_VECTOR_AFTER_EXECUTE,
        E_DUPLICATE_VECTOR_AFTER_FETCH,
        E_DUPLICATE_VECTOR_AFTER_RESULT_SHAPE,
        E_DUPLICATE_VECTOR_AFTER_CLOSE,
        E_DUPLICATE_VECTOR_AFTER_UNCLASSIFIED,
        E_DUPLICATE_RESULT_SHAPE,
        E_DUPLICATE_DOCUMENT_IDENTITY,
        E_DUPLICATE_JOB_IDENTITY,
        E_DUPLICATE_READINESS,
        E_DUPLICATE_REGISTRY_CARDINALITY,
        E_DUPLICATE_VECTOR_CARDINALITY,
        E_DUPLICATE_OTHER_DOCUMENT,
        E_DUPLICATE_SOURCE_KEY,
        E_DUPLICATE_UNCLASSIFIED,
    }
)
E_DOCUMENT_READY_RUNTIME = "E_DOCUMENT_READY_RUNTIME"
E_DOCUMENT_READY_CLOCK = "E_DOCUMENT_READY_CLOCK"
E_DOCUMENT_READY_LIST = "E_DOCUMENT_READY_LIST"
E_DOCUMENT_READY_RESULT_SHAPE = "E_DOCUMENT_READY_RESULT_SHAPE"
E_DOCUMENT_READY_CARDINALITY = "E_DOCUMENT_READY_CARDINALITY"
E_DOCUMENT_READY_TARGET_MISSING = "E_DOCUMENT_READY_TARGET_MISSING"
E_DOCUMENT_READY_OTHER_MISSING = "E_DOCUMENT_READY_OTHER_MISSING"
E_DOCUMENT_READY_TARGET_FAILED = "E_DOCUMENT_READY_TARGET_FAILED"
E_DOCUMENT_READY_OTHER_FAILED = "E_DOCUMENT_READY_OTHER_FAILED"
E_DOCUMENT_READY_TARGET_CANCELLED = "E_DOCUMENT_READY_TARGET_CANCELLED"
E_DOCUMENT_READY_OTHER_CANCELLED = "E_DOCUMENT_READY_OTHER_CANCELLED"
E_DOCUMENT_READY_TARGET_JOB = "E_DOCUMENT_READY_TARGET_JOB"
E_DOCUMENT_READY_OTHER_JOB = "E_DOCUMENT_READY_OTHER_JOB"
E_DOCUMENT_READY_SOURCE_KEY = "E_DOCUMENT_READY_SOURCE_KEY"
E_DOCUMENT_READY_SLEEP = "E_DOCUMENT_READY_SLEEP"
E_DOCUMENT_READY_TIMEOUT = "E_DOCUMENT_READY_TIMEOUT"
E_DOCUMENT_READY_UNCLASSIFIED = "E_DOCUMENT_READY_UNCLASSIFIED"
DOCUMENT_READY_FAILURE_PHASES = frozenset(
    {
        E_DOCUMENT_READY_RUNTIME,
        E_DOCUMENT_READY_CLOCK,
        E_DOCUMENT_READY_LIST,
        E_DOCUMENT_READY_RESULT_SHAPE,
        E_DOCUMENT_READY_CARDINALITY,
        E_DOCUMENT_READY_TARGET_MISSING,
        E_DOCUMENT_READY_OTHER_MISSING,
        E_DOCUMENT_READY_TARGET_FAILED,
        E_DOCUMENT_READY_OTHER_FAILED,
        E_DOCUMENT_READY_TARGET_CANCELLED,
        E_DOCUMENT_READY_OTHER_CANCELLED,
        E_DOCUMENT_READY_TARGET_JOB,
        E_DOCUMENT_READY_OTHER_JOB,
        E_DOCUMENT_READY_SOURCE_KEY,
        E_DOCUMENT_READY_SLEEP,
        E_DOCUMENT_READY_TIMEOUT,
        E_DOCUMENT_READY_UNCLASSIFIED,
    }
)
E_COMPLETION_PROCESSOR_MISSING = "E_COMPLETION_PROCESSOR_MISSING"
E_COMPLETION_NO_AUTHORITATIVE_OUTCOME = "E_COMPLETION_NO_AUTHORITATIVE_OUTCOME"
E_COMPLETION_CARDINALITY = "E_COMPLETION_CARDINALITY"
E_COMPLETION_BACKGROUND_FAILED = "E_COMPLETION_BACKGROUND_FAILED"
E_COMPLETION_NOT_PROCESSED = "E_COMPLETION_NOT_PROCESSED"
E_COMPLETION_RESULT_MISSING = "E_COMPLETION_RESULT_MISSING"
E_COMPLETION_UNCLASSIFIED = "E_COMPLETION_UNCLASSIFIED"
E_COMPLETION_STALLED_RUN_ENTER = "E_COMPLETION_STALLED_RUN_ENTER"
E_COMPLETION_STALLED_EMBED_ENTER = "E_COMPLETION_STALLED_EMBED_ENTER"
E_COMPLETION_STALLED_VECTOR_ENTER = "E_COMPLETION_STALLED_VECTOR_ENTER"
E_COMPLETION_STALLED_PROMPT_ENTER = "E_COMPLETION_STALLED_PROMPT_ENTER"
E_COMPLETION_STALLED_GATEWAY_FACTORY_ENTER = (
    "E_COMPLETION_STALLED_GATEWAY_FACTORY_ENTER"
)
E_COMPLETION_STALLED_HTTP_ENTER = "E_COMPLETION_STALLED_HTTP_ENTER"
E_COMPLETION_STALLED_CALLBACK_ENTER = "E_COMPLETION_STALLED_CALLBACK_ENTER"
E_COMPLETION_STALLED_CALLBACK_NOT_ENTERED = "E_COMPLETION_STALLED_CALLBACK_NOT_ENTERED"
E_COMPLETION_STALLED_FAILED = "E_COMPLETION_STALLED_FAILED"
E_COMPLETION_STALLED_WORKER_NOT_LIVE = "E_COMPLETION_STALLED_WORKER_NOT_LIVE"
E_COMPLETION_STALLED_UNCLASSIFIED = "E_COMPLETION_STALLED_UNCLASSIFIED"
E_COMPLETION_SUBMISSION_NOT_ATTEMPTED = "E_COMPLETION_SUBMISSION_NOT_ATTEMPTED"
E_COMPLETION_SUBMISSION_REJECTED_DUPLICATE = (
    "E_COMPLETION_SUBMISSION_REJECTED_DUPLICATE"
)
E_COMPLETION_SUBMISSION_REJECTED_STALE = "E_COMPLETION_SUBMISSION_REJECTED_STALE"
E_COMPLETION_SUBMISSION_REJECTED_CAPACITY_REJECTED = (
    "E_COMPLETION_SUBMISSION_REJECTED_CAPACITY_REJECTED"
)
E_COMPLETION_SUBMISSION_REJECTED_NOT_STARTED = (
    "E_COMPLETION_SUBMISSION_REJECTED_NOT_STARTED"
)
E_COMPLETION_SUBMISSION_REJECTED_CLOSED = "E_COMPLETION_SUBMISSION_REJECTED_CLOSED"
E_COMPLETION_SUBMIT_FAILED = "E_COMPLETION_SUBMIT_FAILED"
E_COMPLETION_FUTURE_QUEUED = "E_COMPLETION_FUTURE_QUEUED"
E_COMPLETION_FUTURE_RUNNING_NO_STAGE = "E_COMPLETION_FUTURE_RUNNING_NO_STAGE"
E_COMPLETION_FUTURE_TERMINAL_NO_PUBLICATION = (
    "E_COMPLETION_FUTURE_TERMINAL_NO_PUBLICATION"
)
E_COMPLETION_DIAGNOSTIC_UNCLASSIFIED = "E_COMPLETION_DIAGNOSTIC_UNCLASSIFIED"
_SUBMISSION_PHASES = {
    RAGDiagnosticSubmissionState.NOT_ATTEMPTED: (E_COMPLETION_SUBMISSION_NOT_ATTEMPTED),
    RAGDiagnosticSubmissionState.REJECTED_DUPLICATE: (
        E_COMPLETION_SUBMISSION_REJECTED_DUPLICATE
    ),
    RAGDiagnosticSubmissionState.REJECTED_STALE: (
        E_COMPLETION_SUBMISSION_REJECTED_STALE
    ),
    RAGDiagnosticSubmissionState.REJECTED_CAPACITY_REJECTED: (
        E_COMPLETION_SUBMISSION_REJECTED_CAPACITY_REJECTED
    ),
    RAGDiagnosticSubmissionState.REJECTED_NOT_STARTED: (
        E_COMPLETION_SUBMISSION_REJECTED_NOT_STARTED
    ),
    RAGDiagnosticSubmissionState.REJECTED_CLOSED: (
        E_COMPLETION_SUBMISSION_REJECTED_CLOSED
    ),
    RAGDiagnosticSubmissionState.SUBMIT_FAILED: E_COMPLETION_SUBMIT_FAILED,
}
_STALLED_STAGE_PHASES = {
    RAGDiagnosticStage.RUN: E_COMPLETION_STALLED_RUN_ENTER,
    RAGDiagnosticStage.EMBED: E_COMPLETION_STALLED_EMBED_ENTER,
    RAGDiagnosticStage.VECTOR: E_COMPLETION_STALLED_VECTOR_ENTER,
    RAGDiagnosticStage.PROMPT: E_COMPLETION_STALLED_PROMPT_ENTER,
    RAGDiagnosticStage.GATEWAY_FACTORY: E_COMPLETION_STALLED_GATEWAY_FACTORY_ENTER,
    RAGDiagnosticStage.HTTP: E_COMPLETION_STALLED_HTTP_ENTER,
    RAGDiagnosticStage.CALLBACK: E_COMPLETION_STALLED_CALLBACK_ENTER,
}
E_POSTGRES_CHILD_REPOSITORY = "E_POSTGRES_CHILD_REPOSITORY"
E_POSTGRES_CHILD_PREFLIGHT = "E_POSTGRES_CHILD_PREFLIGHT"
E_POSTGRES_CHILD_TLS = "E_POSTGRES_CHILD_TLS"
E_POSTGRES_CHILD_STARTUP = "E_POSTGRES_CHILD_STARTUP"
E_POSTGRES_CHILD_MIGRATION = "E_POSTGRES_CHILD_MIGRATION"
E_POSTGRES_CHILD_READINESS = "E_POSTGRES_CHILD_READINESS"
E_POSTGRES_CHILD_HANDOFF = "E_POSTGRES_CHILD_HANDOFF"
E_POSTGRES_CHILD_CLEANUP = "E_POSTGRES_CHILD_CLEANUP"
E_POSTGRES_CHILD_PROTECTED_RESOURCES = "E_POSTGRES_CHILD_PROTECTED_RESOURCES"
E_POSTGRES_CHILD_UNCLASSIFIED = "E_POSTGRES_CHILD_UNCLASSIFIED"
E_POSTGRES_CHILD_TIMEOUT = "E_POSTGRES_CHILD_TIMEOUT"
E_POSTGRES_CHILD_READY_EXIT = "E_POSTGRES_CHILD_READY_EXIT"
E_POSTGRES_CHILD_HANDOFF_NOT_PRODUCED = "E_POSTGRES_CHILD_HANDOFF_NOT_PRODUCED"
E_POSTGRES_CONFIG = "E_POSTGRES_CONFIG"
E_POSTGRES_DOCKER = "E_POSTGRES_DOCKER"
E_POSTGRES_LAUNCH = "E_POSTGRES_LAUNCH"
E_POSTGRES_READER = "E_POSTGRES_READER"
E_POSTGRES_CLOCK = "E_POSTGRES_CLOCK"
E_POSTGRES_POLL = "E_POSTGRES_POLL"
E_POSTGRES_EVENTS = "E_POSTGRES_EVENTS"
E_POSTGRES_HANDOFF = "E_POSTGRES_HANDOFF"
E_POSTGRES_OWNERSHIP = "E_POSTGRES_OWNERSHIP"
E_POSTGRES_UNCLASSIFIED = "E_POSTGRES_UNCLASSIFIED"
POSTGRES_CHILD_PHASES = {
    "E_REPOSITORY": E_POSTGRES_CHILD_REPOSITORY,
    "E_PREFLIGHT": E_POSTGRES_CHILD_PREFLIGHT,
    "E_TLS": E_POSTGRES_CHILD_TLS,
    "E_STARTUP": E_POSTGRES_CHILD_STARTUP,
    "E_MIGRATION": E_POSTGRES_CHILD_MIGRATION,
    "E_READINESS": E_POSTGRES_CHILD_READINESS,
    "E_HANDOFF": E_POSTGRES_CHILD_HANDOFF,
    "E_CLEANUP": E_POSTGRES_CHILD_CLEANUP,
    "E_PROTECTED_RESOURCES": E_POSTGRES_CHILD_PROTECTED_RESOURCES,
}
POSTGRES_CHILD_FAILURE_PHASES = frozenset(
    {
        *POSTGRES_CHILD_PHASES.values(),
        E_POSTGRES_CHILD_UNCLASSIFIED,
        E_POSTGRES_CHILD_TIMEOUT,
        E_POSTGRES_CHILD_READY_EXIT,
        E_POSTGRES_CHILD_HANDOFF_NOT_PRODUCED,
    }
)
POSTGRES_STARTUP_FAILURE_PHASES = frozenset(
    {
        E_POSTGRES_CONFIG,
        E_POSTGRES_DOCKER,
        E_POSTGRES_LAUNCH,
        E_POSTGRES_READER,
        E_POSTGRES_CLOCK,
        E_POSTGRES_POLL,
        E_POSTGRES_EVENTS,
        E_POSTGRES_HANDOFF,
        E_POSTGRES_OWNERSHIP,
        E_POSTGRES_UNCLASSIFIED,
    }
)
_TLS_CHILD_FAILURE_SUFFIX = " PR54 PostgreSQL TLS service failed"
_TLS_CHILD_READY_PATTERN = re.compile(
    r"PR54 PostgreSQL TLS READY; TTL remaining: [0-9]+ seconds"
)
_TLS_CHILD_LINE_LIMIT = 256
_StartupT = TypeVar("_StartupT")
COMPLETION_FAILURE_PHASES = frozenset(
    {
        E_COMPLETION_PROCESSOR_MISSING,
        E_COMPLETION_NO_AUTHORITATIVE_OUTCOME,
        E_COMPLETION_CARDINALITY,
        E_COMPLETION_BACKGROUND_FAILED,
        E_COMPLETION_NOT_PROCESSED,
        E_COMPLETION_RESULT_MISSING,
        E_COMPLETION_UNCLASSIFIED,
        E_COMPLETION_STALLED_RUN_ENTER,
        E_COMPLETION_STALLED_EMBED_ENTER,
        E_COMPLETION_STALLED_VECTOR_ENTER,
        E_COMPLETION_STALLED_PROMPT_ENTER,
        E_COMPLETION_STALLED_GATEWAY_FACTORY_ENTER,
        E_COMPLETION_STALLED_HTTP_ENTER,
        E_COMPLETION_STALLED_CALLBACK_ENTER,
        E_COMPLETION_STALLED_CALLBACK_NOT_ENTERED,
        E_COMPLETION_STALLED_FAILED,
        E_COMPLETION_STALLED_WORKER_NOT_LIVE,
        E_COMPLETION_STALLED_UNCLASSIFIED,
        *_SUBMISSION_PHASES.values(),
        E_COMPLETION_FUTURE_QUEUED,
        E_COMPLETION_FUTURE_RUNNING_NO_STAGE,
        E_COMPLETION_FUTURE_TERMINAL_NO_PUBLICATION,
        E_COMPLETION_DIAGNOSTIC_UNCLASSIFIED,
    }
)
PHASES = (
    "E_PREFLIGHT",
    "E_POSTGRES_START",
    "E_MIGRATIONS",
    "E_READINESS",
    "E_PROFILE",
    "E_MODEL",
    "E_DOCUMENT_SUBMIT",
    "E_DOCUMENT_READY",
    "E_VECTOR_SCOPE",
    "E_ORCHESTRATION",
    "E_COMPLETION_PUMP",
    "E_ADMISSION",
    "E_CITATION_PROJECTION",
    "E_DUPLICATE",
    "E_DELETE",
    "E_SCOPE_ISOLATION",
    "E_CLEANUP",
)


class DashboardRAGVLLME2EError(RuntimeError):
    """A fixed phase-only E2E failure."""

    def __init__(self, phase: str) -> None:
        self.phase = (
            phase
            if phase in PHASES
            or phase in COMPLETION_FAILURE_PHASES
            or phase in POSTGRES_CHILD_FAILURE_PHASES
            or phase in POSTGRES_STARTUP_FAILURE_PHASES
            or phase in CLEANUP_FAILURE_PHASES
            or phase in DUPLICATE_FAILURE_PHASES
            or phase in DOCUMENT_READY_FAILURE_PHASES
            else "E_PREFLIGHT"
        )
        super().__init__(self.phase)


class _CleanupPhaseError(RuntimeError):
    def __init__(self, phase: str) -> None:
        self.phase = phase
        super().__init__(phase)


class _CompletionPumpError(RuntimeError):
    def __init__(self, phase: str) -> None:
        self.phase = phase
        super().__init__(phase)


class _DuplicatePhaseError(RuntimeError):
    def __init__(self, phase: str) -> None:
        self.phase = phase
        super().__init__(phase)


_DUPLICATE_VECTOR_OPERATIONS = frozenset(
    {"SETTINGS", "CONNECT", "CURSOR", "EXECUTE", "FETCH", "RESULT_SHAPE", "CLOSE"}
)
_DUPLICATE_VECTOR_PHASES = {
    "BEFORE": {
        "SETTINGS": E_DUPLICATE_VECTOR_BEFORE_SETTINGS,
        "CONNECT": E_DUPLICATE_VECTOR_BEFORE_CONNECT,
        "CURSOR": E_DUPLICATE_VECTOR_BEFORE_CURSOR,
        "EXECUTE": E_DUPLICATE_VECTOR_BEFORE_EXECUTE,
        "FETCH": E_DUPLICATE_VECTOR_BEFORE_FETCH,
        "RESULT_SHAPE": E_DUPLICATE_VECTOR_BEFORE_RESULT_SHAPE,
        "CLOSE": E_DUPLICATE_VECTOR_BEFORE_CLOSE,
    },
    "AFTER": {
        "SETTINGS": E_DUPLICATE_VECTOR_AFTER_SETTINGS,
        "CONNECT": E_DUPLICATE_VECTOR_AFTER_CONNECT,
        "CURSOR": E_DUPLICATE_VECTOR_AFTER_CURSOR,
        "EXECUTE": E_DUPLICATE_VECTOR_AFTER_EXECUTE,
        "FETCH": E_DUPLICATE_VECTOR_AFTER_FETCH,
        "RESULT_SHAPE": E_DUPLICATE_VECTOR_AFTER_RESULT_SHAPE,
        "CLOSE": E_DUPLICATE_VECTOR_AFTER_CLOSE,
    },
}


class _DuplicateVectorOperationError(RuntimeError):
    def __init__(self, operation: str) -> None:
        if operation not in _DUPLICATE_VECTOR_OPERATIONS:
            raise ValueError
        self.operation = operation
        super().__init__(operation)


class _DocumentReadyPhaseError(RuntimeError):
    def __init__(self, phase: str) -> None:
        self.phase = phase
        super().__init__(phase)


def _stalled_completion_phase(snapshot: RAGDiagnosticSnapshot) -> str:
    if not isinstance(snapshot, RAGDiagnosticSnapshot) or snapshot.unclassified:
        return E_COMPLETION_DIAGNOSTIC_UNCLASSIFIED
    if snapshot.authoritative_completion_published:
        return E_COMPLETION_NO_AUTHORITATIVE_OUTCOME
    if snapshot.future_state is RAGDiagnosticFutureState.TERMINAL:
        return E_COMPLETION_FUTURE_TERMINAL_NO_PUBLICATION
    submission_phase = _SUBMISSION_PHASES.get(snapshot.submission_state)
    if submission_phase is not None:
        return submission_phase
    if snapshot.submission_state is not RAGDiagnosticSubmissionState.ACCEPTED:
        return E_COMPLETION_DIAGNOSTIC_UNCLASSIFIED
    if snapshot.future_state is RAGDiagnosticFutureState.QUEUED:
        return E_COMPLETION_FUTURE_QUEUED
    if (
        snapshot.future_state is RAGDiagnosticFutureState.RUNNING
        and not snapshot.events
    ):
        return E_COMPLETION_FUTURE_RUNNING_NO_STAGE
    if snapshot.future_state is RAGDiagnosticFutureState.ABSENT:
        return E_COMPLETION_DIAGNOSTIC_UNCLASSIFIED
    active: set[RAGDiagnosticStage] = set()
    last_failed = False
    try:
        for event in snapshot.events:
            if event.status is RAGDiagnosticStatus.ENTER:
                active.add(event.stage)
                last_failed = False
            elif event.status is RAGDiagnosticStatus.OK:
                active.discard(event.stage)
                last_failed = False
            elif event.status is RAGDiagnosticStatus.FAILED:
                active.discard(event.stage)
                last_failed = True
            else:
                return E_COMPLETION_STALLED_UNCLASSIFIED
    except BaseException:
        return E_COMPLETION_STALLED_UNCLASSIFIED
    if active:
        for event in reversed(snapshot.events):
            if event.stage in active:
                return _STALLED_STAGE_PHASES[event.stage]
    if last_failed:
        return E_COMPLETION_STALLED_FAILED
    if not snapshot.worker_live:
        return E_COMPLETION_STALLED_WORKER_NOT_LIVE
    return E_COMPLETION_STALLED_UNCLASSIFIED


class _PostgresChildError(RuntimeError):
    def __init__(self, phase: str) -> None:
        self.phase = (
            phase
            if phase in POSTGRES_CHILD_FAILURE_PHASES
            else E_POSTGRES_CHILD_UNCLASSIFIED
        )
        super().__init__(self.phase)


class _PostgresStartupError(RuntimeError):
    def __init__(self, phase: str) -> None:
        self.phase = (
            phase
            if phase in POSTGRES_STARTUP_FAILURE_PHASES
            else E_POSTGRES_UNCLASSIFIED
        )
        super().__init__(self.phase)


@dataclass(frozen=True, slots=True)
class _TLSChildOutputEvent:
    kind: str
    phase: str | None = None


@dataclass(frozen=True, slots=True)
class _WindowsProcessObservation:
    process_id: int
    parent_process_id: int
    executable_path: str | None = field(repr=False)
    command_line: str | None = field(repr=False)
    creation_time_utc: str | None = field(repr=False)


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    branch: str
    head: str
    baseline: str
    handoff_root: Path = field(repr=False)
    provider: KnowledgeBaseRAGProviderSettings = field(repr=False)
    policy: RAGCoachingIntegrationPolicy = field(repr=False)
    vllm: VLLMOpenAICompatibleSettings = field(repr=False)
    ttl_seconds: int


@dataclass(frozen=True, slots=True)
class PostgreSQLStartupConfig:
    branch: str
    head: str
    baseline: str
    handoff_root: Path = field(repr=False)
    ttl_seconds: int


class LifecycleOperations(Protocol):
    def run_phase(self, phase: str) -> None: ...

    def cleanup(self) -> None: ...


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if value is None or not value or value != value.strip():
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")
    return value


def _strict_float(environment: Mapping[str, str], name: str) -> float:
    raw = _required(environment, name)
    try:
        value = float(raw)
    except ValueError:
        raise DashboardRAGVLLME2EError("E_PREFLIGHT") from None
    if raw.lower() in {"true", "false", "nan", "inf", "+inf", "-inf"}:
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")
    return value


def _strict_int(environment: Mapping[str, str], name: str) -> int:
    raw = _required(environment, name)
    if not raw.isascii() or not raw.isdigit():
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")
    return int(raw)


def _read_json(path_value: str, expected: frozenset[str]) -> dict[str, object]:
    try:
        path = Path(path_value)
        if (
            not path.is_absolute()
            or ".." in path.parts
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > 65_536
        ):
            raise ValueError
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError
        return payload
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        raise DashboardRAGVLLME2EError("E_PREFLIGHT") from None


def _git_output(arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=False,
            shell=False,
            timeout=10,
        )
        if max(len(result.stdout), len(result.stderr)) > _MAX_GIT_OUTPUT_BYTES:
            raise ValueError
        return result.stdout.decode("utf-8", errors="strict").strip()
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError):
        raise DashboardRAGVLLME2EError("E_PREFLIGHT") from None


def preflight(environment: Mapping[str, str] | None = None) -> ControllerConfig:
    try:
        return _preflight(environment)
    except DashboardRAGVLLME2EError:
        raise
    except Exception:
        raise DashboardRAGVLLME2EError("E_PREFLIGHT") from None


def postgres_preflight(
    environment: Mapping[str, str] | None = None,
) -> PostgreSQLStartupConfig:
    try:
        return _postgres_preflight(environment)
    except DashboardRAGVLLME2EError:
        raise
    except Exception:
        raise DashboardRAGVLLME2EError("E_PREFLIGHT") from None


def _postgres_preflight(
    environment: Mapping[str, str] | None = None,
) -> PostgreSQLStartupConfig:
    source = os.environ if environment is None else environment
    if sys.platform != "win32" or Path.cwd().resolve() != REPOSITORY_ROOT:
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")
    branch = _required(source, BRANCH_ENV)
    head = _required(source, HEAD_ENV)
    baseline = _required(source, BASELINE_ENV)
    if (
        not BRANCH_PATTERN.fullmatch(branch)
        or ".." in branch
        or branch.endswith("/")
        or not COMMIT_PATTERN.fullmatch(head)
        or not COMMIT_PATTERN.fullmatch(baseline)
    ):
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")
    if (
        _git_output(["branch", "--show-current"]) != branch
        or _git_output(["rev-parse", "HEAD"]) != head
        or _git_output(["rev-parse", f"origin/{branch}"]) != head
        or _git_output(["status", "--porcelain=v1", "--untracked-files=all"])
    ):
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")
    _git_output(["merge-base", "--is-ancestor", baseline, head])
    if (
        shutil.which("docker") is None
        or not (REPOSITORY_ROOT / "compose.postgres-tls-smoke.yml").is_file()
    ):
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")

    handoff_root = Path(_required(source, HANDOFF_ROOT_ENV))
    if (
        not handoff_root.is_absolute()
        or ".." in handoff_root.parts
        or handoff_root.is_symlink()
        or not handoff_root.is_dir()
        or REPOSITORY_ROOT in handoff_root.resolve(strict=True).parents
    ):
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")
    raw_ttl = _required(source, TTL_ENV)
    if not raw_ttl.isascii() or not raw_ttl.isdigit():
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")
    ttl = int(raw_ttl)
    if not MINIMUM_TTL_SECONDS <= ttl <= MAXIMUM_TTL_SECONDS:
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")
    return PostgreSQLStartupConfig(branch, head, baseline, handoff_root, ttl)


def _preflight(environment: Mapping[str, str] | None = None) -> ControllerConfig:
    source = os.environ if environment is None else environment
    postgres = _postgres_preflight(source)
    if source.get(TTL_ENV) != str(FULL_E2E_TTL_SECONDS):
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")
    provider = KnowledgeBaseRAGProviderSettings.model_validate(
        _read_json(
            _required(source, PROVIDER_ENV),
            frozenset(
                {
                    "tenant_id",
                    "knowledge_base_id",
                    "model_id",
                    "model_name_or_path",
                    "vector_dimension",
                    "normalize_embeddings",
                    "device",
                    "local_files_only",
                }
            ),
        )
    )
    if (
        provider.model_id != MINILM_MODEL
        or provider.vector_dimension != MINILM_DIMENSION
        or provider.normalize_embeddings is not True
        or provider.device != "cpu"
        or provider.local_files_only is not True
        or provider.model_name_or_path == MINILM_MODEL
    ):
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")
    validate_local_minilm_snapshot(provider.model_name_or_path)
    policy = RAGCoachingIntegrationPolicy.model_validate(
        _read_json(
            _required(source, POLICY_ENV),
            frozenset(
                {
                    "rag_llm_enabled_labels",
                    "title",
                    "action",
                    "priority",
                    "label_id",
                    "expires_after_seconds",
                }
            ),
        )
    )
    from pydantic import SecretStr

    try:
        verify_tls = _required(source, "CALLMETRIC_VLLM_VERIFY_TLS")
        if verify_tls != "true":
            raise ValueError
        vllm = VLLMOpenAICompatibleSettings(
            base_url=_required(source, "CALLMETRIC_VLLM_BASE_URL"),
            model_id=_required(source, "CALLMETRIC_VLLM_MODEL_ID"),
            api_token=SecretStr(_required(source, TOKEN_ENV)),
            ca_certificate_path=SecretStr(_required(source, CA_ENV)),
            connect_timeout_seconds=_strict_float(
                source, "CALLMETRIC_VLLM_CONNECT_TIMEOUT_SECONDS"
            ),
            read_timeout_seconds=_strict_float(
                source, "CALLMETRIC_VLLM_READ_TIMEOUT_SECONDS"
            ),
            max_output_tokens=_strict_int(source, "CALLMETRIC_VLLM_MAX_OUTPUT_TOKENS"),
            temperature=_strict_float(source, "CALLMETRIC_VLLM_TEMPERATURE"),
            verify_tls=True,
        )
        if vllm.max_output_tokens < MINIMUM_E2E_OUTPUT_TOKENS:
            raise ValueError
    except (ValueError, TypeError):
        raise DashboardRAGVLLME2EError("E_PREFLIGHT") from None
    parsed = urlsplit(vllm.base_url)
    ca_path = Path(_required(source, CA_ENV))
    if (
        parsed.scheme != "https"
        or parsed.hostname != "localhost"
        or parsed.port is None
        or parsed.path != "/v1"
        or vllm.verify_tls is not True
        or vllm.api_token is None
        or not ca_path.is_absolute()
        or ca_path.is_symlink()
        or not ca_path.is_file()
    ):
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")
    return ControllerConfig(
        postgres.branch,
        postgres.head,
        postgres.baseline,
        postgres.handoff_root,
        provider,
        policy,
        vllm,
        postgres.ttl_seconds,
    )


def run(
    *,
    preflight_only: bool,
    environment: Mapping[str, str] | None = None,
    operations_factory: Callable[[ControllerConfig], LifecycleOperations] | None = None,
) -> str:
    config = preflight(environment)
    if preflight_only:
        return PREFLIGHT_OK
    operations = (
        _ProductionLifecycle(config, os.environ if environment is None else environment)
        if operations_factory is None
        else operations_factory(config)
    )
    functional_primary_error: BaseException | None = None
    try:
        for phase in PHASES[1:-1]:
            try:
                operations.run_phase(phase)
            except BaseException as error:
                if phase == "E_POSTGRES_START" and isinstance(
                    error, (_PostgresChildError, _PostgresStartupError)
                ):
                    raise DashboardRAGVLLME2EError(error.phase) from None
                if phase == "E_POSTGRES_START":
                    raise DashboardRAGVLLME2EError(E_POSTGRES_UNCLASSIFIED) from None
                if phase == "E_COMPLETION_PUMP":
                    completion_phase = (
                        error.phase
                        if isinstance(error, _CompletionPumpError)
                        and error.phase in COMPLETION_FAILURE_PHASES
                        else E_COMPLETION_UNCLASSIFIED
                    )
                    raise DashboardRAGVLLME2EError(completion_phase) from None
                if phase == "E_DOCUMENT_READY":
                    document_ready_phase = (
                        error.phase
                        if isinstance(error, _DocumentReadyPhaseError)
                        and error.phase in DOCUMENT_READY_FAILURE_PHASES
                        else E_DOCUMENT_READY_UNCLASSIFIED
                    )
                    raise DashboardRAGVLLME2EError(document_ready_phase) from None
                if phase == "E_DUPLICATE":
                    duplicate_phase = (
                        error.phase
                        if isinstance(error, _DuplicatePhaseError)
                        and error.phase in DUPLICATE_FAILURE_PHASES
                        else E_DUPLICATE_UNCLASSIFIED
                    )
                    raise DashboardRAGVLLME2EError(duplicate_phase) from None
                raise DashboardRAGVLLME2EError(phase) from None
    except BaseException as error:
        functional_primary_error = error
    try:
        operations.cleanup()
    except BaseException as error:
        if functional_primary_error is None:
            cleanup_phase = (
                error.phase
                if isinstance(error, _CleanupPhaseError)
                and error.phase in CLEANUP_FAILURE_PHASES
                else "E_CLEANUP"
            )
            functional_primary_error = DashboardRAGVLLME2EError(cleanup_phase)
    if functional_primary_error is not None:
        raise functional_primary_error
    return E2E_OK


def run_postgres_startup_only(
    *,
    environment: Mapping[str, str] | None = None,
    operations_factory: (
        Callable[[PostgreSQLStartupConfig], LifecycleOperations] | None
    ) = None,
) -> str:
    config = postgres_preflight(environment)
    operations = (
        _ProductionLifecycle(config, os.environ if environment is None else environment)
        if operations_factory is None
        else operations_factory(config)
    )
    functional_primary_error: BaseException | None = None
    try:
        operations.run_phase("E_POSTGRES_START")
    except (_PostgresChildError, _PostgresStartupError) as error:
        functional_primary_error = DashboardRAGVLLME2EError(error.phase)
    except BaseException:
        functional_primary_error = DashboardRAGVLLME2EError(E_POSTGRES_UNCLASSIFIED)
    try:
        operations.cleanup()
    except BaseException as error:
        if functional_primary_error is None:
            cleanup_phase = (
                error.phase
                if isinstance(error, _CleanupPhaseError)
                and error.phase in CLEANUP_FAILURE_PHASES
                else "E_CLEANUP"
            )
            functional_primary_error = DashboardRAGVLLME2EError(cleanup_phase)
    if functional_primary_error is not None:
        raise functional_primary_error
    return POSTGRES_STARTUP_OK


class _ProductionLifecycle:
    """Stateful adapter around existing production boundaries."""

    def __init__(
        self,
        config: ControllerConfig | PostgreSQLStartupConfig,
        environment: Mapping[str, str],
    ) -> None:
        self._postgres_config = PostgreSQLStartupConfig(
            config.branch,
            config.head,
            config.baseline,
            config.handoff_root,
            config.ttl_seconds,
        )
        self._config = config if isinstance(config, ControllerConfig) else None
        self._environment = dict(environment)
        self._service: subprocess.Popen[bytes] | None = None
        self._postgres_project: str | None = None
        self._handoff: Path | None = None
        self._postgres_settings: PostgreSQLVectorStoreSettings | None = None
        self._document_runtime: PostgreSQLDocumentIngestionRuntime | None = None
        self._rag_manager: BoundedPostgreSQLRAGManager | None = None
        self._rag_diagnostic_observer: BoundedRAGDiagnosticObserver | None = None
        self._target_entry: DocumentRegistryEntry | None = None
        self._other_entry: DocumentRegistryEntry | None = None
        self._outcome: StableCoachingOutcome | None = None
        self._processor: RAGCoachingProcessorDecorator | None = None
        self._vector_count = 0
        self._protected_resources: object | None = None
        self._protected_handoff_entries: frozenset[Path] | None = None
        self._owned_processes: dict[int, int] = {}
        self._owner_marker: str | None = None
        self._postgres_launch_boundary: datetime | None = None
        self._tls_child_events: queue.SimpleQueue[_TLSChildOutputEvent] = (
            queue.SimpleQueue()
        )
        self._tls_child_output_thread: threading.Thread | None = None

    def run_phase(self, phase: str) -> None:
        getattr(self, f"_{phase.removeprefix('E_').lower()}")()

    def _full_config(self) -> ControllerConfig:
        if self._config is None:
            raise RuntimeError
        return self._config

    def _postgres_start(self) -> None:
        try:
            self._start_postgres_service()
        except (_PostgresChildError, _PostgresStartupError):
            raise
        except BaseException:
            raise _PostgresStartupError(E_POSTGRES_UNCLASSIFIED) from None

    @staticmethod
    def _postgres_startup_call(
        phase: str, operation: Callable[[], _StartupT]
    ) -> _StartupT:
        try:
            return operation()
        except (_PostgresChildError, _PostgresStartupError):
            raise
        except BaseException:
            raise _PostgresStartupError(phase) from None

    def _start_postgres_service(self) -> None:
        try:
            from scripts.run_postgres_tls_service import (
                CERTIFICATE_TIMEOUT_SECONDS,
                COMPOSE_CONFIG_TIMEOUT_SECONDS,
                COMPOSE_STARTUP_TIMEOUT_SECONDS,
                IDENTITY_ACL_TIMEOUT_SECONDS,
                MIGRATION_PROOF_TIMEOUT_SECONDS,
                READINESS_COMMAND_TIMEOUT_SECONDS,
                VALIDATION_TIMEOUT_SECONDS,
                snapshot_protected_resources,
            )
        except BaseException:
            raise _PostgresStartupError(E_POSTGRES_CONFIG) from None

        before = self._postgres_startup_call(
            E_POSTGRES_CONFIG,
            lambda: set(self._postgres_config.handoff_root.iterdir()),
        )
        self._protected_handoff_entries = frozenset(before)
        docker = self._postgres_startup_call(
            E_POSTGRES_DOCKER, lambda: shutil.which("docker")
        )
        if docker is None:
            raise _PostgresStartupError(E_POSTGRES_DOCKER)
        self._protected_resources = self._postgres_startup_call(
            E_POSTGRES_DOCKER, lambda: snapshot_protected_resources(docker)
        )
        environment = self._postgres_startup_call(
            E_POSTGRES_CONFIG, lambda: dict(self._environment)
        )
        environment["CALLMETRIC_POSTGRES_TLS_SERVICE_EXPECTED_BRANCH"] = (
            self._postgres_config.branch
        )
        environment["CALLMETRIC_POSTGRES_TLS_SERVICE_EXPECTED_HEAD"] = (
            self._postgres_config.head
        )
        project = self._postgres_startup_call(
            E_POSTGRES_CONFIG,
            lambda: f"callmetric-pgvector-tls-{os.getpid()}-{secrets.token_hex(6)}",
        )
        if not re.fullmatch(r"callmetric-pgvector-tls-[0-9]+-[a-f0-9]{12}", project):
            raise _PostgresStartupError(E_POSTGRES_CONFIG)
        self._postgres_project = project
        environment["CALLMETRIC_POSTGRES_TLS_SERVICE_PROJECT_NAME"] = project
        owner_marker = self._postgres_startup_call(
            E_POSTGRES_CONFIG, lambda: f"callmetric-owner-{secrets.token_hex(16)}"
        )
        if not OWNER_MARKER_PATTERN.fullmatch(owner_marker):
            raise _PostgresStartupError(E_POSTGRES_CONFIG)
        self._owner_marker = owner_marker
        self._postgres_launch_boundary = self._postgres_startup_call(
            E_POSTGRES_CLOCK, lambda: datetime.now(timezone.utc)
        )
        self._service = self._postgres_startup_call(
            E_POSTGRES_LAUNCH,
            lambda: subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scripts.run_postgres_tls_service",
                    "--ttl-seconds",
                    str(self._postgres_config.ttl_seconds),
                    "--owner-marker",
                    owner_marker,
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            ),
        )
        output = self._service.stdout
        if output is None:
            raise _PostgresStartupError(E_POSTGRES_READER)
        self._tls_child_output_thread = self._postgres_startup_call(
            E_POSTGRES_READER,
            lambda: threading.Thread(
                target=self._read_tls_child_output,
                args=(output,),
                name="postgres-tls-safe-output",
                daemon=True,
            ),
        )
        self._postgres_startup_call(
            E_POSTGRES_READER, self._tls_child_output_thread.start
        )
        validation_command_count = 11
        startup_timeout = (
            validation_command_count * VALIDATION_TIMEOUT_SECONDS
            + COMPOSE_CONFIG_TIMEOUT_SECONDS
            + 2 * CERTIFICATE_TIMEOUT_SECONDS
            + COMPOSE_STARTUP_TIMEOUT_SECONDS
            + 2 * READINESS_COMMAND_TIMEOUT_SECONDS
            + MIGRATION_PROOF_TIMEOUT_SECONDS
            + 2 * IDENTITY_ACL_TIMEOUT_SECONDS
            + 60.0
        )
        deadline = self._postgres_startup_call(
            E_POSTGRES_CLOCK, lambda: time.monotonic() + startup_timeout
        )
        ready_count = 0
        failure_phases: list[str] = []
        malformed = False
        while self._postgres_startup_call(E_POSTGRES_CLOCK, time.monotonic) < deadline:
            child_running = (
                self._postgres_startup_call(E_POSTGRES_POLL, self._service.poll) is None
            )
            for event in self._postgres_startup_call(
                E_POSTGRES_EVENTS, self._drain_tls_child_events
            ):
                if event.kind == "ready":
                    ready_count += 1
                elif event.kind == "failure" and event.phase is not None:
                    failure_phases.append(event.phase)
                else:
                    malformed = True
            if not child_running:
                self._postgres_startup_call(
                    E_POSTGRES_READER, self._join_tls_child_output_thread
                )
                for event in self._postgres_startup_call(
                    E_POSTGRES_EVENTS, self._drain_tls_child_events
                ):
                    if event.kind == "ready":
                        ready_count += 1
                    elif event.kind == "failure" and event.phase is not None:
                        failure_phases.append(event.phase)
                    else:
                        malformed = True
                raise _PostgresChildError(
                    self._classify_exited_tls_child(
                        ready_count, tuple(failure_phases), malformed
                    )
                )
            if malformed or ready_count > 1 or len(failure_phases) > 1:
                raise _PostgresChildError(E_POSTGRES_CHILD_UNCLASSIFIED)
            if failure_phases and ready_count:
                raise _PostgresChildError(E_POSTGRES_CHILD_UNCLASSIFIED)
            if len(failure_phases) == 1:
                raise _PostgresChildError(failure_phases[0])
            created = self._postgres_startup_call(
                E_POSTGRES_HANDOFF,
                lambda: set(self._postgres_config.handoff_root.iterdir()) - before,
            )
            ready = self._postgres_startup_call(
                E_POSTGRES_HANDOFF,
                lambda: [
                    item for item in created if (item / "application.dsn").is_file()
                ],
            )
            if self._tls_child_startup_ready(
                ready_count=ready_count,
                failure_count=len(failure_phases),
                malformed=malformed,
                handoff_count=len(ready),
                child_running=child_running,
            ):
                self._handoff = ready[0]
                self._postgres_startup_call(
                    E_POSTGRES_OWNERSHIP, self._refresh_owned_process_ledger
                )
                return
            self._postgres_startup_call(
                E_POSTGRES_CLOCK, lambda: time.sleep(POLL_INTERVAL_SECONDS)
            )
        raise _PostgresChildError(
            self._classify_tls_child_timeout(
                ready_count, tuple(failure_phases), malformed
            )
        )

    def _read_tls_child_output(self, stream: BinaryIO) -> None:
        try:
            while raw_line := stream.readline(_TLS_CHILD_LINE_LIMIT + 1):
                if len(raw_line) > _TLS_CHILD_LINE_LIMIT:
                    self._tls_child_events.put(_TLSChildOutputEvent("malformed"))
                    continue
                try:
                    line = raw_line.decode("utf-8", errors="strict").rstrip("\r\n")
                except UnicodeDecodeError:
                    self._tls_child_events.put(_TLSChildOutputEvent("malformed"))
                    continue
                if _TLS_CHILD_READY_PATTERN.fullmatch(line):
                    self._tls_child_events.put(_TLSChildOutputEvent("ready"))
                    continue
                child_phase = next(
                    (
                        parent_phase
                        for phase, parent_phase in POSTGRES_CHILD_PHASES.items()
                        if line == f"{phase}{_TLS_CHILD_FAILURE_SUFFIX}"
                    ),
                    None,
                )
                if child_phase is None:
                    self._tls_child_events.put(_TLSChildOutputEvent("malformed"))
                else:
                    self._tls_child_events.put(
                        _TLSChildOutputEvent("failure", child_phase)
                    )
        except BaseException:
            self._tls_child_events.put(_TLSChildOutputEvent("malformed"))

    def _drain_tls_child_events(self) -> tuple[_TLSChildOutputEvent, ...]:
        events: list[_TLSChildOutputEvent] = []
        while True:
            try:
                events.append(self._tls_child_events.get_nowait())
            except queue.Empty:
                return tuple(events)

    @staticmethod
    def _classify_exited_tls_child(
        ready_count: int, failure_phases: tuple[str, ...], malformed: bool
    ) -> str:
        if malformed or ready_count > 1 or len(failure_phases) > 1:
            return E_POSTGRES_CHILD_UNCLASSIFIED
        if ready_count == 0 and len(failure_phases) == 1:
            return failure_phases[0]
        if ready_count == 1 and not failure_phases:
            return E_POSTGRES_CHILD_READY_EXIT
        return E_POSTGRES_CHILD_UNCLASSIFIED

    @staticmethod
    def _tls_child_startup_ready(
        *,
        ready_count: int,
        failure_count: int,
        malformed: bool,
        handoff_count: int,
        child_running: bool,
    ) -> bool:
        return (
            ready_count == 1
            and failure_count == 0
            and not malformed
            and handoff_count == 1
            and child_running
        )

    @staticmethod
    def _classify_tls_child_timeout(
        ready_count: int, failure_phases: tuple[str, ...], malformed: bool
    ) -> str:
        if malformed or ready_count > 1 or len(failure_phases) > 1:
            return E_POSTGRES_CHILD_UNCLASSIFIED
        if ready_count == 0 and len(failure_phases) == 1:
            return failure_phases[0]
        if ready_count == 1 and not failure_phases:
            return E_POSTGRES_CHILD_HANDOFF_NOT_PRODUCED
        return E_POSTGRES_CHILD_TIMEOUT

    def _join_tls_child_output_thread(self) -> None:
        thread = self._tls_child_output_thread
        if thread is None:
            return
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise _PostgresChildError(E_POSTGRES_CHILD_UNCLASSIFIED)

    def _application_dsn(self) -> str:
        if self._handoff is None:
            raise RuntimeError
        return (self._handoff / "application.dsn").read_text(encoding="utf-8").strip()

    def _migrations(self) -> None:
        from psycopg import connect

        connection = connect(self._application_dsn(), autocommit=False)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT version, count(*) FROM callmetric_vector.schema_migrations "
                    "GROUP BY version ORDER BY version"
                )
                if cursor.fetchall() != [("0001", 1), ("0002", 1), ("0003", 1)]:
                    raise RuntimeError
        finally:
            connection.close()

    def _settings(self) -> PostgreSQLVectorStoreSettings:
        from app.composition.postgres_rag import PostgreSQLVectorStoreSettings
        from pydantic import SecretStr

        if self._postgres_settings is None:
            self._postgres_settings = PostgreSQLVectorStoreSettings(
                dsn=SecretStr(self._application_dsn()),
                connect_timeout_seconds=5,
                ssl_mode="verify-full",
                application_name="dashboard-rag-e2e",
            )
        settings = self._postgres_settings
        if settings is None:
            raise RuntimeError
        return settings

    def _readiness(self) -> None:
        from app.vector_store.postgres.readiness import PostgreSQLSchemaReadinessChecker
        from psycopg import connect

        settings = self._settings()
        PostgreSQLSchemaReadinessChecker(
            connection_factory=lambda: connect(
                settings.dsn.get_secret_value(),
                connect_timeout=settings.connect_timeout_seconds,
                sslmode=settings.ssl_mode,
                application_name=settings.application_name,
                autocommit=False,
            )
        ).verify()

    def _profile(self) -> None:
        from app.deployment.postgres_rag import provision_profile_bound_postgres_rag
        from psycopg import connect

        provision_profile_bound_postgres_rag(
            postgres_settings=self._settings(),
            knowledge_base_settings=self._full_config().provider,
            psycopg_connect=connect,
        )

    def _model(self) -> None:
        from app.composition.postgres_rag import compose_profile_bound_postgres_rag
        from psycopg import connect

        composition = compose_profile_bound_postgres_rag(
            postgres_settings=self._settings(),
            knowledge_base_settings=self._full_config().provider,
            psycopg_connect=connect,
        )
        vector = composition.embedder.embed_query(
            tenant_id=self._full_config().provider.tenant_id,
            knowledge_base_id=self._full_config().provider.knowledge_base_id,
            text="Synthetic bounded product question.",
        )
        norm = math.sqrt(sum(value * value for value in vector))
        if (
            len(vector) != 384
            or not all(math.isfinite(value) for value in vector)
            or not math.isclose(norm, 1.0, abs_tol=0.00001)
        ):
            raise RuntimeError

    def _document_submit(self) -> None:
        from app.composition.postgres_document_ingestion import (
            PostgreSQLDocumentIngestionSettings,
            compose_postgres_document_ingestion,
        )
        from app.ingestion.document_background import DocumentSubmissionStatus
        from psycopg import connect

        runtime = compose_postgres_document_ingestion(
            postgres_settings=self._settings(),
            knowledge_base_settings=self._full_config().provider,
            ingestion_settings=PostgreSQLDocumentIngestionSettings(capacity=2),
            psycopg_connect=connect,
        )
        self._document_runtime = runtime
        for token, filename, content in (
            (
                "target",
                "synthetic-guide.txt",
                b"Synthetic bounded product return guidance.",
            ),
            ("other", "synthetic-other.txt", b"Synthetic isolated reference material."),
        ):
            result = runtime.manager.submit(
                submission_token=token,
                content=content,
                original_filename=filename,
                declared_media_type="text/plain",
            )
            if result.status is not DocumentSubmissionStatus.ACCEPTED:
                raise RuntimeError

    def _document_ready(self) -> None:
        from app.ingestion.registry_models import (
            DocumentReadiness,
            derive_document_readiness,
        )

        runtime = self._document_runtime
        if runtime is None:
            raise _DocumentReadyPhaseError(E_DOCUMENT_READY_RUNTIME)
        try:
            deadline = time.monotonic() + DOCUMENT_POLL_TIMEOUT_SECONDS
        except Exception:
            raise _DocumentReadyPhaseError(E_DOCUMENT_READY_CLOCK) from None
        while True:
            try:
                now = time.monotonic()
            except Exception:
                raise _DocumentReadyPhaseError(E_DOCUMENT_READY_CLOCK) from None
            if now >= deadline:
                raise _DocumentReadyPhaseError(E_DOCUMENT_READY_TIMEOUT)
            try:
                entries = runtime.registry.list_documents(
                    tenant_id=runtime.tenant_id,
                    knowledge_base_id=runtime.knowledge_base_id,
                )
            except Exception:
                raise _DocumentReadyPhaseError(E_DOCUMENT_READY_LIST) from None
            if type(entries) is not tuple or any(
                not isinstance(item, DocumentRegistryEntry) for item in entries
            ):
                raise _DocumentReadyPhaseError(E_DOCUMENT_READY_RESULT_SHAPE)
            if len(entries) != 2:
                raise _DocumentReadyPhaseError(E_DOCUMENT_READY_CARDINALITY)
            target_matches = tuple(
                item
                for item in entries
                if item.document.original_filename == "synthetic-guide.txt"
            )
            other_matches = tuple(
                item
                for item in entries
                if item.document.original_filename == "synthetic-other.txt"
            )
            if len(target_matches) > 1 or len(other_matches) > 1:
                raise _DocumentReadyPhaseError(E_DOCUMENT_READY_RESULT_SHAPE)
            if not target_matches:
                raise _DocumentReadyPhaseError(E_DOCUMENT_READY_TARGET_MISSING)
            if not other_matches:
                raise _DocumentReadyPhaseError(E_DOCUMENT_READY_OTHER_MISSING)
            target = target_matches[0]
            other = other_matches[0]

            def require_consistent(entry: DocumentRegistryEntry, *, phase: str) -> None:
                try:
                    consistent = (
                        entry.document.tenant_id == entry.job.tenant_id
                        and entry.document.knowledge_base_id
                        == entry.job.knowledge_base_id
                        and entry.document.document_id == entry.job.document_id
                        and entry.readiness
                        is derive_document_readiness(entry.document, entry.job)
                    )
                except Exception:
                    raise _DocumentReadyPhaseError(phase) from None
                if not consistent:
                    raise _DocumentReadyPhaseError(phase)

            require_consistent(target, phase=E_DOCUMENT_READY_TARGET_JOB)
            require_consistent(other, phase=E_DOCUMENT_READY_OTHER_JOB)
            if (
                target.document.storage_object_key is not None
                or other.document.storage_object_key is not None
            ):
                raise _DocumentReadyPhaseError(E_DOCUMENT_READY_SOURCE_KEY)
            if target.readiness is DocumentReadiness.FAILED:
                raise _DocumentReadyPhaseError(E_DOCUMENT_READY_TARGET_FAILED)
            if other.readiness is DocumentReadiness.FAILED:
                raise _DocumentReadyPhaseError(E_DOCUMENT_READY_OTHER_FAILED)
            if target.readiness is DocumentReadiness.CANCELLED:
                raise _DocumentReadyPhaseError(E_DOCUMENT_READY_TARGET_CANCELLED)
            if other.readiness is DocumentReadiness.CANCELLED:
                raise _DocumentReadyPhaseError(E_DOCUMENT_READY_OTHER_CANCELLED)
            if (
                target.readiness is DocumentReadiness.READY
                and other.readiness is DocumentReadiness.READY
            ):
                self._target_entry = target
                self._other_entry = other
                return
            try:
                time.sleep(POLL_INTERVAL_SECONDS)
            except Exception:
                raise _DocumentReadyPhaseError(E_DOCUMENT_READY_SLEEP) from None

    def _vector_scope(self) -> None:
        from psycopg import connect

        entry = self._target_entry
        if entry is None:
            raise RuntimeError
        connection = connect(self._application_dsn(), autocommit=False)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT count(*), bool_and(vector_dims(embedding) = 384), "
                    "bool_and(abs(1 - sqrt(-(embedding <#> embedding))) <= 0.00001) "
                    "FROM callmetric_vector.vector_records WHERE tenant_id = %s "
                    "AND knowledge_base_id = %s AND document_id = %s",
                    (
                        entry.document.tenant_id,
                        entry.document.knowledge_base_id,
                        entry.document.document_id,
                    ),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RuntimeError
                count, dimensions, normalized = row
                if not count or dimensions is not True or normalized is not True:
                    raise RuntimeError
                self._vector_count = count
        finally:
            connection.close()

    def _orchestration(self) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from app.calls.models import CallState
        from app.coaching.llm_result_gate import coaching_wire_json_schema
        from app.coaching.coordinator import CoachingCoordinator
        from app.coaching.rule_engine import RuleBasedCoachingEngine
        from app.composition.postgres_rag_background import BoundedPostgreSQLRAGManager
        from app.composition.postgres_rag_orchestration import (
            compose_profile_bound_postgres_rag_orchestration,
        )
        from app.composition.postgres_rag_runtime import (
            ProfileVerifiedPostgreSQLRAGRunner,
        )
        from app.events.models import (
            ClassificationLabel,
            ClassificationResultEvent,
            CoachingAction,
            TranscriptEvent,
            TranscriptKind,
        )
        from app.integration.citation_projection import SafeCoachingCitationProjector
        from app.integration.composition import (
            RAGCoachingIntegrationDependencies,
            compose_rag_coaching_processor,
        )
        from app.llm.vllm_openai_compatible import VLLMOpenAICompatibleGateway
        from app.tenancy.models import TenantConfig
        from app.vector_store.postgres.readiness import PostgreSQLSchemaReadinessChecker
        from live_dashboard.demo_data import tenant_demos
        from psycopg import connect

        controller_config = self._full_config()
        provider = controller_config.provider
        observer = BoundedRAGDiagnosticObserver()
        self._rag_diagnostic_observer = observer
        composition = compose_profile_bound_postgres_rag_orchestration(
            postgres_settings=self._settings(),
            knowledge_base_settings=provider,
            psycopg_connect=connect,
            llm_gateway_factory=lambda: VLLMOpenAICompatibleGateway(
                controller_config.vllm,
                structured_output_json_schema=coaching_wire_json_schema(),
            ),
            diagnostic_observer=observer,
        )
        settings = self._settings()
        readiness = PostgreSQLSchemaReadinessChecker(
            connection_factory=lambda: connect(
                settings.dsn.get_secret_value(),
                sslmode=settings.ssl_mode,
                autocommit=False,
            )
        )
        runner = ProfileVerifiedPostgreSQLRAGRunner(composition, readiness)
        runner.prepare()
        manager = BoundedPostgreSQLRAGManager(
            runner=runner,
            max_workers=1,
            capacity=2,
            diagnostic_observer=observer,
        )
        manager.start()
        self._rag_manager = manager
        demo = tenant_demos()[provider.tenant_id]
        tenant_config = demo.config.model_copy(deep=True)
        tenant_config.rag = tenant_config.rag.model_copy(
            update={
                "enabled": True,
                "knowledge_base_id": provider.knowledge_base_id,
                "top_k": 3,
                "minimum_score": 0.0,
            }
        )
        tenant_config.coaching = tenant_config.coaching.model_copy(
            update={"enable_llm": True, "cooldown_seconds": 0.0}
        )
        tenant_config = TenantConfig.model_validate(tenant_config.model_dump())
        now = datetime.now(UTC)
        event = TranscriptEvent(
            tenant_id=provider.tenant_id,
            call_id="synthetic-call",
            event_id="synthetic-event",
            kind=TranscriptKind.STABLE,
            text="Synthetic bounded product question.",
            start_seconds=0,
            end_seconds=1,
            revision=1,
            created_at_utc=now,
        )
        classification = ClassificationResultEvent(
            tenant_id=provider.tenant_id,
            call_id=event.call_id,
            transcript_event_id=event.event_id,
            labels=[
                ClassificationLabel(
                    name=controller_config.policy.rag_llm_enabled_labels[0], score=1.0
                )
            ],
            action=CoachingAction.RAG_ACTION,
            model_id="synthetic-fixed-context",
            created_at_utc=now,
        )
        state = CallState(tenant_id=provider.tenant_id, call_id=event.call_id)
        state.apply_transcript(event)
        state.apply_classification(
            classification, transcript_revision=1, source_sequence=None
        )
        coordinator = CoachingCoordinator(
            tenant_config,
            state,
            RuleBasedCoachingEngine(tenant_config, demo.rules),
        )
        runtime = self._document_runtime
        if runtime is None:
            raise RuntimeError
        processor = compose_rag_coaching_processor(
            coordinator=coordinator,
            tenant_config=tenant_config,
            integration=RAGCoachingIntegrationDependencies(
                background_manager=manager,
                policy=controller_config.policy,
                suggestion_id_factory=lambda: uuid4().hex,
                utc_datetime_factory=lambda: datetime.now(UTC),
                citation_projector=SafeCoachingCitationProjector(runtime.registry),
            ),
        )
        if not isinstance(processor, RAGCoachingProcessorDecorator):
            raise RuntimeError
        processor.process_safely(
            event,
            1.0,
            classification_event=classification,
            active_labels=(controller_config.policy.rag_llm_enabled_labels[0],),
        )
        self._processor = processor

    def _admission(self) -> None:
        outcome = self._outcome
        if (
            outcome is None
            or outcome.result is None
            or len(outcome.result.displayed_suggestions) != 1
        ):
            raise RuntimeError

    def _completion_pump(self) -> None:
        processor = self._processor
        if processor is None:
            raise _CompletionPumpError(E_COMPLETION_PROCESSOR_MISSING)
        deadline = time.monotonic() + self._completion_timeout_seconds()
        while time.monotonic() < deadline:
            completed = processor.drain_completed(current_seconds=1.0)
            if completed:
                if len(completed) != 1:
                    raise _CompletionPumpError(E_COMPLETION_CARDINALITY)
                outcome = completed[0]
                if not isinstance(outcome, StableCoachingOutcome):
                    raise _CompletionPumpError(E_COMPLETION_NOT_PROCESSED)
                if outcome.status is CoachingProcessingStatus.FAILED:
                    raise _CompletionPumpError(E_COMPLETION_BACKGROUND_FAILED)
                if outcome.status is not CoachingProcessingStatus.PROCESSED:
                    raise _CompletionPumpError(E_COMPLETION_NOT_PROCESSED)
                if outcome.result is None:
                    raise _CompletionPumpError(E_COMPLETION_RESULT_MISSING)
                self._outcome = outcome
                return
            time.sleep(POLL_INTERVAL_SECONDS)
        raise _CompletionPumpError(self._stalled_completion_phase())

    def _stalled_completion_phase(self) -> str:
        manager = self._rag_manager
        if manager is None or self._rag_diagnostic_observer is None:
            return E_COMPLETION_NO_AUTHORITATIVE_OUTCOME
        try:
            snapshot = manager.diagnostic_snapshot()
        except BaseException:
            return E_COMPLETION_DIAGNOSTIC_UNCLASSIFIED
        if snapshot is None:
            return E_COMPLETION_DIAGNOSTIC_UNCLASSIFIED
        return _stalled_completion_phase(snapshot)

    def _completion_timeout_seconds(self) -> float:
        settings = self._full_config().vllm
        http_phase_bound = (
            settings.connect_timeout_seconds  # pool acquisition
            + settings.connect_timeout_seconds  # connection establishment
            + settings.read_timeout_seconds  # request write
            + settings.read_timeout_seconds  # response read
        )
        return max(
            DOCUMENT_POLL_TIMEOUT_SECONDS,
            http_phase_bound + ORCHESTRATION_MARGIN_SECONDS,
        )

    def _citation_projection(self) -> None:
        from live_dashboard.view_models import suggestion_card

        outcome = self._outcome
        if (
            outcome is None
            or outcome.result is None
            or not 1 <= len(outcome.sources) <= 5
        ):
            raise RuntimeError
        displayed = outcome.result.displayed_suggestions
        if len(displayed) != 1:
            raise RuntimeError
        card = suggestion_card(displayed[0], sources=outcome.sources)
        if card.sources != outcome.sources:
            raise RuntimeError
        approved_filenames = {"synthetic-guide.txt", "synthetic-other.txt"}
        internal_names = (
            "tenant_id",
            "knowledge_base_id",
            "document_id",
            "chunk_id",
            "job_id",
            "storage_object_key",
            "sha256_hex",
        )
        if card.evidence_ids or any(
            source.media_label != "TXT"
            or source.original_filename not in approved_filenames
            or any(hasattr(source, name) for name in internal_names)
            for source in card.sources
        ):
            raise RuntimeError

    def _duplicate(self) -> None:
        from app.ingestion.document_background import (
            DocumentSubmissionResult,
            DocumentSubmissionStatus,
        )
        from app.ingestion.registry_models import DocumentReadiness

        runtime = self._document_runtime
        target_before = self._target_entry
        other_before = self._other_entry
        if (
            runtime is None
            or target_before is None
            or other_before is None
            or self._vector_count <= 0
        ):
            raise _DuplicatePhaseError(E_DUPLICATE_RUNTIME)
        try:
            before_entries = runtime.registry.list_documents(
                tenant_id=runtime.tenant_id,
                knowledge_base_id=runtime.knowledge_base_id,
            )
        except Exception:
            raise _DuplicatePhaseError(E_DUPLICATE_BEFORE_LIST) from None
        if type(before_entries) is not tuple or any(
            not isinstance(entry, DocumentRegistryEntry) for entry in before_entries
        ):
            raise _DuplicatePhaseError(E_DUPLICATE_RESULT_SHAPE)
        if len(before_entries) != 2:
            raise _DuplicatePhaseError(E_DUPLICATE_REGISTRY_CARDINALITY)
        try:
            fresh_target_before = runtime.registry.get_entry(
                tenant_id=target_before.document.tenant_id,
                knowledge_base_id=target_before.document.knowledge_base_id,
                document_id=target_before.document.document_id,
            )
        except Exception:
            raise _DuplicatePhaseError(E_DUPLICATE_TARGET_LOOKUP) from None
        if fresh_target_before is not None and not isinstance(
            fresh_target_before, DocumentRegistryEntry
        ):
            raise _DuplicatePhaseError(E_DUPLICATE_RESULT_SHAPE)
        try:
            fresh_other_before = runtime.registry.get_entry(
                tenant_id=other_before.document.tenant_id,
                knowledge_base_id=other_before.document.knowledge_base_id,
                document_id=other_before.document.document_id,
            )
        except Exception:
            raise _DuplicatePhaseError(E_DUPLICATE_OTHER_LOOKUP) from None
        if fresh_other_before is not None and not isinstance(
            fresh_other_before, DocumentRegistryEntry
        ):
            raise _DuplicatePhaseError(E_DUPLICATE_RESULT_SHAPE)
        if fresh_target_before is None or fresh_other_before is None:
            raise _DuplicatePhaseError(E_DUPLICATE_DOCUMENT_IDENTITY)
        vector_count_before = self._duplicate_vector_snapshot(position="BEFORE")
        try:
            result = runtime.manager.submit(
                submission_token="duplicate",
                content=b"Synthetic bounded product return guidance.",
                original_filename="synthetic-guide.txt",
                declared_media_type="text/plain",
            )
        except Exception:
            raise _DuplicatePhaseError(E_DUPLICATE_SUBMIT) from None
        if not isinstance(result, DocumentSubmissionResult) or not isinstance(
            result.status, DocumentSubmissionStatus
        ):
            raise _DuplicatePhaseError(E_DUPLICATE_RESULT_SHAPE)
        if result.status is not DocumentSubmissionStatus.ACCEPTED:
            raise _DuplicatePhaseError(E_DUPLICATE_STATUS)
        try:
            after_entries = runtime.registry.list_documents(
                tenant_id=runtime.tenant_id,
                knowledge_base_id=runtime.knowledge_base_id,
            )
        except Exception:
            raise _DuplicatePhaseError(E_DUPLICATE_AFTER_LIST) from None
        if type(after_entries) is not tuple or any(
            not isinstance(entry, DocumentRegistryEntry) for entry in after_entries
        ):
            raise _DuplicatePhaseError(E_DUPLICATE_RESULT_SHAPE)
        try:
            target_after = runtime.registry.get_entry(
                tenant_id=fresh_target_before.document.tenant_id,
                knowledge_base_id=fresh_target_before.document.knowledge_base_id,
                document_id=fresh_target_before.document.document_id,
            )
        except Exception:
            raise _DuplicatePhaseError(E_DUPLICATE_TARGET_LOOKUP) from None
        if target_after is not None and not isinstance(
            target_after, DocumentRegistryEntry
        ):
            raise _DuplicatePhaseError(E_DUPLICATE_RESULT_SHAPE)
        try:
            other_after = runtime.registry.get_entry(
                tenant_id=fresh_other_before.document.tenant_id,
                knowledge_base_id=fresh_other_before.document.knowledge_base_id,
                document_id=fresh_other_before.document.document_id,
            )
        except Exception:
            raise _DuplicatePhaseError(E_DUPLICATE_OTHER_LOOKUP) from None
        if other_after is not None and not isinstance(
            other_after, DocumentRegistryEntry
        ):
            raise _DuplicatePhaseError(E_DUPLICATE_RESULT_SHAPE)
        vector_count_after = self._duplicate_vector_snapshot(position="AFTER")
        if len(after_entries) != 2 or len(after_entries) != len(before_entries):
            raise _DuplicatePhaseError(E_DUPLICATE_REGISTRY_CARDINALITY)
        if target_after is None or other_after is None:
            raise _DuplicatePhaseError(E_DUPLICATE_DOCUMENT_IDENTITY)
        if (
            fresh_target_before.document.storage_object_key is not None
            or target_after.document.storage_object_key is not None
            or fresh_other_before.document.storage_object_key is not None
            or other_after.document.storage_object_key is not None
        ):
            raise _DuplicatePhaseError(E_DUPLICATE_SOURCE_KEY)
        if target_after.document != fresh_target_before.document:
            raise _DuplicatePhaseError(E_DUPLICATE_DOCUMENT_IDENTITY)
        if target_after.job != fresh_target_before.job:
            raise _DuplicatePhaseError(E_DUPLICATE_JOB_IDENTITY)
        if target_after.readiness is not DocumentReadiness.READY:
            raise _DuplicatePhaseError(E_DUPLICATE_READINESS)
        if other_after != fresh_other_before:
            raise _DuplicatePhaseError(E_DUPLICATE_OTHER_DOCUMENT)
        if vector_count_after != vector_count_before:
            raise _DuplicatePhaseError(E_DUPLICATE_VECTOR_CARDINALITY)

    def _duplicate_vector_snapshot(self, *, position: str) -> int:
        phases = _DUPLICATE_VECTOR_PHASES.get(position)
        if phases is None:
            raise _DuplicatePhaseError(E_DUPLICATE_VECTOR_BEFORE_UNCLASSIFIED)
        try:
            count = self._target_vector_count()
        except _DuplicateVectorOperationError as error:
            raise _DuplicatePhaseError(phases[error.operation]) from None
        except Exception:
            unclassified = (
                E_DUPLICATE_VECTOR_BEFORE_UNCLASSIFIED
                if position == "BEFORE"
                else E_DUPLICATE_VECTOR_AFTER_UNCLASSIFIED
            )
            raise _DuplicatePhaseError(unclassified) from None
        if type(count) is not int or count < 0:
            raise _DuplicatePhaseError(phases["RESULT_SHAPE"])
        return count

    def _target_vector_count(self) -> int:
        try:
            from psycopg import connect

            entry = self._target_entry
            if entry is None:
                raise RuntimeError
            dsn = self._application_dsn()
            parameters = (
                entry.document.tenant_id,
                entry.document.knowledge_base_id,
                entry.document.document_id,
            )
        except Exception:
            raise _DuplicateVectorOperationError("SETTINGS") from None
        try:
            connection = connect(dsn, autocommit=False)
        except Exception:
            raise _DuplicateVectorOperationError("CONNECT") from None
        primary_error: _DuplicateVectorOperationError | None = None
        result: int | None = None
        cursor_manager = None
        cursor = None
        cursor_entered = False
        try:
            try:
                constructed_cursor = connection.cursor()
                cursor_manager = constructed_cursor
                cursor = constructed_cursor.__enter__()
                cursor_entered = True
            except Exception:
                primary_error = _DuplicateVectorOperationError("CURSOR")
            if primary_error is None:
                if cursor is None:
                    primary_error = _DuplicateVectorOperationError("CURSOR")
            if primary_error is None and cursor is not None:
                try:
                    cursor.execute(
                        "SELECT count(*) FROM callmetric_vector.vector_records "
                        "WHERE tenant_id = %s AND knowledge_base_id = %s "
                        "AND document_id = %s",
                        parameters,
                    )
                except Exception:
                    primary_error = _DuplicateVectorOperationError("EXECUTE")
            row: object = None
            if primary_error is None and cursor is not None:
                try:
                    row = cursor.fetchone()
                except Exception:
                    primary_error = _DuplicateVectorOperationError("FETCH")
            if primary_error is None:
                if (
                    type(row) is not tuple
                    or len(row) != 1
                    or type(row[0]) is not int
                    or row[0] < 0
                ):
                    primary_error = _DuplicateVectorOperationError("RESULT_SHAPE")
                else:
                    result = row[0]
        finally:
            if cursor_entered and cursor_manager is not None:
                try:
                    cursor_manager.__exit__(
                        type(primary_error) if primary_error is not None else None,
                        primary_error,
                        primary_error.__traceback__
                        if primary_error is not None
                        else None,
                    )
                except Exception:
                    if primary_error is None:
                        primary_error = _DuplicateVectorOperationError("CLOSE")
            try:
                connection.close()
            except Exception:
                if primary_error is None:
                    primary_error = _DuplicateVectorOperationError("CLOSE")
        if primary_error is not None:
            raise primary_error
        if result is None:
            raise _DuplicateVectorOperationError("RESULT_SHAPE")
        return result

    def _delete(self) -> None:
        runtime = self._document_runtime
        entry = self._target_entry
        if runtime is None or entry is None:
            raise RuntimeError
        deleted = runtime.registry.delete_document(
            tenant_id=entry.document.tenant_id,
            knowledge_base_id=entry.document.knowledge_base_id,
            document_id=entry.document.document_id,
        )
        if deleted is None or deleted.storage_object_key is not None:
            raise RuntimeError

    def _scope_isolation(self) -> None:
        runtime = self._document_runtime
        other = self._other_entry
        if runtime is None or other is None:
            raise RuntimeError
        if (
            runtime.registry.get_entry(
                tenant_id=other.document.tenant_id,
                knowledge_base_id=other.document.knowledge_base_id,
                document_id=other.document.document_id,
            )
            is None
        ):
            raise RuntimeError
        if self._target_vector_count() != 0:
            raise RuntimeError
        if (
            runtime.postgres_rag.profile_repository.get_profile(
                tenant_id=runtime.tenant_id, knowledge_base_id=runtime.knowledge_base_id
            )
            is None
        ):
            raise RuntimeError

    def cleanup(self) -> None:
        internal_lifecycle_error: BaseException | None = None
        unrecoverable_cleanup_error: BaseException | None = None
        recoverable_action_trigger: BaseException | None = None
        try:
            if self._rag_manager is not None:
                self._rag_manager.close(wait=False)
        except BaseException as error:
            internal_lifecycle_error = error
        try:
            if self._document_runtime is not None:
                for entry in (self._target_entry, self._other_entry):
                    if entry is not None:
                        self._document_runtime.registry.delete_document(
                            tenant_id=entry.document.tenant_id,
                            knowledge_base_id=entry.document.knowledge_base_id,
                            document_id=entry.document.document_id,
                        )
                self._document_runtime.close(wait=False)
        except BaseException as error:
            internal_lifecycle_error = internal_lifecycle_error or error
        service = self._service
        owned_processes = dict(self._owned_processes)
        graceful_recovery_trigger: BaseException | None = None
        if service is not None:
            try:
                if self._owner_marker is None:
                    self._refresh_owned_process_ledger()
                else:
                    self._owned_processes.update(self._discover_marker_processes())
                owned_processes = dict(self._owned_processes)
                if self._owner_marker is None and service.pid not in owned_processes:
                    raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
            except BaseException as error:
                graceful_recovery_trigger = error
                unrecoverable_cleanup_error = (
                    unrecoverable_cleanup_error
                    or _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
                )
            try:
                if service.poll() is None:
                    service.send_signal(signal.CTRL_BREAK_EVENT)
            except BaseException as error:
                graceful_recovery_trigger = graceful_recovery_trigger or error
            try:
                return_code = service.wait(timeout=150)
                if return_code != 0:
                    graceful_recovery_trigger = (
                        graceful_recovery_trigger or RuntimeError()
                    )
            except BaseException as error:
                graceful_recovery_trigger = graceful_recovery_trigger or error
        if graceful_recovery_trigger is None:
            try:
                self._require_postgres_residue_absent()
                self._require_initial_processes_absent(owned_processes)
            except BaseException as error:
                graceful_recovery_trigger = error
        if graceful_recovery_trigger is not None:
            try:
                if service is not None:
                    self._terminate_owned_process_tree(service, owned_processes)
            except BaseException as error:
                if (
                    isinstance(error, _CleanupPhaseError)
                    and error.phase == E_CLEANUP_PROCESS_ACTION
                ):
                    recoverable_action_trigger = error
                else:
                    unrecoverable_cleanup_error = unrecoverable_cleanup_error or error
            try:
                self._cleanup_exact_postgres_project()
            except BaseException as error:
                unrecoverable_cleanup_error = unrecoverable_cleanup_error or error
            try:
                self._cleanup_exact_handoff()
            except BaseException as error:
                unrecoverable_cleanup_error = unrecoverable_cleanup_error or error
        final_verification_error: BaseException | None = None
        try:
            self._require_postgres_residue_absent()
        except BaseException as error:
            final_verification_error = error
        try:
            self._require_initial_processes_absent(owned_processes)
        except BaseException as error:
            final_verification_error = final_verification_error or error
        try:
            self._require_handoff_root_unchanged()
        except BaseException as error:
            final_verification_error = final_verification_error or error
        try:
            self._require_protected_resources_unchanged()
        except BaseException as error:
            final_verification_error = final_verification_error or error
        try:
            self._release_service_handle()
        except BaseException as error:
            final_verification_error = final_verification_error or error
        cleanup_error = (
            internal_lifecycle_error
            or unrecoverable_cleanup_error
            or final_verification_error
        )
        if cleanup_error is not None:
            raise cleanup_error
        _ = recoverable_action_trigger

    @staticmethod
    def _windows_process_observations() -> tuple[_WindowsProcessObservation, ...]:
        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        if powershell is None:
            raise RuntimeError
        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[Console]::OutputEncoding = "
                "[System.Text.UTF8Encoding]::new($false); "
                "$OutputEncoding = [Console]::OutputEncoding; "
                "Get-CimInstance Win32_Process | ForEach-Object { "
                "[PSCustomObject]@{ProcessId=$_.ProcessId;"
                "ParentProcessId=$_.ParentProcessId;"
                "ExecutablePath=$_.ExecutablePath;CommandLine=$_.CommandLine;"
                "CreationDate=$(if ($null -eq $_.CreationDate) {$null} else "
                "{$_.CreationDate.ToUniversalTime().ToString('o')})}} | "
                "ConvertTo-Json -Compress",
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=False,
            shell=False,
            timeout=30,
        )
        if (
            max(len(result.stdout), len(result.stderr)) > _MAX_WMI_OUTPUT_BYTES
            or result.stderr
        ):
            raise RuntimeError
        try:
            payload = json.loads(result.stdout.decode("utf-8", errors="strict"))
        except (UnicodeError, ValueError):
            raise RuntimeError from None
        rows = payload if isinstance(payload, list) else [payload]
        observations: list[_WindowsProcessObservation] = []
        seen: set[int] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError
            if not {
                "ProcessId",
                "ParentProcessId",
                "ExecutablePath",
                "CommandLine",
                "CreationDate",
            }.issubset(row):
                raise RuntimeError
            process_id = row.get("ProcessId")
            parent_id = row.get("ParentProcessId")
            if (
                type(process_id) is not int
                or type(parent_id) is not int
                or process_id < 0
                or parent_id < 0
                or process_id in seen
                or (process_id == 0 and parent_id != 0)
            ):
                raise RuntimeError
            executable_path = row.get("ExecutablePath")
            command_line = row.get("CommandLine")
            creation_time = row.get("CreationDate")
            if executable_path is not None and not isinstance(executable_path, str):
                raise RuntimeError
            if command_line is not None and not isinstance(command_line, str):
                raise RuntimeError
            if creation_time is not None and not isinstance(creation_time, str):
                raise RuntimeError
            seen.add(process_id)
            if process_id == 0:
                continue
            observations.append(
                _WindowsProcessObservation(
                    process_id,
                    parent_id,
                    executable_path,
                    command_line,
                    creation_time,
                )
            )
        return tuple(observations)

    @staticmethod
    def _parse_windows_command_line(command_line: str) -> tuple[str, ...]:
        shell32 = WinDLL("shell32", use_last_error=True)
        kernel32 = WinDLL("kernel32", use_last_error=True)
        parser = shell32.CommandLineToArgvW
        parser.argtypes = [wintypes.LPCWSTR, POINTER(c_int)]
        parser.restype = POINTER(wintypes.LPWSTR)
        local_free = kernel32.LocalFree
        local_free.argtypes = [c_void_p]
        local_free.restype = c_void_p
        count = c_int()
        arguments = parser(command_line, byref(count))
        if not arguments or count.value <= 0:
            raise RuntimeError
        try:
            return tuple(arguments[index] for index in range(count.value))
        finally:
            local_free(arguments)

    @staticmethod
    def _reviewed_python_executables() -> frozenset[str]:
        candidates = [sys.executable]
        base = getattr(sys, "_base_executable", None)
        if isinstance(base, str) and base:
            candidates.append(base)
        return frozenset(
            os.path.normcase(str(Path(candidate).resolve(strict=True)))
            for candidate in candidates
        )

    def _discover_marker_processes(self) -> dict[int, int]:
        marker = self._owner_marker
        launch_boundary = self._postgres_launch_boundary
        if marker is None or launch_boundary is None:
            raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
        reviewed_executables = self._reviewed_python_executables()
        owned: dict[int, int] = {}
        for observation in self._windows_process_observations():
            command_line = observation.command_line
            if command_line is None or marker not in command_line:
                continue
            try:
                arguments = self._parse_windows_command_line(command_line)
            except BaseException:
                raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY) from None
            marker_indexes = [
                index
                for index, argument in enumerate(arguments)
                if argument == "--owner-marker"
            ]
            if len(marker_indexes) != 1:
                raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)
            marker_index = marker_indexes[0]
            if (
                marker_index + 1 >= len(arguments)
                or arguments[marker_index + 1] != marker
                or arguments.count(marker) != 1
            ):
                raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)
            if observation.process_id == os.getpid():
                raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)
            executable_path = observation.executable_path
            creation_time = observation.creation_time_utc
            if executable_path is None or creation_time is None:
                raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)
            try:
                canonical_executable = os.path.normcase(
                    str(Path(executable_path).resolve(strict=True))
                )
                created = datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    raise ValueError
                created = created.astimezone(timezone.utc)
            except (OSError, ValueError):
                raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY) from None
            if (
                canonical_executable not in reviewed_executables
                or created < launch_boundary
            ):
                raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)
            owned[observation.process_id] = observation.parent_process_id
        return owned

    @classmethod
    def _windows_process_table(cls) -> dict[int, int]:
        return {
            item.process_id: item.parent_process_id
            for item in cls._windows_process_observations()
        }

    def _refresh_owned_process_ledger(self) -> None:
        service = self._service
        if service is None:
            return
        if self._owner_marker is not None:
            discovered = self._discover_marker_processes()
            if not discovered:
                raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
            self._owned_processes.update(discovered)
            return
        root = service.pid
        current = self._windows_process_table()
        if not self._owned_processes:
            if root not in current:
                raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
            self._owned_processes = {
                process_id: parent_id
                for process_id, parent_id in current.items()
                if process_id == root or self._is_descendant(current, process_id, root)
            }
            return
        for process_id, parent_id in self._owned_processes.items():
            if process_id in current and current[process_id] != parent_id:
                raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)
        changed = True
        while changed:
            changed = False
            for process_id, parent_id in current.items():
                if process_id in self._owned_processes:
                    continue
                if parent_id in self._owned_processes and parent_id in current:
                    self._owned_processes[process_id] = parent_id
                    changed = True

    @staticmethod
    def _descendant_depth(table: Mapping[int, int], process_id: int, root: int) -> int:
        current = process_id
        seen: set[int] = set()
        depth = 0
        while current != root:
            if current in seen or current not in table:
                raise RuntimeError
            seen.add(current)
            current = table[current]
            depth += 1
            if depth > len(table):
                raise RuntimeError
        return depth

    def _terminate_owned_process_tree(
        self,
        service: subprocess.Popen[bytes],
        initial_processes: Mapping[int, int],
    ) -> None:
        if self._owner_marker is not None:
            self._terminate_marker_processes(service)
            return
        root = service.pid
        if type(root) is not int or root <= 0 or root not in initial_processes:
            raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
        owned = {
            process_id: parent_id
            for process_id, parent_id in initial_processes.items()
            if process_id == root
            or self._is_descendant(initial_processes, process_id, root)
        }
        validation_failure: BaseException | None = None
        deadline = time.monotonic() + 30.0

        def taskkill_path() -> str:
            taskkill = shutil.which("taskkill.exe") or shutil.which("taskkill")
            if taskkill is None:
                raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
            return taskkill

        def observe() -> dict[int, int]:
            try:
                return self._windows_process_table()
            except BaseException:
                raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE) from None

        def record_current_descendants(current: Mapping[int, int]) -> None:
            nonlocal validation_failure
            for process_id, parent_id in owned.items():
                if process_id in current and current[process_id] != parent_id:
                    validation_failure = validation_failure or _CleanupPhaseError(
                        E_CLEANUP_PROCESS_VERIFY
                    )
            changed = True
            while changed:
                changed = False
                for process_id, parent_id in current.items():
                    if process_id in owned:
                        continue
                    if parent_id in owned and parent_id in current:
                        owned[process_id] = parent_id
                        changed = True

        def terminate(process_id: int, current: Mapping[int, int]) -> None:
            nonlocal validation_failure
            if process_id not in current:
                return
            identity_matches = current[process_id] == owned[process_id]
            if not identity_matches:
                validation_failure = validation_failure or _CleanupPhaseError(
                    E_CLEANUP_PROCESS_VERIFY
                )
                return
            try:
                result = subprocess.run(
                    [taskkill_path(), "/PID", str(process_id), "/F"],
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    capture_output=True,
                    text=False,
                    shell=False,
                    timeout=30,
                )
                if (
                    max(len(result.stdout or b""), len(result.stderr or b""))
                    > _MAX_RESOURCE_OUTPUT_BYTES
                ):
                    raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
            except BaseException:
                observed = observe()
                if process_id not in observed:
                    return
                identity_matches = observed[process_id] == owned[process_id]
                validation_failure = validation_failure or _CleanupPhaseError(
                    E_CLEANUP_PROCESS_VERIFY
                    if not identity_matches
                    else E_CLEANUP_PROCESS_ACTION
                )

        while time.monotonic() < deadline:
            current = observe()
            record_current_descendants(current)
            descendants = sorted(
                (
                    (self._descendant_depth(owned, process_id, root), process_id)
                    for process_id in owned
                    if process_id != root and process_id in current
                ),
                reverse=True,
            )
            if not descendants:
                break
            for _depth, process_id in descendants:
                terminate(process_id, current)
            if validation_failure is not None:
                break
        else:
            validation_failure = validation_failure or _CleanupPhaseError(
                E_CLEANUP_PROCESS_ACTION
            )

        current = observe()
        if validation_failure is None:
            terminate(root, current)
        try:
            if service.poll() is None:
                service.wait(timeout=max(0.0, deadline - time.monotonic()))
        except BaseException:
            pass

        observed = observe()
        record_current_descendants(observed)
        if validation_failure is not None:
            raise validation_failure
        if root in observed or any(process_id in observed for process_id in owned):
            raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)
        if service.poll() is None:
            raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)

    def _terminate_marker_processes(self, service: subprocess.Popen[bytes]) -> None:
        taskkill = shutil.which("taskkill.exe") or shutil.which("taskkill")
        if taskkill is None:
            raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            owned = self._discover_marker_processes()
            self._owned_processes.update(owned)
            if not owned:
                break

            def depth(process_id: int) -> int:
                current = process_id
                seen: set[int] = set()
                result = 0
                while owned.get(current) in owned:
                    if current in seen:
                        raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)
                    seen.add(current)
                    current = owned[current]
                    result += 1
                return result

            for process_id in sorted(
                owned, key=lambda item: (depth(item), item), reverse=True
            ):
                try:
                    result = subprocess.run(
                        [taskkill, "/PID", str(process_id), "/F"],
                        cwd=REPOSITORY_ROOT,
                        check=True,
                        capture_output=True,
                        text=False,
                        shell=False,
                        timeout=30,
                    )
                    if (
                        max(len(result.stdout or b""), len(result.stderr or b""))
                        > _MAX_RESOURCE_OUTPUT_BYTES
                    ):
                        raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
                except BaseException:
                    if process_id in self._discover_marker_processes():
                        raise _CleanupPhaseError(E_CLEANUP_PROCESS_ACTION) from None
        else:
            raise _CleanupPhaseError(E_CLEANUP_PROCESS_ACTION)
        if self._discover_marker_processes():
            raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)
        try:
            if service.poll() is None:
                service.wait(timeout=max(0.0, deadline - time.monotonic()))
        except BaseException:
            pass
        if service.poll() is None:
            raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)

    @classmethod
    def _is_descendant(
        cls, table: Mapping[int, int], process_id: int, root: int
    ) -> bool:
        try:
            return cls._descendant_depth(table, process_id, root) > 0
        except RuntimeError:
            return False

    def _cleanup_exact_postgres_project(self) -> None:
        from scripts.run_postgres_tls_service import cleanup_exact_project_resources

        project = self._postgres_project
        docker = shutil.which("docker")
        if project is None or docker is None:
            raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
        try:
            cleanup_exact_project_resources(docker, project)
        except BaseException:
            raise _CleanupPhaseError(E_CLEANUP_PROJECT_ACTION) from None

    def _cleanup_exact_handoff(self) -> None:
        from scripts.run_postgres_tls_service import cleanup_exact_handoff_child

        handoff = self._handoff
        if handoff is None:
            return
        try:
            cleanup_exact_handoff_child(self._postgres_config.handoff_root, handoff)
        except BaseException:
            raise _CleanupPhaseError(E_CLEANUP_HANDOFF_ACTION) from None

    def _require_protected_resources_unchanged(self) -> None:
        from scripts.run_postgres_tls_service import (
            ProtectedResourceSnapshot,
            require_protected_resources_unchanged,
        )

        expected = self._protected_resources
        docker = shutil.which("docker")
        if not isinstance(expected, ProtectedResourceSnapshot) or docker is None:
            raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
        try:
            require_protected_resources_unchanged(docker, expected)
        except BaseException:
            raise _CleanupPhaseError(E_CLEANUP_PROTECTED_VERIFY) from None

    def _require_handoff_root_unchanged(self) -> None:
        expected = self._protected_handoff_entries
        if expected is None:
            return
        root = self._postgres_config.handoff_root
        try:
            resolved = root.resolve(strict=True)
            if root != resolved or root.is_symlink() or not root.is_dir():
                raise RuntimeError
            if frozenset(root.iterdir()) != expected:
                raise RuntimeError
        except BaseException:
            raise _CleanupPhaseError(E_CLEANUP_HANDOFF_VERIFY) from None

    def _require_postgres_residue_absent(self) -> None:
        project = self._postgres_project
        if project is None:
            return
        docker = shutil.which("docker")
        if docker is None:
            raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
        for resource in ("container", "network", "volume"):
            try:
                result = subprocess.run(
                    [
                        docker,
                        resource,
                        "ls",
                        "-q",
                        "--filter",
                        f"label=com.docker.compose.project={project}",
                    ],
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    capture_output=True,
                    text=False,
                    shell=False,
                    timeout=30,
                )
            except BaseException:
                raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE) from None
            if max(len(result.stdout), len(result.stderr)) > _MAX_RESOURCE_OUTPUT_BYTES:
                raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
            try:
                output = result.stdout.decode("ascii", errors="strict").strip()
            except UnicodeError:
                raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE) from None
            if output:
                raise _CleanupPhaseError(E_CLEANUP_PROJECT_VERIFY)
        if self._handoff is not None and os.path.lexists(self._handoff):
            raise _CleanupPhaseError(E_CLEANUP_HANDOFF_VERIFY)
        service = self._service
        if service is not None:
            try:
                running = service.poll() is None
                if self._owner_marker is not None:
                    marker_processes = self._discover_marker_processes()
                    process_residue = bool(marker_processes)
                else:
                    table = self._windows_process_table()
                    process_residue = service.pid in table or any(
                        self._is_descendant(table, process_id, service.pid)
                        for process_id in table
                        if process_id != service.pid
                    )
            except BaseException:
                raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE) from None
            if running or process_residue:
                raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)

    def _require_initial_processes_absent(
        self, initial_processes: Mapping[int, int]
    ) -> None:
        service = self._service
        if not initial_processes or service is None:
            if self._owner_marker is not None and self._discover_marker_processes():
                raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)
            return
        if self._owner_marker is not None:
            if self._discover_marker_processes():
                raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)
            return
        root = service.pid
        owned = {
            process_id
            for process_id in initial_processes
            if process_id == root
            or self._is_descendant(initial_processes, process_id, root)
        }
        if root not in owned:
            raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
        try:
            observed = self._windows_process_table()
        except BaseException:
            raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE) from None
        if any(process_id in observed for process_id in owned):
            raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)

    def _release_service_handle(self) -> None:
        service = self._service
        if service is None:
            return
        try:
            if service.poll() is None:
                raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)
            service.wait(timeout=0)
            self._join_tls_child_output_thread()
            output = getattr(service, "stdout", None)
            if output is not None:
                output.close()
        except _CleanupPhaseError:
            raise
        except BaseException:
            raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE) from None
        self._service = None
        self._tls_child_output_thread = None


def main(arguments: list[str] | None = None) -> int:
    values = sys.argv[1:] if arguments is None else arguments
    if values not in ([], ["--preflight-only"], ["--postgres-startup-only"]):
        print("E_PREFLIGHT")
        return 1
    try:
        if values == ["--postgres-startup-only"]:
            print(run_postgres_startup_only())
        else:
            print(run(preflight_only=bool(values)))
    except DashboardRAGVLLME2EError as error:
        print(error.phase)
        return 1
    except BaseException:
        print("E_PREFLIGHT")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
