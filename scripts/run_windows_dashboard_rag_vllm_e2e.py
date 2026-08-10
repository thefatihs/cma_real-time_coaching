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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Protocol, TypeVar
from urllib.parse import urlsplit

from app.composition.postgres_document_ingestion import (
    MINILM_DIMENSION,
    MINILM_MODEL,
    validate_local_minilm_snapshot,
)
from app.composition.postgres_rag import KnowledgeBaseRAGProviderSettings
from app.composition.postgres_rag import PostgreSQLVectorStoreSettings
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
DOCUMENT_POLL_TIMEOUT_SECONDS = 300.0
POLL_INTERVAL_SECONDS = 0.2
ORCHESTRATION_MARGIN_SECONDS = 60.0
MINIMUM_E2E_OUTPUT_TOKENS = 256

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
E_COMPLETION_PROCESSOR_MISSING = "E_COMPLETION_PROCESSOR_MISSING"
E_COMPLETION_NO_AUTHORITATIVE_OUTCOME = "E_COMPLETION_NO_AUTHORITATIVE_OUTCOME"
E_COMPLETION_CARDINALITY = "E_COMPLETION_CARDINALITY"
E_COMPLETION_BACKGROUND_FAILED = "E_COMPLETION_BACKGROUND_FAILED"
E_COMPLETION_NOT_PROCESSED = "E_COMPLETION_NOT_PROCESSED"
E_COMPLETION_RESULT_MISSING = "E_COMPLETION_RESULT_MISSING"
E_COMPLETION_UNCLASSIFIED = "E_COMPLETION_UNCLASSIFIED"
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
        return subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
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
                raise DashboardRAGVLLME2EError(phase) from None
    except BaseException as error:
        functional_primary_error = error
    try:
        operations.cleanup()
    except BaseException:
        if functional_primary_error is None:
            functional_primary_error = DashboardRAGVLLME2EError("E_CLEANUP")
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
    except BaseException:
        if functional_primary_error is None:
            functional_primary_error = DashboardRAGVLLME2EError("E_CLEANUP")
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
        self._target_entry: DocumentRegistryEntry | None = None
        self._other_entry: DocumentRegistryEntry | None = None
        self._outcome: StableCoachingOutcome | None = None
        self._processor: RAGCoachingProcessorDecorator | None = None
        self._vector_count = 0
        self._protected_resources: dict[str, frozenset[str]] | None = None
        self._owned_processes: dict[int, int] = {}
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
        self._service = self._postgres_startup_call(
            E_POSTGRES_LAUNCH,
            lambda: subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scripts.run_postgres_tls_service",
                    "--ttl-seconds",
                    str(self._postgres_config.ttl_seconds),
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
        from app.ingestion.registry_models import DocumentReadiness

        runtime = self._document_runtime
        if runtime is None:
            raise RuntimeError
        deadline = time.monotonic() + DOCUMENT_POLL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            entries = runtime.registry.list_documents(
                tenant_id=runtime.tenant_id,
                knowledge_base_id=runtime.knowledge_base_id,
            )
            ready = [
                item for item in entries if item.readiness is DocumentReadiness.READY
            ]
            if len(ready) == 2:
                by_name = {item.document.original_filename: item for item in ready}
                self._target_entry = by_name["synthetic-guide.txt"]
                self._other_entry = by_name["synthetic-other.txt"]
                if any(item.document.storage_object_key is not None for item in ready):
                    raise RuntimeError
                return
            time.sleep(POLL_INTERVAL_SECONDS)
        raise RuntimeError

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
        composition = compose_profile_bound_postgres_rag_orchestration(
            postgres_settings=self._settings(),
            knowledge_base_settings=provider,
            psycopg_connect=connect,
            llm_gateway_factory=lambda: VLLMOpenAICompatibleGateway(
                controller_config.vllm,
                structured_output_json_schema=coaching_wire_json_schema(),
            ),
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
        manager = BoundedPostgreSQLRAGManager(runner=runner, max_workers=1, capacity=2)
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
        raise _CompletionPumpError(E_COMPLETION_NO_AUTHORITATIVE_OUTCOME)

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
        from app.ingestion.document_background import DocumentSubmissionStatus

        runtime = self._document_runtime
        if runtime is None:
            raise RuntimeError
        result = runtime.manager.submit(
            submission_token="duplicate",
            content=b"Synthetic bounded product return guidance.",
            original_filename="synthetic-guide.txt",
            declared_media_type="text/plain",
        )
        if result.status is not DocumentSubmissionStatus.ACCEPTED:
            raise RuntimeError
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self._target_vector_count() == self._vector_count:
                runtime_entries = runtime.registry.list_documents(
                    tenant_id=runtime.tenant_id,
                    knowledge_base_id=runtime.knowledge_base_id,
                )
                if len(runtime_entries) == 2:
                    return
            time.sleep(POLL_INTERVAL_SECONDS)
        raise RuntimeError

    def _target_vector_count(self) -> int:
        from psycopg import connect

        entry = self._target_entry
        if entry is None:
            raise RuntimeError
        connection = connect(self._application_dsn(), autocommit=False)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT count(*) FROM callmetric_vector.vector_records "
                    "WHERE tenant_id = %s AND knowledge_base_id = %s "
                    "AND document_id = %s",
                    (
                        entry.document.tenant_id,
                        entry.document.knowledge_base_id,
                        entry.document.document_id,
                    ),
                )
                row = cursor.fetchone()
                if row is None or type(row[0]) is not int:
                    raise RuntimeError
                return row[0]
        finally:
            connection.close()

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
        fallback_action_error: BaseException | None = None
        try:
            if self._rag_manager is not None:
                self._rag_manager.close(wait=False)
        except BaseException as error:
            fallback_action_error = error
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
            fallback_action_error = fallback_action_error or error
        service = self._service
        owned_processes = dict(self._owned_processes)
        graceful_recovery_trigger: BaseException | None = None
        if service is not None:
            try:
                self._refresh_owned_process_ledger()
                owned_processes = dict(self._owned_processes)
                if service.pid not in owned_processes:
                    raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
            except BaseException as error:
                graceful_recovery_trigger = error
                fallback_action_error = fallback_action_error or _CleanupPhaseError(
                    E_CLEANUP_UNVERIFIABLE
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
                fallback_action_error = fallback_action_error or error
            try:
                self._cleanup_exact_postgres_project()
            except BaseException as error:
                fallback_action_error = fallback_action_error or error
            try:
                self._cleanup_exact_handoff()
            except BaseException as error:
                fallback_action_error = fallback_action_error or error
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
            self._require_protected_resources_unchanged()
        except BaseException as error:
            final_verification_error = final_verification_error or error
        try:
            self._release_service_handle()
        except BaseException as error:
            final_verification_error = final_verification_error or error
        cleanup_error = fallback_action_error or final_verification_error
        if cleanup_error is not None:
            raise cleanup_error

    @staticmethod
    def _windows_process_table() -> dict[int, int]:
        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        if powershell is None:
            raise RuntimeError
        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_Process | Select-Object ProcessId,"
                "ParentProcessId | ConvertTo-Json -Compress",
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
            timeout=30,
        )
        payload = json.loads(result.stdout)
        rows = payload if isinstance(payload, list) else [payload]
        table: dict[int, int] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError
            process_id = row.get("ProcessId")
            parent_id = row.get("ParentProcessId")
            if (
                type(process_id) is not int
                or type(parent_id) is not int
                or process_id <= 0
                or parent_id < 0
                or process_id in table
            ):
                raise RuntimeError
            table[process_id] = parent_id
        return table

    def _refresh_owned_process_ledger(self) -> None:
        service = self._service
        if service is None:
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
                subprocess.run(
                    [taskkill_path(), "/PID", str(process_id), "/F"],
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    shell=False,
                    timeout=30,
                )
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
            require_protected_resources_unchanged,
        )

        expected = self._protected_resources
        docker = shutil.which("docker")
        if expected is None or docker is None:
            raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE)
        try:
            require_protected_resources_unchanged(docker, expected)
        except BaseException:
            raise _CleanupPhaseError(E_CLEANUP_PROTECTED_VERIFY) from None

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
                    text=True,
                    shell=False,
                    timeout=30,
                )
            except BaseException:
                raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE) from None
            if result.stdout.strip():
                raise _CleanupPhaseError(E_CLEANUP_PROJECT_VERIFY)
        if self._handoff is not None and os.path.lexists(self._handoff):
            raise _CleanupPhaseError(E_CLEANUP_HANDOFF_VERIFY)
        service = self._service
        if service is not None:
            try:
                table = self._windows_process_table()
                running = service.poll() is None
            except BaseException:
                raise _CleanupPhaseError(E_CLEANUP_UNVERIFIABLE) from None
            if (
                running
                or service.pid in table
                or any(
                    self._is_descendant(table, process_id, service.pid)
                    for process_id in table
                    if process_id != service.pid
                )
            ):
                raise _CleanupPhaseError(E_CLEANUP_PROCESS_VERIFY)

    def _require_initial_processes_absent(
        self, initial_processes: Mapping[int, int]
    ) -> None:
        service = self._service
        if not initial_processes or service is None:
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
