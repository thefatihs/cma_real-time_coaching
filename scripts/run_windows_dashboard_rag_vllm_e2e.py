"""Secret-safe Windows document-backed dashboard RAG/vLLM E2E controller."""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
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
from app.coaching.coordinator import StableCoachingOutcome
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
POLL_TIMEOUT_SECONDS = 300.0
POLL_INTERVAL_SECONDS = 0.2

PREFLIGHT_OK = "PREFLIGHT_OK"
E2E_OK = "E2E_OK"
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
        self.phase = phase if phase in PHASES else "E_PREFLIGHT"
        super().__init__(self.phase)


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


def _preflight(environment: Mapping[str, str] | None = None) -> ControllerConfig:
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
    raw_ttl = _required(source, TTL_ENV)
    if not raw_ttl.isascii() or not raw_ttl.isdigit():
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")
    ttl = int(raw_ttl)
    if not MINIMUM_TTL_SECONDS <= ttl <= MAXIMUM_TTL_SECONDS:
        raise DashboardRAGVLLME2EError("E_PREFLIGHT")
    return ControllerConfig(
        branch, head, baseline, handoff_root, provider, policy, vllm, ttl
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
    primary: BaseException | None = None
    try:
        for phase in PHASES[1:-1]:
            try:
                operations.run_phase(phase)
            except BaseException:
                raise DashboardRAGVLLME2EError(phase) from None
    except BaseException as error:
        primary = error
    try:
        operations.cleanup()
    except BaseException:
        if primary is None:
            primary = DashboardRAGVLLME2EError("E_CLEANUP")
    if primary is not None:
        raise primary
    return E2E_OK


class _ProductionLifecycle:
    """Stateful adapter around existing production boundaries."""

    def __init__(
        self, config: ControllerConfig, environment: Mapping[str, str]
    ) -> None:
        self._config = config
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

    def run_phase(self, phase: str) -> None:
        getattr(self, f"_{phase.removeprefix('E_').lower()}")()

    def _postgres_start(self) -> None:
        before = set(self._config.handoff_root.iterdir())
        environment = dict(self._environment)
        environment["CALLMETRIC_POSTGRES_TLS_SERVICE_EXPECTED_BRANCH"] = (
            self._config.branch
        )
        environment["CALLMETRIC_POSTGRES_TLS_SERVICE_EXPECTED_HEAD"] = self._config.head
        project = f"callmetric-pgvector-tls-{os.getpid()}-{secrets.token_hex(6)}"
        if not re.fullmatch(r"callmetric-pgvector-tls-[0-9]+-[a-f0-9]{12}", project):
            raise RuntimeError
        self._postgres_project = project
        environment["CALLMETRIC_POSTGRES_TLS_SERVICE_PROJECT_NAME"] = project
        self._service = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.run_postgres_tls_service",
                "--ttl-seconds",
                str(self._config.ttl_seconds),
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            created = set(self._config.handoff_root.iterdir()) - before
            ready = [item for item in created if (item / "application.dsn").is_file()]
            if len(ready) == 1:
                self._handoff = ready[0]
                return
            if self._service.poll() is not None:
                break
            time.sleep(POLL_INTERVAL_SECONDS)
        raise RuntimeError

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
            knowledge_base_settings=self._config.provider,
            psycopg_connect=connect,
        )

    def _model(self) -> None:
        from app.composition.postgres_rag import compose_profile_bound_postgres_rag
        from psycopg import connect

        composition = compose_profile_bound_postgres_rag(
            postgres_settings=self._settings(),
            knowledge_base_settings=self._config.provider,
            psycopg_connect=connect,
        )
        vector = composition.embedder.embed_query(
            tenant_id=self._config.provider.tenant_id,
            knowledge_base_id=self._config.provider.knowledge_base_id,
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
            knowledge_base_settings=self._config.provider,
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
        deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
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

        provider = self._config.provider
        composition = compose_profile_bound_postgres_rag_orchestration(
            postgres_settings=self._settings(),
            knowledge_base_settings=provider,
            psycopg_connect=connect,
            llm_gateway_factory=lambda: VLLMOpenAICompatibleGateway(
                self._config.vllm,
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
        config = demo.config.model_copy(deep=True)
        config.rag = config.rag.model_copy(
            update={
                "enabled": True,
                "knowledge_base_id": provider.knowledge_base_id,
                "top_k": 3,
                "minimum_score": 0.0,
            }
        )
        config.coaching = config.coaching.model_copy(update={"enable_llm": True})
        config = TenantConfig.model_validate(config.model_dump())
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
                    name=self._config.policy.rag_llm_enabled_labels[0], score=1.0
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
            config, state, RuleBasedCoachingEngine(config, demo.rules)
        )
        runtime = self._document_runtime
        if runtime is None:
            raise RuntimeError
        processor = compose_rag_coaching_processor(
            coordinator=coordinator,
            tenant_config=config,
            integration=RAGCoachingIntegrationDependencies(
                background_manager=manager,
                policy=self._config.policy,
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
            active_labels=(self._config.policy.rag_llm_enabled_labels[0],),
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
        deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            processor = self._processor
            if processor is None:
                raise RuntimeError
            completed = processor.drain_completed(current_seconds=1.0)
            if completed:
                self._outcome = completed[0]
                if self._outcome.result is None:
                    raise RuntimeError
                return
            time.sleep(POLL_INTERVAL_SECONDS)
        raise RuntimeError

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
        primary: BaseException | None = None
        try:
            if self._rag_manager is not None:
                self._rag_manager.close(wait=False)
        except BaseException as error:
            primary = error
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
            primary = primary or error
        service = self._service
        if service is not None:
            if service.poll() is None:
                try:
                    service.send_signal(signal.CTRL_BREAK_EVENT)
                except BaseException as error:
                    primary = primary or error
            try:
                return_code = service.wait(timeout=150)
                if return_code != 0:
                    primary = primary or RuntimeError()
            except BaseException as error:
                primary = primary or error
        try:
            self._require_postgres_residue_absent()
        except BaseException as error:
            primary = primary or error
        if primary is not None:
            raise primary

    def _require_postgres_residue_absent(self) -> None:
        project = self._postgres_project
        if project is None:
            return
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError
        for resource in ("container", "network", "volume"):
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
            if result.stdout.strip():
                raise RuntimeError
        if self._handoff is not None and self._handoff.exists():
            raise RuntimeError


def main(arguments: list[str] | None = None) -> int:
    values = sys.argv[1:] if arguments is None else arguments
    if values not in ([], ["--preflight-only"]):
        print("E_PREFLIGHT")
        return 1
    try:
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
