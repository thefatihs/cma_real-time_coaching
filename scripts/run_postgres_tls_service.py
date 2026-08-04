"""Run a TTL-bounded Windows PostgreSQL/pgvector TLS service."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import csv
import io
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any, TypeVar

from scripts import run_postgres_tls_smoke as smoke

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "feat/rag-coaching-integration"
EXPECTED_HEAD_VARIABLE = "CALLMETRIC_POSTGRES_TLS_SERVICE_EXPECTED_HEAD"
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
HANDOFF_ROOT_VARIABLE = "CALLMETRIC_POSTGRES_TLS_SERVICE_HANDOFF_ROOT"
MINIMUM_TTL_SECONDS = 300
MAXIMUM_TTL_SECONDS = 7_200
VALIDATION_TIMEOUT_SECONDS = 30.0
IDENTITY_ACL_TIMEOUT_SECONDS = 30.0
CERTIFICATE_TIMEOUT_SECONDS = 60.0
COMPOSE_CONFIG_TIMEOUT_SECONDS = 30.0
COMPOSE_STARTUP_TIMEOUT_SECONDS = 120.0
READINESS_COMMAND_TIMEOUT_SECONDS = 30.0
POSTGRES_READINESS_TIMEOUT_SECONDS = smoke.HEALTH_TIMEOUT_SECONDS
MIGRATION_PROOF_TIMEOUT_SECONDS = 180.0
CLEANUP_TIMEOUT_SECONDS = 120.0
PROJECT_PATTERN = smoke.PROJECT_NAME_PATTERN
HANDOFF_PATTERN = re.compile(r"^callmetric-postgres-tls-[a-z0-9_]{8}$")
HANDOFF_FILES = frozenset({"application.dsn", "ca.crt", "connection.json"})
RESOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
CERTIFICATE_CONTAINER_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+$")
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
COMPOSE_SERVICE_LABEL = "com.docker.compose.service"
COMPOSE_NETWORK_LABEL = "com.docker.compose.network"
COMPOSE_VOLUME_LABEL = "com.docker.compose.volume"
VOLUME_KEY = "postgres-tls-smoke-data"
AUTHORIZED_ARTIFACTS = frozenset(
    {
        "docs/runbooks/postgres_tls_service_controller.md",
        "scripts/run_postgres_tls_service.py",
        "tests/test_postgres_tls_service.py",
    }
)
_FIXED_FAILURE = "PR54 PostgreSQL TLS service failed"
_READY = "PR54 PostgreSQL TLS READY; TTL remaining: {seconds} seconds"

E_REPOSITORY = "E_REPOSITORY"
E_PREFLIGHT = "E_PREFLIGHT"
E_TLS = "E_TLS"
E_STARTUP = "E_STARTUP"
E_MIGRATION = "E_MIGRATION"
E_READINESS = "E_READINESS"
E_HANDOFF = "E_HANDOFF"
E_CLEANUP = "E_CLEANUP"
E_PROTECTED_RESOURCES = "E_PROTECTED_RESOURCES"
PHASE_CODES = frozenset(
    {
        E_REPOSITORY,
        E_PREFLIGHT,
        E_TLS,
        E_STARTUP,
        E_MIGRATION,
        E_READINESS,
        E_HANDOFF,
        E_CLEANUP,
        E_PROTECTED_RESOURCES,
    }
)
T = TypeVar("T")


class PostgreSQLTLSServiceError(RuntimeError):
    """A failure whose public representation contains no runtime detail."""

    def __init__(self, *, phase: str = E_PREFLIGHT) -> None:
        self.phase = phase if phase in PHASE_CODES else E_PREFLIGHT
        super().__init__(_FIXED_FAILURE)


def _phase(phase: str, operation: Callable[[], T]) -> T:
    try:
        return operation()
    except Exception as error:
        raise PostgreSQLTLSServiceError(phase=phase) from error


def _run_command(
    arguments: list[str],
    *,
    environment: dict[str, str] | None = None,
    capture_output: bool = True,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=capture_output,
        shell=False,
        timeout=timeout,
    )


def _output(
    arguments: list[str],
    *,
    environment: dict[str, str] | None = None,
    timeout: float = VALIDATION_TIMEOUT_SECONDS,
) -> str:
    try:
        return _run_command(
            arguments,
            environment=environment,
            capture_output=True,
            timeout=timeout,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise PostgreSQLTLSServiceError() from error


@contextmanager
def _bounded_smoke_runner(timeout: float) -> Iterator[None]:
    original = smoke._run

    def bounded_run(
        arguments: list[str],
        *,
        environment: dict[str, str] | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return _run_command(
            arguments,
            environment=environment,
            capture_output=capture_output,
            timeout=timeout,
        )

    smoke._run = bounded_run
    try:
        yield
    finally:
        smoke._run = original


def _bounded_smoke_call(timeout: float, operation: Callable[[], T]) -> T:
    with _bounded_smoke_runner(timeout):
        return operation()


def _parse_arguments(arguments: list[str]) -> tuple[int, bool]:
    preflight = False
    values = list(arguments)
    if "--preflight-only" in values:
        if values.count("--preflight-only") != 1:
            raise PostgreSQLTLSServiceError()
        values.remove("--preflight-only")
        preflight = True
    if len(values) != 2 or values[0] != "--ttl-seconds":
        raise PostgreSQLTLSServiceError()
    raw = values[1]
    if not raw.isascii() or not raw.isdigit():
        raise PostgreSQLTLSServiceError()
    ttl = int(raw)
    if not MINIMUM_TTL_SECONDS <= ttl <= MAXIMUM_TTL_SECONDS:
        raise PostgreSQLTLSServiceError()
    return ttl, preflight


def _validate_repository() -> None:
    expected_head = os.environ.get(EXPECTED_HEAD_VARIABLE)
    if expected_head is None or not COMMIT_PATTERN.fullmatch(expected_head):
        raise PostgreSQLTLSServiceError(phase=E_REPOSITORY)
    if Path.cwd().resolve() != REPOSITORY_ROOT:
        raise PostgreSQLTLSServiceError(phase=E_REPOSITORY)
    if _output(["git", "branch", "--show-current"]) != EXPECTED_BRANCH:
        raise PostgreSQLTLSServiceError(phase=E_REPOSITORY)
    if _output(["git", "rev-parse", "HEAD"]) != expected_head:
        raise PostgreSQLTLSServiceError(phase=E_REPOSITORY)
    if _output(["git", "rev-parse", f"origin/{EXPECTED_BRANCH}"]) != expected_head:
        raise PostgreSQLTLSServiceError(phase=E_REPOSITORY)
    raw = _output(["git", "status", "--porcelain", "-z", "--untracked-files=all"])
    records = [record for record in raw.split("\0") if record]
    observed = {record[3:] for record in records if record.startswith("?? ")}
    if len(records) != len(observed) or observed not in (
        set(),
        set(AUTHORIZED_ARTIFACTS),
    ):
        raise PostgreSQLTLSServiceError(phase=E_REPOSITORY)


def _docker_preflight() -> str:
    docker = shutil.which("docker")
    if docker is None:
        raise PostgreSQLTLSServiceError()
    _output([docker, "--version"])
    _output([docker, "compose", "version"])
    _output([docker, "info", "--format", "{{.ServerVersion}}"])
    if _output([docker, "context", "show"]) != "desktop-linux":
        raise PostgreSQLTLSServiceError()
    if smoke.IMAGE != f"{smoke.IMAGE_TAG}@{smoke.IMAGE_DIGEST}":
        raise PostgreSQLTLSServiceError()
    return docker


def _resource_snapshot(docker: str) -> dict[str, frozenset[str]]:
    commands = {
        "container": [docker, "container", "ls", "-aq"],
        "network": [docker, "network", "ls", "-q"],
        "volume": [docker, "volume", "ls", "-q"],
    }
    return {
        kind: frozenset(value for value in _output(command).splitlines() if value)
        for kind, command in commands.items()
    }


def _inspect_field(
    docker: str,
    resource: str,
    reference: str,
    template: str,
) -> str:
    if not RESOURCE_ID_PATTERN.fullmatch(reference):
        raise PostgreSQLTLSServiceError(phase=E_CLEANUP)
    return _output(
        [docker, resource, "inspect", "--format", template, reference],
        timeout=CLEANUP_TIMEOUT_SECONDS,
    )


def _project_resource_references(
    docker: str,
    project: str,
) -> dict[str, tuple[str, ...]]:
    if not PROJECT_PATTERN.fullmatch(project):
        raise PostgreSQLTLSServiceError(phase=E_CLEANUP)
    commands = {
        "container": [docker, "container", "ls", "-aq"],
        "network": [docker, "network", "ls", "-q"],
        "volume": [docker, "volume", "ls", "-q"],
    }
    observed: dict[str, tuple[str, ...]] = {}
    for resource, command in commands.items():
        output = _output(
            [
                *command,
                "--filter",
                f"label={COMPOSE_PROJECT_LABEL}={project}",
            ],
            timeout=CLEANUP_TIMEOUT_SECONDS,
        )
        references = tuple(line for line in output.splitlines() if line)
        if len(references) != len(set(references)) or any(
            not RESOURCE_ID_PATTERN.fullmatch(reference) for reference in references
        ):
            raise PostgreSQLTLSServiceError(phase=E_CLEANUP)
        observed[resource] = references
    return observed


def _validate_exact_project_resources(
    docker: str,
    project: str,
    resources: dict[str, tuple[str, ...]],
) -> None:
    containers = resources.get("container", ())
    networks = resources.get("network", ())
    volumes = resources.get("volume", ())
    if set(resources) != {"container", "network", "volume"}:
        raise PostgreSQLTLSServiceError(phase=E_CLEANUP)
    if len(containers) > 2 or len(networks) > 1 or len(volumes) > 1:
        raise PostgreSQLTLSServiceError(phase=E_CLEANUP)

    observed_services: set[str] = set()
    for reference in containers:
        name = _inspect_field(docker, "container", reference, "{{.Name}}").removeprefix(
            "/"
        )
        labeled_project = _inspect_field(
            docker,
            "container",
            reference,
            f'{{{{index .Config.Labels "{COMPOSE_PROJECT_LABEL}"}}}}',
        )
        service = _inspect_field(
            docker,
            "container",
            reference,
            f'{{{{index .Config.Labels "{COMPOSE_SERVICE_LABEL}"}}}}',
        )
        if labeled_project != project or service in observed_services:
            raise PostgreSQLTLSServiceError(phase=E_CLEANUP)
        observed_services.add(service)
        if service == smoke.SERVICE:
            if name != f"{project}-{smoke.SERVICE}-1":
                raise PostgreSQLTLSServiceError(phase=E_CLEANUP)
        elif service == "certificate-init":
            prefix = f"{project}-certificate-init-run-"
            suffix = name.removeprefix(prefix)
            if name == suffix or not CERTIFICATE_CONTAINER_SUFFIX_PATTERN.fullmatch(
                suffix
            ):
                raise PostgreSQLTLSServiceError(phase=E_CLEANUP)
        else:
            raise PostgreSQLTLSServiceError(phase=E_CLEANUP)

    for resource, references, expected_name, label_name, expected_label in (
        (
            "network",
            networks,
            f"{project}_default",
            COMPOSE_NETWORK_LABEL,
            "default",
        ),
        (
            "volume",
            volumes,
            f"{project}_{VOLUME_KEY}",
            COMPOSE_VOLUME_LABEL,
            VOLUME_KEY,
        ),
    ):
        for reference in references:
            if (
                _inspect_field(docker, resource, reference, "{{.Name}}")
                != expected_name
                or _inspect_field(
                    docker,
                    resource,
                    reference,
                    f'{{{{index .Labels "{COMPOSE_PROJECT_LABEL}"}}}}',
                )
                != project
                or _inspect_field(
                    docker,
                    resource,
                    reference,
                    f'{{{{index .Labels "{label_name}"}}}}',
                )
                != expected_label
            ):
                raise PostgreSQLTLSServiceError(phase=E_CLEANUP)


def _exact_project_fallback(docker: str, project: str) -> None:
    resources = _project_resource_references(docker, project)
    _validate_exact_project_resources(docker, project, resources)
    for resource, remove_arguments in (
        ("container", ("container", "rm", "--force")),
        ("network", ("network", "rm")),
        ("volume", ("volume", "rm")),
    ):
        for reference in resources[resource]:
            _run_command(
                [docker, *remove_arguments, reference],
                capture_output=True,
                timeout=CLEANUP_TIMEOUT_SECONDS,
            )


def _cleanup_project(
    docker: str,
    project: str,
    environment: dict[str, str],
) -> None:
    down_failed = False
    try:
        _bounded_smoke_call(
            CLEANUP_TIMEOUT_SECONDS,
            lambda: smoke._cleanup(docker, project, environment),
        )
    except BaseException:
        down_failed = True
    if not down_failed:
        try:
            _bounded_smoke_call(
                CLEANUP_TIMEOUT_SECONDS,
                lambda: smoke._require_no_project_resources(docker, project),
            )
            return
        except BaseException:
            pass
    _exact_project_fallback(docker, project)
    _bounded_smoke_call(
        CLEANUP_TIMEOUT_SECONDS,
        lambda: smoke._require_no_project_resources(docker, project),
    )


def _private_root(raw: str | None) -> Path:
    if not raw:
        raise PostgreSQLTLSServiceError(phase=E_HANDOFF)
    root = Path(raw)
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise PostgreSQLTLSServiceError(phase=E_HANDOFF) from error
    if (
        root != resolved
        or root.is_symlink()
        or not root.is_dir()
        or REPOSITORY_ROOT in resolved.parents
    ):
        raise PostgreSQLTLSServiceError(phase=E_HANDOFF)
    return resolved


def _write_private(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
    path.chmod(0o600)
    _restrict_owner(path, directory=False)


def _restrict_owner(path: Path, *, directory: bool) -> None:
    if os.name != "nt":
        path.chmod(0o700 if directory else 0o600)
        return
    permission = "(OI)(CI)F" if directory else "F"
    try:
        identity = subprocess.run(
            ["whoami", "/user", "/fo", "csv", "/nh"],
            check=True,
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            shell=False,
            timeout=IDENTITY_ACL_TIMEOUT_SECONDS,
        ).stdout
        row = next(csv.reader(io.StringIO(identity)))
        if len(row) != 2 or not row[1].startswith("S-"):
            raise PostgreSQLTLSServiceError(phase=E_HANDOFF)
        subprocess.run(
            [
                "icacls",
                str(path),
                "/inheritance:r",
                "/grant:r",
                f"*{row[1]}:{permission}",
            ],
            check=True,
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            shell=False,
            timeout=IDENTITY_ACL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PostgreSQLTLSServiceError(phase=E_HANDOFF) from error


def _create_handoff(
    root: Path, certificate_directory: Path, dsn: str, ttl: int
) -> Path:
    handoff = Path(tempfile.mkdtemp(prefix="callmetric-postgres-tls-", dir=root))
    handoff.chmod(0o700)
    try:
        _restrict_owner(handoff, directory=True)
        _write_private(
            handoff / "ca.crt", (certificate_directory / "ca.crt").read_bytes()
        )
        _write_private(handoff / "application.dsn", dsn.encode("utf-8"))
        metadata = {
            "ca_file": "ca.crt",
            "dsn_file": "application.dsn",
            "host": smoke.TLS_HOST,
            "sslmode": "verify-full",
            "ttl_seconds": ttl,
        }
        _write_private(
            handoff / "connection.json",
            (
                json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode(),
        )
        if frozenset(item.name for item in handoff.iterdir()) != HANDOFF_FILES:
            raise PostgreSQLTLSServiceError(phase=E_HANDOFF)
        if os.name != "nt" and (
            handoff.stat().st_mode & 0o077
            or any(item.stat().st_mode & 0o077 for item in handoff.iterdir())
        ):
            raise PostgreSQLTLSServiceError(phase=E_HANDOFF)
        return handoff
    except Exception:
        shutil.rmtree(handoff)
        raise


def _remove_handoff(handoff: Path | None, root: Path) -> None:
    if handoff is None:
        return
    resolved = handoff.resolve(strict=True)
    if resolved.parent != root or not HANDOFF_PATTERN.fullmatch(resolved.name):
        raise PostgreSQLTLSServiceError(phase=E_CLEANUP)
    shutil.rmtree(resolved)


def _signals() -> tuple[signal.Signals, ...]:
    selected = [signal.SIGINT, signal.SIGTERM]
    sighup = getattr(signal, "SIGHUP", None)
    if isinstance(sighup, signal.Signals):
        selected.append(sighup)
    return tuple(selected)


def _install_handlers(stop: threading.Event) -> dict[signal.Signals, Any]:
    previous: dict[signal.Signals, Any] = {}

    def request_stop(_number: int, _frame: FrameType | None) -> None:
        stop.set()

    try:
        for item in _signals():
            previous[item] = signal.signal(item, request_stop)
    except BaseException:
        _restore_handlers(previous)
        raise
    return previous


def _restore_handlers(previous: dict[signal.Signals, Any]) -> None:
    for item, handler in previous.items():
        signal.signal(item, handler)


def _wait(ttl: int, stop: threading.Event) -> None:
    deadline = time.monotonic() + ttl
    print(
        _READY.format(seconds=max(0, math.ceil(deadline - time.monotonic()))),
        flush=True,
    )
    stop.wait(timeout=max(0.0, deadline - time.monotonic()))


def _capture(
    current: BaseException | None, phase: str, operation: Callable[[], object]
) -> BaseException | None:
    try:
        _phase(phase, operation)
    except BaseException as error:
        return current or error
    return current


def run(ttl: int, *, preflight_only: bool = False) -> None:
    if not MINIMUM_TTL_SECONDS <= ttl <= MAXIMUM_TTL_SECONDS:
        raise PostgreSQLTLSServiceError()
    _phase(E_REPOSITORY, _validate_repository)
    docker = _phase(E_PREFLIGHT, _docker_preflight)
    before = _phase(E_PROTECTED_RESOURCES, lambda: _resource_snapshot(docker))
    if preflight_only:
        return
    root = _phase(
        E_HANDOFF, lambda: _private_root(os.environ.get(HANDOFF_ROOT_VARIABLE))
    )
    project = f"callmetric-pgvector-tls-{os.getpid()}-{secrets.token_hex(6)}"
    if not PROJECT_PATTERN.fullmatch(project):
        raise PostgreSQLTLSServiceError()
    migration_password = secrets.token_urlsafe(32)
    application_password = secrets.token_urlsafe(32)
    stop = threading.Event()
    previous = _phase(E_PREFLIGHT, lambda: _install_handlers(stop))
    handoff: Path | None = None
    primary: BaseException | None = None
    cleanup: BaseException | None = None
    environment: dict[str, str] | None = None
    resource_creation_attempted = False
    try:
        with tempfile.TemporaryDirectory(
            prefix="callmetric-postgres-tls-service-"
        ) as raw_tls:
            tls = Path(raw_tls)
            environment = smoke._smoke_environment(
                certificate_directory=tls, migration_password=migration_password
            )
            try:
                _phase(
                    E_PREFLIGHT,
                    lambda: _run_command(
                        smoke._compose_arguments(docker, project, "config", "--quiet"),
                        environment=environment,
                        capture_output=True,
                        timeout=COMPOSE_CONFIG_TIMEOUT_SECONDS,
                    ),
                )
                resource_creation_attempted = True
                _phase(
                    E_TLS,
                    lambda: _bounded_smoke_call(
                        CERTIFICATE_TIMEOUT_SECONDS,
                        lambda: smoke._generate_certificates(
                            docker, project, environment
                        ),
                    ),
                )
                _phase(
                    E_TLS,
                    lambda: _bounded_smoke_call(
                        CERTIFICATE_TIMEOUT_SECONDS,
                        lambda: smoke._validate_certificates(docker, tls),
                    ),
                )
                _phase(
                    E_STARTUP,
                    lambda: _run_command(
                        smoke._compose_arguments(
                            docker, project, "up", "-d", smoke.SERVICE
                        ),
                        environment=environment,
                        capture_output=True,
                        timeout=COMPOSE_STARTUP_TIMEOUT_SECONDS,
                    ),
                )
                _phase(
                    E_READINESS,
                    lambda: _bounded_smoke_call(
                        READINESS_COMMAND_TIMEOUT_SECONDS,
                        lambda: smoke._wait_until_healthy(docker, project, environment),
                    ),
                )
                port = _phase(
                    E_READINESS,
                    lambda: _bounded_smoke_call(
                        READINESS_COMMAND_TIMEOUT_SECONDS,
                        lambda: smoke._published_port(docker, project, environment),
                    ),
                )
                test_environment = smoke._pytest_environment(
                    port=port,
                    certificate_directory=tls,
                    migration_password=migration_password,
                    application_password=application_password,
                )
                _phase(
                    E_MIGRATION,
                    lambda: _run_command(
                        [
                            sys.executable,
                            "-m",
                            "pytest",
                            "-m",
                            "postgres_integration",
                            str(smoke.INTEGRATION_TEST),
                        ],
                        environment=test_environment,
                        capture_output=True,
                        timeout=MIGRATION_PROOF_TIMEOUT_SECONDS,
                    ),
                )
                dsn = (
                    smoke._connection_dsn(
                        user=smoke.APPLICATION_USER,
                        password=application_password,
                        port=port,
                        certificate_directory=tls,
                    )
                    + "&sslmode=verify-full"
                )
                handoff = _phase(
                    E_HANDOFF, lambda: _create_handoff(root, tls, dsn, ttl)
                )
                _phase(E_READINESS, lambda: _wait(ttl, stop))
            except BaseException as error:
                primary = error
            finally:
                if resource_creation_attempted and environment is not None:
                    cleanup = _capture(
                        cleanup,
                        E_CLEANUP,
                        lambda: _cleanup_project(docker, project, environment),
                    )
                cleanup = _capture(
                    cleanup, E_CLEANUP, lambda: _remove_handoff(handoff, root)
                )
    except BaseException as error:
        primary = primary or error

    def require_preserved_resources() -> None:
        if _resource_snapshot(docker) != before:
            raise PostgreSQLTLSServiceError(phase=E_PROTECTED_RESOURCES)

    protected = _capture(
        None,
        E_PROTECTED_RESOURCES,
        require_preserved_resources,
    )
    restored = _capture(None, E_CLEANUP, lambda: _restore_handlers(previous))
    if primary:
        raise primary
    if cleanup:
        raise cleanup
    if protected:
        raise protected
    if restored:
        raise restored


def main(arguments: list[str] | None = None) -> int:
    try:
        ttl, preflight = _parse_arguments(
            sys.argv[1:] if arguments is None else arguments
        )
        run(ttl, preflight_only=preflight)
    except PostgreSQLTLSServiceError as error:
        print(f"{error.phase} {_FIXED_FAILURE}", file=sys.stderr)
        return 1
    except Exception:
        print(f"{E_PREFLIGHT} {_FIXED_FAILURE}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
