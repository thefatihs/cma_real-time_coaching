"""Run a TTL-bounded, loopback-only vLLM service for the PR54 E2E test."""

from __future__ import annotations

from collections.abc import Callable
import json
import math
import os
import re
import secrets
import shutil
import signal
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any, TypeVar

from scripts import run_vllm_loopback_smoke as smoke

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ubuntu/beyza/cma_real-time_coaching-pr54")
EXPECTED_HOST = "ip-172-31-46-151"
EXPECTED_BRANCH = "feat/rag-coaching-integration"
BASELINE_COMMIT = "09dc12df0c06184435054e038988cb3ede596761"
EXPECTED_HEAD_ENVIRONMENT_VARIABLE = "CALLMETRIC_VLLM_E2E_EXPECTED_HEAD"
HANDOFF_ROOT_ENVIRONMENT_VARIABLE = "CALLMETRIC_VLLM_E2E_HANDOFF_ROOT"
CACHE_ENVIRONMENT_VARIABLE = smoke.CACHE_ENVIRONMENT_VARIABLE
HF_OFFLINE_ENVIRONMENT_VARIABLE = "CALLMETRIC_VLLM_HF_HUB_OFFLINE"
TRANSFORMERS_OFFLINE_ENVIRONMENT_VARIABLE = "CALLMETRIC_VLLM_TRANSFORMERS_OFFLINE"
COMMIT_PATTERN = smoke.COMMIT_PATTERN

IMAGE_TAG = smoke.IMAGE_TAG
IMAGE_INDEX_DIGEST = smoke.IMAGE_INDEX_DIGEST
IMAGE_AMD64_DIGEST = smoke.IMAGE_AMD64_DIGEST
IMAGE = smoke.IMAGE
MODEL = smoke.MODEL
MODEL_REVISION = smoke.MODEL_REVISION
SERVED_MODEL = smoke.SERVED_MODEL
LOOPBACK_HOST = smoke.LOOPBACK_HOST
TLS_HOST = smoke.TLS_HOST
PORT = smoke.PORT
SERVICE = smoke.SERVICE
COMPOSE_FILE = smoke.COMPOSE_FILE

GPU_DEVICE = "0"
MAX_MODEL_LENGTH = 8192
GPU_MEMORY_UTILIZATION = 0.80
MAX_SEQUENCES = 2
MAX_OUTPUT_TOKENS = 256
MINIMUM_TTL_SECONDS = 300
MAXIMUM_TTL_SECONDS = 7_200
COLD_MINIMUM_FREE_BYTES = smoke.MINIMUM_FREE_BYTES
WARM_RUNTIME_OVERHEAD_BYTES = 2 * 1024**3
WARM_MINIMUM_FREE_BYTES = (
    smoke.RESERVE_BYTES + smoke.MARGIN_BYTES + WARM_RUNTIME_OVERHEAD_BYTES
)
MODEL_SNAPSHOT_RELATIVE_PATH = (
    Path("hub") / f"models--{MODEL.replace('/', '--')}" / "snapshots" / MODEL_REVISION
)
SUBPROCESS_TIMEOUT_SECONDS = 30.0
PULL_TIMEOUT_SECONDS = smoke.PULL_TIMEOUT_SECONDS
START_TIMEOUT_SECONDS = smoke.START_TIMEOUT_SECONDS
HEALTH_TIMEOUT_SECONDS = smoke.HEALTH_TIMEOUT_SECONDS
HTTP_TIMEOUT_SECONDS = smoke.HTTP_TIMEOUT_SECONDS
SHUTDOWN_TIMEOUT_SECONDS = 120.0
GPU_IDLE_TIMEOUT_SECONDS = 30.0
GPU_IDLE_POLL_SECONDS = 0.5
PROJECT_PATTERN = re.compile(r"^callmetric-vllm-e2e-[0-9]+-[a-f0-9]{12}$")
HANDOFF_DIRECTORY_PATTERN = re.compile(r"^callmetric-vllm-e2e-[a-z0-9_]{8}$")
AUTHORIZED_DEVELOPMENT_ARTIFACTS = frozenset(
    {
        "docs/runbooks/vllm_e2e_service_controller.md",
        "scripts/run_vllm_e2e_service.py",
        "tests/test_vllm_e2e_service.py",
    }
)
HANDOFF_FILES = frozenset({"ca.crt", "token", "connection.json"})
_FIXED_FAILURE = "PR54 vLLM service failed"
_READY_MESSAGE = "PR54 vLLM READY; TTL remaining: {seconds} seconds"


E_REPOSITORY = "E_REPOSITORY"
E_RUNTIME_CONTRACT = "E_RUNTIME_CONTRACT"
E_GPU = "E_GPU"
E_GPU_ACTIVITY = "E_GPU_ACTIVITY"
E_GPU_CLEANUP = "E_GPU_CLEANUP"
E_DISK_CAPACITY = "E_DISK_CAPACITY"
E_CACHE_METADATA = "E_CACHE_METADATA"
E_MODEL_METADATA = "E_MODEL_METADATA"
E_IMAGE_METADATA = "E_IMAGE_METADATA"
E_TLS = "E_TLS"
E_STARTUP = "E_STARTUP"
E_READINESS = "E_READINESS"
E_CLEANUP = "E_CLEANUP"
E_PROTECTED_CONTAINERS = "E_PROTECTED_CONTAINERS"
PHASE_CODES = frozenset(
    {
        E_REPOSITORY,
        E_RUNTIME_CONTRACT,
        E_GPU,
        E_GPU_ACTIVITY,
        E_GPU_CLEANUP,
        E_DISK_CAPACITY,
        E_CACHE_METADATA,
        E_MODEL_METADATA,
        E_IMAGE_METADATA,
        E_TLS,
        E_STARTUP,
        E_READINESS,
        E_CLEANUP,
        E_PROTECTED_CONTAINERS,
    }
)
T = TypeVar("T")


class VLLME2EServiceError(RuntimeError):
    """A fixed, secret-safe service lifecycle failure."""

    def __init__(
        self,
        _message: str = _FIXED_FAILURE,
        *,
        phase: str = E_RUNTIME_CONTRACT,
    ) -> None:
        self.phase = phase if phase in PHASE_CODES else E_RUNTIME_CONTRACT
        super().__init__(_FIXED_FAILURE)


def _phase_call(phase: str, operation: Callable[[], T]) -> T:
    try:
        return operation()
    except Exception as error:
        raise VLLME2EServiceError(phase=phase) from error


def _run(
    arguments: list[str],
    *,
    environment: dict[str, str] | None = None,
    cwd: Path = REPOSITORY_ROOT,
    timeout: float = SUBPROCESS_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            check=True,
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            shell=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise VLLME2EServiceError(_FIXED_FAILURE) from error


def _output(arguments: list[str], **kwargs: Any) -> str:
    return _run(arguments, **kwargs).stdout.strip()


def _compose_arguments(project: str, *arguments: str) -> list[str]:
    if not PROJECT_PATTERN.fullmatch(project):
        raise VLLME2EServiceError(_FIXED_FAILURE)
    return [
        "docker",
        "compose",
        "--project-name",
        project,
        "--file",
        str(COMPOSE_FILE),
        *arguments,
    ]


def _parse_ttl(arguments: list[str]) -> int:
    if len(arguments) != 2 or arguments[0] != "--ttl-seconds":
        raise VLLME2EServiceError(_FIXED_FAILURE)
    raw_ttl = arguments[1]
    if not raw_ttl.isascii() or not raw_ttl.isdigit():
        raise VLLME2EServiceError(_FIXED_FAILURE)
    ttl = int(raw_ttl)
    if not MINIMUM_TTL_SECONDS <= ttl <= MAXIMUM_TTL_SECONDS:
        raise VLLME2EServiceError(_FIXED_FAILURE)
    return ttl


def _validate_worktree_status(raw_status: str) -> None:
    if raw_status == "":
        return
    if not raw_status.endswith("\0"):
        raise VLLME2EServiceError(_FIXED_FAILURE)
    records = raw_status.split("\0")
    if records[-1] or len(records) - 1 != len(AUTHORIZED_DEVELOPMENT_ARTIFACTS):
        raise VLLME2EServiceError(_FIXED_FAILURE)
    observed: set[str] = set()
    for record in records[:-1]:
        if len(record) < 4 or record[:2] != "??" or record[2] != " ":
            raise VLLME2EServiceError(_FIXED_FAILURE)
        path = record[3:]
        if path not in AUTHORIZED_DEVELOPMENT_ARTIFACTS or path in observed:
            raise VLLME2EServiceError(_FIXED_FAILURE)
        observed.add(path)
    if observed != AUTHORIZED_DEVELOPMENT_ARTIFACTS:
        raise VLLME2EServiceError(_FIXED_FAILURE)


def _validate_repository() -> None:
    if Path.cwd().resolve() != EXPECTED_ROOT or REPOSITORY_ROOT != EXPECTED_ROOT:
        raise VLLME2EServiceError(_FIXED_FAILURE)
    expected_head = os.environ.get(EXPECTED_HEAD_ENVIRONMENT_VARIABLE)
    if expected_head is None or not COMMIT_PATTERN.fullmatch(expected_head):
        raise VLLME2EServiceError(_FIXED_FAILURE)
    checks = (
        (["hostname"], EXPECTED_HOST),
        (["git", "branch", "--show-current"], EXPECTED_BRANCH),
        (["git", "rev-parse", "HEAD"], expected_head),
        (["git", "rev-parse", "origin/main"], BASELINE_COMMIT),
    )
    for command, expected in checks:
        if _output(command) != expected:
            raise VLLME2EServiceError(_FIXED_FAILURE)
    _run(["git", "merge-base", "--is-ancestor", BASELINE_COMMIT, expected_head])
    status = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "-z"]
    ).stdout
    _validate_worktree_status(status)


def _validate_runtime_contract() -> None:
    compose = COMPOSE_FILE.read_text(encoding="utf-8")
    required = (
        f'"{LOOPBACK_HOST}:{PORT}:{PORT}"',
        IMAGE,
        f'device_ids: ["{GPU_DEVICE}"]',
        '- "8192"',
        '- "0.80"',
        '- "2"',
        MODEL,
        MODEL_REVISION,
        SERVED_MODEL,
        "subjectAltName=DNS:localhost,IP:127.0.0.1",
        'HF_HUB_OFFLINE: "${CALLMETRIC_VLLM_HF_HUB_OFFLINE:-0}"',
        'TRANSFORMERS_OFFLINE: "${CALLMETRIC_VLLM_TRANSFORMERS_OFFLINE:-0}"',
    )
    combined = compose + Path(smoke.__file__).read_text(encoding="utf-8")
    if any(value not in combined for value in required):
        raise VLLME2EServiceError(_FIXED_FAILURE)
    forbidden = ("0.0.0.0:8001", "[::]", "container_name")
    if any(value in compose for value in forbidden):
        raise VLLME2EServiceError(_FIXED_FAILURE)


def _exact_local_image_available() -> bool:
    try:
        metadata = _output(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Id}}|{{.Architecture}}|{{.Os}}",
                IMAGE,
            ]
        )
    except VLLME2EServiceError:
        return False
    return metadata == f"{IMAGE_AMD64_DIGEST}|amd64|linux"


def _exact_model_snapshot_available(cache: Path) -> bool:
    snapshot = cache / MODEL_SNAPSHOT_RELATIVE_PATH
    try:
        resolved = snapshot.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    return (
        not snapshot.is_symlink()
        and resolved == snapshot
        and resolved.is_dir()
        and cache in resolved.parents
    )


def _validate_free_disk(required_bytes: int) -> None:
    if required_bytes <= 0 or smoke._free_disk_bytes() < required_bytes:
        raise VLLME2EServiceError(_FIXED_FAILURE)


def _validated_cache_and_disk() -> tuple[Path, bool]:
    raw_cache = os.environ.get(CACHE_ENVIRONMENT_VARIABLE)
    try:
        cache = smoke._validate_cache_directory(raw_cache)
    except Exception as error:
        _phase_call(
            E_DISK_CAPACITY,
            lambda: _validate_free_disk(COLD_MINIMUM_FREE_BYTES),
        )
        raise VLLME2EServiceError(phase=E_CACHE_METADATA) from error

    warm = _exact_local_image_available() and _exact_model_snapshot_available(cache)
    required_bytes = WARM_MINIMUM_FREE_BYTES if warm else COLD_MINIMUM_FREE_BYTES
    _phase_call(
        E_DISK_CAPACITY,
        lambda: _validate_free_disk(required_bytes),
    )
    return cache, warm


def _validate_private_directory(raw_path: str | None) -> Path:
    if raw_path is None or not raw_path or "\x00" in raw_path:
        raise VLLME2EServiceError(_FIXED_FAILURE)
    candidate = Path(raw_path)
    if not candidate.is_absolute() or candidate == Path("/"):
        raise VLLME2EServiceError(_FIXED_FAILURE)
    try:
        resolved = candidate.resolve(strict=True)
        stat = resolved.stat()
    except OSError as error:
        raise VLLME2EServiceError(_FIXED_FAILURE) from error
    if (
        candidate != resolved
        or candidate.is_symlink()
        or not resolved.is_dir()
        or resolved == REPOSITORY_ROOT
        or REPOSITORY_ROOT in resolved.parents
        or stat.st_uid != os.getuid()
        or stat.st_mode & 0o077
    ):
        raise VLLME2EServiceError(_FIXED_FAILURE)
    return resolved


def _write_owner_only(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    except OSError as error:
        raise VLLME2EServiceError(_FIXED_FAILURE) from error


def _create_handoff(root: Path, tls_directory: Path, token: str, ttl: int) -> Path:
    raw_directory = tempfile.mkdtemp(prefix="callmetric-vllm-e2e-", dir=root)
    handoff = Path(raw_directory)
    handoff.chmod(0o700)
    try:
        _write_owner_only(handoff / "ca.crt", (tls_directory / "ca.crt").read_bytes())
        _write_owner_only(handoff / "token", token.encode("utf-8"))
        metadata = {
            "host": TLS_HOST,
            "port": PORT,
            "model": SERVED_MODEL,
            "ca_file": "ca.crt",
            "token_file": "token",
            "ttl_seconds": ttl,
        }
        _write_owner_only(
            handoff / "connection.json",
            (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )
        if (
            frozenset(path.name for path in handoff.iterdir()) != HANDOFF_FILES
            or handoff.stat().st_mode & 0o077
            or any(path.stat().st_mode & 0o077 for path in handoff.iterdir())
        ):
            raise VLLME2EServiceError(_FIXED_FAILURE)
        return handoff
    except Exception:
        shutil.rmtree(handoff)
        raise


def _remove_handoff(handoff: Path | None, root: Path) -> None:
    if handoff is None:
        return
    try:
        resolved = handoff.resolve(strict=True)
    except OSError as error:
        raise VLLME2EServiceError(_FIXED_FAILURE) from error
    if resolved.parent != root or not HANDOFF_DIRECTORY_PATTERN.fullmatch(
        resolved.name
    ):
        raise VLLME2EServiceError(_FIXED_FAILURE)
    shutil.rmtree(resolved)


def _environment(cache: Path, tls_directory: Path, token: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            CACHE_ENVIRONMENT_VARIABLE: str(cache),
            "CALLMETRIC_VLLM_SMOKE_TLS_DIR": str(tls_directory),
            "CALLMETRIC_VLLM_SMOKE_API_KEY": token,
            HF_OFFLINE_ENVIRONMENT_VARIABLE: "1",
            TRANSFORMERS_OFFLINE_ENVIRONMENT_VARIABLE: "1",
        }
    )
    return environment


def _wait_for_ready(context: ssl.SSLContext, token: str) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            health_status, _ = smoke._request("/health", context=context, token=token)
            model_status, payload = smoke._request(
                "/v1/models", context=context, token=token
            )
            identifiers = [
                item.get("id")
                for item in payload.get("data", [])
                if isinstance(item, dict)
            ]
            if (
                health_status == 200
                and model_status == 200
                and identifiers == [SERVED_MODEL]
            ):
                return
        except smoke.VLLMSmokeError:
            pass
        time.sleep(1.0)
    raise VLLME2EServiceError(_FIXED_FAILURE)


def _wait_foreground(ttl: int, stop_event: threading.Event) -> None:
    deadline = time.monotonic() + ttl
    previous_handlers: dict[signal.Signals, Any] = {}

    def request_stop(_signal_number: int, _frame: FrameType | None) -> None:
        stop_event.set()

    try:
        for selected_signal in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[selected_signal] = signal.signal(
                selected_signal, request_stop
            )
        remaining = max(0, math.ceil(deadline - time.monotonic()))
        print(_READY_MESSAGE.format(seconds=remaining), flush=True)
        stop_event.wait(timeout=max(0.0, deadline - time.monotonic()))
    finally:
        for selected_signal, previous in previous_handlers.items():
            signal.signal(selected_signal, previous)


def _cleanup(project: str, environment: dict[str, str]) -> None:
    _run(
        _compose_arguments(project, "down", "--remove-orphans", "--timeout", "30"),
        environment=environment,
        timeout=SHUTDOWN_TIMEOUT_SECONDS,
    )


def _validate_gpu_idle() -> None:
    if smoke._gpu_memory_used() != 0:
        raise VLLME2EServiceError(_FIXED_FAILURE)


def _validate_gpu_active() -> None:
    if smoke._gpu_memory_used() <= 0:
        raise VLLME2EServiceError(_FIXED_FAILURE)


def _wait_for_gpu_idle() -> None:
    deadline = time.monotonic() + GPU_IDLE_TIMEOUT_SECONDS
    while smoke._gpu_memory_used() != 0:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise VLLME2EServiceError(_FIXED_FAILURE)
        time.sleep(min(GPU_IDLE_POLL_SECONDS, remaining))


def _capture_failure(
    current: BaseException | None,
    phase: str,
    operation: Callable[[], object],
) -> BaseException | None:
    try:
        _phase_call(phase, operation)
    except BaseException as error:
        return current if current is not None else error
    return current


def run(ttl: int) -> None:
    if not MINIMUM_TTL_SECONDS <= ttl <= MAXIMUM_TTL_SECONDS:
        raise VLLME2EServiceError(phase=E_RUNTIME_CONTRACT)
    _phase_call(E_REPOSITORY, _validate_repository)
    _phase_call(E_RUNTIME_CONTRACT, _validate_runtime_contract)
    _phase_call(E_RUNTIME_CONTRACT, smoke._validate_tools)
    _phase_call(E_GPU, smoke._validate_gpu)
    _phase_call(E_GPU, _validate_gpu_idle)
    cache, _warm = _validated_cache_and_disk()
    _phase_call(E_RUNTIME_CONTRACT, smoke._validate_port_free)
    handoff_root = _phase_call(
        E_TLS,
        lambda: _validate_private_directory(
            os.environ.get(HANDOFF_ROOT_ENVIRONMENT_VARIABLE)
        ),
    )
    protected_before = _phase_call(
        E_PROTECTED_CONTAINERS,
        smoke._protected_container_snapshot,
    )
    _phase_call(E_IMAGE_METADATA, smoke._validate_image_metadata)
    _phase_call(E_MODEL_METADATA, smoke._validate_model_metadata)

    project = f"callmetric-vllm-e2e-{os.getpid()}-{secrets.token_hex(6)}"
    token = secrets.token_urlsafe(48)
    if not PROJECT_PATTERN.fullmatch(project) or len(token) < 48:
        raise VLLME2EServiceError(phase=E_RUNTIME_CONTRACT)

    environment: dict[str, str] | None = None
    handoff: Path | None = None
    startup_attempted = False
    primary_failure: BaseException | None = None
    cleanup_failure: BaseException | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="callmetric-vllm-e2e-tls-") as raw_tls:
            tls_directory = Path(raw_tls)
            try:
                _phase_call(E_TLS, lambda: smoke._generate_certificates(tls_directory))
                handoff = _phase_call(
                    E_TLS,
                    lambda: _create_handoff(handoff_root, tls_directory, token, ttl),
                )
                environment = _environment(cache, tls_directory, token)
                _phase_call(
                    E_RUNTIME_CONTRACT,
                    lambda: _run(
                        _compose_arguments(project, "config", "--quiet"),
                        environment=environment,
                    ),
                )
                _phase_call(
                    E_IMAGE_METADATA,
                    lambda: _run(
                        _compose_arguments(project, "pull", SERVICE),
                        environment=environment,
                        timeout=PULL_TIMEOUT_SECONDS,
                    ),
                )
                startup_attempted = True
                _phase_call(
                    E_STARTUP,
                    lambda: _run(
                        _compose_arguments(project, "up", "--detach", SERVICE),
                        environment=environment,
                        timeout=START_TIMEOUT_SECONDS,
                    ),
                )
                context = _phase_call(
                    E_TLS,
                    lambda: ssl.create_default_context(
                        cafile=str(tls_directory / "ca.crt")
                    ),
                )
                _phase_call(E_READINESS, lambda: _wait_for_ready(context, token))
                _phase_call(E_GPU_ACTIVITY, _validate_gpu_active)
                _phase_call(
                    E_RUNTIME_CONTRACT,
                    lambda: _wait_foreground(ttl, threading.Event()),
                )
            except BaseException as error:
                primary_failure = error
            finally:
                if startup_attempted and environment is not None:
                    cleanup_failure = _capture_failure(
                        cleanup_failure,
                        E_CLEANUP,
                        lambda: _cleanup(project, environment),
                    )
                    cleanup_failure = _capture_failure(
                        cleanup_failure,
                        E_GPU_CLEANUP,
                        _wait_for_gpu_idle,
                    )
                cleanup_failure = _capture_failure(
                    cleanup_failure,
                    E_CLEANUP,
                    lambda: _remove_handoff(handoff, handoff_root),
                )
    except BaseException:
        cleanup_failure = cleanup_failure or VLLME2EServiceError(phase=E_CLEANUP)

    protected_failure: BaseException | None = None
    try:
        protected_after = _phase_call(
            E_PROTECTED_CONTAINERS,
            smoke._protected_container_snapshot,
        )
        if protected_after != protected_before:
            raise VLLME2EServiceError(phase=E_PROTECTED_CONTAINERS)
    except BaseException as error:
        protected_failure = error

    if primary_failure is not None:
        raise primary_failure
    if cleanup_failure is not None:
        raise cleanup_failure
    if protected_failure is not None:
        raise protected_failure
    _phase_call(E_RUNTIME_CONTRACT, smoke._validate_port_free)
    _phase_call(E_REPOSITORY, _validate_repository)


def main(arguments: list[str] | None = None) -> int:
    try:
        ttl = _parse_ttl(sys.argv[1:] if arguments is None else arguments)
        run(ttl)
    except VLLME2EServiceError as error:
        print(f"{error.phase} {_FIXED_FAILURE}", file=sys.stderr)
        return 1
    except Exception:
        print(f"{E_RUNTIME_CONTRACT} {_FIXED_FAILURE}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
