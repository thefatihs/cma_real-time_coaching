"""Run the isolated, synthetic-only vLLM loopback TLS smoke lifecycle."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ubuntu/beyza/cma_real-time_coaching-pr53")
COMPOSE_FILE = REPOSITORY_ROOT / "compose.vllm-loopback-smoke.yml"
EXPECTED_HOST = "ip-172-31-46-151"
EXPECTED_BRANCH = "feat/vllm-deployment"
BASELINE_COMMIT = "0418228e189a5b968654a6308fe520b936256ac4"
EXPECTED_HEAD_ENVIRONMENT_VARIABLE = "CALLMETRIC_VLLM_SMOKE_EXPECTED_HEAD"
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
EXPECTED_ARTIFACT_PATHS = frozenset(
    {
        "compose.vllm-loopback-smoke.yml",
        "docs/runbooks/vllm_loopback_smoke.md",
        "scripts/run_vllm_loopback_smoke.py",
        "tests/test_vllm_loopback_smoke_runner.py",
    }
)

IMAGE_TAG = "vllm/vllm-openai:v0.26.0-ubuntu2404"
IMAGE_INDEX_DIGEST = (
    "sha256:ef7bfc14df9233e3e5d41e733e3be0afa6abbe5ae5f14ee0758110030f6cd53e"
)
IMAGE_AMD64_DIGEST = (
    "sha256:1161da8a5edbdff239ab1812784d7fe5d28775c675809a8420e8a0a05d0e56d1"
)
IMAGE = f"{IMAGE_TAG}@{IMAGE_AMD64_DIGEST}"
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
MODEL_REVISION = "b25037543e9394b818fdfca67ab2a00ecc7dd641"
SERVED_MODEL = "callmetric-qwen25-7b-awq"
MODEL_API_URL = f"https://huggingface.co/api/models/{MODEL}/revision/{MODEL_REVISION}"

LOOPBACK_HOST = "127.0.0.1"
TLS_HOST = "localhost"
PORT = 8001
SERVICE = "vllm"
PROJECT_PATTERN = re.compile(r"^callmetric-vllm-smoke-[0-9]+-[a-f0-9]{12}$")
CACHE_ENVIRONMENT_VARIABLE = "CALLMETRIC_VLLM_SMOKE_CACHE_DIR"
COMPRESSED_IMAGE_BYTES = 8_945_177_971
EXPANDED_IMAGE_BYTES = 22_362_944_928
MODEL_REPOSITORY_BYTES = 5_582_400_357
TRANSIENT_BYTES = (
    COMPRESSED_IMAGE_BYTES
    + EXPANDED_IMAGE_BYTES
    + MODEL_REPOSITORY_BYTES
    + MODEL_REPOSITORY_BYTES
)
RESERVE_BYTES = 15 * 1024**3
MARGIN_BYTES = 2 * 1024**3
MINIMUM_FREE_BYTES = TRANSIENT_BYTES + RESERVE_BYTES + MARGIN_BYTES
SUBPROCESS_TIMEOUT_SECONDS = 30.0
PULL_TIMEOUT_SECONDS = 1_800.0
START_TIMEOUT_SECONDS = 120.0
HEALTH_TIMEOUT_SECONDS = 900.0
HTTP_TIMEOUT_SECONDS = 30.0
EXPECTED_CERTIFICATE_FILES = frozenset(
    {"ca.crt", "ca.key", "ca.srl", "server.crt", "server.csr", "server.key"}
)
PROTECTED_CONTAINERS = {
    "rag-redis": "ce50a0ac3175c2643d8cdbe47eb8dbafc2da26370837ea59138f750a3f52d107",
    "rag-postgres": "872307591cfc873c544ebb5db4075dec848161ee2225ed38c542ebe198e1defc",
    "rag-adminer": "20381a2f2b25bbed344d547254b05cee3c97c58dff21c5d10a4ae833571ad490",
    "callmetric-local-rabbitmq": (
        "01d593e1b68b211ec49f7c154d545be9115c2cbf306e294d2e2292d3024f85df"
    ),
}
SYNTHETIC_PROMPT = (
    "Sentetik bir çağrı merkezi örneği için yalnızca JSON üret: "
    '{"coaching":"Müşterinin sorusunu kısa ve nazikçe yeniden doğrulayın."}'
)
_FIXED_FAILURE = "vLLM loopback smoke failed"


class VLLMSmokeError(RuntimeError):
    """A fixed, secret-safe smoke lifecycle failure."""


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
        raise VLLMSmokeError(_FIXED_FAILURE) from error


def _output(arguments: list[str], **kwargs: Any) -> str:
    return _run(arguments, **kwargs).stdout.strip()


def _compose_arguments(project: str, *arguments: str) -> list[str]:
    if not PROJECT_PATTERN.fullmatch(project):
        raise VLLMSmokeError(_FIXED_FAILURE)
    return [
        "docker",
        "compose",
        "--project-name",
        project,
        "--file",
        str(COMPOSE_FILE),
        *arguments,
    ]


def _validate_worktree_status(raw_status: str) -> None:
    if not raw_status.endswith("\0"):
        raise VLLMSmokeError(_FIXED_FAILURE)
    records = raw_status.split("\0")
    if records[-1] or len(records) - 1 != len(EXPECTED_ARTIFACT_PATHS):
        raise VLLMSmokeError(_FIXED_FAILURE)
    observed: set[str] = set()
    for record in records[:-1]:
        if len(record) < 4 or record[2] != " ":
            raise VLLMSmokeError(_FIXED_FAILURE)
        status = record[:2]
        path = record[3:]
        if status != "??" or path not in EXPECTED_ARTIFACT_PATHS or path in observed:
            raise VLLMSmokeError(_FIXED_FAILURE)
        observed.add(path)
    if observed != EXPECTED_ARTIFACT_PATHS:
        raise VLLMSmokeError(_FIXED_FAILURE)


def _validate_current_worktree() -> None:
    result = _run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
        ]
    )
    _validate_worktree_status(result.stdout)


def _validate_repository() -> None:
    if Path.cwd().resolve() != EXPECTED_ROOT or REPOSITORY_ROOT != EXPECTED_ROOT:
        raise VLLMSmokeError(_FIXED_FAILURE)
    expected_head = os.environ.get(EXPECTED_HEAD_ENVIRONMENT_VARIABLE)
    if expected_head is None or not COMMIT_PATTERN.fullmatch(expected_head):
        raise VLLMSmokeError(_FIXED_FAILURE)
    checks = (
        (["hostname"], EXPECTED_HOST),
        (["git", "branch", "--show-current"], EXPECTED_BRANCH),
        (["git", "rev-parse", "HEAD"], expected_head),
        (["git", "rev-parse", "origin/main"], BASELINE_COMMIT),
    )
    for command, expected in checks:
        if _output(command) != expected:
            raise VLLMSmokeError(_FIXED_FAILURE)
    _run(["git", "merge-base", "--is-ancestor", BASELINE_COMMIT, expected_head])
    _validate_current_worktree()


def _validate_tools() -> None:
    if shutil.which("docker") is None or shutil.which("openssl") is None:
        raise VLLMSmokeError(_FIXED_FAILURE)
    _run(["docker", "--version"])
    _run(["docker", "compose", "version"])
    _run(["docker", "buildx", "version"])
    _run(["openssl", "version"])


def _validate_gpu() -> None:
    rows = _output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    if not rows or [part.strip() for part in rows[0].split(",")] != [
        "0",
        "NVIDIA L40S",
        "46068",
        "8.9",
    ]:
        raise VLLMSmokeError(_FIXED_FAILURE)


def _validate_port_free() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((LOOPBACK_HOST, PORT))
        except OSError as error:
            raise VLLMSmokeError(_FIXED_FAILURE) from error


def _free_disk_bytes() -> int:
    return shutil.disk_usage("/").free


def _validate_disk() -> None:
    if _free_disk_bytes() < MINIMUM_FREE_BYTES:
        raise VLLMSmokeError(_FIXED_FAILURE)


def _validate_cache_directory(raw_path: str | None) -> Path:
    if raw_path is None or not raw_path or "\x00" in raw_path:
        raise VLLMSmokeError(_FIXED_FAILURE)
    candidate = Path(raw_path)
    if not candidate.is_absolute() or candidate == Path("/"):
        raise VLLMSmokeError(_FIXED_FAILURE)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise VLLMSmokeError(_FIXED_FAILURE) from error
    if (
        not resolved.is_dir()
        or resolved == REPOSITORY_ROOT
        or REPOSITORY_ROOT in resolved.parents
        or candidate != resolved
        or candidate.is_symlink()
        or resolved.stat().st_uid != os.getuid()
        or resolved.stat().st_mode & 0o022
    ):
        raise VLLMSmokeError(_FIXED_FAILURE)
    return resolved


def _protected_container_snapshot() -> dict[str, tuple[str, str]]:
    raw = _output(
        [
            "docker",
            "ps",
            "--no-trunc",
            "--format",
            "{{json .}}",
        ]
    )
    observed: dict[str, tuple[str, str]] = {}
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise VLLMSmokeError(_FIXED_FAILURE) from error
        name = row.get("Names")
        if name in PROTECTED_CONTAINERS:
            status = str(row.get("Status"))
            normalized_status = (
                "running-healthy" if "(healthy)" in status else "running"
            )
            observed[name] = (str(row.get("ID")), normalized_status)
    if set(observed) != set(PROTECTED_CONTAINERS):
        raise VLLMSmokeError(_FIXED_FAILURE)
    for name, expected_id in PROTECTED_CONTAINERS.items():
        container_id, status = observed[name]
        if container_id != expected_id or not status.startswith("running"):
            raise VLLMSmokeError(_FIXED_FAILURE)
    return observed


def _validate_sha256_digest(value: str) -> None:
    if not SHA256_DIGEST_PATTERN.fullmatch(value):
        raise VLLMSmokeError(_FIXED_FAILURE)


def _validate_image_metadata() -> None:
    _validate_sha256_digest(IMAGE_INDEX_DIGEST)
    _validate_sha256_digest(IMAGE_AMD64_DIGEST)
    summary = _output(["docker", "buildx", "imagetools", "inspect", IMAGE_TAG])
    digests = {
        line.split(":", maxsplit=1)[1].strip()
        for line in summary.splitlines()
        if line.strip().startswith("Digest:")
    }
    if IMAGE_INDEX_DIGEST not in digests:
        raise VLLMSmokeError(_FIXED_FAILURE)
    raw_index = _output(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            "--raw",
            f"{IMAGE_TAG}@{IMAGE_INDEX_DIGEST}",
        ]
    )
    try:
        index = json.loads(raw_index)
        manifests = index["manifests"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise VLLMSmokeError(_FIXED_FAILURE) from error
    amd64 = [
        item
        for item in manifests
        if item.get("platform") == {"architecture": "amd64", "os": "linux"}
    ]
    if len(amd64) != 1 or amd64[0].get("digest") != IMAGE_AMD64_DIGEST:
        raise VLLMSmokeError(_FIXED_FAILURE)


def _validate_model_metadata() -> None:
    request = urllib.request.Request(
        MODEL_API_URL,
        headers={"Accept": "application/json", "User-Agent": "callmetric-pr53-smoke"},
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except (OSError, ValueError) as error:
        raise VLLMSmokeError(_FIXED_FAILURE) from error
    tags = payload.get("tags", [])
    config = payload.get("config", {})
    quantization = config.get("quantization_config", {})
    if (
        payload.get("id") != MODEL
        or payload.get("author") != "Qwen"
        or payload.get("sha") != MODEL_REVISION
        or "license:apache-2.0" not in tags
        or "awq" not in tags
        or config.get("architectures") != ["Qwen2ForCausalLM"]
        or quantization.get("bits") != 4
        or quantization.get("quant_method") != "awq"
    ):
        raise VLLMSmokeError(_FIXED_FAILURE)


def _generate_certificates(directory: Path) -> None:
    environment = os.environ.copy()
    environment["OPENSSL_CONF"] = os.devnull
    _run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:3072",
            "-sha256",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=CallMetric PR53 Ephemeral CA",
            "-keyout",
            "ca.key",
            "-out",
            "ca.crt",
        ],
        environment=environment,
        cwd=directory,
    )
    _run(
        [
            "openssl",
            "req",
            "-new",
            "-newkey",
            "rsa:3072",
            "-sha256",
            "-nodes",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
            "-keyout",
            "server.key",
            "-out",
            "server.csr",
        ],
        environment=environment,
        cwd=directory,
    )
    _run(
        [
            "openssl",
            "x509",
            "-req",
            "-sha256",
            "-days",
            "1",
            "-in",
            "server.csr",
            "-CA",
            "ca.crt",
            "-CAkey",
            "ca.key",
            "-CAcreateserial",
            "-copy_extensions",
            "copy",
            "-out",
            "server.crt",
        ],
        environment=environment,
        cwd=directory,
    )
    for name in ("ca.key", "server.key"):
        (directory / name).chmod(0o600)
    observed = frozenset(path.name for path in directory.iterdir())
    if observed != EXPECTED_CERTIFICATE_FILES:
        raise VLLMSmokeError(_FIXED_FAILURE)
    certificate = _output(
        ["openssl", "x509", "-in", "server.crt", "-noout", "-ext", "subjectAltName"],
        cwd=directory,
    )
    if "DNS:localhost, IP Address:127.0.0.1" not in certificate:
        raise VLLMSmokeError(_FIXED_FAILURE)


def _environment(
    cache_directory: Path,
    tls_directory: Path,
    token: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            CACHE_ENVIRONMENT_VARIABLE: str(cache_directory),
            "CALLMETRIC_VLLM_SMOKE_TLS_DIR": str(tls_directory),
            "CALLMETRIC_VLLM_SMOKE_API_KEY": token,
        }
    )
    return environment


def _request(
    path: str,
    *,
    context: ssl.SSLContext,
    token: str | None,
    body: dict[str, object] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"https://{TLS_HOST}:{PORT}{path}",
        data=data,
        headers=headers,
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(
            request, context=context, timeout=HTTP_TIMEOUT_SECONDS
        ) as response:
            raw_payload = response.read()
            payload = {} if not raw_payload else json.loads(raw_payload)
            return response.status, payload
    except urllib.error.HTTPError as error:
        return error.code, {}
    except (OSError, ValueError) as error:
        raise VLLMSmokeError(_FIXED_FAILURE) from error


def _wait_for_health(context: ssl.SSLContext, token: str) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            status, _payload = _request("/health", context=context, token=token)
        except VLLMSmokeError:
            time.sleep(1.0)
            continue
        if status == 200:
            return
        time.sleep(1.0)
    raise VLLMSmokeError(_FIXED_FAILURE)


def _verify_tls_failures(ca_file: Path) -> None:
    try:
        _request("/health", context=ssl.create_default_context(), token=None)
    except VLLMSmokeError:
        pass
    else:
        raise VLLMSmokeError(_FIXED_FAILURE)
    mismatch_context = ssl.create_default_context(cafile=str(ca_file))
    try:
        with socket.create_connection(
            (LOOPBACK_HOST, PORT), timeout=HTTP_TIMEOUT_SECONDS
        ) as connection:
            with mismatch_context.wrap_socket(
                connection, server_hostname="mismatch.invalid"
            ):
                raise VLLMSmokeError(_FIXED_FAILURE)
    except ssl.SSLCertVerificationError:
        pass
    except OSError as error:
        raise VLLMSmokeError(_FIXED_FAILURE) from error


def _verify_api(context: ssl.SSLContext, token: str) -> None:
    for rejected_token in (None, "synthetic-incorrect-token"):
        status, _payload = _request("/v1/models", context=context, token=rejected_token)
        if status not in {401, 403}:
            raise VLLMSmokeError(_FIXED_FAILURE)
    status, models = _request("/v1/models", context=context, token=token)
    identifiers = [
        item.get("id") for item in models.get("data", []) if isinstance(item, dict)
    ]
    if status != 200 or identifiers != [SERVED_MODEL]:
        raise VLLMSmokeError(_FIXED_FAILURE)
    status, completion = _request(
        "/v1/completions",
        context=context,
        token=token,
        body={
            "model": SERVED_MODEL,
            "prompt": SYNTHETIC_PROMPT,
            "max_tokens": 256,
            "temperature": 0,
            "stream": False,
        },
    )
    choices = completion.get("choices", [])
    if (
        status != 200
        or not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or not isinstance(choices[0].get("text"), str)
        or not choices[0]["text"].strip()
    ):
        raise VLLMSmokeError(_FIXED_FAILURE)


def _gpu_memory_used() -> int:
    raw = _output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if not raw.isascii() or not raw.isdigit():
        raise VLLMSmokeError(_FIXED_FAILURE)
    return int(raw)


def _cleanup(project: str, environment: dict[str, str]) -> None:
    _run(
        _compose_arguments(project, "down", "--remove-orphans", "--timeout", "30"),
        environment=environment,
        timeout=START_TIMEOUT_SECONDS,
    )


def run() -> None:
    _validate_repository()
    _validate_tools()
    _validate_gpu()
    _validate_port_free()
    _validate_disk()
    cache_directory = _validate_cache_directory(
        os.environ.get(CACHE_ENVIRONMENT_VARIABLE)
    )
    protected_before = _protected_container_snapshot()
    _validate_image_metadata()
    _validate_model_metadata()
    project = f"callmetric-vllm-smoke-{os.getpid()}-{secrets.token_hex(6)}"
    token = secrets.token_urlsafe(48)
    if not PROJECT_PATTERN.fullmatch(project) or len(token) < 48:
        raise VLLMSmokeError(_FIXED_FAILURE)
    environment: dict[str, str] | None = None
    with tempfile.TemporaryDirectory(prefix="callmetric-vllm-smoke-") as raw_tls:
        tls_directory = Path(raw_tls)
        try:
            _generate_certificates(tls_directory)
            environment = _environment(cache_directory, tls_directory, token)
            _run(
                _compose_arguments(project, "config", "--quiet"),
                environment=environment,
            )
            baseline_memory = _gpu_memory_used()
            _run(
                _compose_arguments(project, "pull", SERVICE),
                environment=environment,
                timeout=PULL_TIMEOUT_SECONDS,
            )
            _run(
                _compose_arguments(project, "up", "--detach", SERVICE),
                environment=environment,
                timeout=START_TIMEOUT_SECONDS,
            )
            context = ssl.create_default_context(cafile=str(tls_directory / "ca.crt"))
            _wait_for_health(context, token)
            _verify_tls_failures(tls_directory / "ca.crt")
            _verify_api(context, token)
            if _gpu_memory_used() <= baseline_memory:
                raise VLLMSmokeError(_FIXED_FAILURE)
        finally:
            try:
                if environment is not None:
                    _cleanup(project, environment)
            finally:
                if _protected_container_snapshot() != protected_before:
                    raise VLLMSmokeError(_FIXED_FAILURE)
    _validate_current_worktree()


def main() -> int:
    try:
        run()
    except Exception:
        print(_FIXED_FAILURE, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
