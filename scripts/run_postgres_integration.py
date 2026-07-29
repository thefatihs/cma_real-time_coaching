"""Run the opt-in PostgreSQL/pgvector integration test in isolated Docker."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

IMAGE_TAG = "pgvector/pgvector:0.8.5-pg16-bookworm"
IMAGE_DIGEST = "sha256:1d533553fefe4f12e5d80c7b80622ba0c382abb5758856f52983d8789179f0fb"
IMAGE = f"{IMAGE_TAG}@{IMAGE_DIGEST}"
SERVICE = "postgres-vector-integration"
DATABASE = "callmetric_vector_test"
USER = "callmetric_test"
PASSWORD = "callmetric_test_password"
LOOPBACK_HOST = "127.0.0.1"
HEALTH_TIMEOUT_SECONDS = 60.0
PROJECT_NAME_PATTERN = re.compile(r"^callmetric-pgvector-[0-9]+-[a-f0-9]{12}$")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPOSITORY_ROOT / "compose.postgres-integration.yml"
INTEGRATION_TEST = (
    REPOSITORY_ROOT / "tests" / "integration" / "test_postgres_vector_integration.py"
)


class IntegrationRunError(RuntimeError):
    """An isolated PostgreSQL integration lifecycle step failed."""


def _run(
    arguments: list[str],
    *,
    input_text: str | None = None,
    environment: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        input=input_text,
        text=True,
        capture_output=capture_output,
        shell=False,
    )


def _compose_arguments(
    docker: str,
    project_name: str,
    *arguments: str,
) -> list[str]:
    if not PROJECT_NAME_PATTERN.fullmatch(project_name):
        raise IntegrationRunError("generated Compose project name is unsafe")
    return [
        docker,
        "compose",
        "--project-name",
        project_name,
        "--file",
        str(COMPOSE_FILE),
        *arguments,
    ]


def _container_id(docker: str, project_name: str) -> str:
    result = _run(
        _compose_arguments(docker, project_name, "ps", "-q", SERVICE),
        capture_output=True,
    )
    container_id = result.stdout.strip()
    if not container_id or "\n" in container_id or "\r" in container_id:
        raise IntegrationRunError("Compose did not return exactly one container")
    return container_id


def _wait_until_healthy(
    docker: str,
    project_name: str,
) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    container_id = _container_id(docker, project_name)
    while time.monotonic() < deadline:
        result = _run(
            [
                docker,
                "inspect",
                "--format",
                "{{json .State}}",
                container_id,
            ],
            capture_output=True,
        )
        try:
            state = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise IntegrationRunError(
                "Docker returned malformed container state"
            ) from error
        if not isinstance(state, dict):
            raise IntegrationRunError("Docker returned malformed container state")
        health = state.get("Health")
        if isinstance(health, dict) and health.get("Status") == "healthy":
            return
        if state.get("Running") is not True:
            raise IntegrationRunError("PostgreSQL integration container stopped")
        time.sleep(0.5)
    raise IntegrationRunError("PostgreSQL health check timed out after 60 seconds")


def _published_port(docker: str, project_name: str) -> int:
    result = _run(
        _compose_arguments(docker, project_name, "port", SERVICE, "5432"),
        capture_output=True,
    )
    endpoint = result.stdout.strip()
    if endpoint.count(":") != 1:
        raise IntegrationRunError("Compose returned a malformed published port")
    host, raw_port = endpoint.split(":", maxsplit=1)
    if host != LOOPBACK_HOST or not raw_port.isascii() or not raw_port.isdigit():
        raise IntegrationRunError("PostgreSQL port is not loopback-only")
    port = int(raw_port)
    if not 1 <= port <= 65535:
        raise IntegrationRunError("Compose returned an invalid published port")
    return port


def _pytest_environment(port: int) -> dict[str, str]:
    environment = os.environ.copy()
    migration_dsn = f"postgresql://{USER}:{PASSWORD}@{LOOPBACK_HOST}:{port}/{DATABASE}"
    environment.update(
        {
            "CALLMETRIC_POSTGRES_INTEGRATION": "1",
            "CALLMETRIC_POSTGRES_HOST": LOOPBACK_HOST,
            "CALLMETRIC_POSTGRES_PORT": str(port),
            "CALLMETRIC_POSTGRES_DATABASE": DATABASE,
            "CALLMETRIC_POSTGRES_USER": USER,
            "CALLMETRIC_POSTGRES_PASSWORD": PASSWORD,
            "CALLMETRIC_POSTGRES_CONNECT_TIMEOUT": "5",
            "CALLMETRIC_POSTGRES_MIGRATION_DSN": migration_dsn,
            "CALLMETRIC_POSTGRES_MIGRATION_CONNECT_TIMEOUT_SECONDS": "5",
            "CALLMETRIC_POSTGRES_MIGRATION_SSL_MODE": "require",
            "CALLMETRIC_POSTGRES_MIGRATION_APPLICATION_NAME": (
                "callmetric-integration-migration"
            ),
            "CALLMETRIC_POSTGRES_MIGRATION_LOCK_TIMEOUT_SECONDS": "10",
            "CALLMETRIC_POSTGRES_MIGRATION_STATEMENT_TIMEOUT_SECONDS": "30",
        }
    )
    return environment


def _print_project_logs(docker: str, project_name: str) -> None:
    result = subprocess.run(
        _compose_arguments(docker, project_name, "logs", "--no-color"),
        check=False,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        shell=False,
    )
    if result.stdout:
        print(result.stdout, file=sys.stderr, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise IntegrationRunError("failed to collect project-scoped Compose logs")


def _cleanup(docker: str, project_name: str) -> None:
    _run(
        _compose_arguments(
            docker,
            project_name,
            "down",
            "--volumes",
            "--remove-orphans",
        )
    )


def run() -> None:
    if IMAGE != (
        "pgvector/pgvector:0.8.5-pg16-bookworm@"
        "sha256:1d533553fefe4f12e5d80c7b80622ba0c382abb5758856f52983d8789179f0fb"
    ):
        raise IntegrationRunError("PostgreSQL integration image pin is invalid")
    docker = shutil.which("docker")
    if docker is None:
        raise IntegrationRunError("docker executable is unavailable")
    project_name = f"callmetric-pgvector-{os.getpid()}-{secrets.token_hex(6)}"
    primary_error: BaseException | None = None
    cleanup_errors: list[Exception] = []
    started = False
    try:
        _run([docker, "--version"])
        _run([docker, "compose", "version"])
        _run([docker, "info", "--format", "{{.ServerVersion}}"])
        _run(_compose_arguments(docker, project_name, "config", "--quiet"))
        _run([docker, "pull", IMAGE])
        started = True
        _run(_compose_arguments(docker, project_name, "up", "-d"))
        _wait_until_healthy(docker, project_name)
        port = _published_port(docker, project_name)
        _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "postgres_integration",
                str(INTEGRATION_TEST),
            ],
            environment=_pytest_environment(port),
        )
    except BaseException as error:
        primary_error = error
        if started:
            try:
                _print_project_logs(docker, project_name)
            except Exception as logs_error:
                cleanup_errors.append(logs_error)
    finally:
        try:
            _cleanup(docker, project_name)
        except Exception as cleanup_error:
            cleanup_errors.append(cleanup_error)

    if primary_error is not None:
        if cleanup_errors:
            raise primary_error from ExceptionGroup(
                "PostgreSQL integration diagnostics or cleanup failed",
                cleanup_errors,
            )
        raise primary_error
    if cleanup_errors:
        raise ExceptionGroup("PostgreSQL integration cleanup failed", cleanup_errors)


def main() -> int:
    try:
        run()
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"PostgreSQL integration failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
