"""Run the isolated PostgreSQL/pgvector TLS smoke lifecycle."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

IMAGE_TAG = "pgvector/pgvector:0.8.5-pg16-bookworm"
IMAGE_DIGEST = "sha256:1d533553fefe4f12e5d80c7b80622ba0c382abb5758856f52983d8789179f0fb"
IMAGE = f"{IMAGE_TAG}@{IMAGE_DIGEST}"
SERVICE = "postgres-vector-tls-smoke"
DATABASE = "callmetric_vector_tls_smoke"
MIGRATION_USER = "callmetric_tls_migration"
APPLICATION_USER = "callmetric_tls_application"
LOOPBACK_HOST = "127.0.0.1"
TLS_HOST = "localhost"
HEALTH_TIMEOUT_SECONDS = 60.0
PROJECT_NAME_PATTERN = re.compile(r"^callmetric-pgvector-tls-[0-9]+-[a-f0-9]{12}$")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPOSITORY_ROOT / "compose.postgres-tls-smoke.yml"
INTEGRATION_TEST = (
    REPOSITORY_ROOT / "tests" / "integration" / "test_postgres_tls_smoke.py"
)
EXPECTED_CERTIFICATE_FILES = frozenset(
    {
        "ca.crt",
        "ca.key",
        "ca.srl",
        "server.crt",
        "server.csr",
        "server.key",
    }
)
_FIXED_FAILURE = "PostgreSQL TLS smoke failed"


class TLSSmokeRunError(RuntimeError):
    """A fixed secret-safe TLS smoke lifecycle failure."""


def _run(
    arguments: list[str],
    *,
    environment: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
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
        raise TLSSmokeRunError(_FIXED_FAILURE)
    return [
        docker,
        "compose",
        "--project-name",
        project_name,
        "--file",
        str(COMPOSE_FILE),
        *arguments,
    ]


def _smoke_environment(
    *,
    certificate_directory: Path,
    migration_password: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CALLMETRIC_POSTGRES_TLS_CERT_DIR": str(certificate_directory),
            "CALLMETRIC_POSTGRES_TLS_MIGRATION_PASSWORD": migration_password,
        }
    )
    return environment


def _generate_certificates(
    docker: str,
    project_name: str,
    environment: dict[str, str],
) -> None:
    _run(
        _compose_arguments(
            docker,
            project_name,
            "run",
            "--rm",
            "--no-deps",
            "certificate-init",
        ),
        environment=environment,
        capture_output=True,
    )


def _validate_certificates(
    docker: str,
    certificate_directory: Path,
) -> None:
    observed = frozenset(path.name for path in certificate_directory.iterdir())
    if observed != EXPECTED_CERTIFICATE_FILES:
        raise TLSSmokeRunError(_FIXED_FAILURE)
    volume = f"{certificate_directory}:/certificates:ro"
    commands = (
        [
            "openssl",
            "verify",
            "-CAfile",
            "/certificates/ca.crt",
            "/certificates/server.crt",
        ],
        [
            "openssl",
            "x509",
            "-in",
            "/certificates/server.crt",
            "-noout",
            "-checkhost",
            TLS_HOST,
        ],
        [
            "openssl",
            "x509",
            "-in",
            "/certificates/server.crt",
            "-noout",
            "-checkip",
            LOOPBACK_HOST,
        ],
        [
            "/bin/sh",
            "-ceu",
            'test "$(stat -c %a /certificates/ca.key)" = 600 && '
            'test "$(stat -c %a /certificates/server.key)" = 600',
        ],
    )
    for command in commands:
        _run(
            [
                docker,
                "run",
                "--rm",
                "--network",
                "none",
                "--volume",
                volume,
                IMAGE,
                *command,
            ],
            capture_output=True,
        )


def _container_id(
    docker: str,
    project_name: str,
    environment: dict[str, str],
) -> str | None:
    result = _run(
        _compose_arguments(docker, project_name, "ps", "-a", "-q", SERVICE),
        environment=environment,
        capture_output=True,
    )
    container_id = result.stdout.strip()
    if not container_id:
        return None
    if "\n" in container_id or "\r" in container_id:
        raise TLSSmokeRunError(_FIXED_FAILURE)
    return container_id


def _wait_until_healthy(
    docker: str,
    project_name: str,
    environment: dict[str, str],
) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        container_id = _container_id(docker, project_name, environment)
        if container_id is None:
            time.sleep(0.5)
            continue
        result = _run(
            [docker, "inspect", "--format", "{{json .State}}", container_id],
            capture_output=True,
        )
        try:
            state = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise TLSSmokeRunError(_FIXED_FAILURE) from error
        if not isinstance(state, dict):
            raise TLSSmokeRunError(_FIXED_FAILURE)
        health = state.get("Health")
        if isinstance(health, dict) and health.get("Status") == "healthy":
            return
        if state.get("Running") is not True:
            raise TLSSmokeRunError(_FIXED_FAILURE)
        time.sleep(0.5)
    raise TLSSmokeRunError(_FIXED_FAILURE)


def _published_port(
    docker: str,
    project_name: str,
    environment: dict[str, str],
) -> int:
    result = _run(
        _compose_arguments(docker, project_name, "port", SERVICE, "5432"),
        environment=environment,
        capture_output=True,
    )
    endpoint = result.stdout.strip()
    if endpoint.count(":") != 1:
        raise TLSSmokeRunError(_FIXED_FAILURE)
    host, raw_port = endpoint.split(":", maxsplit=1)
    if host != LOOPBACK_HOST or not raw_port.isascii() or not raw_port.isdigit():
        raise TLSSmokeRunError(_FIXED_FAILURE)
    port = int(raw_port)
    if not 1 <= port <= 65535:
        raise TLSSmokeRunError(_FIXED_FAILURE)
    return port


def _connection_dsn(
    *,
    user: str,
    password: str,
    port: int,
    certificate_directory: Path,
) -> str:
    encoded_user = quote(user, safe="")
    encoded_password = quote(password, safe="")
    encoded_root = quote(str(certificate_directory / "ca.crt"), safe="")
    return (
        f"postgresql://{encoded_user}:{encoded_password}@{TLS_HOST}:{port}/{DATABASE}"
        f"?sslrootcert={encoded_root}"
    )


def _pytest_environment(
    *,
    port: int,
    certificate_directory: Path,
    migration_password: str,
    application_password: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    migration_dsn = _connection_dsn(
        user=MIGRATION_USER,
        password=migration_password,
        port=port,
        certificate_directory=certificate_directory,
    )
    application_dsn = _connection_dsn(
        user=APPLICATION_USER,
        password=application_password,
        port=port,
        certificate_directory=certificate_directory,
    )
    environment.update(
        {
            "CALLMETRIC_POSTGRES_TLS_SMOKE": "1",
            "CALLMETRIC_POSTGRES_TLS_PORT": str(port),
            "CALLMETRIC_POSTGRES_TLS_CERT_DIR": str(certificate_directory),
            "CALLMETRIC_POSTGRES_TLS_MIGRATION_PASSWORD": migration_password,
            "CALLMETRIC_POSTGRES_TLS_APPLICATION_PASSWORD": application_password,
            "CALLMETRIC_POSTGRES_MIGRATION_DSN": migration_dsn,
            "CALLMETRIC_POSTGRES_MIGRATION_CONNECT_TIMEOUT_SECONDS": "5",
            "CALLMETRIC_POSTGRES_MIGRATION_SSL_MODE": "verify-full",
            "CALLMETRIC_POSTGRES_MIGRATION_APPLICATION_NAME": (
                "callmetric-tls-smoke-migration"
            ),
            "CALLMETRIC_POSTGRES_MIGRATION_LOCK_TIMEOUT_SECONDS": "10",
            "CALLMETRIC_POSTGRES_MIGRATION_STATEMENT_TIMEOUT_SECONDS": "30",
            "CALLMETRIC_POSTGRES_DSN": application_dsn,
            "CALLMETRIC_POSTGRES_CONNECT_TIMEOUT_SECONDS": "5",
            "CALLMETRIC_POSTGRES_SSL_MODE": "verify-full",
            "CALLMETRIC_POSTGRES_APPLICATION_NAME": "callmetric-tls-smoke",
        }
    )
    return environment


def _cleanup(docker: str, project_name: str, environment: dict[str, str]) -> None:
    _run(
        _compose_arguments(
            docker,
            project_name,
            "down",
            "--volumes",
            "--remove-orphans",
        ),
        environment=environment,
        capture_output=True,
    )


def _require_no_project_resources(docker: str, project_name: str) -> None:
    filters = (
        ("container", "ls", "-aq"),
        ("network", "ls", "-q"),
        ("volume", "ls", "-q"),
    )
    for resource, *arguments in filters:
        result = _run(
            [
                docker,
                resource,
                *arguments,
                "--filter",
                f"label=com.docker.compose.project={project_name}",
            ],
            capture_output=True,
        )
        if result.stdout.strip():
            raise TLSSmokeRunError(_FIXED_FAILURE)


def run() -> None:
    if IMAGE != (
        "pgvector/pgvector:0.8.5-pg16-bookworm@"
        "sha256:1d533553fefe4f12e5d80c7b80622ba0c382abb5758856f52983d8789179f0fb"
    ):
        raise TLSSmokeRunError(_FIXED_FAILURE)
    docker = shutil.which("docker")
    if docker is None:
        raise TLSSmokeRunError(_FIXED_FAILURE)
    project_name = f"callmetric-pgvector-tls-{os.getpid()}-{secrets.token_hex(6)}"
    migration_password = secrets.token_urlsafe(32)
    application_password = secrets.token_urlsafe(32)
    primary_error: BaseException | None = None
    cleanup_errors: list[Exception] = []
    certificate_path: Path | None = None
    environment: dict[str, str] | None = None
    with tempfile.TemporaryDirectory(prefix="callmetric-postgres-tls-") as temporary:
        certificate_path = Path(temporary)
        environment = _smoke_environment(
            certificate_directory=certificate_path,
            migration_password=migration_password,
        )
        try:
            _run([docker, "--version"], capture_output=True)
            _run([docker, "compose", "version"], capture_output=True)
            _run(
                [docker, "info", "--format", "{{.ServerVersion}}"], capture_output=True
            )
            _run(
                _compose_arguments(docker, project_name, "config", "--quiet"),
                environment=environment,
                capture_output=True,
            )
            _generate_certificates(docker, project_name, environment)
            _validate_certificates(docker, certificate_path)
            _run(
                _compose_arguments(docker, project_name, "up", "-d", SERVICE),
                environment=environment,
                capture_output=True,
            )
            _wait_until_healthy(docker, project_name, environment)
            port = _published_port(docker, project_name, environment)
            _run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-m",
                    "postgres_integration",
                    str(INTEGRATION_TEST),
                ],
                environment=_pytest_environment(
                    port=port,
                    certificate_directory=certificate_path,
                    migration_password=migration_password,
                    application_password=application_password,
                ),
            )
        except BaseException as error:
            primary_error = error
        finally:
            try:
                _cleanup(docker, project_name, environment)
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                _require_no_project_resources(docker, project_name)
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)

    if certificate_path is not None and certificate_path.exists():
        cleanup_errors.append(TLSSmokeRunError(_FIXED_FAILURE))
    if primary_error is not None:
        if cleanup_errors:
            raise primary_error from ExceptionGroup(_FIXED_FAILURE, cleanup_errors)
        raise primary_error
    if cleanup_errors:
        raise ExceptionGroup(_FIXED_FAILURE, cleanup_errors)


def main() -> int:
    try:
        run()
    except KeyboardInterrupt:
        return 130
    except BaseException:
        print(_FIXED_FAILURE, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
