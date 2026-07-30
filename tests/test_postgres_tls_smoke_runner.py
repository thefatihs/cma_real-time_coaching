from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import run_postgres_tls_smoke as subject


def _completed(
    arguments: list[str],
    *,
    stdout: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr="")


class _DeterministicCommandRunner:
    def __init__(self, *, fail_stage: str | None = None) -> None:
        self.fail_stage = fail_stage
        self.calls: list[tuple[list[str], dict[str, str] | None]] = []
        self.certificate_directory: Path | None = None

    def __call__(
        self,
        arguments: list[str],
        *,
        environment: dict[str, str] | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output
        self.calls.append((arguments, environment))
        rendered = " ".join(arguments)
        is_cleanup = " down " in f" {rendered} " or " ls " in f" {rendered} "
        if (
            self.fail_stage is not None
            and self.fail_stage in rendered
            and not is_cleanup
        ):
            raise RuntimeError("synthetic-sensitive-provider-detail")
        if "certificate-init" in arguments:
            assert environment is not None
            directory = Path(environment["CALLMETRIC_POSTGRES_TLS_CERT_DIR"])
            self.certificate_directory = directory
            for name in subject.EXPECTED_CERTIFICATE_FILES:
                (directory / name).write_text("synthetic", encoding="utf-8")
        if arguments[:2] == ["docker", "inspect"]:
            return _completed(
                arguments,
                stdout='{"Running":true,"Health":{"Status":"healthy"}}',
            )
        if "port" in arguments:
            return _completed(arguments, stdout="127.0.0.1:54321\n")
        if "ps" in arguments and "-q" in arguments:
            return _completed(arguments, stdout="synthetic-container-id\n")
        return _completed(arguments)


def _run_with_fake(
    monkeypatch: pytest.MonkeyPatch,
    fake: _DeterministicCommandRunner,
) -> None:
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "docker")
    monkeypatch.setattr(subject, "_run", fake)
    monkeypatch.setattr(subject.os, "getpid", lambda: 123)
    monkeypatch.setattr(subject.secrets, "token_hex", lambda _size: "abcdef123456")
    monkeypatch.setattr(
        subject.secrets,
        "token_urlsafe",
        lambda size: f"synthetic-password-{size}",
    )
    subject.run()


def test_image_is_exactly_digest_pinned() -> None:
    assert subject.IMAGE == (
        "pgvector/pgvector:0.8.5-pg16-bookworm@"
        "sha256:1d533553fefe4f12e5d80c7b80622ba0c382abb5758856f52983d8789179f0fb"
    )


@pytest.mark.parametrize(
    "project_name",
    [
        "",
        "callmetric-pgvector-tls",
        "callmetric-pgvector-tls-1-ABCDEF123456",
        "callmetric-pgvector-tls-1-abcdef12345",
        "unsafe project",
    ],
)
def test_compose_arguments_reject_unsafe_project_names(project_name: str) -> None:
    with pytest.raises(subject.TLSSmokeRunError):
        subject._compose_arguments("docker", project_name, "config")


def test_compose_arguments_use_only_tls_compose() -> None:
    arguments = subject._compose_arguments(
        "docker",
        "callmetric-pgvector-tls-1-abcdef123456",
        "up",
        "-d",
    )

    assert str(subject.COMPOSE_FILE) in arguments
    assert "compose.postgres-tls-smoke.yml" in str(subject.COMPOSE_FILE)
    assert "compose.postgres-integration.yml" not in " ".join(arguments)


def test_connection_dsn_uses_localhost_and_trust_root_without_printing(
    tmp_path: Path,
) -> None:
    dsn = subject._connection_dsn(
        user=subject.APPLICATION_USER,
        password="synthetic password",
        port=54321,
        certificate_directory=tmp_path,
    )

    assert "@localhost:54321/" in dsn
    assert "sslrootcert=" in dsn
    assert "synthetic password" not in dsn
    assert "synthetic%20password" in dsn


@pytest.mark.parametrize(
    ("endpoint", "valid"),
    [
        ("127.0.0.1:1\n", True),
        ("127.0.0.1:65535\n", True),
        ("0.0.0.0:5432\n", False),
        ("localhost:5432\n", False),
        ("127.0.0.1:0\n", False),
        ("127.0.0.1:65536\n", False),
        ("127.0.0.1:not-a-port\n", False),
        ("127.0.0.1:5432\n127.0.0.1:5433\n", False),
    ],
)
def test_dynamic_loopback_port_parsing(
    endpoint: str,
    valid: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "_run",
        lambda *_args, **_kwargs: _completed([], stdout=endpoint),
    )

    if valid:
        assert subject._published_port(
            "docker",
            "callmetric-pgvector-tls-1-abcdef123456",
            {},
        ) == int(endpoint.rsplit(":", maxsplit=1)[1])
    else:
        with pytest.raises(subject.TLSSmokeRunError):
            subject._published_port(
                "docker",
                "callmetric-pgvector-tls-1-abcdef123456",
                {},
            )


def test_health_wait_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            _completed([], stdout="synthetic-container\n"),
            _completed([], stdout='{"Running":true,"Health":{"Status":"starting"}}'),
        ]
    )
    times = iter([0.0, 61.0])
    monkeypatch.setattr(subject, "_run", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(subject.time, "monotonic", lambda: next(times))

    with pytest.raises(subject.TLSSmokeRunError):
        subject._wait_until_healthy(
            "docker",
            "callmetric-pgvector-tls-1-abcdef123456",
            {},
        )


def test_complete_fake_lifecycle_uses_tls_commands_and_removes_temp_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _DeterministicCommandRunner()

    _run_with_fake(monkeypatch, fake)

    assert fake.certificate_directory is not None
    assert not fake.certificate_directory.exists()
    rendered_calls = tuple(" ".join(arguments) for arguments, _env in fake.calls)
    assert any("certificate-init" in call for call in rendered_calls)
    assert any("openssl verify" in call for call in rendered_calls)
    assert any("-checkhost localhost" in call for call in rendered_calls)
    assert any("-checkip 127.0.0.1" in call for call in rendered_calls)
    assert any(" up -d postgres-vector-tls-smoke" in call for call in rendered_calls)
    assert any(" down --volumes --remove-orphans" in call for call in rendered_calls)
    assert any("test_postgres_tls_smoke.py" in call for call in rendered_calls)
    assert all(
        "compose.postgres-integration.yml" not in call for call in rendered_calls
    )


@pytest.mark.parametrize(
    "failed_stage",
    [
        "docker --version",
        "compose version",
        "config --quiet",
        "certificate-init",
        "openssl verify",
        "up -d",
        "inspect",
        "port",
        "test_postgres_tls_smoke.py",
    ],
)
def test_every_lifecycle_failure_attempts_cleanup_and_removes_certificates(
    failed_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _DeterministicCommandRunner(fail_stage=failed_stage)

    with pytest.raises(RuntimeError):
        _run_with_fake(monkeypatch, fake)

    rendered_calls = tuple(" ".join(arguments) for arguments, _env in fake.calls)
    assert any(" down --volumes --remove-orphans" in call for call in rendered_calls)
    assert any("container ls -aq" in call for call in rendered_calls)
    assert any("network ls -q" in call for call in rendered_calls)
    assert any("volume ls -q" in call for call in rendered_calls)
    if fake.certificate_directory is not None:
        assert not fake.certificate_directory.exists()


def test_main_reports_only_fixed_secret_free_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> None:
        raise RuntimeError(
            "synthetic-password postgresql://synthetic private/temp/path"
        )

    monkeypatch.setattr(subject, "run", fail)

    assert subject.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "PostgreSQL TLS smoke failed\n"
    assert "synthetic" not in captured.err


def test_compose_contract_has_exact_tls_and_safety_guards() -> None:
    content = subject.COMPOSE_FILE.read_text(encoding="utf-8")

    assert content.count(subject.IMAGE) == 2
    assert "DNS:localhost,IP:127.0.0.1" in content
    assert "ssl=on" in content
    assert "127.0.0.1::5432" in content
    assert 'restart: "no"' in content
    assert "container_name" not in content
    assert "privileged" not in content
    assert "/var/run/docker.sock" not in content
    assert "compose.postgres-integration.yml" not in content
    assert "vllm" not in content.casefold()
    assert "callmetric_test_password" not in content


def test_runner_source_contains_no_production_host_or_committed_secret() -> None:
    content = Path(subject.__file__).read_text(encoding="utf-8")

    assert "CALLMETRIC_POSTGRES_TLS_MIGRATION_PASSWORD" in content
    assert "token_urlsafe" in content
    assert "amazonaws.com" not in content
    assert "api_token" not in content.casefold()
    assert "callmetric_test_password" not in content
