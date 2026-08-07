from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import threading
from pathlib import Path

import pytest

from scripts import run_postgres_tls_service as subject

CURRENT_COMMIT = "cf3932dcb43911b2a5f5ff139f6ce07c88caf389"


@pytest.mark.parametrize("ttl", [300, 600, 7200])
def test_ttl_bounds(ttl: int) -> None:
    assert subject._parse_arguments(["--ttl-seconds", str(ttl)]) == (ttl, False)
    assert subject._parse_arguments(
        ["--preflight-only", "--ttl-seconds", str(ttl)]
    ) == (ttl, True)


@pytest.mark.parametrize("value", ["299", "7201", "-1", "٣٠٠", "3.0"])
def test_invalid_ttl_is_rejected(value: str) -> None:
    with pytest.raises(subject.PostgreSQLTLSServiceError):
        subject._parse_arguments(["--ttl-seconds", value])


def test_handoff_is_owner_only_verify_full_and_removable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restrictions: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        subject,
        "_restrict_owner",
        lambda path, *, directory: restrictions.append((path, directory)),
    )
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    tls = tmp_path / "tls"
    tls.mkdir(mode=0o700)
    (tls / "ca.crt").write_text("synthetic-ca", encoding="utf-8")
    handoff = subject._create_handoff(root, tls, "synthetic-dsn", 300)
    assert {item.name for item in handoff.iterdir()} == subject.HANDOFF_FILES
    if os.name != "nt":
        assert handoff.stat().st_mode & 0o077 == 0
        assert all(item.stat().st_mode & 0o077 == 0 for item in handoff.iterdir())
    metadata = json.loads((handoff / "connection.json").read_text(encoding="utf-8"))
    assert metadata["sslmode"] == "verify-full"
    assert metadata["host"] == "localhost"
    assert (handoff / "application.dsn").read_text(encoding="utf-8") == "synthetic-dsn"
    assert restrictions[0] == (handoff, True)
    assert {path.name for path, directory in restrictions[1:] if not directory} == (
        subject.HANDOFF_FILES
    )
    subject._remove_handoff(handoff, root.resolve())
    assert not handoff.exists()


def test_public_protected_resource_snapshot_comparison_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = [
        {
            "container": frozenset({"container123"}),
            "network": frozenset({"network123"}),
            "volume": frozenset({"volume123"}),
        },
        {
            "container": frozenset({"container123"}),
            "network": frozenset({"network123"}),
            "volume": frozenset({"volume123"}),
        },
    ]
    monkeypatch.setattr(subject, "_resource_snapshot", lambda _docker: snapshots.pop(0))

    expected = subject.snapshot_protected_resources("docker")
    subject.require_protected_resources_unchanged("docker", expected)

    assert snapshots == []


def test_public_protected_resource_comparison_rejects_change_or_invalid_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "container": frozenset({"container123"}),
        "network": frozenset({"network123"}),
        "volume": frozenset({"volume123"}),
    }
    monkeypatch.setattr(
        subject,
        "_resource_snapshot",
        lambda _docker: {**expected, "volume": frozenset()},
    )

    with pytest.raises(subject.PostgreSQLTLSServiceError):
        subject.require_protected_resources_unchanged("docker", expected)
    with pytest.raises(subject.PostgreSQLTLSServiceError):
        subject.require_protected_resources_unchanged("docker", {})


def test_exact_project_cleanup_is_idempotent_when_targets_are_already_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = "callmetric-pgvector-tls-123-abcdef123456"
    monkeypatch.setattr(
        subject,
        "_project_resource_references",
        lambda _docker, _project: {
            "container": (),
            "network": (),
            "volume": (),
        },
    )
    monkeypatch.setattr(
        subject,
        "_run_command",
        lambda *_args, **_kwargs: pytest.fail("absent resource removal attempted"),
    )
    monkeypatch.setattr(subject.smoke, "_require_no_project_resources", lambda *_: None)

    subject.cleanup_exact_project_resources("docker", project)


def test_exact_project_fallback_attempts_volume_and_network_after_container_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = "callmetric-pgvector-tls-123-abcdef123456"
    resources = {
        "container": ("container123",),
        "network": ("network123",),
        "volume": ("volume123",),
    }
    monkeypatch.setattr(
        subject, "_project_resource_references", lambda _docker, _project: resources
    )
    monkeypatch.setattr(
        subject, "_validate_exact_project_resources", lambda *_args: None
    )
    attempted: list[str] = []

    def remove(arguments: list[str], **_kwargs: object) -> None:
        attempted.append(arguments[1])
        if arguments[1] == "container":
            raise RuntimeError

    monkeypatch.setattr(subject, "_run_command", remove)

    with pytest.raises(RuntimeError):
        subject._exact_project_fallback("docker", project)

    assert attempted == ["container", "volume", "network"]


def test_exact_handoff_cleanup_removes_only_validated_run_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    handoff = root / "callmetric-postgres-tls-abcdefgh"
    handoff.mkdir()
    for name in subject.HANDOFF_FILES:
        (handoff / name).write_text("synthetic", encoding="utf-8")
    sibling = root / "operator-owned-sibling"
    sibling.mkdir()
    (sibling / "preserved.txt").write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(subject, "_private_root", lambda _raw: root.resolve())

    subject.cleanup_exact_handoff_child(root, handoff)

    assert not handoff.exists()
    assert sibling.exists()
    assert (sibling / "preserved.txt").exists()


def test_exact_handoff_cleanup_refuses_sibling_or_incomplete_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    sibling = root / "operator-owned-sibling"
    sibling.mkdir()
    handoff = root / "callmetric-postgres-tls-abcdefgh"
    handoff.mkdir()
    (handoff / "application.dsn").write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(subject, "_private_root", lambda _raw: root.resolve())

    with pytest.raises(subject.PostgreSQLTLSServiceError):
        subject.cleanup_exact_handoff_child(root, sibling)
    with pytest.raises(subject.PostgreSQLTLSServiceError):
        subject.cleanup_exact_handoff_child(root, handoff)

    assert sibling.exists()
    assert handoff.exists()


def test_preflight_stops_before_secrets_signals_or_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_validate_repository", lambda: None)
    monkeypatch.setattr(subject, "_docker_preflight", lambda: "docker")
    monkeypatch.setattr(subject, "_resource_snapshot", lambda _docker: {})
    monkeypatch.setattr(
        subject.secrets, "token_urlsafe", lambda _size: pytest.fail("secret generated")
    )
    monkeypatch.setattr(
        subject, "_install_handlers", lambda _stop: pytest.fail("handler installed")
    )
    subject.run(300, preflight_only=True)


def _mock_clean_repository(
    monkeypatch: pytest.MonkeyPatch,
    *,
    current: str = CURRENT_COMMIT,
    remote: str = CURRENT_COMMIT,
) -> None:
    monkeypatch.setenv(subject.EXPECTED_HEAD_VARIABLE, CURRENT_COMMIT)
    monkeypatch.setattr(subject.Path, "cwd", lambda: subject.REPOSITORY_ROOT)

    def output(arguments: list[str], **_kwargs: object) -> str:
        command = tuple(arguments)
        responses = {
            ("git", "branch", "--show-current"): subject.DEFAULT_EXPECTED_BRANCH,
            ("git", "rev-parse", "HEAD"): current,
            (
                "git",
                "rev-parse",
                f"origin/{subject.DEFAULT_EXPECTED_BRANCH}",
            ): remote,
            (
                "git",
                "status",
                "--porcelain",
                "-z",
                "--untracked-files=all",
            ): "",
        }
        return responses[command]

    monkeypatch.setattr(subject, "_output", output)


def test_exact_configured_current_and_remote_head_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_clean_repository(monkeypatch)
    subject._validate_repository()


def test_explicit_reviewed_document_branch_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = "feat/dashboard-rag-document-upload"
    monkeypatch.setenv(subject.EXPECTED_BRANCH_VARIABLE, branch)
    monkeypatch.setenv(subject.EXPECTED_HEAD_VARIABLE, CURRENT_COMMIT)
    monkeypatch.setattr(subject.Path, "cwd", lambda: subject.REPOSITORY_ROOT)

    def output(arguments: list[str], **_kwargs: object) -> str:
        responses = {
            ("git", "branch", "--show-current"): branch,
            ("git", "rev-parse", "HEAD"): CURRENT_COMMIT,
            ("git", "rev-parse", f"origin/{branch}"): CURRENT_COMMIT,
            ("git", "status", "--porcelain", "-z", "--untracked-files=all"): "",
        }
        return responses[tuple(arguments)]

    monkeypatch.setattr(subject, "_output", output)
    subject._validate_repository()


@pytest.mark.parametrize("branch", ["", "../unsafe", "-unsafe", "unsafe/"])
def test_unsafe_expected_branch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, branch: str
) -> None:
    _mock_clean_repository(monkeypatch)
    monkeypatch.setenv(subject.EXPECTED_BRANCH_VARIABLE, branch)
    with pytest.raises(subject.PostgreSQLTLSServiceError):
        subject._validate_repository()


@pytest.mark.parametrize(
    "configured",
    [
        None,
        "",
        "cf3932d",
        "CF3932DCB43911B2A5F5FF139F6CE07C88CAF389",
        "g" * 40,
        f"{CURRENT_COMMIT}0",
        "0" * 40,
    ],
)
def test_missing_malformed_uppercase_abbreviated_or_stale_head_fails_closed(
    configured: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_clean_repository(monkeypatch)
    if configured is None:
        monkeypatch.delenv(subject.EXPECTED_HEAD_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(subject.EXPECTED_HEAD_VARIABLE, configured)
    with pytest.raises(subject.PostgreSQLTLSServiceError) as captured:
        subject._validate_repository()
    assert captured.value.phase == subject.E_REPOSITORY


@pytest.mark.parametrize("mismatch", ["current", "remote"])
def test_current_or_remote_mismatch_fails_repository_phase(
    mismatch: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = "0" * 40
    _mock_clean_repository(
        monkeypatch,
        current=stale if mismatch == "current" else CURRENT_COMMIT,
        remote=stale if mismatch == "remote" else CURRENT_COMMIT,
    )
    with pytest.raises(subject.PostgreSQLTLSServiceError) as captured:
        subject._validate_repository()
    assert captured.value.phase == subject.E_REPOSITORY


def test_controller_contains_no_fixed_repository_commit_and_accepts_future_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert CURRENT_COMMIT not in source
    assert "EXPECTED_HEAD =" not in source
    future = "1" * 40
    _mock_clean_repository(monkeypatch, current=future, remote=future)
    monkeypatch.setenv(subject.EXPECTED_HEAD_VARIABLE, future)
    subject._validate_repository()


def test_ready_is_emitted_only_from_foreground_wait(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    stop = threading.Event()
    stop.set()
    monkeypatch.setattr(subject.time, "monotonic", lambda: 10.0)
    subject._wait(300, stop)
    assert (
        capsys.readouterr().out
        == "PR54 PostgreSQL TLS READY; TTL remaining: 300 seconds\n"
    )


def test_profile_proof_failure_prevents_handoff_and_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_bounded_lifecycle(monkeypatch, tmp_path, failures=frozenset({"pytest"}))
    monkeypatch.setattr(
        subject,
        "_create_handoff",
        lambda *_args: pytest.fail("handoff published before profile proof passed"),
    )
    monkeypatch.setattr(
        subject,
        "_wait",
        lambda *_args: pytest.fail("READY emitted before profile proof passed"),
    )

    with pytest.raises(subject.PostgreSQLTLSServiceError) as captured:
        subject.run(300)

    assert captured.value.phase == subject.E_MIGRATION


def test_controlled_signals_include_available_contract() -> None:
    assert signal.SIGINT in subject._signals()
    assert signal.SIGTERM in subject._signals()
    sighup = getattr(signal, "SIGHUP", None)
    if isinstance(sighup, signal.Signals):
        assert sighup in subject._signals()
    sigbreak = getattr(signal, "SIGBREAK", None)
    if isinstance(sigbreak, signal.Signals):
        assert sigbreak in subject._signals()


def test_sigbreak_requests_normal_shutdown_and_handlers_are_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sigbreak = getattr(signal, "SIGBREAK", None)
    if not isinstance(sigbreak, signal.Signals):
        pytest.skip("SIGBREAK is not supported")
    installed: dict[signal.Signals, object] = {}
    restored: list[tuple[signal.Signals, object]] = []

    def install(item: signal.Signals, handler: object) -> object:
        if callable(handler):
            installed[item] = handler
            return f"previous-{item.value}"
        restored.append((item, handler))
        return object()

    monkeypatch.setattr(subject.signal, "signal", install)
    stop = threading.Event()
    previous = subject._install_handlers(stop)
    handler = installed[sigbreak]
    assert callable(handler)
    handler(sigbreak.value, None)
    assert stop.is_set()

    subject._restore_handlers(previous)
    assert {item for item, _handler in restored} == set(previous)


def test_explicit_controller_owned_project_is_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = "callmetric-pgvector-tls-123-abcdef123456"
    calls = _stub_bounded_lifecycle(monkeypatch, tmp_path)
    monkeypatch.setenv(subject.PROJECT_NAME_VARIABLE, project)
    subject.run(300)
    assert any(f"--project-name {project}" in rendered for rendered, _ in calls)


def test_cleanup_is_exact_project_and_includes_volumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subject.smoke, "_run", lambda arguments, **_kwargs: calls.append(arguments)
    )
    subject.smoke._cleanup("docker", "callmetric-pgvector-tls-1-abcdef123456", {})
    rendered = " ".join(calls[0])
    assert "--project-name callmetric-pgvector-tls-1-abcdef123456" in rendered
    assert "down --volumes --remove-orphans" in rendered
    assert "prune" not in rendered


@pytest.mark.parametrize("phase", sorted(subject.PHASE_CODES))
def test_main_reports_only_fixed_phase(
    phase: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(_ttl: int, *, preflight_only: bool = False) -> None:
        del preflight_only
        raise subject.PostgreSQLTLSServiceError(phase=phase)

    monkeypatch.setattr(subject, "run", fail)
    assert subject.main(["--ttl-seconds", "300"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"{phase} PR54 PostgreSQL TLS service failed\n"


def test_controller_reuses_pr52_without_modifying_it() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "from scripts import run_postgres_tls_smoke as smoke" in source
    assert "127.0.0.1" not in source
    assert "sslmode=verify-full" in source
    assert "docker system prune" not in source


def test_every_direct_subprocess_call_has_an_explicit_timeout() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    subprocess_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    ]
    assert subprocess_calls
    assert all(
        any(keyword.arg == "timeout" for keyword in call.keywords)
        for call in subprocess_calls
    )
    timeout_values = [
        ast.unparse(
            next(keyword.value for keyword in call.keywords if keyword.arg == "timeout")
        )
        for call in subprocess_calls
    ]
    assert timeout_values.count("timeout") == 1
    assert timeout_values.count("IDENTITY_ACL_TIMEOUT_SECONDS") == 2


def _stub_bounded_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    failures: frozenset[str] = frozenset(),
) -> list[tuple[str, float]]:
    calls: list[tuple[str, float]] = []
    root = tmp_path / "handoff-root"
    root.mkdir()
    handoff = root / "callmetric-postgres-tls-abcdefgh"

    monkeypatch.setattr(subject, "_validate_repository", lambda: None)
    monkeypatch.setattr(subject, "_docker_preflight", lambda: "docker")
    monkeypatch.setattr(subject, "_resource_snapshot", lambda _docker: {})
    monkeypatch.setattr(subject, "_private_root", lambda _raw: root)
    monkeypatch.setattr(subject, "_install_handlers", lambda _stop: {})
    monkeypatch.setattr(subject, "_restore_handlers", lambda _previous: None)
    monkeypatch.setattr(subject, "_create_handoff", lambda *_args: handoff)
    monkeypatch.setattr(subject, "_remove_handoff", lambda *_args: None)
    monkeypatch.setattr(subject, "_wait", lambda *_args: None)
    monkeypatch.setattr(subject.os, "getpid", lambda: 123)
    monkeypatch.setattr(subject.secrets, "token_hex", lambda _size: "abcdef123456")
    monkeypatch.setattr(
        subject.secrets, "token_urlsafe", lambda size: f"synthetic-{size}"
    )
    monkeypatch.setattr(subject.smoke, "_smoke_environment", lambda **_kwargs: {})
    monkeypatch.setattr(subject.smoke, "_pytest_environment", lambda **_kwargs: {})
    monkeypatch.setattr(
        subject.smoke,
        "_generate_certificates",
        lambda *_args: subject.smoke._run(
            ["certificate-generation"], capture_output=True
        ),
    )
    monkeypatch.setattr(
        subject.smoke,
        "_validate_certificates",
        lambda *_args: subject.smoke._run(
            ["certificate-validation"], capture_output=True
        ),
    )
    monkeypatch.setattr(
        subject.smoke,
        "_wait_until_healthy",
        lambda *_args: subject.smoke._run(["readiness-inspect"], capture_output=True),
    )

    def published_port(*_args: object) -> int:
        subject.smoke._run(["published-port"], capture_output=True)
        return 54321

    monkeypatch.setattr(subject.smoke, "_published_port", published_port)

    def run_command(
        arguments: list[str],
        *,
        environment: dict[str, str] | None = None,
        capture_output: bool = True,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del environment, capture_output
        rendered = " ".join(arguments)
        calls.append((rendered, timeout))
        if any(marker in rendered for marker in failures):
            raise subprocess.TimeoutExpired(
                arguments,
                timeout,
                output="synthetic-secret-output",
                stderr="postgresql://synthetic-secret",
            )
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(subject, "_run_command", run_command)
    return calls


def _timeout_for(calls: list[tuple[str, float]], marker: str) -> float:
    return next(timeout for rendered, timeout in calls if marker in rendered)


def test_phase_specific_subprocess_timeout_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub_bounded_lifecycle(monkeypatch, tmp_path)
    subject.run(300)
    assert (
        _timeout_for(calls, "config --quiet") == subject.COMPOSE_CONFIG_TIMEOUT_SECONDS
    )
    assert (
        _timeout_for(calls, "certificate-generation")
        == subject.CERTIFICATE_TIMEOUT_SECONDS
    )
    assert (
        _timeout_for(calls, "certificate-validation")
        == subject.CERTIFICATE_TIMEOUT_SECONDS
    )
    assert _timeout_for(calls, "up -d") == subject.COMPOSE_STARTUP_TIMEOUT_SECONDS
    assert (
        _timeout_for(calls, "readiness-inspect")
        == subject.READINESS_COMMAND_TIMEOUT_SECONDS
    )
    assert (
        _timeout_for(calls, "published-port")
        == subject.READINESS_COMMAND_TIMEOUT_SECONDS
    )
    assert _timeout_for(calls, "pytest") == subject.MIGRATION_PROOF_TIMEOUT_SECONDS
    assert _timeout_for(calls, "down --volumes") == subject.CLEANUP_TIMEOUT_SECONDS
    assert _timeout_for(calls, "container ls") == subject.CLEANUP_TIMEOUT_SECONDS
    assert _timeout_for(calls, "network ls") == subject.CLEANUP_TIMEOUT_SECONDS
    assert _timeout_for(calls, "volume ls") == subject.CLEANUP_TIMEOUT_SECONDS
    assert (
        subject.POSTGRES_READINESS_TIMEOUT_SECONDS
        == subject.smoke.HEALTH_TIMEOUT_SECONDS
    )


@pytest.mark.parametrize(
    ("failure", "expected_phase"),
    [("up -d", subject.E_STARTUP), ("pytest", subject.E_MIGRATION)],
)
def test_primary_timeout_triggers_exact_cleanup(
    failure: str,
    expected_phase: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _stub_bounded_lifecycle(
        monkeypatch, tmp_path, failures=frozenset({failure})
    )
    with pytest.raises(subject.PostgreSQLTLSServiceError) as captured:
        subject.run(300)
    assert captured.value.phase == expected_phase
    downs = [rendered for rendered, _timeout in calls if "down --volumes" in rendered]
    assert len(downs) == 1
    assert "--project-name callmetric-pgvector-tls-123-abcdef123456" in downs[0]


def test_cleanup_timeout_does_not_mask_primary_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_bounded_lifecycle(
        monkeypatch,
        tmp_path,
        failures=frozenset({"up -d", "down --volumes"}),
    )
    with pytest.raises(subject.PostgreSQLTLSServiceError) as captured:
        subject.run(300)
    assert captured.value.phase == subject.E_STARTUP


def test_cleanup_timeout_is_reported_when_it_is_the_only_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_bounded_lifecycle(
        monkeypatch,
        tmp_path,
        failures=frozenset({"down --volumes", "container ls"}),
    )
    with pytest.raises(subject.PostgreSQLTLSServiceError) as captured:
        subject.run(300)
    assert captured.value.phase == subject.E_CLEANUP


def test_timeout_output_never_exposes_secret_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def timeout(_ttl: int, *, preflight_only: bool = False) -> None:
        del preflight_only
        raise subprocess.TimeoutExpired(
            ["secret-command", "postgresql://secret"],
            30,
            output="secret-output",
            stderr="private-key-path",
        )

    monkeypatch.setattr(subject, "run", timeout)
    assert subject.main(["--ttl-seconds", "300"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "E_PREFLIGHT PR54 PostgreSQL TLS service failed\n"


def test_compose_down_timeout_uses_only_validated_exact_project_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = "callmetric-pgvector-tls-123-abcdef123456"
    resources = {
        "container": ("container123",),
        "network": ("network123",),
        "volume": ("volume123",),
    }
    removals: list[list[str]] = []
    residue_checks = 0

    monkeypatch.setattr(
        subject,
        "_project_resource_references",
        lambda _docker, _project: resources,
    )

    def inspect(_docker: str, resource: str, _reference: str, template: str) -> str:
        if template == "{{.Name}}":
            return {
                "container": f"/{project}-{subject.smoke.SERVICE}-1",
                "network": f"{project}_default",
                "volume": f"{project}_{subject.VOLUME_KEY}",
            }[resource]
        if subject.COMPOSE_PROJECT_LABEL in template:
            return project
        if subject.COMPOSE_SERVICE_LABEL in template:
            return subject.smoke.SERVICE
        if subject.COMPOSE_NETWORK_LABEL in template:
            return "default"
        if subject.COMPOSE_VOLUME_LABEL in template:
            return subject.VOLUME_KEY
        pytest.fail("unexpected inspect template")

    monkeypatch.setattr(subject, "_inspect_field", inspect)

    def run_command(arguments: list[str], **_kwargs: object) -> object:
        removals.append(arguments)
        return object()

    monkeypatch.setattr(subject, "_run_command", run_command)

    def down(*_args: object) -> None:
        raise subprocess.TimeoutExpired(["docker", "compose", "down"], 120)

    def residue(*_args: object) -> None:
        nonlocal residue_checks
        residue_checks += 1

    monkeypatch.setattr(subject.smoke, "_cleanup", down)
    monkeypatch.setattr(subject.smoke, "_require_no_project_resources", residue)

    subject._cleanup_project("docker", project, {})

    assert removals == [
        ["docker", "container", "rm", "--force", "container123"],
        ["docker", "volume", "rm", "volume123"],
        ["docker", "network", "rm", "network123"],
    ]
    assert residue_checks == 1
    rendered = " ".join(" ".join(arguments) for arguments in removals)
    assert "prune" not in rendered
    assert "cache" not in rendered


@pytest.mark.parametrize(
    "resources",
    [
        {
            "container": ("one", "two", "three"),
            "network": (),
            "volume": (),
        },
        {"container": (), "network": ("one", "two"), "volume": ()},
        {"container": (), "network": (), "volume": ("one", "two")},
    ],
)
def test_unexpected_exact_project_resource_counts_fail_closed(
    resources: dict[str, tuple[str, ...]],
) -> None:
    with pytest.raises(subject.PostgreSQLTLSServiceError) as captured:
        subject._validate_exact_project_resources(
            "docker",
            "callmetric-pgvector-tls-123-abcdef123456",
            resources,
        )
    assert captured.value.phase == subject.E_CLEANUP


def test_unexpected_project_resource_name_or_label_fails_before_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = "callmetric-pgvector-tls-123-abcdef123456"
    resources = {"container": ("container123",), "network": (), "volume": ()}

    def inspect(_docker: str, _resource: str, _reference: str, template: str) -> str:
        if template == "{{.Name}}":
            return "/unrelated-container"
        if subject.COMPOSE_PROJECT_LABEL in template:
            return project
        return subject.smoke.SERVICE

    monkeypatch.setattr(subject, "_inspect_field", inspect)
    with pytest.raises(subject.PostgreSQLTLSServiceError) as captured:
        subject._validate_exact_project_resources("docker", project, resources)
    assert captured.value.phase == subject.E_CLEANUP


@pytest.mark.parametrize(
    "signal_timing",
    ["before_ready", "foreground", "after_ttl", "during_cleanup"],
)
def test_controlled_interrupt_timings_share_one_ordered_cleanup_lifecycle(
    signal_timing: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_bounded_lifecycle(monkeypatch, tmp_path)
    events: list[str] = []
    stop = threading.Event()

    def install(candidate: threading.Event) -> dict[signal.Signals, object]:
        nonlocal stop
        stop = candidate
        events.append("install")
        return {}

    def request_stop() -> None:
        stop.set()
        events.append("signal")

    def certificates(*_args: object) -> None:
        events.append("certificates")
        if signal_timing == "before_ready":
            request_stop()

    def wait(*_args: object) -> None:
        events.append("wait")
        if signal_timing == "foreground":
            request_stop()
        events.append("ttl-return")

    cleanup_calls = 0

    def cleanup(*_args: object) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        events.append("docker-cleanup-start")
        if signal_timing == "after_ttl":
            request_stop()
        events.append("docker-down-fallback-residue")
        if signal_timing == "during_cleanup":
            request_stop()
        events.append("docker-cleanup-end")

    snapshots = 0

    def snapshot(_docker: str) -> dict[str, frozenset[str]]:
        nonlocal snapshots
        snapshots += 1
        events.append(f"protected-{snapshots}")
        return {"container": frozenset(), "network": frozenset(), "volume": frozenset()}

    monkeypatch.setattr(subject, "_install_handlers", install)
    monkeypatch.setattr(
        subject, "_restore_handlers", lambda _value: events.append("restore")
    )
    monkeypatch.setattr(subject.smoke, "_generate_certificates", certificates)
    monkeypatch.setattr(subject, "_wait", wait)
    monkeypatch.setattr(subject, "_cleanup_project", cleanup)
    monkeypatch.setattr(
        subject, "_remove_handoff", lambda *_args: events.append("handoff")
    )
    monkeypatch.setattr(subject, "_resource_snapshot", snapshot)

    subject.run(300)

    assert cleanup_calls == 1
    assert stop.is_set()
    assert events.index("docker-cleanup-end") < events.index("handoff")
    assert events.index("handoff") < events.index("protected-2")
    assert events.index("protected-2") < events.index("restore")
