from __future__ import annotations

import json
import os
import signal
import threading
from pathlib import Path

import pytest

from scripts import run_postgres_tls_service as subject


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


def test_controlled_signals_include_available_contract() -> None:
    assert signal.SIGINT in subject._signals()
    assert signal.SIGTERM in subject._signals()
    sighup = getattr(signal, "SIGHUP", None)
    if isinstance(sighup, signal.Signals):
        assert sighup in subject._signals()


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
