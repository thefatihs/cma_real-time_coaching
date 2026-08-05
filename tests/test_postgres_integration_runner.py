"""Secret-safe unit coverage for the isolated PostgreSQL controller."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import scripts.run_postgres_integration as subject


_SYNTHETIC_PASSWORD = "generated-synthetic-password"
_SYNTHETIC_DSN = "postgresql://synthetic:generated-synthetic-password@local/db"


def _completed(
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["synthetic-command"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_literal_redaction_preserves_unrelated_output() -> None:
    source = f"test_file.py:12 useful {_SYNTHETIC_PASSWORD} {_SYNTHETIC_DSN}"

    sanitized = subject._sanitize_output(  # noqa: SLF001
        source,
        (_SYNTHETIC_PASSWORD, _SYNTHETIC_DSN),
    )

    assert _SYNTHETIC_PASSWORD not in sanitized
    assert _SYNTHETIC_DSN not in sanitized
    assert sanitized.count(subject._REDACTION_PLACEHOLDER) >= 1  # noqa: SLF001
    assert "test_file.py:12 useful" in sanitized


@pytest.mark.parametrize("failure_phase", ("pytest", "Compose", "cleanup"))
def test_subprocess_failure_is_sanitized_and_phase_remains_useful(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_phase: str,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return _completed(
            returncode=1,
            stdout=f"test_file.py:44 {_SYNTHETIC_DSN}\n",
            stderr=f"failure {_SYNTHETIC_PASSWORD}\n",
        )

    monkeypatch.setattr(subject.subprocess, "run", fake_run)

    with pytest.raises(subject.IntegrationRunError, match=failure_phase):
        subject._run(  # noqa: SLF001
            ["synthetic-command"],
            sensitive_values=(_SYNTHETIC_PASSWORD, _SYNTHETIC_DSN),
            failure_phase=failure_phase,
        )

    output = capsys.readouterr()
    combined = output.out + output.err
    assert _SYNTHETIC_PASSWORD not in combined
    assert _SYNTHETIC_DSN not in combined
    assert subject._REDACTION_PLACEHOLDER in combined  # noqa: SLF001
    assert "test_file.py:44" in combined


def test_success_output_is_useful_secret_free_and_command_has_no_credentials(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        arguments: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(arguments)
        return _completed(stdout=f"12 passed {_SYNTHETIC_PASSWORD}\n")

    monkeypatch.setattr(subject.subprocess, "run", fake_run)

    subject._run(  # noqa: SLF001
        ["synthetic-command", "--safe"],
        sensitive_values=(_SYNTHETIC_PASSWORD, _SYNTHETIC_DSN),
    )

    output = capsys.readouterr().out
    assert output == f"12 passed {subject._REDACTION_PLACEHOLDER}\n"  # noqa: SLF001
    assert calls == [["synthetic-command", "--safe"]]
    assert all(_SYNTHETIC_PASSWORD not in value for value in calls[0])
    assert all(_SYNTHETIC_DSN not in value for value in calls[0])


def test_pytest_failure_still_attempts_project_cleanup_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls: list[tuple[str, str, tuple[str, ...]]] = []

    def fake_run(
        arguments: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        if "pytest" in arguments:
            raise subject.IntegrationRunError(
                "PostgreSQL integration failed during pytest"
            )
        return _completed()

    monkeypatch.setattr(subject, "_run", fake_run)
    monkeypatch.setattr(subject.shutil, "which", lambda value: value)
    monkeypatch.setattr(subject, "_wait_until_healthy", lambda *args: None)
    monkeypatch.setattr(subject, "_published_port", lambda *args: 49152)
    monkeypatch.setattr(subject, "_print_project_logs", lambda *args: None)
    monkeypatch.setattr(
        subject,
        "_cleanup",
        lambda docker, project, sensitive: cleanup_calls.append(
            (docker, project, sensitive)
        ),
    )

    with pytest.raises(subject.IntegrationRunError, match="pytest"):
        subject.run()

    assert len(cleanup_calls) == 1
    assert cleanup_calls[0][1].startswith("callmetric-pgvector-")
    assert subject.PASSWORD in cleanup_calls[0][2]


def test_project_log_failure_output_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda *args, **kwargs: _completed(
            returncode=1,
            stdout=f"lifecycle {_SYNTHETIC_PASSWORD}\n",
            stderr=f"compose {_SYNTHETIC_DSN}\n",
        ),
    )

    with pytest.raises(subject.IntegrationRunError, match="Compose logs"):
        subject._print_project_logs(  # noqa: SLF001
            "docker",
            "callmetric-pgvector-123-abcdefabcdef",
            (_SYNTHETIC_PASSWORD, _SYNTHETIC_DSN),
        )

    combined = capsys.readouterr().err
    assert _SYNTHETIC_PASSWORD not in combined
    assert _SYNTHETIC_DSN not in combined
    assert "lifecycle" in combined and "compose" in combined


def test_runner_module_path_is_repository_owned() -> None:
    assert Path(subject.__file__).name == "run_postgres_integration.py"
