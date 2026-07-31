from __future__ import annotations

import json
import signal
import ssl
import subprocess
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import run_vllm_e2e_service as subject


def _completed(
    arguments: list[str], stdout: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr="")


def test_exact_pins_runtime_limits_and_gpu_are_reused() -> None:
    assert subject.IMAGE_TAG == "vllm/vllm-openai:v0.26.0-ubuntu2404"
    assert subject.IMAGE_INDEX_DIGEST == (
        "sha256:ef7bfc14df9233e3e5d41e733e3be0afa6abbe5ae5f14ee0758110030f6cd53e"
    )
    assert subject.IMAGE_AMD64_DIGEST == (
        "sha256:1161da8a5edbdff239ab1812784d7fe5d28775c675809a8420e8a0a05d0e56d1"
    )
    assert subject.MODEL == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert subject.MODEL_REVISION == "b25037543e9394b818fdfca67ab2a00ecc7dd641"
    assert subject.SERVED_MODEL == "callmetric-qwen25-7b-awq"
    assert (
        subject.GPU_DEVICE,
        subject.MAX_MODEL_LENGTH,
        subject.GPU_MEMORY_UTILIZATION,
        subject.MAX_SEQUENCES,
        subject.MAX_OUTPUT_TOKENS,
    ) == ("0", 8192, 0.80, 2, 256)


def _create_exact_snapshot(cache: Path) -> Path:
    snapshot = cache / subject.MODEL_SNAPSHOT_RELATIVE_PATH
    snapshot.mkdir(parents=True)
    return snapshot


def test_cold_and_warm_disk_thresholds_are_exact() -> None:
    assert subject.COLD_MINIMUM_FREE_BYTES == subject.smoke.MINIMUM_FREE_BYTES
    assert subject.COLD_MINIMUM_FREE_BYTES == 60_726_534_621
    assert subject.WARM_RUNTIME_OVERHEAD_BYTES == 2 * 1024**3
    assert subject.WARM_MINIMUM_FREE_BYTES == 19 * 1024**3
    assert subject.WARM_MINIMUM_FREE_BYTES == (
        subject.smoke.RESERVE_BYTES
        + subject.smoke.MARGIN_BYTES
        + subject.WARM_RUNTIME_OVERHEAD_BYTES
    )


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            (f"{subject.IMAGE_AMD64_DIGEST}|amd64|linux"),
            True,
        ),
        ("sha256:" + "0" * 64 + "|amd64|linux", False),
        (f"{subject.IMAGE_AMD64_DIGEST}|arm64|linux", False),
        (f"{subject.IMAGE_AMD64_DIGEST}|amd64|windows", False),
    ],
)
def test_warm_image_requires_exact_local_amd64_digest(
    metadata: str,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_output", lambda *_args, **_kwargs: metadata)
    assert subject._exact_local_image_available() is expected


def test_warm_path_requires_exact_revision_snapshot(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    wrong = (
        cache
        / "hub"
        / f"models--{subject.MODEL.replace('/', '--')}"
        / "snapshots"
        / ("0" * 40)
    )
    wrong.mkdir(parents=True)

    assert not subject._exact_model_snapshot_available(cache)
    _create_exact_snapshot(cache)
    assert subject._exact_model_snapshot_available(cache)


def test_verified_warm_path_enforces_reserve_margin_and_runtime_overhead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    _create_exact_snapshot(cache)
    monkeypatch.setattr(
        subject.smoke,
        "_validate_cache_directory",
        lambda _raw: cache.resolve(),
    )
    monkeypatch.setattr(subject, "_exact_local_image_available", lambda: True)
    monkeypatch.setattr(
        subject.smoke,
        "_free_disk_bytes",
        lambda: subject.WARM_MINIMUM_FREE_BYTES,
    )

    validated_cache, warm = subject._validated_cache_and_disk()

    assert validated_cache == cache.resolve()
    assert warm is True

    monkeypatch.setattr(
        subject.smoke,
        "_free_disk_bytes",
        lambda: subject.WARM_MINIMUM_FREE_BYTES - 1,
    )
    with pytest.raises(subject.VLLME2EServiceError) as captured:
        subject._validated_cache_and_disk()
    assert captured.value.phase == subject.E_DISK_CAPACITY


@pytest.mark.parametrize("missing_proof", ["image", "snapshot"])
def test_missing_warm_proof_uses_unchanged_cold_guard(
    missing_proof: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    if missing_proof != "snapshot":
        _create_exact_snapshot(cache)
    monkeypatch.setattr(
        subject.smoke,
        "_validate_cache_directory",
        lambda _raw: cache.resolve(),
    )
    monkeypatch.setattr(
        subject,
        "_exact_local_image_available",
        lambda: missing_proof != "image",
    )
    monkeypatch.setattr(
        subject.smoke,
        "_free_disk_bytes",
        lambda: subject.COLD_MINIMUM_FREE_BYTES - 1,
    )

    with pytest.raises(subject.VLLME2EServiceError) as captured:
        subject._validated_cache_and_disk()

    assert captured.value.phase == subject.E_DISK_CAPACITY


def test_invalid_cache_metadata_falls_back_to_cold_then_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject.smoke,
        "_validate_cache_directory",
        lambda _raw: (_ for _ in ()).throw(RuntimeError("sensitive path")),
    )
    monkeypatch.setattr(
        subject.smoke,
        "_free_disk_bytes",
        lambda: subject.COLD_MINIMUM_FREE_BYTES,
    )

    with pytest.raises(subject.VLLME2EServiceError) as captured:
        subject._validated_cache_and_disk()

    assert captured.value.phase == subject.E_CACHE_METADATA
    assert "sensitive" not in str(captured.value)


def test_incomplete_exact_snapshot_is_forced_offline_and_cannot_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    _create_exact_snapshot(cache)
    monkeypatch.setattr(
        subject.smoke,
        "_validate_cache_directory",
        lambda _raw: cache.resolve(),
    )
    monkeypatch.setattr(subject, "_exact_local_image_available", lambda: True)
    monkeypatch.setattr(
        subject.smoke,
        "_free_disk_bytes",
        lambda: subject.WARM_MINIMUM_FREE_BYTES,
    )

    _cache, warm = subject._validated_cache_and_disk()
    environment = subject._environment(cache, tmp_path, "synthetic-token")

    assert warm is True
    assert environment[subject.HF_OFFLINE_ENVIRONMENT_VARIABLE] == "1"
    assert environment[subject.TRANSFORMERS_OFFLINE_ENVIRONMENT_VARIABLE] == "1"


def test_offline_mode_is_mandatory_and_has_no_download_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = subject.COMPOSE_FILE.read_text(encoding="utf-8")
    assert 'HF_HUB_OFFLINE: "${CALLMETRIC_VLLM_HF_HUB_OFFLINE:-0}"' in compose
    assert (
        'TRANSFORMERS_OFFLINE: "${CALLMETRIC_VLLM_TRANSFORMERS_OFFLINE:-0}"' in compose
    )
    monkeypatch.setenv(subject.HF_OFFLINE_ENVIRONMENT_VARIABLE, "0")
    monkeypatch.setenv(subject.TRANSFORMERS_OFFLINE_ENVIRONMENT_VARIABLE, "0")
    environment = subject._environment(
        tmp_path,
        tmp_path,
        "synthetic-token",
    )
    assert environment[subject.HF_OFFLINE_ENVIRONMENT_VARIABLE] == "1"
    assert environment[subject.TRANSFORMERS_OFFLINE_ENVIRONMENT_VARIABLE] == "1"

    unsafe = tmp_path / "compose.yml"
    unsafe.write_text(
        compose.replace("CALLMETRIC_VLLM_HF_HUB_OFFLINE", "WRONG_OFFLINE_VARIABLE"),
        encoding="utf-8",
    )
    monkeypatch.setattr(subject, "COMPOSE_FILE", unsafe)
    with pytest.raises(subject.VLLME2EServiceError):
        subject._validate_runtime_contract()


@pytest.mark.parametrize(
    "phase",
    [
        subject.E_REPOSITORY,
        subject.E_RUNTIME_CONTRACT,
        subject.E_GPU,
        subject.E_GPU_ACTIVITY,
        subject.E_GPU_CLEANUP,
        subject.E_DISK_CAPACITY,
        subject.E_CACHE_METADATA,
        subject.E_IMAGE_METADATA,
        subject.E_MODEL_METADATA,
        subject.E_TLS,
        subject.E_STARTUP,
        subject.E_READINESS,
        subject.E_CLEANUP,
        subject.E_PROTECTED_CONTAINERS,
    ],
)
def test_main_emits_only_fixed_phase_code_and_message(
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        subject,
        "run",
        lambda _ttl: (_ for _ in ()).throw(
            subject.VLLME2EServiceError(
                "sensitive token /private/path",
                phase=phase,
            )
        ),
    )

    assert subject.main(["--ttl-seconds", "300"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"{phase} PR54 vLLM service failed\n"
    assert "sensitive" not in captured.err


def test_runtime_contract_is_loopback_only() -> None:
    subject._validate_runtime_contract()
    assert (subject.LOOPBACK_HOST, subject.TLS_HOST, subject.PORT) == (
        "127.0.0.1",
        "localhost",
        8001,
    )


@pytest.mark.parametrize(
    "unsafe_binding",
    [
        "0.0.0.0:8001:8001",
        "[::]:8001:8001",
        "192.0.2.10:8001:8001",
    ],
)
def test_runtime_contract_rejects_non_loopback_bindings(
    unsafe_binding: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = subject.COMPOSE_FILE.read_text(encoding="utf-8").replace(
        "127.0.0.1:8001:8001",
        unsafe_binding,
    )
    unsafe_compose = tmp_path / "compose.yml"
    unsafe_compose.write_text(compose, encoding="utf-8")
    monkeypatch.setattr(subject, "COMPOSE_FILE", unsafe_compose)

    with pytest.raises(subject.VLLME2EServiceError):
        subject._validate_runtime_contract()


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--ttl-seconds"],
        ["300"],
        ["--ttl-seconds", "0"],
        ["--ttl-seconds", "299"],
        ["--ttl-seconds", "7201"],
        ["--ttl-seconds", "-1"],
        ["--ttl-seconds", "forever"],
        ["--ttl-seconds", "３００"],
    ],
)
def test_ttl_is_required_ascii_bounded_and_nonzero(arguments: list[str]) -> None:
    with pytest.raises(subject.VLLME2EServiceError, match="^PR54 vLLM service failed$"):
        subject._parse_ttl(arguments)


@pytest.mark.parametrize("ttl", [300, 600, 7200])
def test_valid_ttl_is_accepted(ttl: int) -> None:
    assert subject._parse_ttl(["--ttl-seconds", str(ttl)]) == ttl


def test_handoff_is_owner_only_and_contains_exact_files(tmp_path: Path) -> None:
    root = tmp_path / "handoff-root"
    root.mkdir(mode=0o700)
    tls = tmp_path / "tls"
    tls.mkdir(mode=0o700)
    (tls / "ca.crt").write_text("synthetic-ca", encoding="utf-8")

    handoff = subject._create_handoff(root, tls, "synthetic-token", 600)

    assert {path.name for path in handoff.iterdir()} == subject.HANDOFF_FILES
    assert handoff.stat().st_mode & 0o077 == 0
    assert all(path.stat().st_mode & 0o077 == 0 for path in handoff.iterdir())
    metadata = json.loads((handoff / "connection.json").read_text(encoding="utf-8"))
    assert metadata == {
        "ca_file": "ca.crt",
        "host": "localhost",
        "model": subject.SERVED_MODEL,
        "port": 8001,
        "token_file": "token",
        "ttl_seconds": 600,
    }
    subject._remove_handoff(handoff, root.resolve())
    assert not handoff.exists()


def test_private_directory_rejects_permissive_or_repository_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(subject.os, "getuid", lambda: root.stat().st_uid)
    assert subject._validate_private_directory(str(root)) == root.resolve()
    root.chmod(0o750)
    with pytest.raises(subject.VLLME2EServiceError):
        subject._validate_private_directory(str(root))
    with pytest.raises(subject.VLLME2EServiceError):
        subject._validate_private_directory(str(subject.REPOSITORY_ROOT))


def test_gpu_must_be_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject.smoke, "_gpu_memory_used", lambda: 0)
    subject._validate_gpu_idle()
    monkeypatch.setattr(subject.smoke, "_gpu_memory_used", lambda: 1)
    with pytest.raises(
        subject.VLLME2EServiceError,
        match="^PR54 vLLM service failed$",
    ):
        subject._validate_gpu_idle()


def test_gpu_activity_requires_only_bounded_positive_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject.smoke, "_gpu_memory_used", lambda: 6008)
    subject._validate_gpu_active()
    monkeypatch.setattr(subject.smoke, "_gpu_memory_used", lambda: 0)
    with pytest.raises(
        subject.VLLME2EServiceError,
        match="^PR54 vLLM service failed$",
    ):
        subject._validate_gpu_active()


def test_post_cleanup_gpu_idle_polling_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([0.0, 0.0, 0.5, subject.GPU_IDLE_TIMEOUT_SECONDS])
    sleeps: list[float] = []
    monkeypatch.setattr(subject.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(subject.time, "sleep", sleeps.append)
    monkeypatch.setattr(subject.smoke, "_gpu_memory_used", lambda: 1)

    with pytest.raises(
        subject.VLLME2EServiceError,
        match="^PR54 vLLM service failed$",
    ):
        subject._wait_for_gpu_idle()

    assert sleeps == [subject.GPU_IDLE_POLL_SECONDS] * 2


def test_post_cleanup_gpu_idle_polling_accepts_delayed_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = iter([6008, 1, 0])
    clock = iter([0.0, 0.0, 0.5])
    sleeps: list[float] = []
    monkeypatch.setattr(subject.smoke, "_gpu_memory_used", lambda: next(memory))
    monkeypatch.setattr(subject.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(subject.time, "sleep", sleeps.append)

    subject._wait_for_gpu_idle()

    assert sleeps == [subject.GPU_IDLE_POLL_SECONDS] * 2


def test_readiness_requires_health_and_exact_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            (200, {}),
            (200, {"data": [{"id": "wrong-model"}]}),
            (200, {}),
            (200, {"data": [{"id": subject.SERVED_MODEL}]}),
        ]
    )
    monkeypatch.setattr(subject.smoke, "_request", lambda *_a, **_k: next(responses))
    monkeypatch.setattr(subject.time, "sleep", lambda _seconds: None)
    subject._wait_for_ready(ssl.create_default_context(), "synthetic-token")


def test_foreground_wait_is_bounded_and_restores_signal_handlers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installed: list[tuple[signal.Signals, Any]] = []
    waits: list[float | None] = []
    clock = iter([100.0, 100.0, 100.0])

    class FakeEvent:
        def set(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> bool:
            waits.append(timeout)
            return False

    def fake_signal(selected: signal.Signals, handler: Any) -> Any:
        installed.append((selected, handler))
        return f"previous-{selected}"

    monkeypatch.setattr(subject.signal, "signal", fake_signal)
    monkeypatch.setattr(subject.time, "monotonic", lambda: next(clock))
    subject._wait_foreground(300, FakeEvent())  # type: ignore[arg-type]

    assert waits == [300.0]
    assert [selected for selected, _handler in installed[:2]] == [
        signal.SIGINT,
        signal.SIGTERM,
    ]
    assert len(installed) == 4
    assert capsys.readouterr().out == "PR54 vLLM READY; TTL remaining: 300 seconds\n"


def _stub_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    protected_snapshots: list[dict[str, tuple[str, str]]] | None = None,
) -> tuple[list[list[str]], Path]:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    handoff_root = tmp_path / "handoff-root"
    handoff_root.mkdir(mode=0o700)
    calls: list[list[str]] = []
    snapshots = iter(
        protected_snapshots
        or [
            {"protected": ("id", "running")},
            {"protected": ("id", "running")},
        ]
    )

    monkeypatch.setattr(subject, "_validate_repository", lambda: None)
    monkeypatch.setattr(subject, "_validate_runtime_contract", lambda: None)
    monkeypatch.setattr(subject.smoke, "_validate_tools", lambda: None)
    monkeypatch.setattr(subject.smoke, "_validate_gpu", lambda: None)
    monkeypatch.setattr(subject, "_validate_gpu_idle", lambda: None)
    monkeypatch.setattr(subject, "_validate_gpu_active", lambda: None)
    monkeypatch.setattr(subject, "_wait_for_gpu_idle", lambda: None)
    monkeypatch.setattr(subject.smoke, "_validate_disk", lambda: None)
    monkeypatch.setattr(subject.smoke, "_validate_port_free", lambda: None)
    monkeypatch.setattr(
        subject.smoke, "_validate_cache_directory", lambda _raw: cache.resolve()
    )
    monkeypatch.setattr(
        subject,
        "_validated_cache_and_disk",
        lambda: (cache.resolve(), False),
    )
    monkeypatch.setattr(
        subject, "_validate_private_directory", lambda _raw: handoff_root.resolve()
    )
    monkeypatch.setattr(
        subject.smoke, "_protected_container_snapshot", lambda: next(snapshots)
    )
    monkeypatch.setattr(subject.smoke, "_validate_image_metadata", lambda: None)
    monkeypatch.setattr(subject.smoke, "_validate_model_metadata", lambda: None)

    def fake_certificates(directory: Path) -> None:
        (directory / "ca.crt").write_text("synthetic-ca", encoding="utf-8")

    monkeypatch.setattr(subject.smoke, "_generate_certificates", fake_certificates)
    monkeypatch.setattr(
        subject.ssl, "create_default_context", lambda **_kwargs: object()
    )
    monkeypatch.setattr(subject, "_wait_for_ready", lambda *_args: None)
    monkeypatch.setattr(subject, "_wait_foreground", lambda *_args: None)
    monkeypatch.setattr(
        subject,
        "_run",
        lambda arguments, **_kwargs: calls.append(arguments) or _completed(arguments),
    )
    return calls, handoff_root


def test_run_uses_unique_project_and_isolated_cleanup_without_real_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, handoff_root = _stub_lifecycle(monkeypatch, tmp_path)

    subject.run(300)

    rendered = [" ".join(call) for call in calls]
    assert any(" pull vllm" in call for call in rendered)
    assert any(" up --detach vllm" in call for call in rendered)
    down = [call for call in rendered if " down " in call]
    assert len(down) == 1
    assert "--remove-orphans --timeout 30" in down[0]
    assert "--volumes" not in down[0]
    assert not tuple(handoff_root.iterdir())
    assert (tmp_path / "cache").is_dir()


def test_nonzero_prestart_gpu_fails_before_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    validate_gpu_idle = subject._validate_gpu_idle
    calls, _ = _stub_lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(subject, "_validate_gpu_idle", validate_gpu_idle)
    monkeypatch.setattr(subject.smoke, "_gpu_memory_used", lambda: 1)

    with pytest.raises(subject.VLLME2EServiceError) as raised:
        subject.run(300)

    assert raised.value.phase == subject.E_GPU
    assert not calls


def test_lifecycle_gpu_and_cleanup_order_is_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, _ = _stub_lifecycle(monkeypatch, tmp_path)
    events: list[str] = []

    monkeypatch.setattr(
        subject, "_validate_gpu_idle", lambda: events.append("pre-idle")
    )
    monkeypatch.setattr(
        subject, "_validate_gpu_active", lambda: events.append("post-active")
    )
    monkeypatch.setattr(
        subject, "_wait_for_gpu_idle", lambda: events.append("post-idle")
    )
    monkeypatch.setattr(
        subject, "_wait_for_ready", lambda *_args: events.append("readiness")
    )
    monkeypatch.setattr(
        subject, "_wait_foreground", lambda *_args: events.append("foreground")
    )

    def ordered_run(arguments: list[str], **_kwargs: object) -> object:
        calls.append(arguments)
        if "up" in arguments:
            events.append("up")
        if "down" in arguments:
            events.append("down")
        return _completed(arguments)

    monkeypatch.setattr(subject, "_run", ordered_run)

    subject.run(300)

    assert events == [
        "pre-idle",
        "up",
        "readiness",
        "post-active",
        "foreground",
        "down",
        "post-idle",
    ]


def test_zero_post_readiness_gpu_activity_fails_and_runs_exact_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    validate_gpu_active = subject._validate_gpu_active
    calls, _ = _stub_lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(subject, "_validate_gpu_active", validate_gpu_active)
    monkeypatch.setattr(subject.smoke, "_gpu_memory_used", lambda: 0)

    with pytest.raises(subject.VLLME2EServiceError) as raised:
        subject.run(300)

    assert raised.value.phase == subject.E_GPU_ACTIVITY
    up = [call for call in calls if "up" in call]
    down = [call for call in calls if "down" in call]
    assert len(up) == len(down) == 1
    assert (
        up[0][up[0].index("--project-name") + 1]
        == down[0][down[0].index("--project-name") + 1]
    )


@pytest.mark.parametrize(
    ("failure", "expected_phase"),
    [
        ("startup", subject.E_STARTUP),
        ("readiness", subject.E_READINESS),
        ("foreground", subject.E_RUNTIME_CONTRACT),
    ],
)
def test_failures_from_startup_onward_attempt_exact_down_once(
    failure: str,
    expected_phase: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls, _ = _stub_lifecycle(monkeypatch, tmp_path)

    def selective_run(arguments: list[str], **_kwargs: object) -> object:
        calls.append(arguments)
        if failure == "startup" and "up" in arguments:
            raise RuntimeError("synthetic startup failure")
        return _completed(arguments)

    monkeypatch.setattr(subject, "_run", selective_run)
    if failure == "readiness":
        monkeypatch.setattr(
            subject,
            "_wait_for_ready",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic")),
        )
    if failure == "foreground":
        monkeypatch.setattr(
            subject,
            "_wait_foreground",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic")),
        )

    with pytest.raises(subject.VLLME2EServiceError) as raised:
        subject.run(300)

    assert raised.value.phase == expected_phase
    up = [call for call in calls if "up" in call]
    down = [call for call in calls if "down" in call]
    assert len(up) == len(down) == 1
    assert (
        up[0][up[0].index("--project-name") + 1]
        == down[0][down[0].index("--project-name") + 1]
    )


def test_cleanup_failure_does_not_mask_primary_and_later_checks_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, handoff_root = _stub_lifecycle(monkeypatch, tmp_path)
    protected_calls = 0

    def protected_snapshot() -> dict[str, tuple[str, str]]:
        nonlocal protected_calls
        protected_calls += 1
        return {"protected": ("id", "running")}

    cleanup_calls = 0

    def failing_cleanup(_project: str, _environment: dict[str, str]) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setattr(
        subject.smoke, "_protected_container_snapshot", protected_snapshot
    )
    monkeypatch.setattr(subject, "_cleanup", failing_cleanup)
    monkeypatch.setattr(
        subject,
        "_wait_for_ready",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic readiness")),
    )

    with pytest.raises(subject.VLLME2EServiceError) as raised:
        subject.run(300)

    assert raised.value.phase == subject.E_READINESS
    assert cleanup_calls == 1
    assert protected_calls == 2
    assert not tuple(handoff_root.iterdir())
    assert (tmp_path / "cache").is_dir()
    assert not any("down" in call for call in calls)


def test_cleanup_failure_without_primary_reports_cleanup_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _calls, handoff_root = _stub_lifecycle(monkeypatch, tmp_path)
    protected_calls = 0

    def protected_snapshot() -> dict[str, tuple[str, str]]:
        nonlocal protected_calls
        protected_calls += 1
        return {"protected": ("id", "running")}

    monkeypatch.setattr(
        subject.smoke, "_protected_container_snapshot", protected_snapshot
    )
    monkeypatch.setattr(
        subject,
        "_cleanup",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic cleanup")),
    )

    with pytest.raises(subject.VLLME2EServiceError) as raised:
        subject.run(300)

    assert raised.value.phase == subject.E_CLEANUP
    assert protected_calls == 2
    assert not tuple(handoff_root.iterdir())
    assert (tmp_path / "cache").is_dir()


@pytest.mark.parametrize("secondary_failure", ["handoff", "protected"])
def test_secondary_failures_do_not_mask_primary_phase(
    secondary_failure: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _calls, _handoff_root = _stub_lifecycle(monkeypatch, tmp_path)
    protected_calls = 0

    def protected_snapshot() -> dict[str, tuple[str, str]]:
        nonlocal protected_calls
        protected_calls += 1
        if secondary_failure == "protected" and protected_calls == 2:
            raise RuntimeError("synthetic protected failure")
        return {"protected": ("id", "running")}

    monkeypatch.setattr(
        subject.smoke, "_protected_container_snapshot", protected_snapshot
    )
    if secondary_failure == "handoff":
        monkeypatch.setattr(
            subject,
            "_remove_handoff",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic handoff")),
        )
    monkeypatch.setattr(
        subject,
        "_wait_for_ready",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic readiness")),
    )

    with pytest.raises(subject.VLLME2EServiceError) as raised:
        subject.run(300)

    assert raised.value.phase == subject.E_READINESS
    assert protected_calls == 2
    assert (tmp_path / "cache").is_dir()


def test_persistent_post_cleanup_gpu_use_has_fixed_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wait_for_gpu_idle = subject._wait_for_gpu_idle
    _stub_lifecycle(monkeypatch, tmp_path)
    clock = iter([0.0, 0.0, subject.GPU_IDLE_TIMEOUT_SECONDS])
    monkeypatch.setattr(subject.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(subject.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(subject.smoke, "_gpu_memory_used", lambda: 1)
    monkeypatch.setattr(subject, "_wait_for_gpu_idle", wait_for_gpu_idle)

    with pytest.raises(subject.VLLME2EServiceError) as raised:
        subject.run(300)

    assert raised.value.phase == subject.E_GPU_CLEANUP


def test_protected_container_change_fails_after_isolated_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, _ = _stub_lifecycle(
        monkeypatch,
        tmp_path,
        protected_snapshots=[
            {"protected": ("original", "running")},
            {"protected": ("changed", "running")},
        ],
    )

    with pytest.raises(subject.VLLME2EServiceError):
        subject.run(300)

    assert sum("down" in call for call in calls) == 1


def test_subprocess_timeout_is_bounded_and_secret_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired("sensitive-command", 1)

    monkeypatch.setattr(subject.subprocess, "run", timeout)
    with pytest.raises(subject.VLLME2EServiceError, match="^PR54 vLLM service failed$"):
        subject._run(["synthetic-command"])
    assert subject.SUBPROCESS_TIMEOUT_SECONDS > 0
    assert subject.PULL_TIMEOUT_SECONDS > 0
    assert subject.START_TIMEOUT_SECONDS > 0
    assert subject.HEALTH_TIMEOUT_SECONDS > 0
    assert subject.SHUTDOWN_TIMEOUT_SECONDS > 0


def test_source_preserves_cache_and_rejects_unsafe_docker_or_binding() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "prune" not in source
    assert '"--volumes"' not in source
    assert '"0.0.0.0"' not in source
    compose = subject.COMPOSE_FILE.read_text(encoding="utf-8")
    assert '"[::]"' not in compose
    assert "container_name" not in compose
    assert "rmtree(cache" not in source
    assert "glob(" not in source
    assert "shell=False" in source
    assert "--query-compute-apps" not in source
    assert "docker kill" not in source


def test_worktree_accepts_clean_or_exact_development_artifacts() -> None:
    exact_status = "".join(
        f"?? {path}\0" for path in sorted(subject.AUTHORIZED_DEVELOPMENT_ARTIFACTS)
    )
    subject._validate_worktree_status("")
    subject._validate_worktree_status(exact_status)


@pytest.mark.parametrize(
    "status",
    [
        "?? scripts/run_vllm_e2e_service.py\0",
        " M scripts/run_vllm_e2e_service.py\0",
        "?? ../scripts/run_vllm_e2e_service.py\0",
        "?? scripts/run_vllm_e2e_service.py.backup\0",
    ],
)
def test_worktree_rejects_non_exact_development_state(status: str) -> None:
    with pytest.raises(subject.VLLME2EServiceError):
        subject._validate_worktree_status(status)


def test_canonical_module_entrypoint_imports_without_running_lifecycle() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_vllm_e2e_service",
            "--ttl-seconds",
            "0",
        ],
        cwd=subject.REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "E_RUNTIME_CONTRACT PR54 vLLM service failed\n"
    assert "ModuleNotFoundError" not in result.stderr


def test_runbook_uses_only_canonical_module_invocation() -> None:
    runbook = (
        subject.REPOSITORY_ROOT / "docs" / "runbooks" / "vllm_e2e_service_controller.md"
    ).read_text(encoding="utf-8")

    assert (
        "uv run python -m scripts.run_vllm_e2e_service --ttl-seconds <300-7200>"
        in runbook
    )
    assert "uv run python scripts/run_vllm_e2e_service.py" not in runbook


def test_main_emits_only_fixed_secret_free_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        subject,
        "run",
        lambda _ttl: (_ for _ in ()).throw(
            RuntimeError("token certificate /sensitive/path environment")
        ),
    )

    assert subject.main(["--ttl-seconds", "300"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "E_RUNTIME_CONTRACT PR54 vLLM service failed\n"
    assert "sensitive" not in captured.err
