from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import check_gpu_demo_readiness as readiness


SCRIPT = Path("scripts/check_gpu_demo_readiness.py")


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return 1 if self._available else 0

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "Synthetic GPU"


def _dependencies(
    tmp_path: Path,
    *,
    system: str = "Linux",
    cuda: bool = True,
    cache: bool = True,
    artifacts: str = "ready",
    relay_free: bool = True,
    missing_module: str | None = None,
    environment_ready: bool = True,
) -> readiness.ProbeDependencies:
    cache_root = tmp_path / "hub"
    if cache:
        (
            cache_root / "models--Systran--faster-whisper-large-v3" / "snapshots" / "x"
        ).mkdir(parents=True)
    modules = {
        "torch": SimpleNamespace(cuda=FakeCuda(cuda)),
        "ctranslate2": SimpleNamespace(
            get_supported_compute_types=lambda device: {"float16", "int8"}
        ),
        "faster_whisper": object(),
        "setfit": object(),
        "app.audio_ingress.tcp_microphone_relay": SimpleNamespace(
            RELAY_LOOPBACK_HOST="127.0.0.1"
        ),
        **{module: object() for _, module in readiness.DASHBOARD_PACKAGES},
    }

    def import_module(name: str) -> object:
        if name == missing_module or name not in modules:
            raise ImportError
        return modules[name]

    artifact_status = {
        "ready": readiness.Check("SetFit artifacts", "PASS", "SETFIT_READY"),
        "missing": readiness.Check(
            "SetFit artifacts", "FAIL", "SETFIT_ARTIFACT_MISSING"
        ),
        "incompatible": readiness.Check(
            "SetFit artifacts", "FAIL", "SETFIT_ARTIFACT_INCOMPATIBLE"
        ),
    }[artifacts]
    environment = dict(readiness.FEATURE_GATES) if environment_ready else {}
    return readiness.ProbeDependencies(
        system=lambda: system,
        python_version=lambda: (3, 12, 1),
        import_module=import_module,
        package_version=lambda name: "1.0",
        environment=environment,
        cache_roots=lambda: (cache_root,),
        relay_port_free=lambda: relay_free,
        artifact_probe=lambda: artifact_status,
    )


def _reason(report: readiness.PreflightReport, label: str) -> str:
    return next(check.reason for check in report.checks if check.label == label)


def test_ready_linux_gpu_scenario(tmp_path: Path) -> None:
    report = readiness.run_preflight(deps=_dependencies(tmp_path))
    assert (report.overall, report.exit_code) == ("READY", 0)


def test_missing_cuda_fails_linux(tmp_path: Path) -> None:
    report = readiness.run_preflight(deps=_dependencies(tmp_path, cuda=False))
    assert (report.overall, report.exit_code) == ("NOT READY", 1)
    assert _reason(report, "CUDA") == "CUDA_UNAVAILABLE"


def test_windows_without_cuda_is_target_pending(tmp_path: Path) -> None:
    report = readiness.run_preflight(
        deps=_dependencies(tmp_path, system="Windows", cuda=False)
    )
    assert (report.overall, report.exit_code) == ("TARGET CHECK PENDING", 2)


@pytest.mark.parametrize(
    ("artifact", "reason"),
    [
        ("missing", "SETFIT_ARTIFACT_MISSING"),
        ("incompatible", "SETFIT_ARTIFACT_INCOMPATIBLE"),
    ],
)
def test_bad_setfit_artifacts_fail(tmp_path: Path, artifact: str, reason: str) -> None:
    report = readiness.run_preflight(deps=_dependencies(tmp_path, artifacts=artifact))
    assert report.exit_code == 1
    assert _reason(report, "SetFit artifacts") == reason


def test_missing_whisper_cache_fails(tmp_path: Path) -> None:
    report = readiness.run_preflight(deps=_dependencies(tmp_path, cache=False))
    assert _reason(report, "Whisper cache") == "WHISPER_CACHE_MISSING"
    assert report.exit_code == 1


def test_relay_port_collision_fails(tmp_path: Path) -> None:
    report = readiness.run_preflight(deps=_dependencies(tmp_path, relay_free=False))
    assert _reason(report, "Relay") == "RELAY_PORT_BUSY"


def test_missing_dashboard_dependency_fails(tmp_path: Path) -> None:
    report = readiness.run_preflight(
        deps=_dependencies(tmp_path, missing_module="aioice")
    )
    assert _reason(report, "Dashboard deps") == "DASHBOARD_DEPENDENCY_MISSING"


def test_environment_warning_does_not_fail(tmp_path: Path) -> None:
    report = readiness.run_preflight(
        deps=_dependencies(tmp_path, environment_ready=False)
    )
    assert _reason(report, "Demo environment") == "DEMO_ENVIRONMENT_WARNING"
    assert report.exit_code == 0


def test_default_output_has_no_details_secrets_or_home_path(tmp_path: Path) -> None:
    deps = _dependencies(tmp_path)
    deps.environment = {
        **readiness.FEATURE_GATES,
        "SECRET_TOKEN": "never-print-this",
        "HOME": r"C:\Users\PrivateName",
    }
    output = readiness.render_report(readiness.run_preflight(deps=deps))
    assert "never-print-this" not in output
    assert "PrivateName" not in output
    assert str(tmp_path) not in output


def test_optional_setfit_load_probe_is_bounded(tmp_path: Path) -> None:
    calls: list[bool] = []
    deps = _dependencies(tmp_path)
    deps.setfit_load_probe = lambda verbose: (
        calls.append(verbose) is None,
        "Synthetic inference completed",
    )
    report = readiness.run_preflight(deps=deps, verify_setfit_load=True)
    assert calls == [False]
    assert _reason(report, "SetFit runtime") == "SETFIT_LOAD_VERIFIED"


def test_normal_check_does_not_call_load_probe(tmp_path: Path) -> None:
    deps = _dependencies(tmp_path)

    def forbidden(verbose: bool) -> tuple[bool, str]:
        raise AssertionError("load probe must be opt-in")

    deps.setfit_load_probe = forbidden
    readiness.run_preflight(deps=deps)


def test_no_network_or_download_api_is_used() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from_pretrained(" not in source
    assert "requests." not in source
    assert "urlopen(" not in source
