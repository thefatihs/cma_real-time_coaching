"""Read-only readiness checks for the Linux GPU dashboard demo."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from dataclasses import dataclass, field
from io import StringIO
import importlib
from importlib import metadata
import logging
import os
from pathlib import Path
import platform
import socket
import sys
from typing import Callable, Final, Literal, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.classification.artifacts import load_training_metadata  # noqa: E402
from app.classification.calibration import sha256_directory  # noqa: E402
from app.classification.runtime import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    DEFAULT_TAXONOMY,
    DEFAULT_THRESHOLD_PROFILE,
    RuntimeSetFitClassifier,
)

Status = Literal["PASS", "WARN", "FAIL", "PENDING"]
OVERALL_READY: Final = "READY"
OVERALL_NOT_READY: Final = "NOT READY"
OVERALL_PENDING: Final = "TARGET CHECK PENDING"
RELAY_PORT: Final = 18_765
WHISPER_REPOSITORY: Final = "Systran/faster-whisper-large-v3"
FEATURE_GATES: Final[Mapping[str, str]] = {
    "CALLMETRIC_DASHBOARD_LOCAL_MIC_TEST": "1",
    "CALLMETRIC_DASHBOARD_SSH_MIC_RELAY_TEST": "1",
    "CALLMETRIC_LOCAL_MIC_ASR_PROFILE": "gpu-large-v3",
    "CALLMETRIC_UPLOADED_ASR_PROFILE": "gpu-large-v3",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
DASHBOARD_PACKAGES: Final[tuple[tuple[str, str], ...]] = (
    ("streamlit", "streamlit"),
    ("streamlit-webrtc", "streamlit_webrtc"),
    ("aiortc", "aiortc"),
    ("av", "av"),
    ("aioice", "aioice"),
    ("pyee", "pyee"),
)


@dataclass(frozen=True, slots=True)
class Check:
    label: str
    status: Status
    reason: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreflightReport:
    checks: tuple[Check, ...]
    overall: str
    exit_code: int


@dataclass(slots=True)
class ProbeDependencies:
    system: Callable[[], str] = platform.system
    python_version: Callable[[], tuple[int, int, int]] = lambda: (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )
    import_module: Callable[[str], object] = importlib.import_module
    package_version: Callable[[str], str] = metadata.version
    environment: Mapping[str, str] = field(default_factory=lambda: os.environ)
    cache_roots: Callable[[], Sequence[Path]] | None = None
    relay_port_free: Callable[[], bool] | None = None
    artifact_probe: Callable[[], Check] | None = None
    setfit_load_probe: Callable[[bool], tuple[bool, str]] | None = None


def _safe_import(deps: ProbeDependencies, name: str) -> object | None:
    try:
        return deps.import_module(name)
    except Exception:
        return None


def _runtime_check(deps: ProbeDependencies) -> Check:
    version = deps.python_version()
    passed = version >= (3, 12, 0)
    return Check(
        "Runtime",
        "PASS" if passed else "FAIL",
        "RUNTIME_READY" if passed else "PYTHON_VERSION_UNSUPPORTED",
        (f"Python {'.'.join(map(str, version))}", f"Platform {deps.system()}"),
    )


def _cuda_checks(deps: ProbeDependencies) -> tuple[Check, Check]:
    torch = _safe_import(deps, "torch")
    cuda_available = False
    count = 0
    cuda_details: list[str] = []
    if torch is not None:
        try:
            cuda = getattr(torch, "cuda")
            cuda_available = bool(cuda.is_available())
            count = int(cuda.device_count()) if cuda_available else 0
            cuda_details.append(f"torch {deps.package_version('torch')}")
            if count:
                cuda_details.extend(
                    f"CUDA device {index}: {cuda.get_device_name(index)}"
                    for index in range(count)
                )
        except Exception:
            cuda_available = False
    local_windows = deps.system().lower() == "windows"
    if cuda_available and count:
        cuda_check = Check("CUDA", "PASS", "CUDA_READY", tuple(cuda_details))
    elif local_windows:
        cuda_check = Check(
            "CUDA", "PENDING", "TARGET_CHECK_PENDING", tuple(cuda_details)
        )
    else:
        cuda_check = Check("CUDA", "FAIL", "CUDA_UNAVAILABLE", tuple(cuda_details))

    ct2 = _safe_import(deps, "ctranslate2")
    faster_whisper = _safe_import(deps, "faster_whisper")
    runtime_details: list[str] = []
    compute_types: set[str] = set()
    if ct2 is not None:
        try:
            runtime_details.append(f"ctranslate2 {deps.package_version('ctranslate2')}")
            if cuda_available:
                compute_types = set(ct2.get_supported_compute_types("cuda"))  # type: ignore[attr-defined]
                runtime_details.append(
                    f"CUDA compute types {', '.join(sorted(compute_types))}"
                )
        except Exception:
            compute_types = set()
    if faster_whisper is not None:
        try:
            runtime_details.append(
                f"faster-whisper {deps.package_version('faster-whisper')}"
            )
        except Exception:
            pass
    imports_ready = ct2 is not None and faster_whisper is not None
    float16_ready = not cuda_available or "float16" in compute_types
    if imports_ready and float16_ready:
        whisper = Check(
            "Whisper runtime", "PASS", "WHISPER_RUNTIME_READY", tuple(runtime_details)
        )
    else:
        reason = (
            "WHISPER_RUNTIME_MISSING"
            if not imports_ready
            else "WHISPER_FLOAT16_UNSUPPORTED"
        )
        whisper = Check("Whisper runtime", "FAIL", reason, tuple(runtime_details))
    return cuda_check, whisper


def _huggingface_cache_roots(environment: Mapping[str, str]) -> tuple[Path, ...]:
    if value := environment.get("HF_HUB_CACHE"):
        return (Path(value),)
    if value := environment.get("HF_HOME"):
        return (Path(value) / "hub",)
    if value := environment.get("XDG_CACHE_HOME"):
        return (Path(value) / "huggingface" / "hub",)
    try:
        return (Path.home() / ".cache" / "huggingface" / "hub",)
    except Exception:
        return ()


def _whisper_cache_check(deps: ProbeDependencies) -> Check:
    roots = (
        tuple(deps.cache_roots())
        if deps.cache_roots is not None
        else _huggingface_cache_roots(deps.environment)
    )
    model_name = "models--Systran--faster-whisper-large-v3"
    try:
        present = any(
            (root / model_name / "snapshots").is_dir()
            and any(
                item.is_dir() for item in (root / model_name / "snapshots").iterdir()
            )
            for root in roots
        )
    except OSError:
        return Check("Whisper cache", "WARN", "WHISPER_CACHE_UNKNOWN")
    return Check(
        "Whisper cache",
        "PASS" if present else "FAIL",
        "WHISPER_CACHE_PRESENT" if present else "WHISPER_CACHE_MISSING",
        (f"Expected model {WHISPER_REPOSITORY}",),
    )


def _artifact_check() -> Check:
    model_exists = DEFAULT_MODEL_DIR.is_dir()
    profile_exists = DEFAULT_THRESHOLD_PROFILE.is_file()
    taxonomy_exists = DEFAULT_TAXONOMY.is_file()
    details = (
        f"Model directory exists: {model_exists}",
        f"Threshold profile exists: {profile_exists}",
        f"Taxonomy exists: {taxonomy_exists}",
    )
    if not (model_exists and profile_exists and taxonomy_exists):
        return Check("SetFit artifacts", "FAIL", "SETFIT_ARTIFACT_MISSING", details)
    try:
        previous_logging_threshold = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with redirect_stderr(StringIO()):
                from live_dashboard.runtime_wiring import inspect_default_artifacts
        finally:
            logging.disable(previous_logging_threshold)
    except Exception:
        return Check(
            "SetFit artifacts", "FAIL", "SETFIT_ARTIFACT_INCOMPATIBLE", details
        )
    if not inspect_default_artifacts().compatible:
        return Check(
            "SetFit artifacts", "FAIL", "SETFIT_ARTIFACT_INCOMPATIBLE", details
        )
    try:
        artifact = load_training_metadata(DEFAULT_MODEL_DIR)
        checksum = sha256_directory(DEFAULT_MODEL_DIR)
        details += (
            f"model_id {artifact.model_id}",
            f"Label count {len(artifact.label_order)}",
            f"Model checksum {checksum}",
        )
    except Exception:
        return Check(
            "SetFit artifacts", "FAIL", "SETFIT_ARTIFACT_INCOMPATIBLE", details
        )
    return Check("SetFit artifacts", "PASS", "SETFIT_READY", details)


def _setfit_runtime_check(
    deps: ProbeDependencies, *, verify_load: bool, verbose: bool
) -> Check:
    if _safe_import(deps, "setfit") is None:
        return Check("SetFit runtime", "FAIL", "SETFIT_RUNTIME_MISSING")
    details = ("RuntimeSetFitClassifier available", "SetFit device cpu")
    if not verify_load:
        return Check("SetFit runtime", "PASS", "SETFIT_RUNTIME_READY", details)
    probe = deps.setfit_load_probe or _default_setfit_load_probe
    success, detail = probe(verbose)
    if detail:
        details += (detail,)
    return Check(
        "SetFit runtime",
        "PASS" if success else "FAIL",
        "SETFIT_LOAD_VERIFIED" if success else "SETFIT_LOAD_FAILED",
        details,
    )


def _default_setfit_load_probe(verbose: bool) -> tuple[bool, str]:
    try:
        result = RuntimeSetFitClassifier().classify(
            tenant_id="synthetic-preflight",
            call_id="synthetic-preflight",
            text="Aboneliğimi iptal etmek istiyorum.",
        )
        detail = (
            f"Synthetic probability count {len(result.probabilities)}"
            if verbose
            else "Synthetic inference completed"
        )
        return True, detail
    except Exception:
        return False, ""


def _default_relay_port_free() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", RELAY_PORT))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _relay_check(deps: ProbeDependencies) -> Check:
    relay = _safe_import(deps, "app.audio_ingress.tcp_microphone_relay")
    if relay is None or getattr(relay, "RELAY_LOOPBACK_HOST", None) != "127.0.0.1":
        return Check("Relay", "FAIL", "RELAY_IMPORT_FAILED")
    free = (
        deps.relay_port_free()
        if deps.relay_port_free is not None
        else _default_relay_port_free()
    )
    return Check(
        "Relay",
        "PASS" if free else "FAIL",
        "RELAY_READY" if free else "RELAY_PORT_BUSY",
        (f"Loopback port {RELAY_PORT}",),
    )


def _dashboard_check(deps: ProbeDependencies) -> Check:
    details: list[str] = []
    missing: list[str] = []
    previous_logging_threshold = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with redirect_stderr(StringIO()):
            for distribution, module in DASHBOARD_PACKAGES:
                if _safe_import(deps, module) is None:
                    missing.append(distribution)
                    continue
                try:
                    details.append(
                        f"{distribution} {deps.package_version(distribution)}"
                    )
                except Exception:
                    details.append(f"{distribution} version unknown")
    finally:
        logging.disable(previous_logging_threshold)
    return Check(
        "Dashboard deps",
        "PASS" if not missing else "FAIL",
        "DASHBOARD_DEPS_READY" if not missing else "DASHBOARD_DEPENDENCY_MISSING",
        tuple(details + ([f"Missing {', '.join(missing)}"] if missing else [])),
    )


def _environment_check(deps: ProbeDependencies) -> Check:
    mismatches = tuple(
        f"{name}=expected:{expected},actual:{deps.environment.get(name, '<missing>')}"
        for name, expected in FEATURE_GATES.items()
        if deps.environment.get(name) != expected
    )
    return Check(
        "Demo environment",
        "WARN" if mismatches else "PASS",
        "DEMO_ENVIRONMENT_WARNING" if mismatches else "DEMO_ENVIRONMENT_READY",
        mismatches,
    )


def run_preflight(
    *,
    deps: ProbeDependencies | None = None,
    verify_setfit_load: bool = False,
    verbose: bool = False,
) -> PreflightReport:
    dependencies = deps or ProbeDependencies()
    cuda, whisper = _cuda_checks(dependencies)
    artifacts = (
        dependencies.artifact_probe()
        if dependencies.artifact_probe is not None
        else _artifact_check()
    )
    checks = (
        _runtime_check(dependencies),
        cuda,
        whisper,
        _whisper_cache_check(dependencies),
        artifacts,
        _setfit_runtime_check(
            dependencies, verify_load=verify_setfit_load, verbose=verbose
        ),
        _relay_check(dependencies),
        _dashboard_check(dependencies),
        _environment_check(dependencies),
    )
    if any(check.status == "FAIL" for check in checks):
        return PreflightReport(checks, OVERALL_NOT_READY, 1)
    if any(check.status == "PENDING" for check in checks):
        return PreflightReport(checks, OVERALL_PENDING, 2)
    return PreflightReport(checks, OVERALL_READY, 0)


def render_report(report: PreflightReport, *, verbose: bool = False) -> str:
    lines = ["CallMetric GPU Demo Preflight", ""]
    for check in report.checks:
        lines.append(f"{check.label:<20} {check.status}")
        if verbose:
            lines.append(f"  {check.reason}")
            for detail in check.details:
                if detail.startswith("Model checksum "):
                    detail = (
                        f"Model checksum {detail.removeprefix('Model checksum ')[:12]}"
                    )
                lines.append(f"  {detail}")
    lines.extend(("", f"Overall: {report.overall}"))
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only CallMetric Linux GPU demo readiness checks.",
        epilog="Exit codes: 0 READY, 1 NOT READY, 2 TARGET CHECK PENDING.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show safe versions and diagnostics."
    )
    parser.add_argument(
        "--verify-setfit-load",
        action="store_true",
        help="Load trusted local SetFit artifacts and run one synthetic inference.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_preflight(
        verify_setfit_load=arguments.verify_setfit_load,
        verbose=arguments.verbose,
    )
    print(render_report(report, verbose=arguments.verbose))
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
