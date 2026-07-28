"""Fail safely when tracked text files contain unresolved merge markers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
import subprocess


DIAGNOSTIC_CODE = "repository_conflict_marker_detected"
SCAN_FAILED_CODE = "repository_conflict_marker_scan_failed"
_MARKER_WIDTH = 7
_IGNORED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
_BINARY_SUFFIXES = frozenset(
    {
        ".7z",
        ".bin",
        ".dll",
        ".exe",
        ".flac",
        ".gif",
        ".gz",
        ".jpeg",
        ".jpg",
        ".m4a",
        ".mp3",
        ".ogg",
        ".onnx",
        ".parquet",
        ".pdf",
        ".png",
        ".pt",
        ".pth",
        ".safetensors",
        ".tar",
        ".wav",
        ".webp",
        ".whl",
        ".zip",
    }
)


class MarkerType(str, Enum):
    LESS_THAN = "less_than"
    EQUALS = "equals"
    GREATER_THAN = "greater_than"


@dataclass(frozen=True, slots=True)
class ConflictMarkerFinding:
    relative_path: str
    line_number: int
    marker_type: MarkerType

    def diagnostic(self) -> str:
        return (
            f"{DIAGNOSTIC_CODE} {self.relative_path} "
            f"{self.line_number} {self.marker_type.value}"
        )


def marker_prefixes() -> tuple[tuple[str, MarkerType], ...]:
    return (
        ("<" * _MARKER_WIDTH, MarkerType.LESS_THAN),
        ("=" * _MARKER_WIDTH, MarkerType.EQUALS),
        (">" * _MARKER_WIDTH, MarkerType.GREATER_THAN),
    )


def scan_tracked_repository(repository_root: Path) -> tuple[ConflictMarkerFinding, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    paths = tuple(
        item.decode("utf-8") for item in completed.stdout.split(b"\0") if item
    )
    return scan_relative_paths(repository_root, paths)


def scan_relative_paths(
    repository_root: Path,
    relative_paths: Iterable[str],
) -> tuple[ConflictMarkerFinding, ...]:
    findings: list[ConflictMarkerFinding] = []
    for relative_path in sorted(set(relative_paths)):
        normalized = PurePosixPath(relative_path.replace("\\", "/"))
        if _is_ignored(normalized):
            continue
        path = repository_root.joinpath(*normalized.parts)
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for prefix, marker_type in marker_prefixes():
                if line.startswith(prefix):
                    findings.append(
                        ConflictMarkerFinding(
                            relative_path=normalized.as_posix(),
                            line_number=line_number,
                            marker_type=marker_type,
                        )
                    )
                    break
    return tuple(findings)


def _is_ignored(path: PurePosixPath) -> bool:
    return (
        any(part in _IGNORED_PARTS for part in path.parts)
        or path.suffix.lower() in _BINARY_SUFFIXES
    )


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    try:
        findings = scan_tracked_repository(Path.cwd())
    except (OSError, subprocess.SubprocessError, UnicodeError):
        print(SCAN_FAILED_CODE)
        return 2
    for finding in findings:
        print(finding.diagnostic())
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
