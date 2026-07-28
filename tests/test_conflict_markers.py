from pathlib import Path

from scripts.check_conflict_markers import (
    DIAGNOSTIC_CODE,
    MarkerType,
    marker_prefixes,
    scan_relative_paths,
    scan_tracked_repository,
)


def _marker(character: str) -> str:
    return character * 7


def test_all_marker_types_are_detected_without_surrounding_content(
    tmp_path: Path,
) -> None:
    private_content = "private-content"
    source = tmp_path / "source.py"
    source.write_text(
        "\n".join(
            (
                f"{_marker('<')} branch",
                private_content,
                _marker("="),
                f"{_marker('>')} branch",
            )
        ),
        encoding="utf-8",
    )

    findings = scan_relative_paths(tmp_path, ("source.py",))

    assert [finding.marker_type for finding in findings] == [
        MarkerType.LESS_THAN,
        MarkerType.EQUALS,
        MarkerType.GREATER_THAN,
    ]
    assert [finding.line_number for finding in findings] == [1, 3, 4]
    assert all(DIAGNOSTIC_CODE in finding.diagnostic() for finding in findings)
    assert all(private_content not in finding.diagnostic() for finding in findings)


def test_normal_separators_and_comparisons_are_allowed(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text(
        "value <= other\nlabel = value\nheading\n------\n",
        encoding="utf-8",
    )

    assert scan_relative_paths(tmp_path, ("source.py",)) == ()


def test_binary_files_and_ignored_directories_are_skipped(tmp_path: Path) -> None:
    binary = tmp_path / "audio.wav"
    binary.write_bytes((_marker("<") + " private").encode())
    cache = tmp_path / ".pytest_cache"
    cache.mkdir()
    ignored = cache / "state.txt"
    ignored.write_text(_marker(">"), encoding="utf-8")

    assert (
        scan_relative_paths(
            tmp_path,
            ("audio.wav", ".pytest_cache/state.txt"),
        )
        == ()
    )


def test_marker_patterns_are_constructed_without_source_literals() -> None:
    prefixes = marker_prefixes()

    assert tuple(len(prefix) for prefix, _ in prefixes) == (7, 7, 7)
    assert tuple(marker_type for _, marker_type in prefixes) == tuple(MarkerType)


def test_current_tracked_repository_has_no_conflict_markers() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    assert scan_tracked_repository(repository_root) == ()
