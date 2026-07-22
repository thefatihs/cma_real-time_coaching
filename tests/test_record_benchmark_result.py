import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.record_benchmark_result import main


def base_arguments(reference: Path, hypothesis: Path, results: Path) -> list[str]:
    return [
        "--reference",
        str(reference),
        "--hypothesis",
        str(hypothesis),
        "--results-dir",
        str(results),
        "--experiment-id",
        "sentetik-deney",
        "--recording-id",
        "call_001",
        "--segment-id",
        "0030-0115",
        "--start",
        "30",
        "--end",
        "75",
        "--vad-filter",
        "--no-condition-on-previous-text",
    ]


def test_successful_synthetic_recording_stores_only_file_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    private_like = tmp_path / "private"
    private_like.mkdir()
    reference = private_like / "çağrı_reference.txt"
    hypothesis = private_like / "çağrı_hypothesis.txt"
    reference.write_text("Müşteri bugün aradı.", encoding="utf-8")
    hypothesis.write_text("Müşteri dün aradı.", encoding="utf-8")
    results = tmp_path / "results"

    exit_code = main(
        base_arguments(reference, hypothesis, results),
        clock=lambda: datetime(2026, 7, 22, tzinfo=UTC),
    )

    assert exit_code == 0
    json_paths = list(results.glob("*.json"))
    assert len(json_paths) == 1
    saved_text = json_paths[0].read_text(encoding="utf-8")
    values = json.loads(saved_text)
    assert values["reference_filename"] == reference.name
    assert values["hypothesis_filename"] == hypothesis.name
    assert str(tmp_path) not in saved_text
    assert "Müşteri bugün aradı" not in saved_text
    assert (results / "benchmark_runs.csv").exists()
    assert "Saved benchmark run" in capsys.readouterr().out


@pytest.mark.parametrize("case", ["missing", "empty"])
def test_reference_failures_return_nonzero(
    case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference = tmp_path / "reference.txt"
    hypothesis = tmp_path / "hypothesis.txt"
    hypothesis.write_text("sentetik hipotez", encoding="utf-8")
    if case == "empty":
        reference.write_text("", encoding="utf-8")

    exit_code = main(base_arguments(reference, hypothesis, tmp_path / "results"))

    assert exit_code == 1
    assert "Error:" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()
