import json
from pathlib import Path

import pytest

from app.benchmark.models import BenchmarkRun
from app.benchmark.repository import (
    BenchmarkRepository,
    DuplicateRunError,
    InvalidBenchmarkFileError,
)
from tests.benchmark_helpers import make_benchmark_run


def test_json_round_trip_is_deterministic_and_utf8_safe(tmp_path: Path) -> None:
    repository = BenchmarkRepository(tmp_path)
    run = make_benchmark_run()

    saved_path = repository.save(run)
    loaded = repository.load(saved_path)
    serialized = saved_path.read_text(encoding="utf-8")

    assert loaded == run
    assert "çağrı_001_reference.txt" in serialized
    assert "\\u00e7" not in serialized
    assert serialized == saved_path.read_text(encoding="utf-8")


def test_duplicate_is_rejected_unless_overwrite_is_explicit(tmp_path: Path) -> None:
    repository = BenchmarkRepository(tmp_path)
    original = make_benchmark_run()
    replacement = make_benchmark_run(wer=0.1)
    repository.save(original)

    with pytest.raises(DuplicateRunError):
        repository.save(replacement)

    repository.save(replacement, overwrite=True)
    assert repository.load(tmp_path / "run-001.json").wer == 0.1


def test_invalid_json_has_clear_error(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(InvalidBenchmarkFileError, match="broken.json"):
        BenchmarkRepository(tmp_path).load_all()


def test_csv_is_rebuilt_from_json_and_unrelated_files_are_ignored(
    tmp_path: Path,
) -> None:
    repository = BenchmarkRepository(tmp_path)
    repository.save(make_benchmark_run())
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    csv_path = repository.rebuild_csv()
    csv_text = csv_path.read_text(encoding="utf-8")

    assert "run_id" in csv_text
    assert "run-001" in csv_text
    assert "çağrı_001_reference.txt" in csv_text
    assert "ignore me" not in csv_text


def test_persisted_result_rejects_paths_and_contains_no_transcript(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="only a file name"):
        make_benchmark_run(reference_filename="C:\\Private\\reference.txt")

    path = BenchmarkRepository(tmp_path).save(make_benchmark_run())
    values = json.loads(path.read_text(encoding="utf-8"))

    assert "reference_text" not in values
    assert "hypothesis_text" not in values
    assert all("C:\\" not in str(value) for value in values.values())


def test_model_from_dict_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="Invalid benchmark run"):
        BenchmarkRun.from_dict({"run_id": "incomplete"})
