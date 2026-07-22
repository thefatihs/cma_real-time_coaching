import csv
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.benchmark.models import BenchmarkRun


class BenchmarkRepositoryError(ValueError):
    pass


class DuplicateRunError(BenchmarkRepositoryError):
    pass


class InvalidBenchmarkFileError(BenchmarkRepositoryError):
    pass


class BenchmarkRepository:
    SUMMARY_FILENAME = "benchmark_runs.csv"

    def __init__(
        self,
        results_dir: Path,
        json_reader: Callable[..., Any] = json.loads,
    ) -> None:
        self.results_dir = results_dir
        self._json_reader = json_reader

    def save(self, run: BenchmarkRun, *, overwrite: bool = False) -> Path:
        self._validate_run_id(run.run_id)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        destination = self.results_dir / f"{run.run_id}.json"
        if destination.exists() and not overwrite:
            raise DuplicateRunError(f"Benchmark run already exists: {run.run_id}")

        temporary = self.results_dir / f".{run.run_id}.json.tmp"
        serialized = json.dumps(
            run.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
        )
        temporary.write_text(f"{serialized}\n", encoding="utf-8")
        temporary.replace(destination)
        return destination

    def load(self, path: Path) -> BenchmarkRun:
        try:
            raw = path.read_text(encoding="utf-8")
            values = self._json_reader(raw)
            if not isinstance(values, dict):
                raise ValueError("JSON root must be an object")
            return BenchmarkRun.from_dict(values)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise InvalidBenchmarkFileError(
                f"Invalid benchmark JSON file: {path.name}: {error}"
            ) from error

    def load_all(self) -> list[BenchmarkRun]:
        if not self.results_dir.exists():
            return []
        if not self.results_dir.is_dir():
            raise BenchmarkRepositoryError("Benchmark result path is not a directory")

        runs = [self.load(path) for path in sorted(self.results_dir.glob("*.json"))]
        run_ids = [run.run_id for run in runs]
        if len(run_ids) != len(set(run_ids)):
            raise DuplicateRunError("Duplicate run_id found in benchmark JSON files")
        return runs

    def rebuild_csv(self) -> Path:
        runs = self.load_all()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        destination = self.results_dir / self.SUMMARY_FILENAME
        fieldnames = list(BenchmarkRun.__dataclass_fields__)
        temporary = self.results_dir / f".{self.SUMMARY_FILENAME}.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for run in runs:
                writer.writerow(run.to_dict())
        temporary.replace(destination)
        return destination

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
            raise BenchmarkRepositoryError("run_id contains unsafe characters")
