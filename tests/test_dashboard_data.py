from pathlib import Path

from app.benchmark.repository import BenchmarkRepository
from dashboard.common import comparison_rows, filter_runs, load_runs
from tests.benchmark_helpers import make_benchmark_run


def test_dashboard_loads_synthetic_json_and_handles_empty_directory(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    assert load_runs(empty) == []

    repository = BenchmarkRepository(tmp_path / "results")
    repository.save(make_benchmark_run())
    assert load_runs(repository.results_dir) == [make_benchmark_run()]


def test_dashboard_filtering_and_grouping() -> None:
    large = make_benchmark_run(run_id="large")
    small = make_benchmark_run(
        run_id="small",
        model_name="small",
        experiment_id="deney-2",
        wer=0.4,
        word_error_count=4,
    )

    assert filter_runs([large, small], model_name="small") == [small]
    rows = comparison_rows([small, large])
    assert [row["Model"] for row in rows] == ["large-v3", "small"]
