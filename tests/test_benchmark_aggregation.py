import pytest

from app.benchmark.aggregation import (
    aggregate_runs,
    group_by_model_configuration,
    runs_for_experiment,
    runs_for_model,
    runs_for_model_configuration,
)
from tests.benchmark_helpers import make_benchmark_run


def test_weighted_rates_use_total_error_counts() -> None:
    short = make_benchmark_run(
        run_id="short",
        wer=1.0,
        cer=0.5,
        word_error_count=1,
        reference_word_count=1,
        character_error_count=1,
        reference_character_count=2,
    )
    long = make_benchmark_run(
        run_id="long",
        wer=1 / 9,
        cer=1 / 18,
        word_error_count=1,
        reference_word_count=9,
        character_error_count=1,
        reference_character_count=18,
    )

    metrics = aggregate_runs([short, long])

    assert metrics.weighted_wer == pytest.approx(0.2)
    assert metrics.weighted_cer == pytest.approx(0.1)
    assert metrics.weighted_wer != pytest.approx((short.wer + long.wer) / 2)


def test_grouping_and_filters() -> None:
    large = make_benchmark_run(run_id="large")
    small = make_benchmark_run(
        run_id="small", model_name="small", experiment_id="deney-2"
    )

    groups = group_by_model_configuration([large, small])

    assert len(groups) == 2
    assert runs_for_model([large, small], "small") == [small]
    assert runs_for_experiment([large, small], "deney-1") == [large]
    large_configuration = next(key for key in groups if key.model_name == "large-v3")
    assert runs_for_model_configuration([large, small], large_configuration) == [large]


def test_empty_aggregation_is_safe() -> None:
    metrics = aggregate_runs([])

    assert metrics.run_count == 0
    assert metrics.total_reference_words == 0
    assert metrics.weighted_wer is None
    assert metrics.weighted_cer is None
    assert metrics.average_processing_time_seconds is None
