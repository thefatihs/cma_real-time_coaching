import os
from pathlib import Path

from app.benchmark.aggregation import (
    AggregateMetrics,
    ModelConfiguration,
    group_by_model_configuration,
)
from app.benchmark.models import BenchmarkRun
from app.benchmark.repository import BenchmarkRepository


def default_results_directory() -> Path:
    return Path(os.environ.get("CALLMETRIC_BENCHMARK_DIR", "benchmark_results"))


def load_runs(results_directory: Path) -> list[BenchmarkRun]:
    return BenchmarkRepository(results_directory).load_all()


def filter_runs(
    runs: list[BenchmarkRun],
    *,
    experiment_id: str | None = None,
    model_name: str | None = None,
) -> list[BenchmarkRun]:
    return [
        run
        for run in runs
        if (experiment_id is None or run.experiment_id == experiment_id)
        and (model_name is None or run.model_name == model_name)
    ]


def comparison_rows(runs: list[BenchmarkRun]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for configuration, metrics in group_by_model_configuration(runs).items():
        rows.append(_comparison_row(configuration, metrics))
    return sorted(
        rows,
        key=lambda row: (
            row["Ağırlıklı WER"] is None,
            row["Ağırlıklı WER"] if row["Ağırlıklı WER"] is not None else 0.0,
        ),
    )


def _comparison_row(
    configuration: ModelConfiguration, metrics: AggregateMetrics
) -> dict[str, object]:
    return {
        "Model": configuration.model_name,
        "Dil": configuration.language,
        "Beam": configuration.beam_size,
        "VAD": configuration.vad_filter,
        "Önceki metin": configuration.condition_on_previous_text,
        "Başlangıç prompt'u": configuration.initial_prompt or "—",
        "Cihaz": configuration.device,
        "Hesaplama": configuration.compute_type,
        "CPU thread": configuration.cpu_threads,
        "Çalıştırma": metrics.run_count,
        "Ağırlıklı WER": metrics.weighted_wer,
        "Ağırlıklı CER": metrics.weighted_cer,
        "Doğru kelime": metrics.total_correct_words,
    }


def select_results_directory() -> Path:
    import streamlit as st

    default = default_results_directory()
    entered = st.sidebar.text_input("Benchmark sonuç klasörü", value=str(default))
    st.sidebar.info(
        "Uygulama yerel çalışır. Özel kayıt ve transkriptleri Git deposuna koymayın."
    )
    return Path(entered)
