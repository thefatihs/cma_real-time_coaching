from dataclasses import dataclass

from app.benchmark.models import BenchmarkRun


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    run_count: int
    total_reference_words: int
    weighted_wer: float | None
    weighted_cer: float | None
    total_substitutions: int
    total_deletions: int
    total_insertions: int
    total_correct_words: int
    average_processing_time_seconds: float | None
    average_real_time_factor: float | None


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    model_name: str
    language: str
    beam_size: int
    vad_filter: bool
    condition_on_previous_text: bool
    initial_prompt: str | None
    device: str
    compute_type: str
    cpu_threads: int


def aggregate_runs(runs: list[BenchmarkRun]) -> AggregateMetrics:
    total_words = sum(run.reference_word_count for run in runs)
    total_characters = sum(run.reference_character_count for run in runs)
    processing_times = [
        run.processing_time_seconds
        for run in runs
        if run.processing_time_seconds is not None
    ]
    real_time_factors = [
        run.real_time_factor for run in runs if run.real_time_factor is not None
    ]

    return AggregateMetrics(
        run_count=len(runs),
        total_reference_words=total_words,
        weighted_wer=(
            sum(run.word_error_count for run in runs) / total_words
            if total_words
            else None
        ),
        weighted_cer=(
            sum(run.character_error_count for run in runs) / total_characters
            if total_characters
            else None
        ),
        total_substitutions=sum(run.substitutions for run in runs),
        total_deletions=sum(run.deletions for run in runs),
        total_insertions=sum(run.insertions for run in runs),
        total_correct_words=sum(run.correct_words for run in runs),
        average_processing_time_seconds=_average(processing_times),
        average_real_time_factor=_average(real_time_factors),
    )


def runs_for_model(runs: list[BenchmarkRun], model_name: str) -> list[BenchmarkRun]:
    return [run for run in runs if run.model_name == model_name]


def runs_for_experiment(
    runs: list[BenchmarkRun], experiment_id: str
) -> list[BenchmarkRun]:
    return [run for run in runs if run.experiment_id == experiment_id]


def runs_for_model_configuration(
    runs: list[BenchmarkRun], configuration: ModelConfiguration
) -> list[BenchmarkRun]:
    return [run for run in runs if _configuration_for(run) == configuration]


def group_by_model_configuration(
    runs: list[BenchmarkRun],
) -> dict[ModelConfiguration, AggregateMetrics]:
    grouped: dict[ModelConfiguration, list[BenchmarkRun]] = {}
    for run in runs:
        key = _configuration_for(run)
        grouped.setdefault(key, []).append(run)
    return {key: aggregate_runs(group) for key, group in grouped.items()}


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _configuration_for(run: BenchmarkRun) -> ModelConfiguration:
    return ModelConfiguration(
        model_name=run.model_name,
        language=run.language,
        beam_size=run.beam_size,
        vad_filter=run.vad_filter,
        condition_on_previous_text=run.condition_on_previous_text,
        initial_prompt=run.initial_prompt,
        device=run.device,
        compute_type=run.compute_type,
        cpu_threads=run.cpu_threads,
    )
