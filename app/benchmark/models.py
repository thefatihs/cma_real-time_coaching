from dataclasses import asdict, dataclass
from typing import Any, Self


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    schema_version: int
    run_id: str
    experiment_id: str
    recording_id: str
    segment_id: str
    created_at_utc: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    model_name: str
    language: str
    beam_size: int
    vad_filter: bool
    condition_on_previous_text: bool
    initial_prompt: str | None
    device: str
    compute_type: str
    cpu_threads: int
    processing_time_seconds: float | None
    real_time_factor: float | None
    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    correct_words: int
    reference_word_count: int
    word_error_count: int
    character_error_count: int
    reference_character_count: int
    codec_name: str | None
    sample_rate_hz: int | None
    channel_count: int | None
    channel_layout: str | None
    sample_format: str | None
    bit_rate: int | None
    reference_filename: str
    hypothesis_filename: str

    def __post_init__(self) -> None:
        _validate_filename(self.reference_filename, "reference_filename")
        _validate_filename(self.hypothesis_filename, "hypothesis_filename")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Self:
        try:
            return cls(**values)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid benchmark run: {error}") from error


def _validate_filename(value: str, field_name: str) -> None:
    if not value or "/" in value or "\\" in value or ":" in value:
        raise ValueError(f"{field_name} must contain only a file name")
