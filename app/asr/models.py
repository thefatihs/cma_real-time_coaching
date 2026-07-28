from dataclasses import dataclass
from enum import Enum
from math import isfinite


@dataclass(frozen=True, slots=True)
class ASRWordTimestamp:
    text: str
    start_seconds: float
    end_seconds: float
    probability: float | None = None

    def __post_init__(self) -> None:
        normalized_text = self.text.strip()
        if not normalized_text:
            raise ASRWordTimestampError(ASRWordTimestampErrorCategory.INVALID_TEXT)
        if (
            not isfinite(self.start_seconds)
            or not isfinite(self.end_seconds)
            or self.start_seconds < 0
            or self.end_seconds <= self.start_seconds
        ):
            raise ASRWordTimestampError(ASRWordTimestampErrorCategory.INVALID_TIMESTAMP)
        if self.probability is not None and (
            not isfinite(self.probability) or not 0.0 <= self.probability <= 1.0
        ):
            raise ASRWordTimestampError(
                ASRWordTimestampErrorCategory.INVALID_PROBABILITY
            )
        object.__setattr__(self, "text", normalized_text)


class ASRWordTimestampErrorCategory(str, Enum):
    INVALID_TEXT = "invalid_word_text"
    INVALID_TIMESTAMP = "invalid_word_timestamp"
    INVALID_PROBABILITY = "invalid_word_probability"
    OUTSIDE_PARENT_SEGMENT = "word_outside_parent_segment"
    NONDETERMINISTIC_ORDER = "nondeterministic_word_order"
    MALFORMED_PROVIDER_OUTPUT = "malformed_word_output"


class ASRWordTimestampError(ValueError):
    def __init__(self, category: ASRWordTimestampErrorCategory) -> None:
        self.category = category
        super().__init__(category.value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(category={self.category.value!r})"


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    start_seconds: float
    end_seconds: float
    text: str
    words: tuple[ASRWordTimestamp, ...] = ()


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration_seconds: float
    processing_time_seconds: float
    segments: list[TranscriptionSegment]
