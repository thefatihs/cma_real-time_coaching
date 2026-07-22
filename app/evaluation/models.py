from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranscriptEvaluationResult:
    reference_text: str
    hypothesis_text: str
    normalized_reference: str
    normalized_hypothesis: str
    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    correct_words: int
    reference_word_count: int
    character_substitutions: int = 0
    character_deletions: int = 0
    character_insertions: int = 0
    character_error_count: int = 0
    reference_character_count: int = 0
