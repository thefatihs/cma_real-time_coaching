import jiwer

from app.evaluation.models import TranscriptEvaluationResult
from app.evaluation.normalization import normalize_turkish_text


def evaluate_transcript(
    reference_text: str, hypothesis_text: str
) -> TranscriptEvaluationResult:
    normalized_reference = normalize_turkish_text(reference_text)
    normalized_hypothesis = normalize_turkish_text(hypothesis_text)

    if not normalized_reference:
        raise ValueError("Reference transcript is empty after normalization")

    word_result = jiwer.process_words(normalized_reference, normalized_hypothesis)
    character_result = jiwer.process_characters(
        normalized_reference, normalized_hypothesis
    )

    return TranscriptEvaluationResult(
        reference_text=reference_text,
        hypothesis_text=hypothesis_text,
        normalized_reference=normalized_reference,
        normalized_hypothesis=normalized_hypothesis,
        wer=word_result.wer,
        cer=character_result.cer,
        substitutions=word_result.substitutions,
        deletions=word_result.deletions,
        insertions=word_result.insertions,
        correct_words=word_result.hits,
        reference_word_count=len(normalized_reference.split()),
        character_substitutions=character_result.substitutions,
        character_deletions=character_result.deletions,
        character_insertions=character_result.insertions,
        character_error_count=(
            character_result.substitutions
            + character_result.deletions
            + character_result.insertions
        ),
        reference_character_count=(
            character_result.hits
            + character_result.substitutions
            + character_result.deletions
        ),
    )
