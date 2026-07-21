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

    return TranscriptEvaluationResult(
        reference_text=reference_text,
        hypothesis_text=hypothesis_text,
        normalized_reference=normalized_reference,
        normalized_hypothesis=normalized_hypothesis,
        wer=word_result.wer,
        cer=jiwer.cer(normalized_reference, normalized_hypothesis),
        substitutions=word_result.substitutions,
        deletions=word_result.deletions,
        insertions=word_result.insertions,
        correct_words=word_result.hits,
        reference_word_count=len(normalized_reference.split()),
    )
