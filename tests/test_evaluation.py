import pytest

from app.evaluation.metrics import evaluate_transcript
from app.evaluation.normalization import normalize_turkish_text


def test_turkish_i_characters_are_normalized_correctly() -> None:
    assert normalize_turkish_text("I İ ı i") == "ı i ı i"


def test_punctuation_and_whitespace_are_normalized() -> None:
    text = "  Merhaba,\tİstanbul!\nÇağrı—merkezi...  "

    assert normalize_turkish_text(text) == "merhaba istanbul çağrı merkezi"


def test_unicode_forms_are_normalized_consistently() -> None:
    composed = "Çözüm"
    decomposed = "C\u0327o\u0308zu\u0308m"

    assert normalize_turkish_text(composed) == normalize_turkish_text(decomposed)


def test_identical_transcripts_have_zero_error_rates() -> None:
    result = evaluate_transcript("Müşteri aradı.", "müşteri aradı")

    assert result.wer == 0.0
    assert result.cer == 0.0
    assert result.correct_words == 2
    assert result.reference_word_count == 2


@pytest.mark.parametrize(
    ("reference", "hypothesis", "substitutions", "deletions", "insertions"),
    [
        ("müşteri aradı", "müşteri bekledi", 1, 0, 0),
        ("müşteri bugün aradı", "müşteri aradı", 0, 1, 0),
        ("müşteri aradı", "müşteri bugün aradı", 0, 0, 1),
    ],
)
def test_word_error_types(
    reference: str,
    hypothesis: str,
    substitutions: int,
    deletions: int,
    insertions: int,
) -> None:
    result = evaluate_transcript(reference, hypothesis)

    assert result.substitutions == substitutions
    assert result.deletions == deletions
    assert result.insertions == insertions


@pytest.mark.parametrize("reference", ["", "  ", "... !!!"])
def test_empty_reference_is_rejected(reference: str) -> None:
    with pytest.raises(ValueError, match="Reference transcript is empty"):
        evaluate_transcript(reference, "örnek metin")
