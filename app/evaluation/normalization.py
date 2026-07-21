import unicodedata


def normalize_turkish_text(text: str) -> str:
    """Normalize Turkish transcript text for deterministic metric calculation.

    Punctuation characters from every Unicode punctuation category are replaced
    with spaces so removing punctuation never joins neighboring words.
    """
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("I", "ı").replace("İ", "i").lower()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(without_punctuation.split())
