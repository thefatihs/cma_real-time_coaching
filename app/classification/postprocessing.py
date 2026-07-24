"""Deterministic contrast guards for final active classification labels."""

from dataclasses import dataclass
import re
import unicodedata

from app.events.models import ClassificationResultEvent


PRICE_INFORMATION_SIGNALS = (
    "fiyati ne kadar",
    "aylik fiyat",
    "ucreti ne",
    "ucret secenek",
    "fiyat bilgisi",
    "kac lira",
    "paket ucret",
    "tarife fiyati",
)
PRICE_OBJECTION_SIGNALS = (
    "cok pahali",
    "fiyat cok yuksek",
    "butcemi asiyor",
    "bu ucret fazla",
    "odeyemem",
    "uygun degil",
    "bu fiyatla devam etmem",
    "indirim yapilmazsa",
    "verdigim paraya degmiyor",
    "ucretiniz cok pahali",
)


@dataclass(frozen=True, slots=True)
class ClassificationPostProcessingMetadata:
    applied_guards: tuple[str, ...] = ()
    suppressed_labels: tuple[str, ...] = ()


def apply_classification_contrast_guards(
    text: str,
    result: ClassificationResultEvent,
) -> tuple[ClassificationResultEvent, ClassificationPostProcessingMetadata]:
    normalized = _normalize_turkish(text)
    active_names = {label.name for label in result.labels}
    price_query = any(signal in normalized for signal in PRICE_INFORMATION_SIGNALS)
    price_objection = any(signal in normalized for signal in PRICE_OBJECTION_SIGNALS)
    if "price_objection" not in active_names or not price_query or price_objection:
        return result, ClassificationPostProcessingMetadata()
    filtered = result.model_copy(
        update={
            "labels": [
                label for label in result.labels if label.name != "price_objection"
            ]
        }
    )
    return (
        filtered,
        ClassificationPostProcessingMetadata(
            applied_guards=("turkish_price_query_vs_objection",),
            suppressed_labels=("price_objection",),
        ),
    )


def _normalize_turkish(text: str) -> str:
    casefolded = text.casefold().replace("ı", "i")
    decomposed = unicodedata.normalize("NFKD", casefolded)
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", without_marks)).strip()
