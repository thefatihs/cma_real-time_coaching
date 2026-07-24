"""Canonical business-label normalization at runtime boundaries."""

from collections.abc import Iterable
from enum import Enum


CANONICAL_LABELS = frozenset(
    {
        "product_information",
        "price_objection",
        "cancellation_request",
        "technical_issue",
        "complaint",
        "renewal_interest",
        "churn_risk",
        "no_action",
    }
)

_LABEL_ALIASES = {
    "urun_bilgisi": "product_information",
    "paket_sorusu": "product_information",
    "fiyat_itirazi": "price_objection",
    "butce_endisesi": "price_objection",
    "iptal_riski": "cancellation_request",
    "ayrilma_talebi": "cancellation_request",
}


class ClassificationViewSource(str, Enum):
    DELTA = "delta"
    BOUNDED_CONTEXT = "bounded_context"
    BOTH = "both"


def canonical_label(label: str) -> str | None:
    cleaned = label.strip()
    if cleaned in CANONICAL_LABELS:
        return cleaned
    return _LABEL_ALIASES.get(cleaned.casefold())


def canonical_labels(labels: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for label in labels:
        normalized = canonical_label(label)
        if normalized is not None and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    if any(label != "no_action" for label in result):
        return [label for label in result if label != "no_action"]
    return result
