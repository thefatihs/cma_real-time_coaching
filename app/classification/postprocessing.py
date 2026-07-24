"""Deterministic contrast guards for final active classification labels."""

from dataclasses import dataclass
import re
import unicodedata

from app.events.labels import ClassificationViewSource, canonical_label
from app.events.models import ClassificationResultEvent, CoachingAction


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


_ACTION_STRENGTH = {
    CoachingAction.NO_ACTION: 0,
    CoachingAction.TEMPLATE_ACTION: 1,
    CoachingAction.RAG_ACTION: 2,
    CoachingAction.ESCALATE: 3,
}


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


def canonicalize_classification_result(
    result: ClassificationResultEvent,
) -> ClassificationResultEvent:
    labels_by_name = {}
    for label in result.labels:
        name = canonical_label(label.name)
        if name is None:
            continue
        previous = labels_by_name.get(name)
        if previous is None or label.score > previous.score:
            labels_by_name[name] = label.model_copy(update={"name": name})

    probabilities = _canonical_values(result.probabilities)
    thresholds = _canonical_values(result.thresholds)
    active_names = set(labels_by_name)
    if probabilities and thresholds:
        common_names = set(probabilities).intersection(thresholds)
        probabilities = {name: probabilities[name] for name in common_names}
        thresholds = {name: thresholds[name] for name in common_names}
        labels_by_name = {
            name: label
            for name, label in labels_by_name.items()
            if name in common_names
        }
    if any(name != "no_action" for name in active_names):
        labels_by_name.pop("no_action", None)

    return result.model_copy(
        update={
            "labels": list(labels_by_name.values()),
            "probabilities": probabilities,
            "thresholds": thresholds,
        }
    )


def merge_classification_views(
    delta_result: ClassificationResultEvent,
    context_result: ClassificationResultEvent,
) -> tuple[ClassificationResultEvent, dict[str, ClassificationViewSource]]:
    delta_labels = {label.name: label for label in delta_result.labels}
    context_labels = {label.name: label for label in context_result.labels}
    active_names = set(delta_labels).union(context_labels)
    if any(name != "no_action" for name in active_names):
        active_names.discard("no_action")

    merged_labels = [
        max(
            (
                label
                for label in (delta_labels.get(name), context_labels.get(name))
                if label is not None
            ),
            key=lambda label: label.score,
        )
        for name in sorted(active_names)
    ]
    label_sources = {
        name: (
            ClassificationViewSource.BOTH
            if name in delta_labels and name in context_labels
            else ClassificationViewSource.DELTA
            if name in delta_labels
            else ClassificationViewSource.BOUNDED_CONTEXT
        )
        for name in sorted(active_names)
    }
    probabilities = _merge_values(
        delta_result.probabilities,
        context_result.probabilities,
        maximum=True,
    )
    thresholds = _merge_values(
        delta_result.thresholds,
        context_result.thresholds,
        maximum=False,
    )
    if probabilities and thresholds:
        common_names = set(probabilities).intersection(thresholds)
        probabilities = {name: probabilities[name] for name in common_names}
        thresholds = {name: thresholds[name] for name in common_names}
        merged_labels = [label for label in merged_labels if label.name in common_names]
        label_sources = {
            name: source
            for name, source in label_sources.items()
            if name in common_names
        }

    processing_times = [
        value
        for value in (
            delta_result.processing_time_ms,
            context_result.processing_time_ms,
        )
        if value is not None
    ]
    return (
        context_result.model_copy(
            update={
                "labels": merged_labels,
                "action": max(
                    (delta_result.action, context_result.action),
                    key=_ACTION_STRENGTH.__getitem__,
                ),
                "threshold_profile_id": (
                    context_result.threshold_profile_id
                    or delta_result.threshold_profile_id
                ),
                "probabilities": probabilities,
                "thresholds": thresholds,
                "processing_time_ms": (
                    sum(processing_times) if processing_times else None
                ),
            }
        ),
        label_sources,
    )


def _canonical_values(values: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for label, value in values.items():
        name = canonical_label(label)
        if name is not None:
            normalized[name] = max(value, normalized.get(name, 0.0))
    return normalized


def _merge_values(
    first: dict[str, float],
    second: dict[str, float],
    *,
    maximum: bool,
) -> dict[str, float]:
    merged = dict(first)
    for label, value in second.items():
        if label not in merged:
            merged[label] = value
        elif maximum:
            merged[label] = max(merged[label], value)
        else:
            merged[label] = min(merged[label], value)
    return merged


def _normalize_turkish(text: str) -> str:
    casefolded = text.casefold().replace("ı", "i")
    decomposed = unicodedata.normalize("NFKD", casefolded)
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", without_marks)).strip()
