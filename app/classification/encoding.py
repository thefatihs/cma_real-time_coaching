"""Deterministic multi-label encoding shared by training and evaluation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.classification.models import ClassificationTaxonomy

NO_ACTION_LABEL = "no_action"


@dataclass(frozen=True, slots=True)
class MultiLabelEncoder:
    label_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.label_order:
            raise ValueError("label_order cannot be empty")
        if len(self.label_order) != len(set(self.label_order)):
            raise ValueError("label_order must be unique")
        if NO_ACTION_LABEL not in self.label_order:
            raise ValueError("label_order must contain no_action")

    @classmethod
    def from_taxonomy(cls, taxonomy: ClassificationTaxonomy) -> "MultiLabelEncoder":
        return cls(taxonomy.label_ids)

    def encode(self, labels: tuple[str, ...]) -> tuple[int, ...]:
        unknown = set(labels) - set(self.label_order)
        if unknown:
            raise ValueError(f"Unknown classification labels: {sorted(unknown)}")
        _validate_no_action(labels)
        return tuple(int(label in labels) for label in self.label_order)

    def encode_many(
        self, labels: Sequence[tuple[str, ...]]
    ) -> tuple[tuple[int, ...], ...]:
        return tuple(self.encode(item) for item in labels)

    def decode(self, vector: Sequence[int | bool]) -> tuple[str, ...]:
        if len(vector) != len(self.label_order):
            raise ValueError("prediction vector length does not match label order")
        selected = tuple(
            label
            for label, enabled in zip(self.label_order, vector, strict=True)
            if enabled
        )
        return enforce_no_action_exclusivity(selected)

    def threshold_probabilities(
        self,
        probabilities: Sequence[float],
        thresholds: Mapping[str, float],
    ) -> tuple[int, ...]:
        if len(probabilities) != len(self.label_order):
            raise ValueError("probability vector length does not match label order")
        if set(thresholds) != set(self.label_order):
            raise ValueError("threshold labels must exactly match label order")
        predicted = [
            int(probability >= thresholds[label])
            for label, probability in zip(self.label_order, probabilities, strict=True)
        ]
        decoded = enforce_no_action_exclusivity(
            tuple(
                label
                for label, enabled in zip(self.label_order, predicted, strict=True)
                if enabled
            )
        )
        return tuple(int(label in decoded) for label in self.label_order)


def taxonomy_thresholds(taxonomy: ClassificationTaxonomy) -> dict[str, float]:
    return {label.id: label.default_threshold for label in taxonomy.labels}


def enforce_no_action_exclusivity(labels: tuple[str, ...]) -> tuple[str, ...]:
    if NO_ACTION_LABEL in labels and len(labels) > 1:
        return tuple(label for label in labels if label != NO_ACTION_LABEL)
    return labels


def _validate_no_action(labels: tuple[str, ...]) -> None:
    if NO_ACTION_LABEL in labels and len(labels) > 1:
        raise ValueError("no_action cannot appear with another label")
