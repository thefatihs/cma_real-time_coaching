"""Safe multi-label evaluation without retaining source texts."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from app.classification.encoding import MultiLabelEncoder
from app.classification.models import ClassificationExample


class ProbabilityModel(Protocol):
    def predict_proba(self, inputs: list[str]) -> object: ...


@dataclass(frozen=True, slots=True)
class LabelMetrics:
    precision: float
    recall: float
    f1: float
    support: int
    minimum_probability: float
    mean_probability: float
    maximum_probability: float
    predictions_above_threshold: int
    true_positives: int
    false_positives: int
    false_negatives: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
            "minimum_probability": self.minimum_probability,
            "mean_probability": self.mean_probability,
            "maximum_probability": self.maximum_probability,
            "predictions_above_threshold": self.predictions_above_threshold,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass(frozen=True, slots=True)
class NoActionDiagnostics:
    no_business_label_above_threshold: int
    no_action_above_threshold: int
    pre_exclusivity_conflicts: int

    def as_dict(self) -> dict[str, int]:
        return {
            "no_business_label_above_threshold": (
                self.no_business_label_above_threshold
            ),
            "no_action_above_threshold": self.no_action_above_threshold,
            "pre_exclusivity_conflicts": self.pre_exclusivity_conflicts,
        }


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_label: Mapping[str, LabelMetrics]
    exact_match_ratio: float
    hamming_loss: float
    average_inference_time_ms: float
    example_count: int
    no_predicted_labels: int
    no_action_diagnostics: NoActionDiagnostics

    def as_dict(self) -> dict[str, object]:
        return {
            "micro": {
                "precision": self.micro_precision,
                "recall": self.micro_recall,
                "f1": self.micro_f1,
            },
            "macro": {
                "precision": self.macro_precision,
                "recall": self.macro_recall,
                "f1": self.macro_f1,
            },
            "per_label": {
                label: metrics.as_dict() for label, metrics in self.per_label.items()
            },
            "exact_match_ratio": self.exact_match_ratio,
            "hamming_loss": self.hamming_loss,
            "average_inference_time_ms": self.average_inference_time_ms,
            "example_count": self.example_count,
            "no_predicted_labels": self.no_predicted_labels,
            "no_action_diagnostics": self.no_action_diagnostics.as_dict(),
        }


def evaluate_probabilities(
    true_vectors: Sequence[Sequence[int]],
    probabilities: Sequence[Sequence[float]],
    encoder: MultiLabelEncoder,
    thresholds: Mapping[str, float],
    *,
    elapsed_seconds: float = 0.0,
) -> EvaluationMetrics:
    if len(true_vectors) != len(probabilities):
        raise ValueError("true and probability row counts must match")
    if not true_vectors:
        raise ValueError("evaluation requires at least one example")
    probability_rows = tuple(
        _validate_probability_vector(row, len(encoder.label_order))
        for row in probabilities
    )
    raw_predictions = tuple(
        tuple(
            int(probability >= thresholds[label])
            for label, probability in zip(encoder.label_order, row, strict=True)
        )
        for row in probability_rows
    )
    predicted = tuple(
        encoder.threshold_probabilities(row, thresholds) for row in probability_rows
    )
    truth = tuple(
        _validate_binary_vector(row, len(encoder.label_order)) for row in true_vectors
    )

    per_label: dict[str, LabelMetrics] = {}
    total_tp = total_fp = total_fn = 0
    for index, label in enumerate(encoder.label_order):
        tp = sum(
            row[index] == 1 and guess[index] == 1
            for row, guess in zip(truth, predicted, strict=True)
        )
        fp = sum(
            row[index] == 0 and guess[index] == 1
            for row, guess in zip(truth, predicted, strict=True)
        )
        fn = sum(
            row[index] == 1 and guess[index] == 0
            for row, guess in zip(truth, predicted, strict=True)
        )
        support = sum(row[index] for row in truth)
        label_probabilities = tuple(row[index] for row in probability_rows)
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        per_label[label] = LabelMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
            minimum_probability=min(label_probabilities),
            mean_probability=sum(label_probabilities) / len(label_probabilities),
            maximum_probability=max(label_probabilities),
            predictions_above_threshold=sum(row[index] for row in raw_predictions),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro = _precision_recall_f1(total_tp, total_fp, total_fn)
    count = len(truth)
    label_count = len(encoder.label_order)
    exact_matches = sum(
        actual == guess for actual, guess in zip(truth, predicted, strict=True)
    )
    differences = sum(
        actual != guess
        for row, predicted_row in zip(truth, predicted, strict=True)
        for actual, guess in zip(row, predicted_row, strict=True)
    )
    no_action_index = encoder.label_order.index("no_action")
    business_indexes = tuple(
        index for index, label in enumerate(encoder.label_order) if label != "no_action"
    )
    return EvaluationMetrics(
        micro_precision=micro[0],
        micro_recall=micro[1],
        micro_f1=micro[2],
        macro_precision=sum(item.precision for item in per_label.values())
        / label_count,
        macro_recall=sum(item.recall for item in per_label.values()) / label_count,
        macro_f1=sum(item.f1 for item in per_label.values()) / label_count,
        per_label=per_label,
        exact_match_ratio=exact_matches / count,
        hamming_loss=differences / (count * label_count),
        average_inference_time_ms=(elapsed_seconds * 1000) / count,
        example_count=count,
        no_predicted_labels=sum(not any(row) for row in predicted),
        no_action_diagnostics=NoActionDiagnostics(
            no_business_label_above_threshold=sum(
                not any(row[index] for index in business_indexes)
                for row in raw_predictions
            ),
            no_action_above_threshold=sum(
                row[no_action_index] for row in raw_predictions
            ),
            pre_exclusivity_conflicts=sum(
                row[no_action_index] and any(row[index] for index in business_indexes)
                for row in raw_predictions
            ),
        ),
    )


def evaluate_model(
    model: ProbabilityModel,
    examples: tuple[ClassificationExample, ...],
    encoder: MultiLabelEncoder,
    thresholds: Mapping[str, float],
    *,
    timer: Callable[[], float] = perf_counter,
) -> EvaluationMetrics:
    if not examples:
        raise ValueError("evaluation split cannot be empty")
    texts = [example.text for example in examples]
    start = timer()
    raw_probabilities = model.predict_proba(texts)
    elapsed = timer() - start
    probabilities = _to_float_rows(raw_probabilities)
    truth = encoder.encode_many(tuple(example.labels for example in examples))
    return evaluate_probabilities(
        truth,
        probabilities,
        encoder,
        thresholds,
        elapsed_seconds=elapsed,
    )


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _validate_binary_vector(
    values: Sequence[int], expected_length: int
) -> tuple[int, ...]:
    vector = tuple(values)
    if len(vector) != expected_length or any(value not in {0, 1} for value in vector):
        raise ValueError("true vectors must be binary and match label order")
    return vector


def _validate_probability_vector(
    values: Sequence[float], expected_length: int
) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if len(vector) != expected_length:
        raise ValueError("probability vectors must match label order")
    if any(value < 0 or value > 1 for value in vector):
        raise ValueError("probabilities must be between 0 and 1")
    return vector


def _to_float_rows(values: object) -> tuple[tuple[float, ...], ...]:
    to_list = getattr(values, "tolist", None)
    if callable(to_list):
        values = to_list()
    if not isinstance(values, Sequence):
        raise ValueError("model probabilities must be a two-dimensional sequence")
    rows: list[tuple[float, ...]] = []
    for row in values:
        row_to_list = getattr(row, "tolist", None)
        if callable(row_to_list):
            row = row_to_list()
        if not isinstance(row, Sequence):
            raise ValueError("model probabilities must be a two-dimensional sequence")
        rows.append(tuple(float(value) for value in row))
    return tuple(rows)
