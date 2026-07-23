"""Validation-only deterministic threshold calibration."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from app.classification.artifacts import TrainingArtifactMetadata
from app.classification.dataset import ClassificationDataset
from app.classification.encoding import MultiLabelEncoder
from app.classification.evaluation import (
    EvaluationMetrics,
    ProbabilityModel,
    evaluate_probabilities,
)
from app.classification.models import ClassificationTaxonomy, DatasetSplit
from app.classification.training import examples_for_split

CRITICAL_LABELS = frozenset({"cancellation_request", "churn_risk", "complaint"})
CRITICAL_RECALL_TARGET = 0.70


@dataclass(frozen=True, slots=True)
class CalibrationConfiguration:
    minimum_threshold: float = 0.30
    maximum_threshold: float = 0.90
    step: float = 0.05
    critical_recall_target: float = CRITICAL_RECALL_TARGET

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_threshold <= 1:
            raise ValueError("minimum_threshold must be between 0 and 1")
        if not 0 <= self.maximum_threshold <= 1:
            raise ValueError("maximum_threshold must be between 0 and 1")
        if self.minimum_threshold > self.maximum_threshold:
            raise ValueError("minimum_threshold cannot exceed maximum_threshold")
        if self.step <= 0:
            raise ValueError("step must be positive")
        if not 0 <= self.critical_recall_target <= 1:
            raise ValueError("critical_recall_target must be between 0 and 1")

    def candidates(self) -> tuple[float, ...]:
        minimum = Decimal(str(self.minimum_threshold))
        maximum = Decimal(str(self.maximum_threshold))
        step = Decimal(str(self.step))
        values: list[float] = []
        current = minimum
        while current <= maximum:
            values.append(float(current))
            current += step
        return tuple(values)


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    label: str
    threshold: float
    metrics: CandidateMetrics
    critical_recall_target_met: bool | None


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    original_thresholds: Mapping[str, float]
    calibrated_thresholds: Mapping[str, float]
    original_metrics: EvaluationMetrics
    calibrated_metrics: EvaluationMetrics
    critical_recall_target_status: Mapping[str, bool]
    no_action_both_absent_before: int
    no_action_both_absent_after: int


def calibrate_probabilities(
    true_vectors: Sequence[Sequence[int]],
    probabilities: Sequence[Sequence[float]],
    encoder: MultiLabelEncoder,
    original_thresholds: Mapping[str, float],
    configuration: CalibrationConfiguration,
) -> CalibrationResult:
    truth = tuple(tuple(int(value) for value in row) for row in true_vectors)
    probability_rows = tuple(
        tuple(float(value) for value in row) for row in probabilities
    )
    if len(truth) != len(probability_rows) or not truth:
        raise ValueError("calibration requires matching non-empty validation rows")
    if set(original_thresholds) != set(encoder.label_order):
        raise ValueError("original thresholds must exactly match label order")

    calibrated: dict[str, float] = {}
    critical_status: dict[str, bool] = {}
    for index, label in enumerate(encoder.label_order):
        selection = select_threshold(
            label=label,
            true_values=tuple(row[index] for row in truth),
            probabilities=tuple(row[index] for row in probability_rows),
            candidates=configuration.candidates(),
            critical=label in CRITICAL_LABELS,
            recall_target=configuration.critical_recall_target,
        )
        calibrated[label] = selection.threshold
        if selection.critical_recall_target_met is not None:
            critical_status[label] = selection.critical_recall_target_met

    original_metrics = evaluate_probabilities(
        truth, probability_rows, encoder, original_thresholds
    )
    calibrated_metrics = evaluate_probabilities(
        truth, probability_rows, encoder, calibrated
    )
    return CalibrationResult(
        original_thresholds=dict(original_thresholds),
        calibrated_thresholds=calibrated,
        original_metrics=original_metrics,
        calibrated_metrics=calibrated_metrics,
        critical_recall_target_status=critical_status,
        no_action_both_absent_before=_count_no_action_both_absent(
            probability_rows, encoder, original_thresholds
        ),
        no_action_both_absent_after=_count_no_action_both_absent(
            probability_rows, encoder, calibrated
        ),
    )


def select_threshold(
    *,
    label: str,
    true_values: Sequence[int],
    probabilities: Sequence[float],
    candidates: Sequence[float],
    critical: bool,
    recall_target: float = CRITICAL_RECALL_TARGET,
) -> ThresholdSelection:
    if len(true_values) != len(probabilities) or not true_values:
        raise ValueError("label calibration requires matching non-empty values")
    if not candidates:
        raise ValueError("at least one threshold candidate is required")
    metrics = tuple(
        _candidate_metrics(true_values, probabilities, threshold)
        for threshold in candidates
    )
    if critical:
        meeting_target = tuple(item for item in metrics if item.recall >= recall_target)
        if meeting_target:
            selected = max(
                meeting_target,
                key=lambda item: (
                    item.f1,
                    item.precision,
                    item.recall,
                    item.threshold,
                ),
            )
            target_met = True
        else:
            selected = max(
                metrics,
                key=lambda item: (
                    item.recall,
                    item.precision,
                    item.f1,
                    item.threshold,
                ),
            )
            target_met = False
    else:
        selected = max(
            metrics,
            key=lambda item: (
                item.f1,
                item.precision,
                item.recall,
                item.threshold,
            ),
        )
        target_met = None
    return ThresholdSelection(label, selected.threshold, selected, target_met)


def calibrate_validation_model(
    model: ProbabilityModel,
    dataset: ClassificationDataset,
    taxonomy: ClassificationTaxonomy,
    configuration: CalibrationConfiguration,
) -> CalibrationResult:
    examples = examples_for_split(dataset, DatasetSplit.VALIDATION)
    if not examples:
        raise ValueError("validation split cannot be empty")
    encoder = MultiLabelEncoder.from_taxonomy(taxonomy)
    probabilities = _to_probability_rows(
        model.predict_proba([example.text for example in examples])
    )
    truth = encoder.encode_many(tuple(example.labels for example in examples))
    original_thresholds = {
        label.id: label.default_threshold for label in taxonomy.labels
    }
    return calibrate_probabilities(
        truth,
        probabilities,
        encoder,
        original_thresholds,
        configuration,
    )


def save_calibration_report(
    path: str | Path,
    *,
    metadata: TrainingArtifactMetadata,
    result: CalibrationResult,
    configuration: CalibrationConfiguration,
    model_checksum: str,
    taxonomy_checksum: str,
    dataset_checksum: str,
) -> None:
    report = {
        "model_id": metadata.model_id,
        "split": "validation",
        "original_thresholds": dict(result.original_thresholds),
        "calibrated_thresholds": dict(result.calibrated_thresholds),
        "per_label_original_metrics": _per_label_metrics(result.original_metrics),
        "per_label_calibrated_metrics": _per_label_metrics(result.calibrated_metrics),
        "micro_macro_before": _micro_macro(result.original_metrics),
        "micro_macro_after": _micro_macro(result.calibrated_metrics),
        "exact_match_ratio_before": result.original_metrics.exact_match_ratio,
        "exact_match_ratio_after": result.calibrated_metrics.exact_match_ratio,
        "hamming_loss_before": result.original_metrics.hamming_loss,
        "hamming_loss_after": result.calibrated_metrics.hamming_loss,
        "critical_label_recall_target_status": dict(
            result.critical_recall_target_status
        ),
        "model_checksum": model_checksum,
        "taxonomy_checksum": taxonomy_checksum,
        "dataset_checksum": dataset_checksum,
        "calibration_configuration": {
            "minimum_threshold": configuration.minimum_threshold,
            "maximum_threshold": configuration.maximum_threshold,
            "step": configuration.step,
            "critical_recall_target": configuration.critical_recall_target,
            "no_action_both_absent_before": (result.no_action_both_absent_before),
            "no_action_both_absent_after": result.no_action_both_absent_after,
        },
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_directory(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise ValueError("model directory does not exist")
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError("model directory contains no files")
    for item in files:
        relative = item.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as source:
            for block in iter(lambda: source.read(65536), b""):
                digest.update(block)
    return digest.hexdigest()


def _candidate_metrics(
    true_values: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> CandidateMetrics:
    if not 0 <= threshold <= 1:
        raise ValueError("candidate thresholds must be between 0 and 1")
    predicted = tuple(int(value >= threshold) for value in probabilities)
    tp = sum(
        actual == 1 and guess == 1
        for actual, guess in zip(true_values, predicted, strict=True)
    )
    fp = sum(
        actual == 0 and guess == 1
        for actual, guess in zip(true_values, predicted, strict=True)
    )
    fn = sum(
        actual == 1 and guess == 0
        for actual, guess in zip(true_values, predicted, strict=True)
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return CandidateMetrics(threshold, precision, recall, f1, tp, fp, fn)


def _count_no_action_both_absent(
    probabilities: Sequence[Sequence[float]],
    encoder: MultiLabelEncoder,
    thresholds: Mapping[str, float],
) -> int:
    no_action_index = encoder.label_order.index("no_action")
    business_indexes = tuple(
        index for index, label in enumerate(encoder.label_order) if label != "no_action"
    )
    return sum(
        row[no_action_index] < thresholds["no_action"]
        and not any(
            row[index] >= thresholds[encoder.label_order[index]]
            for index in business_indexes
        )
        for row in probabilities
    )


def _per_label_metrics(metrics: EvaluationMetrics) -> dict[str, object]:
    return {label: values.as_dict() for label, values in metrics.per_label.items()}


def _micro_macro(metrics: EvaluationMetrics) -> dict[str, dict[str, float]]:
    return {
        "micro": {
            "precision": metrics.micro_precision,
            "recall": metrics.micro_recall,
            "f1": metrics.micro_f1,
        },
        "macro": {
            "precision": metrics.macro_precision,
            "recall": metrics.macro_recall,
            "f1": metrics.macro_f1,
        },
    }


def _to_probability_rows(values: object) -> tuple[tuple[float, ...], ...]:
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
