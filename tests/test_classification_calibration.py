from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from app.classification.artifacts import MODEL_ID, TrainingArtifactMetadata
from app.classification.calibration import (
    CalibrationConfiguration,
    calibrate_probabilities,
    calibrate_validation_model,
    save_calibration_report,
    select_threshold,
)
from app.classification.dataset import (
    ClassificationDataset,
    load_classification_taxonomy,
)
from app.classification.encoding import MultiLabelEncoder
from app.classification.models import ClassificationExample, DatasetSplit

CHECKSUM = "a" * 64


def test_best_f1_and_deterministic_tie_select_higher_threshold() -> None:
    selection = select_threshold(
        label="product_information",
        true_values=(1, 1, 0, 0),
        probabilities=(0.9, 0.6, 0.55, 0.1),
        candidates=(0.3, 0.5, 0.7),
        critical=False,
    )
    assert selection.threshold == 0.5
    assert selection.metrics.f1 == 0.8


def test_critical_label_prefers_recall_target_over_higher_f1() -> None:
    selection = select_threshold(
        label="complaint",
        true_values=(1, 1, 1, 0, 0, 0),
        probabilities=(0.9, 0.85, 0.4, 0.6, 0.55, 0.5),
        candidates=(0.3, 0.8),
        critical=True,
        recall_target=0.70,
    )
    assert selection.threshold == 0.3
    assert selection.critical_recall_target_met is True
    assert selection.metrics.recall == 1.0


def test_critical_fallback_uses_recall_then_precision() -> None:
    selection = select_threshold(
        label="churn_risk",
        true_values=(1, 1, 0),
        probabilities=(0.9, 0.2, 0.5),
        candidates=(0.3, 0.8),
        critical=True,
    )
    assert selection.critical_recall_target_met is False
    assert selection.threshold == 0.8
    assert selection.metrics.recall == 0.5
    assert selection.metrics.precision == 1.0


def test_threshold_configuration_bounds_and_decimal_steps() -> None:
    configuration = CalibrationConfiguration(0.30, 0.40, 0.05)
    assert configuration.candidates() == (0.3, 0.35, 0.4)
    with pytest.raises(ValueError, match="cannot exceed"):
        CalibrationConfiguration(0.9, 0.3, 0.05)
    with pytest.raises(ValueError, match="positive"):
        CalibrationConfiguration(0.3, 0.9, 0.0)
    with pytest.raises(ValueError, match="between"):
        select_threshold(
            label="x",
            true_values=(1,),
            probabilities=(0.5,),
            candidates=(1.1,),
            critical=False,
        )


def test_calibration_preserves_no_action_exclusivity_and_improves_metrics() -> None:
    encoder = MultiLabelEncoder(("product_information", "no_action"))
    result = calibrate_probabilities(
        ((1, 0), (0, 1)),
        ((0.8, 0.8), (0.2, 0.8)),
        encoder,
        {"product_information": 0.9, "no_action": 0.9},
        CalibrationConfiguration(0.5, 0.5, 0.05),
    )
    assert result.calibrated_metrics.micro_f1 > result.original_metrics.micro_f1
    assert result.calibrated_metrics.per_label["no_action"].false_positives == 0
    assert result.calibrated_metrics.exact_match_ratio == 1.0
    assert result.no_action_both_absent_before == 2
    assert result.no_action_both_absent_after == 0


def test_report_preserves_metadata_model_id_and_contains_no_examples(
    tmp_path: Path,
) -> None:
    encoder = MultiLabelEncoder(("product_information", "no_action"))
    result = calibrate_probabilities(
        ((1, 0),),
        ((0.8, 0.1),),
        encoder,
        {"product_information": 0.7, "no_action": 0.7},
        CalibrationConfiguration(0.5, 0.5, 0.05),
    )
    metadata = TrainingArtifactMetadata(
        model_id="preserved_model_identity",
        backbone="synthetic-backbone",
        label_order=encoder.label_order,
        taxonomy_checksum=CHECKSUM,
        dataset_checksum=CHECKSUM,
        training_parameters={"seed": 42},
        training_timestamp=datetime(2026, 7, 23, tzinfo=UTC),
        split_counts={"validation": 1},
        package_versions={"setfit": "1.1.3"},
    )
    output = tmp_path / "calibration.json"
    save_calibration_report(
        output,
        metadata=metadata,
        result=result,
        configuration=CalibrationConfiguration(0.5, 0.5, 0.05),
        model_checksum=CHECKSUM,
        taxonomy_checksum=CHECKSUM,
        dataset_checksum=CHECKSUM,
    )
    content = output.read_text("utf-8")
    report = json.loads(content)
    assert MODEL_ID == "common_turkish_setfit_v2"
    assert report["model_id"] == "preserved_model_identity"
    assert report["split"] == "validation"
    assert "SENTETİK GİZLİ METİN" not in content
    assert "individual_predictions" not in content
    assert "texts" not in content
    assert set(report) == {
        "model_id",
        "split",
        "original_thresholds",
        "calibrated_thresholds",
        "per_label_original_metrics",
        "per_label_calibrated_metrics",
        "micro_macro_before",
        "micro_macro_after",
        "exact_match_ratio_before",
        "exact_match_ratio_after",
        "hamming_loss_before",
        "hamming_loss_after",
        "critical_label_recall_target_status",
        "model_checksum",
        "taxonomy_checksum",
        "dataset_checksum",
        "calibration_configuration",
    }


def test_validation_model_never_sends_train_or_test_text_to_model() -> None:
    taxonomy = load_classification_taxonomy("config/classification_taxonomy.json")
    examples = (
        ClassificationExample(
            example_id="train",
            text="TRAIN_SENTINEL",
            labels=("no_action",),
            split=DatasetSplit.TRAIN,
        ),
        ClassificationExample(
            example_id="validation",
            text="VALIDATION_SENTINEL",
            labels=("no_action",),
            split=DatasetSplit.VALIDATION,
        ),
        ClassificationExample(
            example_id="test",
            text="TEST_SENTINEL",
            labels=("no_action",),
            split=DatasetSplit.TEST,
        ),
    )
    dataset = ClassificationDataset(
        examples=examples,
        label_counts={},
        split_counts={"train": 1, "validation": 1, "test": 1},
        label_split_counts={},
    )

    class FakeModel:
        def predict_proba(self, inputs: list[str]) -> list[list[float]]:
            assert inputs == ["VALIDATION_SENTINEL"]
            return [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]]

    result = calibrate_validation_model(
        FakeModel(),
        dataset,
        taxonomy,
        CalibrationConfiguration(0.5, 0.5, 0.05),
    )
    assert result.calibrated_metrics.example_count == 1
