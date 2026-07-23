from collections.abc import Callable
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pytest

from app.classification.artifacts import (
    TrainingArtifactMetadata,
    save_evaluation_report,
)
from app.classification.dataset import load_classification_taxonomy
from app.classification.encoding import MultiLabelEncoder
from app.classification.evaluation import evaluate_probabilities
from app.classification.threshold_profiles import (
    ThresholdProfile,
    load_threshold_profile,
    resolve_evaluation_thresholds,
)

PROFILE_PATH = Path("config/classification_thresholds/common_turkish_setfit_v2.json")
MODEL_CHECKSUM = "78b82e686046308cb428c77a852761b66ef6474751af1df301e367f9357758bb"
DATASET_CHECKSUM = "f045635e83bb3d724fc9a6d2969d1600736bdd3342095e6a4eae2c470375db35"
TAXONOMY_CHECKSUM = "2afc2d711215a287696ac7c1fe90140256e5483f49109c00a0d8e2925088794a"


def metadata(**changes: object) -> TrainingArtifactMetadata:
    taxonomy = load_classification_taxonomy("config/classification_taxonomy.json")
    values: dict[str, object] = {
        "model_id": "common_turkish_setfit_v2",
        "backbone": "synthetic-backbone",
        "label_order": taxonomy.label_ids,
        "taxonomy_checksum": TAXONOMY_CHECKSUM,
        "dataset_checksum": DATASET_CHECKSUM,
        "training_parameters": {"seed": 42},
        "training_timestamp": datetime(2026, 7, 23, tzinfo=UTC),
        "split_counts": {"validation": 54},
        "package_versions": {"setfit": "1.1.3"},
    }
    values.update(changes)
    return TrainingArtifactMetadata.model_validate(values)


def load_profile(
    path: Path = PROFILE_PATH,
    *,
    metadata_override: TrainingArtifactMetadata | None = None,
    dataset_checksum: str = DATASET_CHECKSUM,
    taxonomy_checksum: str = TAXONOMY_CHECKSUM,
    model_checksum: str = MODEL_CHECKSUM,
) -> ThresholdProfile:
    return load_threshold_profile(
        path,
        taxonomy=load_classification_taxonomy("config/classification_taxonomy.json"),
        metadata=metadata_override or metadata(),
        dataset_checksum=dataset_checksum,
        taxonomy_checksum=taxonomy_checksum,
        model_checksum=model_checksum,
    )


def write_profile(tmp_path: Path, mutate: Callable[[dict[str, Any]], object]) -> Path:
    payload = json.loads(PROFILE_PATH.read_text("utf-8"))
    mutate(payload)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_versioned_profile_loading() -> None:
    profile = load_profile()
    assert profile.model_id == "common_turkish_setfit_v2"
    assert profile.source_split == "validation"
    assert profile.profile_id == "common_turkish_setfit_v2:calibrated:v1"
    assert profile.calibrated_thresholds["product_information"] == 0.90
    assert profile.critical_recall_target == 0.70


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["calibrated_thresholds"].pop("complaint"),
        lambda payload: payload["calibrated_thresholds"].update({"extra": 0.5}),
    ],
)
def test_missing_and_extra_labels_are_rejected(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], object]
) -> None:
    with pytest.raises(ValueError, match="labels are incompatible"):
        load_profile(write_profile(tmp_path, mutate))


def test_invalid_threshold_range_is_rejected(tmp_path: Path) -> None:
    path = write_profile(
        tmp_path,
        lambda payload: payload["calibrated_thresholds"].update({"complaint": 1.1}),
    )
    with pytest.raises(ValueError, match="Invalid threshold profile"):
        load_profile(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "other", "model_id"),
        ("dataset_checksum", "b" * 64, "dataset checksum"),
        ("taxonomy_checksum", "b" * 64, "taxonomy checksum"),
        ("model_checksum", "b" * 64, "model checksum"),
    ],
)
def test_profile_compatibility_mismatches_are_rejected(
    field: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_profile(
            metadata_override=metadata(model_id=value) if field == "model_id" else None,
            dataset_checksum=value if field == "dataset_checksum" else DATASET_CHECKSUM,
            taxonomy_checksum=value
            if field == "taxonomy_checksum"
            else TAXONOMY_CHECKSUM,
            model_checksum=value if field == "model_checksum" else MODEL_CHECKSUM,
        )


def test_evaluator_uses_profile_and_falls_back_to_taxonomy_thresholds() -> None:
    taxonomy = load_classification_taxonomy("config/classification_taxonomy.json")
    profile_resolution = resolve_evaluation_thresholds(taxonomy, load_profile())
    default_resolution = resolve_evaluation_thresholds(taxonomy)
    assert profile_resolution.threshold_source == "threshold_profile"
    assert profile_resolution.thresholds["churn_risk"] == 0.35
    assert profile_resolution.threshold_profile_id is not None
    assert default_resolution.threshold_source == "taxonomy_defaults"
    assert default_resolution.thresholds["churn_risk"] == 0.72
    assert default_resolution.threshold_profile_id is None


def test_profile_thresholds_preserve_no_action_exclusivity() -> None:
    taxonomy = load_classification_taxonomy("config/classification_taxonomy.json")
    encoder = MultiLabelEncoder.from_taxonomy(taxonomy)
    thresholds = resolve_evaluation_thresholds(taxonomy, load_profile()).thresholds
    probabilities = [[0.95, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]]
    metrics = evaluate_probabilities(
        [[1, 0, 0, 0, 0, 0, 0, 0]],
        probabilities,
        encoder,
        thresholds,
    )
    assert metrics.per_label["product_information"].true_positives == 1
    assert metrics.per_label["no_action"].false_positives == 0
    assert metrics.no_action_diagnostics.pre_exclusivity_conflicts == 1


def test_profile_report_has_safe_provenance_without_predictions(
    tmp_path: Path,
) -> None:
    taxonomy = load_classification_taxonomy("config/classification_taxonomy.json")
    profile = load_profile()
    resolution = resolve_evaluation_thresholds(taxonomy, profile)
    encoder = MultiLabelEncoder.from_taxonomy(taxonomy)
    metrics = evaluate_probabilities(
        [[0, 0, 0, 0, 0, 0, 0, 1]],
        [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]],
        encoder,
        resolution.thresholds,
    )
    output = tmp_path / "report.json"
    save_evaluation_report(
        output,
        metadata=metadata(),
        split="validation",
        thresholds=dict(resolution.thresholds),
        metrics=metrics,
        threshold_source=resolution.threshold_source,
        threshold_profile_id=resolution.threshold_profile_id,
    )
    content = output.read_text("utf-8")
    report = json.loads(content)
    assert report["threshold_source"] == "threshold_profile"
    assert report["threshold_profile_id"] == profile.profile_id
    assert "individual_predictions" not in content
    assert "texts" not in content
    assert "TEST_SENTINEL" not in content
