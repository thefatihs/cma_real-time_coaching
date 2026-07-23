from datetime import UTC, datetime
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.classification.artifacts import (
    MODEL_ID,
    TrainingArtifactMetadata,
    load_training_metadata,
    save_evaluation_report,
    save_training_artifacts,
)
from app.classification.dataset import (
    load_classification_dataset,
    load_classification_taxonomy,
)
from app.classification.encoding import MultiLabelEncoder
from app.classification.evaluation import evaluate_probabilities
from app.classification.training import TrainingParameters, train_setfit_baseline

CHECKSUM = "a" * 64


def metadata(**changes: object) -> TrainingArtifactMetadata:
    values: dict[str, object] = {
        "model_id": MODEL_ID,
        "backbone": "synthetic-backbone",
        "label_order": ("a", "no_action"),
        "taxonomy_checksum": CHECKSUM,
        "dataset_checksum": CHECKSUM,
        "training_parameters": {"seed": 42},
        "training_timestamp": datetime(2026, 7, 23, tzinfo=UTC),
        "split_counts": {"train": 1, "validation": 1, "test": 1},
        "package_versions": {"setfit": "1.1.3"},
    }
    values.update(changes)
    return TrainingArtifactMetadata.model_validate(values)


def test_metadata_round_trip_and_validation(tmp_path: Path) -> None:
    save_training_artifacts(tmp_path, metadata())
    assert load_training_metadata(tmp_path) == metadata()
    with pytest.raises(ValidationError):
        metadata(taxonomy_checksum="short")
    with pytest.raises(ValidationError):
        metadata(label_order=("a", "a"))


def test_evaluation_report_contains_no_transcript_text(tmp_path: Path) -> None:
    secret = "BU SENTETİK TRANSKRİPT RAPORDA OLMAMALI"
    path = tmp_path / "report.json"
    metrics = evaluate_probabilities(
        ((1, 0),),
        ((0.9, 0.1),),
        MultiLabelEncoder(("a", "no_action")),
        {"a": 0.5, "no_action": 0.7},
    )
    save_evaluation_report(
        path,
        metadata=metadata(),
        split="test",
        thresholds={"a": 0.5, "no_action": 0.7},
        metrics=metrics,
    )
    content = path.read_text("utf-8")
    assert secret not in content
    report = json.loads(content)
    assert set(report) == {
        "model_id",
        "backbone",
        "label_order",
        "taxonomy_checksum",
        "dataset_checksum",
        "split",
        "thresholds",
        "metrics",
    }


def test_training_factory_never_receives_test_split(tmp_path: Path) -> None:
    taxonomy_path = Path("config/classification_taxonomy.json")
    dataset_path = Path("data/synthetic/classification_seed.jsonl")
    taxonomy = load_classification_taxonomy(taxonomy_path)
    dataset = load_classification_dataset(dataset_path, taxonomy)
    captured: dict[str, object] = {}

    class FakeModel:
        def save_pretrained(self, save_directory: str | Path) -> None:
            Path(save_directory).mkdir(parents=True, exist_ok=True)

    class FakeTrainer:
        def train(self) -> None:
            captured["trained"] = True

    def factory(
        backbone: str,
        label_order: tuple[str, ...],
        train_payload: dict[str, list[object]],
        validation_payload: dict[str, list[object]],
        parameters: TrainingParameters,
        output_dir: Path,
    ) -> tuple[FakeModel, FakeTrainer]:
        assert output_dir == tmp_path / "model"
        captured["train_count"] = len(train_payload["text"])
        captured["validation_count"] = len(validation_payload["text"])
        captured["all_text"] = tuple(train_payload["text"] + validation_payload["text"])
        return FakeModel(), FakeTrainer()

    train_setfit_baseline(
        dataset=dataset,
        encoder=MultiLabelEncoder.from_taxonomy(taxonomy),
        taxonomy_path=taxonomy_path,
        dataset_path=dataset_path,
        output_dir=tmp_path / "model",
        backbone="fake",
        parameters=TrainingParameters(),
        components_factory=factory,
        timestamp_factory=lambda: datetime(2026, 7, 23, tzinfo=UTC),
    )
    test_texts = {
        example.text for example in dataset.examples if example.split.value == "test"
    }
    assert captured["train_count"] == dataset.split_counts["train"]
    assert captured["validation_count"] == dataset.split_counts["validation"]
    assert not test_texts.intersection(captured["all_text"])  # type: ignore[arg-type]
    assert captured["trained"] is True
