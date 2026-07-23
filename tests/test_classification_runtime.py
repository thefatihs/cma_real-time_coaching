from datetime import UTC, datetime
import json
import logging
from pathlib import Path

import pytest

from app.classification.artifacts import (
    MODEL_ID,
    TrainingArtifactMetadata,
    save_training_artifacts,
    sha256_file,
)
from app.classification.calibration import sha256_directory
from app.classification.dataset import load_classification_taxonomy
from app.classification.runtime import (
    RuntimeArtifactPaths,
    RuntimeClassifierConfig,
    RuntimeSetFitClassifier,
)
from app.events.models import ClassificationResultEvent, CoachingAction

DATASET_CHECKSUM = "f" * 64
NOW = datetime(2026, 7, 23, tzinfo=UTC)


class FakeModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.inputs: list[list[str]] = []

    def predict_proba(self, inputs: list[str]) -> list[list[float]]:
        self.inputs.append(inputs)
        return [self.probabilities]


def artifacts(
    tmp_path: Path,
    *,
    profile_model_id: str = MODEL_ID,
    profile_model_checksum: str | None = None,
) -> RuntimeArtifactPaths:
    taxonomy_path = Path("config/classification_taxonomy.json")
    taxonomy = load_classification_taxonomy(taxonomy_path)
    model_dir = tmp_path / "model"
    metadata = TrainingArtifactMetadata(
        model_id=MODEL_ID,
        backbone="synthetic-backbone",
        label_order=taxonomy.label_ids,
        taxonomy_checksum=sha256_file(taxonomy_path),
        dataset_checksum=DATASET_CHECKSUM,
        training_parameters={"seed": 42},
        training_timestamp=NOW,
        split_counts={"train": 1},
        package_versions={"setfit": "fake"},
    )
    save_training_artifacts(model_dir, metadata)
    (model_dir / "synthetic-weights.bin").write_bytes(b"synthetic")
    model_checksum = profile_model_checksum or sha256_directory(model_dir)
    profile_path = tmp_path / "thresholds.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": profile_model_id,
                "source_split": "validation",
                "calibrated_thresholds": {
                    "product_information": 0.90,
                    "price_objection": 0.35,
                    "cancellation_request": 0.70,
                    "technical_issue": 0.75,
                    "complaint": 0.45,
                    "renewal_interest": 0.65,
                    "churn_risk": 0.35,
                    "no_action": 0.30,
                },
                "model_checksum": model_checksum,
                "dataset_checksum": DATASET_CHECKSUM,
                "taxonomy_checksum": sha256_file(taxonomy_path),
                "critical_recall_target": 0.70,
            }
        ),
        encoding="utf-8",
    )
    return RuntimeArtifactPaths(
        model_dir=model_dir,
        threshold_profile_path=profile_path,
        taxonomy_path=taxonomy_path,
    )


def classifier(
    paths: RuntimeArtifactPaths,
    model: FakeModel,
    load_calls: list[Path],
    *,
    logger: logging.Logger | None = None,
) -> RuntimeSetFitClassifier:
    moments = iter((10.0, 10.012, 20.0, 20.008, 30.0, 30.004))

    def loader(model_dir: Path) -> FakeModel:
        load_calls.append(model_dir)
        return model

    return RuntimeSetFitClassifier(
        RuntimeClassifierConfig(default_artifacts=paths),
        model_loader=loader,
        logger=logger,
        timer=lambda: next(moments),
        utc_datetime_factory=lambda: NOW,
    )


def test_model_is_loaded_lazily_and_reused_between_calls(tmp_path: Path) -> None:
    paths = artifacts(tmp_path)
    model = FakeModel([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8])
    load_calls: list[Path] = []
    subject = classifier(paths, model, load_calls)
    assert load_calls == []
    subject.classify(tenant_id="tenant-a", call_id="call-1", text="Sentetik bir")
    subject.classify(tenant_id="tenant-a", call_id="call-2", text="Sentetik iki")
    assert load_calls == [paths.model_dir]
    assert len(model.inputs) == 2


def test_thresholds_multi_label_event_and_identity_are_preserved(
    tmp_path: Path,
) -> None:
    paths = artifacts(tmp_path)
    model = FakeModel([0.95, 0.40, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
    result = classifier(paths, model, []).classify(
        tenant_id="tenant-alpha",
        call_id="call-42",
        text="Sentetik sınıflandırma metni",
        transcript_event_id="transcript-7",
        revision=7,
    )
    assert isinstance(result, ClassificationResultEvent)
    assert (result.tenant_id, result.call_id, result.transcript_event_id) == (
        "tenant-alpha",
        "call-42",
        "transcript-7",
    )
    assert [label.name for label in result.labels] == [
        "product_information",
        "price_objection",
    ]
    assert result.action is CoachingAction.TEMPLATE_ACTION
    assert result.model_id == MODEL_ID
    assert result.threshold_profile_id == f"{MODEL_ID}:calibrated:v1"
    assert result.probabilities["product_information"] == 0.95
    assert result.thresholds["product_information"] == 0.90
    assert result.processing_time_ms == pytest.approx(12.0)
    assert result.created_at_utc == NOW


def test_no_action_is_only_emitted_from_its_profile_threshold(
    tmp_path: Path,
) -> None:
    paths = artifacts(tmp_path)
    model = FakeModel([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
    result = classifier(paths, model, []).classify(
        tenant_id="tenant-a",
        call_id="call-a",
        text="Sıradan sentetik konuşma",
        sequence_number=12,
    )
    assert [label.name for label in result.labels] == ["no_action"]
    assert result.action is CoachingAction.NO_ACTION
    assert result.transcript_event_id == "call-a:sequence:12"


def test_no_action_cannot_coexist_with_business_label(tmp_path: Path) -> None:
    paths = artifacts(tmp_path)
    model = FakeModel([0.95, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.95])
    result = classifier(paths, model, []).classify(
        tenant_id="tenant-a",
        call_id="call-a",
        text="Sentetik bilgi sorusu",
    )
    assert [label.name for label in result.labels] == ["product_information"]


def test_empty_transcript_is_rejected_before_model_loading(tmp_path: Path) -> None:
    paths = artifacts(tmp_path)
    load_calls: list[Path] = []
    subject = classifier(paths, FakeModel([0.1] * 8), load_calls)
    with pytest.raises(ValueError, match="text cannot be empty"):
        subject.classify(tenant_id="tenant-a", call_id="call-a", text="  ")
    assert load_calls == []


def test_incompatible_profile_fails_before_model_loading(tmp_path: Path) -> None:
    paths = artifacts(tmp_path, profile_model_id="other-model")
    load_calls: list[Path] = []
    subject = classifier(paths, FakeModel([0.1] * 8), load_calls)
    with pytest.raises(ValueError, match="model_id"):
        subject.classify(
            tenant_id="tenant-a",
            call_id="call-a",
            text="Sentetik metin",
        )
    assert load_calls == []


def test_tenant_specific_artifacts_are_selected_and_cached(tmp_path: Path) -> None:
    default_paths = artifacts(tmp_path / "default")
    tenant_paths = artifacts(tmp_path / "tenant")
    models = {
        default_paths.model_dir: FakeModel([0.1] * 7 + [0.8]),
        tenant_paths.model_dir: FakeModel([0.95] + [0.1] * 7),
    }
    load_calls: list[Path] = []

    def loader(model_dir: Path) -> FakeModel:
        load_calls.append(model_dir)
        return models[model_dir]

    subject = RuntimeSetFitClassifier(
        RuntimeClassifierConfig(
            default_artifacts=default_paths,
            tenant_artifacts={"tenant-special": tenant_paths},
        ),
        model_loader=loader,
        timer=iter((1.0, 1.1, 2.0, 2.1)).__next__,
        utc_datetime_factory=lambda: NOW,
    )
    special = subject.classify(
        tenant_id="tenant-special",
        call_id="call-1",
        text="Sentetik özel",
    )
    default = subject.classify(
        tenant_id="tenant-other",
        call_id="call-2",
        text="Sentetik genel",
    )
    assert [label.name for label in special.labels] == ["product_information"]
    assert [label.name for label in default.labels] == ["no_action"]
    assert load_calls == [tenant_paths.model_dir, default_paths.model_dir]


def test_safe_logging_never_contains_transcript_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "BU_SENTETİK_METİN_LOGA_GİRMEMELİ"
    logger = logging.getLogger("classification-runtime-test")
    caplog.set_level(logging.INFO, logger=logger.name)
    subject = classifier(
        artifacts(tmp_path),
        FakeModel([0.1] * 7 + [0.8]),
        [],
        logger=logger,
    )
    subject.classify(tenant_id="tenant-safe", call_id="call-safe", text=secret)
    assert secret not in caplog.text
    assert "classification inference completed" in caplog.text


def test_mocked_probability_inference_is_deterministic(tmp_path: Path) -> None:
    paths = artifacts(tmp_path)
    probabilities = [0.1, 0.8, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1]
    subject = classifier(paths, FakeModel(probabilities), [])
    first = subject.classify(
        tenant_id="tenant-a", call_id="call-a", text="Sentetik aynı"
    )
    second = subject.classify(
        tenant_id="tenant-a", call_id="call-a", text="Sentetik aynı"
    )
    assert [(label.name, label.score) for label in first.labels] == [
        (label.name, label.score) for label in second.labels
    ]
    assert first.probabilities == second.probabilities
