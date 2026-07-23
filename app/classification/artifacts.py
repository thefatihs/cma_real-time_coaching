"""Metadata and text-free evaluation report artifacts."""

from datetime import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.classification.evaluation import EvaluationMetrics

MODEL_ID = "common_turkish_setfit_v1"
DEFAULT_BACKBONE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
METADATA_FILENAME = "training_metadata.json"
LABEL_ORDER_FILENAME = "label_order.json"


class TrainingArtifactMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    backbone: str
    label_order: tuple[str, ...]
    taxonomy_checksum: str
    dataset_checksum: str
    training_parameters: dict[str, int | float | str]
    training_timestamp: datetime
    split_counts: dict[str, int]
    package_versions: dict[str, str]

    @field_validator("model_id", "backbone")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned

    @field_validator("taxonomy_checksum", "dataset_checksum")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("checksums must be lowercase SHA-256 hex digests")
        return value

    @model_validator(mode="after")
    def validate_collections(self) -> Self:
        if not self.label_order or len(self.label_order) != len(set(self.label_order)):
            raise ValueError("label_order must be non-empty and unique")
        if self.training_timestamp.tzinfo is None:
            raise ValueError("training_timestamp must be timezone-aware")
        if any(value < 0 for value in self.split_counts.values()):
            raise ValueError("split counts cannot be negative")
        return self


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_package_versions() -> dict[str, str]:
    packages = (
        "setfit",
        "sentence-transformers",
        "transformers",
        "tokenizers",
        "torch",
        "scikit-learn",
        "datasets",
    )
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def save_training_artifacts(
    output_dir: str | Path, metadata: TrainingArtifactMetadata
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / METADATA_FILENAME, metadata.model_dump(mode="json"))
    _write_json(destination / LABEL_ORDER_FILENAME, list(metadata.label_order))


def load_training_metadata(model_dir: str | Path) -> TrainingArtifactMetadata:
    path = Path(model_dir) / METADATA_FILENAME
    try:
        return TrainingArtifactMetadata.model_validate_json(path.read_text("utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Invalid or missing training metadata") from error


def save_evaluation_report(
    path: str | Path,
    *,
    metadata: TrainingArtifactMetadata,
    split: str,
    thresholds: dict[str, float],
    metrics: EvaluationMetrics,
) -> None:
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    report = {
        "model_id": metadata.model_id,
        "backbone": metadata.backbone,
        "label_order": list(metadata.label_order),
        "taxonomy_checksum": metadata.taxonomy_checksum,
        "dataset_checksum": metadata.dataset_checksum,
        "split": split,
        "thresholds": thresholds,
        "metrics": metrics.as_dict(),
    }
    _write_json(Path(path), report)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
