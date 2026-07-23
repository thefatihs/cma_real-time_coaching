"""Versioned calibrated threshold profiles and compatibility checks."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.classification.artifacts import TrainingArtifactMetadata
from app.classification.encoding import taxonomy_thresholds
from app.classification.models import ClassificationTaxonomy


class ThresholdProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    model_id: str
    source_split: Literal["validation"]
    calibrated_thresholds: dict[str, float]
    model_checksum: str
    dataset_checksum: str
    taxonomy_checksum: str
    critical_recall_target: float

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model_id cannot be empty")
        return cleaned

    @field_validator("model_checksum", "dataset_checksum", "taxonomy_checksum")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("checksums must be lowercase SHA-256 hex digests")
        return value

    @field_validator("calibrated_thresholds")
    @classmethod
    def validate_thresholds(cls, values: dict[str, float]) -> dict[str, float]:
        if not values:
            raise ValueError("calibrated_thresholds cannot be empty")
        for label, threshold in values.items():
            if not label.strip():
                raise ValueError("threshold label cannot be empty")
            if not 0 <= threshold <= 1:
                raise ValueError(f"threshold for {label} must be between 0 and 1")
        return values

    @field_validator("critical_recall_target")
    @classmethod
    def validate_recall_target(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("critical_recall_target must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def validate_profile_identity(self) -> Self:
        if self.source_split != "validation":
            raise ValueError("threshold profiles must use the validation split")
        return self

    @property
    def profile_id(self) -> str:
        return f"{self.model_id}:calibrated:v{self.schema_version}"


@dataclass(frozen=True, slots=True)
class ThresholdResolution:
    thresholds: Mapping[str, float]
    threshold_source: Literal["taxonomy_defaults", "threshold_profile"]
    threshold_profile_id: str | None


def load_threshold_profile(
    path: str | Path,
    *,
    taxonomy: ClassificationTaxonomy,
    metadata: TrainingArtifactMetadata,
    dataset_checksum: str,
    taxonomy_checksum: str,
    model_checksum: str | None = None,
) -> ThresholdProfile:
    profile_path = Path(path)
    try:
        payload = json.loads(
            profile_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
        profile = ThresholdProfile.model_validate(payload)
    except (OSError, ValueError) as error:
        raise ValueError("Invalid threshold profile") from error
    validate_threshold_profile(
        profile,
        taxonomy=taxonomy,
        metadata=metadata,
        dataset_checksum=dataset_checksum,
        taxonomy_checksum=taxonomy_checksum,
        model_checksum=model_checksum,
    )
    return profile


def validate_threshold_profile(
    profile: ThresholdProfile,
    *,
    taxonomy: ClassificationTaxonomy,
    metadata: TrainingArtifactMetadata,
    dataset_checksum: str,
    taxonomy_checksum: str,
    model_checksum: str | None = None,
) -> None:
    expected_labels = set(taxonomy.label_ids)
    actual_labels = set(profile.calibrated_thresholds)
    if actual_labels != expected_labels:
        missing = sorted(expected_labels - actual_labels)
        extra = sorted(actual_labels - expected_labels)
        raise ValueError(
            f"threshold profile labels are incompatible; missing={missing}, extra={extra}"
        )
    if profile.model_id != metadata.model_id:
        raise ValueError("threshold profile model_id does not match model metadata")
    if profile.dataset_checksum != dataset_checksum:
        raise ValueError("threshold profile dataset checksum is stale")
    if profile.taxonomy_checksum != taxonomy_checksum:
        raise ValueError("threshold profile taxonomy checksum is stale")
    if model_checksum is not None and profile.model_checksum != model_checksum:
        raise ValueError("threshold profile model checksum is stale")


def resolve_evaluation_thresholds(
    taxonomy: ClassificationTaxonomy,
    profile: ThresholdProfile | None = None,
) -> ThresholdResolution:
    if profile is None:
        return ThresholdResolution(
            thresholds=taxonomy_thresholds(taxonomy),
            threshold_source="taxonomy_defaults",
            threshold_profile_id=None,
        )
    if set(profile.calibrated_thresholds) != set(taxonomy.label_ids):
        raise ValueError("threshold profile labels do not match taxonomy")
    return ThresholdResolution(
        thresholds=dict(profile.calibrated_thresholds),
        threshold_source="threshold_profile",
        threshold_profile_id=profile.profile_id,
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
