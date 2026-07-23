"""Immutable models for the future SetFit text classifier."""

from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.events.models import CoachingAction


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ClassificationLabelDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    display_name: str
    definition: str
    positive_guidance: str
    negative_guidance: str
    default_threshold: float
    default_coaching_action: CoachingAction
    critical: bool

    @field_validator(
        "id",
        "display_name",
        "definition",
        "positive_guidance",
        "negative_guidance",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned

    @field_validator("default_threshold")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("default_threshold must be between 0 and 1")
        return value


class ClassificationTaxonomy(BaseModel):
    model_config = ConfigDict(frozen=True)

    labels: tuple[ClassificationLabelDefinition, ...]

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        if not self.labels:
            raise ValueError("taxonomy must contain at least one label")
        identifiers = [label.id for label in self.labels]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("taxonomy label IDs must be unique")
        return self

    @property
    def label_ids(self) -> tuple[str, ...]:
        return tuple(label.id for label in self.labels)

    def label(self, label_id: str) -> ClassificationLabelDefinition:
        for definition in self.labels:
            if definition.id == label_id:
                return definition
        raise ValueError(f"Unknown classification label: {label_id}")


class ClassificationExample(BaseModel):
    model_config = ConfigDict(frozen=True)

    example_id: str
    text: str
    labels: tuple[str, ...]
    split: DatasetSplit
    tenant_id: str | None = None
    source: Literal["synthetic"] = "synthetic"
    notes: str | None = None

    @field_validator("example_id", "text")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("tenant_id cannot be whitespace")
        return cleaned

    @field_validator("notes")
    @classmethod
    def normalize_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        if not self.labels:
            raise ValueError("at least one label is required")
        if any(not label.strip() for label in self.labels):
            raise ValueError("labels cannot be empty")
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("labels must be unique")
        if "no_action" in self.labels and len(self.labels) != 1:
            raise ValueError("no_action cannot appear with another label")
        return self
