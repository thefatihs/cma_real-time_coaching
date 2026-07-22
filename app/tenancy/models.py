from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def _optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _unique_non_empty(values: list[str], field_name: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _required_text(value, field_name)
        if item not in seen:
            seen.add(item)
            cleaned.append(item)
    return cleaned


class TenantContext(BaseModel):
    tenant_id: str
    tenant_name: str
    user_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    correlation_id: str | None = None

    @field_validator("tenant_id", "tenant_name")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "value")
        return _required_text(value, field_name)

    @field_validator("user_id", "correlation_id")
    @classmethod
    def validate_optional_text(cls, value: str | None, info: object) -> str | None:
        field_name = getattr(info, "field_name", "value")
        return _optional_text(value, field_name)

    @field_validator("roles")
    @classmethod
    def normalize_roles(cls, values: list[str]) -> list[str]:
        return _unique_non_empty(values, "role")


class TenantASRConfig(BaseModel):
    model_name: str = "large-v3"
    language: str = "tr"
    beam_size: int = 5
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    initial_prompt: str | None = None
    rolling_window_seconds: float = 20.0
    chunk_duration_seconds: float = 2.0
    stable_region_seconds: float = 5.0

    @field_validator("model_name", "language")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))

    @field_validator("beam_size")
    @classmethod
    def validate_beam_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("beam_size must be positive")
        return value

    @field_validator(
        "rolling_window_seconds", "chunk_duration_seconds", "stable_region_seconds"
    )
    @classmethod
    def validate_positive_time(cls, value: float, info: object) -> float:
        if value <= 0:
            raise ValueError(f"{getattr(info, 'field_name', 'time')} must be positive")
        return value

    @field_validator("initial_prompt", mode="before")
    @classmethod
    def normalize_prompt(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def validate_timing_relationships(self) -> Self:
        if self.stable_region_seconds >= self.rolling_window_seconds:
            raise ValueError(
                "stable_region_seconds must be smaller than rolling_window_seconds"
            )
        if self.chunk_duration_seconds > self.rolling_window_seconds:
            raise ValueError(
                "chunk_duration_seconds cannot exceed rolling_window_seconds"
            )
        return self


class TenantClassificationConfig(BaseModel):
    model_id: str
    labels: list[str]
    thresholds: dict[str, float] = Field(default_factory=dict)
    default_threshold: float = 0.70
    multi_label: bool = True

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        return _required_text(value, "model_id")

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, values: list[str]) -> list[str]:
        cleaned = [_required_text(value, "label") for value in values]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("labels must be unique")
        return cleaned

    @field_validator("default_threshold")
    @classmethod
    def validate_default_threshold(cls, value: float) -> float:
        return _probability(value, "default_threshold")

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        unknown = set(self.thresholds) - set(self.labels)
        if unknown:
            raise ValueError(f"threshold labels are not configured: {sorted(unknown)}")
        for label, threshold in self.thresholds.items():
            _probability(threshold, f"threshold for {label}")
        return self

    def threshold_for(self, label: str) -> float:
        if label not in self.labels:
            raise ValueError(f"Unknown classification label: {label}")
        return self.thresholds.get(label, self.default_threshold)


class TenantRAGConfig(BaseModel):
    enabled: bool = True
    knowledge_base_id: str | None = None
    top_k: int = 5
    minimum_score: float = 0.60

    @field_validator("knowledge_base_id", mode="before")
    @classmethod
    def normalize_knowledge_base_id(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k must be positive")
        return value

    @field_validator("minimum_score")
    @classmethod
    def validate_minimum_score(cls, value: float) -> float:
        return _probability(value, "minimum_score")

    @model_validator(mode="after")
    def validate_enabled_knowledge_base(self) -> Self:
        if self.enabled and self.knowledge_base_id is None:
            raise ValueError("knowledge_base_id is required when RAG is enabled")
        if self.knowledge_base_id is not None:
            self.knowledge_base_id = self.knowledge_base_id.strip()
        return self


class TenantCoachingConfig(BaseModel):
    cooldown_seconds: float = 20.0
    max_active_suggestions: int = 2
    enable_templates: bool = True
    enable_llm: bool = True
    allowed_actions: list[str]

    @field_validator("cooldown_seconds")
    @classmethod
    def validate_cooldown(cls, value: float) -> float:
        if value < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        return value

    @field_validator("max_active_suggestions")
    @classmethod
    def validate_max_active(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_active_suggestions must be positive")
        return value

    @field_validator("allowed_actions")
    @classmethod
    def normalize_actions(cls, values: list[str]) -> list[str]:
        cleaned = _unique_non_empty(values, "allowed action")
        if not cleaned:
            raise ValueError("at least one allowed action is required")
        return cleaned


class TenantConfig(BaseModel):
    context: TenantContext
    asr: TenantASRConfig
    classification: TenantClassificationConfig
    rag: TenantRAGConfig
    coaching: TenantCoachingConfig


def _probability(value: float, field_name: str) -> float:
    if not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return value
