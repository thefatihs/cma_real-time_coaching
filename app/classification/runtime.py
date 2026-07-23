"""Tenant-aware lazy runtime adapter for local SetFit artifacts."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
from pathlib import Path
from threading import Lock
from time import perf_counter
from types import MappingProxyType
from typing import Protocol

from app.classification.artifacts import (
    MODEL_ID,
    TrainingArtifactMetadata,
    load_training_metadata,
    sha256_file,
)
from app.classification.calibration import sha256_directory
from app.classification.dataset import load_classification_taxonomy
from app.classification.encoding import MultiLabelEncoder
from app.classification.evaluation import ProbabilityModel
from app.classification.models import ClassificationTaxonomy
from app.classification.threshold_profiles import (
    ThresholdProfile,
    load_threshold_profile,
)
from app.events.models import (
    ClassificationLabel,
    ClassificationResultEvent,
    CoachingAction,
)

DEFAULT_MODEL_DIR = Path("local_artifacts/models/common_turkish_setfit_v2")
DEFAULT_THRESHOLD_PROFILE = Path(
    "config/classification_thresholds/common_turkish_setfit_v2.json"
)
DEFAULT_TAXONOMY = Path("config/classification_taxonomy.json")


class SetFitModelLoader(Protocol):
    def __call__(self, model_dir: Path) -> ProbabilityModel: ...


@dataclass(frozen=True, slots=True)
class RuntimeArtifactPaths:
    model_dir: Path = DEFAULT_MODEL_DIR
    threshold_profile_path: Path = DEFAULT_THRESHOLD_PROFILE
    taxonomy_path: Path = DEFAULT_TAXONOMY
    expected_model_id: str = MODEL_ID

    def __post_init__(self) -> None:
        if not self.expected_model_id.strip():
            raise ValueError("expected_model_id cannot be empty")


@dataclass(frozen=True, slots=True)
class RuntimeClassifierConfig:
    default_artifacts: RuntimeArtifactPaths = field(
        default_factory=RuntimeArtifactPaths
    )
    tenant_artifacts: Mapping[str, RuntimeArtifactPaths] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, RuntimeArtifactPaths] = {}
        for tenant_id, artifacts in self.tenant_artifacts.items():
            cleaned = tenant_id.strip()
            if not cleaned:
                raise ValueError("tenant artifact key cannot be empty")
            if cleaned in normalized:
                raise ValueError("tenant artifact keys must be unique")
            normalized[cleaned] = artifacts
        object.__setattr__(self, "tenant_artifacts", MappingProxyType(normalized))

    def artifacts_for(self, tenant_id: str) -> RuntimeArtifactPaths:
        cleaned = _required_text(tenant_id, "tenant_id")
        return self.tenant_artifacts.get(cleaned, self.default_artifacts)


@dataclass(slots=True)
class _LoadedClassifier:
    model: ProbabilityModel
    metadata: TrainingArtifactMetadata
    taxonomy: ClassificationTaxonomy
    profile: ThresholdProfile
    encoder: MultiLabelEncoder
    inference_lock: Lock = field(default_factory=Lock)


class RuntimeSetFitClassifier:
    def __init__(
        self,
        config: RuntimeClassifierConfig | None = None,
        *,
        model_loader: SetFitModelLoader | None = None,
        logger: logging.Logger | None = None,
        timer: Callable[[], float] = perf_counter,
        utc_datetime_factory: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config or RuntimeClassifierConfig()
        self._model_loader = model_loader or _load_setfit_model
        self._logger = logger or logging.getLogger(__name__)
        self._timer = timer
        self._utc_datetime_factory = utc_datetime_factory
        self._cache: dict[tuple[Path, Path, Path, str], _LoadedClassifier] = {}
        self._load_lock = Lock()

    def classify(
        self,
        *,
        tenant_id: str,
        call_id: str,
        text: str,
        transcript_event_id: str | None = None,
        revision: int | None = None,
        sequence_number: int | None = None,
    ) -> ClassificationResultEvent:
        tenant = _required_text(tenant_id, "tenant_id")
        call = _required_text(call_id, "call_id")
        transcript = _required_text(text, "text")
        _validate_optional_sequence(revision, "revision")
        _validate_optional_sequence(sequence_number, "sequence_number")
        event_id = _transcript_event_id(
            call,
            transcript_event_id=transcript_event_id,
            revision=revision,
            sequence_number=sequence_number,
        )
        try:
            loaded = self._loaded_for(tenant)
            with loaded.inference_lock:
                started = self._timer()
                probabilities = _single_probability_row(
                    loaded.model.predict_proba([transcript]),
                    expected_length=len(loaded.encoder.label_order),
                )
                elapsed_ms = (self._timer() - started) * 1000
            thresholds = loaded.profile.calibrated_thresholds
            active_vector = loaded.encoder.threshold_probabilities(
                probabilities, thresholds
            )
            active_labels = [
                ClassificationLabel(name=label, score=probability)
                for label, probability, active in zip(
                    loaded.encoder.label_order,
                    probabilities,
                    active_vector,
                    strict=True,
                )
                if active
            ]
            action = _strongest_action(
                tuple(label.name for label in active_labels),
                loaded.taxonomy,
            )
            result = ClassificationResultEvent(
                tenant_id=tenant,
                call_id=call,
                transcript_event_id=event_id,
                labels=active_labels,
                action=action,
                model_id=loaded.metadata.model_id,
                threshold_profile_id=loaded.profile.profile_id,
                probabilities=dict(
                    zip(
                        loaded.encoder.label_order,
                        probabilities,
                        strict=True,
                    )
                ),
                thresholds=dict(thresholds),
                processing_time_ms=elapsed_ms,
                created_at_utc=self._utc_datetime_factory(),
            )
            self._logger.info(
                "classification inference completed",
                extra={
                    "tenant_id": tenant,
                    "call_id": call,
                    "model_id": result.model_id,
                    "threshold_profile_id": result.threshold_profile_id,
                    "labels": [label.name for label in result.labels],
                    "inference_time_ms": elapsed_ms,
                },
            )
            return result
        except Exception as error:
            self._logger.error(
                "classification inference failed",
                extra={
                    "tenant_id": tenant,
                    "call_id": call,
                    "error_type": type(error).__name__,
                },
            )
            raise

    def _loaded_for(self, tenant_id: str) -> _LoadedClassifier:
        artifacts = self._config.artifacts_for(tenant_id)
        key = (
            artifacts.model_dir,
            artifacts.threshold_profile_path,
            artifacts.taxonomy_path,
            artifacts.expected_model_id,
        )
        loaded = self._cache.get(key)
        if loaded is not None:
            return loaded
        with self._load_lock:
            loaded = self._cache.get(key)
            if loaded is None:
                loaded = self._load_artifacts(artifacts)
                self._cache[key] = loaded
        return loaded

    def _load_artifacts(self, artifacts: RuntimeArtifactPaths) -> _LoadedClassifier:
        metadata = load_training_metadata(artifacts.model_dir)
        if metadata.model_id != artifacts.expected_model_id:
            raise ValueError("model metadata does not match configured model identity")
        taxonomy = load_classification_taxonomy(artifacts.taxonomy_path)
        encoder = MultiLabelEncoder.from_taxonomy(taxonomy)
        if metadata.label_order != encoder.label_order:
            raise ValueError("model label order does not match taxonomy")
        taxonomy_checksum = sha256_file(artifacts.taxonomy_path)
        if metadata.taxonomy_checksum != taxonomy_checksum:
            raise ValueError("model taxonomy checksum is stale")
        profile = load_threshold_profile(
            artifacts.threshold_profile_path,
            taxonomy=taxonomy,
            metadata=metadata,
            dataset_checksum=metadata.dataset_checksum,
            taxonomy_checksum=taxonomy_checksum,
            model_checksum=sha256_directory(artifacts.model_dir),
        )
        model = self._model_loader(artifacts.model_dir)
        return _LoadedClassifier(
            model=model,
            metadata=metadata,
            taxonomy=taxonomy,
            profile=profile,
            encoder=encoder,
        )


def _load_setfit_model(model_dir: Path) -> ProbabilityModel:
    from setfit import SetFitModel

    return SetFitModel.from_pretrained(model_dir, device="cpu")


def _single_probability_row(
    values: object, *, expected_length: int
) -> tuple[float, ...]:
    to_list = getattr(values, "tolist", None)
    if callable(to_list):
        values = to_list()
    if not isinstance(values, Sequence) or len(values) != 1:
        raise ValueError("model must return exactly one probability row")
    row = values[0]
    row_to_list = getattr(row, "tolist", None)
    if callable(row_to_list):
        row = row_to_list()
    if not isinstance(row, Sequence):
        raise ValueError("model probability row must be a sequence")
    probabilities = tuple(float(value) for value in row)
    if len(probabilities) != expected_length:
        raise ValueError("model probability row does not match label order")
    if any(value < 0 or value > 1 for value in probabilities):
        raise ValueError("model probabilities must be between 0 and 1")
    return probabilities


def _strongest_action(
    labels: tuple[str, ...], taxonomy: ClassificationTaxonomy
) -> CoachingAction:
    strengths = {
        CoachingAction.NO_ACTION: 0,
        CoachingAction.TEMPLATE_ACTION: 1,
        CoachingAction.RAG_ACTION: 2,
        CoachingAction.ESCALATE: 3,
    }
    actions = tuple(taxonomy.label(label).default_coaching_action for label in labels)
    return (
        max(actions, key=strengths.__getitem__) if actions else CoachingAction.NO_ACTION
    )


def _transcript_event_id(
    call_id: str,
    *,
    transcript_event_id: str | None,
    revision: int | None,
    sequence_number: int | None,
) -> str:
    if transcript_event_id is not None:
        return _required_text(transcript_event_id, "transcript_event_id")
    if revision is not None:
        return f"{call_id}:revision:{revision}"
    if sequence_number is not None:
        return f"{call_id}:sequence:{sequence_number}"
    return f"{call_id}:runtime"


def _validate_optional_sequence(value: int | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} cannot be negative")


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
