"""Dashboard-only construction of existing classification and coaching services."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.calls.models import CallState
from app.classification.artifacts import load_training_metadata, sha256_file
from app.classification.calibration import sha256_directory
from app.classification.dataset import load_classification_taxonomy
from app.classification.encoding import MultiLabelEncoder
from app.classification.runtime import (
    DEFAULT_MODEL_DIR,
    DEFAULT_TAXONOMY,
    DEFAULT_THRESHOLD_PROFILE,
    RuntimeSetFitClassifier,
)
from app.classification.threshold_profiles import load_threshold_profile
from app.coaching.coordinator import CoachingCoordinator
from app.coaching.rule_engine import RuleBasedCoachingEngine
from app.coaching.safe_processor import SafeCoachingProcessorAdapter
from app.streaming.pipeline import (
    CoachingProcessorProtocol,
    StreamingASRPipeline,
    WindowTranscriberProtocol,
)
from live_dashboard.demo_data import TenantDemo
from live_dashboard.view_models import DashboardRuntime


@dataclass(frozen=True, slots=True)
class ArtifactAvailability:
    compatible: bool
    safe_message: str | None = None


@dataclass(frozen=True, slots=True)
class DashboardServiceSelection:
    enable_setfit: bool
    enable_coaching: bool


ClassifierProvider = Callable[[], RuntimeSetFitClassifier]


def inspect_default_artifacts(
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    threshold_profile: Path = DEFAULT_THRESHOLD_PROFILE,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
) -> ArtifactAvailability:
    """Validate local metadata/profile compatibility without loading model weights."""
    try:
        if not model_dir.is_dir() or not threshold_profile.is_file():
            raise ValueError("required artifact is missing")
        metadata = load_training_metadata(model_dir)
        taxonomy = load_classification_taxonomy(taxonomy_path)
        encoder = MultiLabelEncoder.from_taxonomy(taxonomy)
        if metadata.label_order != encoder.label_order:
            raise ValueError("artifact label order is incompatible")
        taxonomy_checksum = sha256_file(taxonomy_path)
        if metadata.taxonomy_checksum != taxonomy_checksum:
            raise ValueError("artifact taxonomy is stale")
        load_threshold_profile(
            threshold_profile,
            taxonomy=taxonomy,
            metadata=metadata,
            dataset_checksum=metadata.dataset_checksum,
            taxonomy_checksum=taxonomy_checksum,
            model_checksum=sha256_directory(model_dir),
        )
    except Exception:
        return ArtifactAvailability(
            compatible=False,
            safe_message=(
                "SetFit modeli şu anda kullanılamıyor; kural tabanlı koçluk "
                "ve ses işleme devam edecek."
            ),
        )
    return ArtifactAvailability(compatible=True)


def default_service_selection(
    availability: ArtifactAvailability,
    *,
    deterministic_rules_available: bool,
) -> DashboardServiceSelection:
    return DashboardServiceSelection(
        enable_setfit=availability.compatible,
        enable_coaching=availability.compatible or deterministic_rules_available,
    )


def build_live_pipeline(
    runtime: DashboardRuntime,
    window_transcriber: WindowTranscriberProtocol,
    *,
    selection: DashboardServiceSelection,
    availability: ArtifactAvailability,
    classifier_provider: ClassifierProvider,
) -> StreamingASRPipeline:
    """Build one uploaded-audio pipeline with optional existing services."""
    classifier = (
        classifier_provider()
        if selection.enable_setfit and availability.compatible
        else None
    )
    coaching_factory = (
        _coaching_factory(runtime.tenant) if selection.enable_coaching else None
    )
    runtime.setfit_enabled = classifier is not None
    runtime.coaching_enabled = coaching_factory is not None
    runtime.rule_engine_enabled = coaching_factory is not None
    runtime.classification_failure = (
        selection.enable_setfit and not availability.compatible
    )
    runtime.service_status_message = (
        availability.safe_message if runtime.classification_failure else None
    )
    return StreamingASRPipeline(
        runtime.tenant.config.context,
        runtime.tenant.config.asr,
        window_transcriber,
        runtime_classifier=classifier,
        coaching_coordinator_factory=coaching_factory,
    )


def _coaching_factory(
    tenant: TenantDemo,
) -> Callable[[CallState], CoachingProcessorProtocol]:
    def create(call_state: CallState) -> CoachingProcessorProtocol:
        coordinator = CoachingCoordinator(
            tenant.config,
            call_state,
            RuleBasedCoachingEngine(tenant.config, tenant.rules),
        )
        return SafeCoachingProcessorAdapter(coordinator)

    return create
