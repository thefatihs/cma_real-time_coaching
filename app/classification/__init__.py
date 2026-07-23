"""Text-classification taxonomy and synthetic dataset foundations."""

from app.classification.dataset import (
    ClassificationDataset,
    load_classification_dataset,
    load_classification_taxonomy,
    normalize_example_text,
    validate_required_label_counts,
)
from app.classification.encoding import MultiLabelEncoder, taxonomy_thresholds
from app.classification.models import (
    ClassificationExample,
    ClassificationLabelDefinition,
    ClassificationTaxonomy,
    DatasetSplit,
)
from app.classification.runtime import (
    RuntimeArtifactPaths,
    RuntimeClassifierConfig,
    RuntimeSetFitClassifier,
)

__all__ = [
    "ClassificationDataset",
    "ClassificationExample",
    "ClassificationLabelDefinition",
    "ClassificationTaxonomy",
    "DatasetSplit",
    "MultiLabelEncoder",
    "RuntimeArtifactPaths",
    "RuntimeClassifierConfig",
    "RuntimeSetFitClassifier",
    "load_classification_dataset",
    "load_classification_taxonomy",
    "normalize_example_text",
    "taxonomy_thresholds",
    "validate_required_label_counts",
]
