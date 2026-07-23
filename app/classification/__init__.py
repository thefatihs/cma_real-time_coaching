"""Text-classification taxonomy and synthetic dataset foundations."""

from app.classification.dataset import (
    ClassificationDataset,
    load_classification_dataset,
    load_classification_taxonomy,
    normalize_example_text,
)
from app.classification.encoding import MultiLabelEncoder, taxonomy_thresholds
from app.classification.models import (
    ClassificationExample,
    ClassificationLabelDefinition,
    ClassificationTaxonomy,
    DatasetSplit,
)

__all__ = [
    "ClassificationDataset",
    "ClassificationExample",
    "ClassificationLabelDefinition",
    "ClassificationTaxonomy",
    "DatasetSplit",
    "MultiLabelEncoder",
    "load_classification_dataset",
    "load_classification_taxonomy",
    "normalize_example_text",
    "taxonomy_thresholds",
]
