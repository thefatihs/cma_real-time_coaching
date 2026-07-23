import pytest
from pydantic import ValidationError

from app.classification.models import ClassificationExample, DatasetSplit


def example(**changes: object) -> ClassificationExample:
    values: dict[str, object] = {
        "example_id": "synthetic_001",
        "text": "Paket hakkında bilgi alabilir miyim?",
        "labels": ("product_information",),
        "split": "train",
    }
    values.update(changes)
    return ClassificationExample.model_validate(values)


def test_classification_example_is_immutable_and_normalized() -> None:
    item = example(example_id=" synthetic_001 ", tenant_id=" tenant_demo ")
    assert item.example_id == "synthetic_001"
    assert item.tenant_id == "tenant_demo"
    assert item.split is DatasetSplit.TRAIN
    assert item.source == "synthetic"
    with pytest.raises(ValidationError):
        item.text = "changed"


@pytest.mark.parametrize(
    "changes",
    [
        {"example_id": " "},
        {"text": ""},
        {"labels": ()},
        {"labels": ("complaint", "complaint")},
        {"labels": ("no_action", "complaint")},
        {"tenant_id": " \t "},
        {"source": "customer"},
        {"split": "holdout"},
    ],
)
def test_invalid_example_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        example(**changes)


def test_multiple_non_no_action_labels_are_supported() -> None:
    item = example(labels=("technical_issue", "complaint"))
    assert item.labels == ("technical_issue", "complaint")
