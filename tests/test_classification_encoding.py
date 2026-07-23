import pytest

from app.classification.dataset import load_classification_taxonomy
from app.classification.encoding import MultiLabelEncoder, taxonomy_thresholds


def encoder() -> MultiLabelEncoder:
    taxonomy = load_classification_taxonomy("config/classification_taxonomy.json")
    return MultiLabelEncoder.from_taxonomy(taxonomy)


def test_taxonomy_order_produces_deterministic_multi_hot_vectors() -> None:
    subject = encoder()
    assert subject.label_order[0] == "product_information"
    assert subject.label_order[-1] == "no_action"
    assert subject.encode(("technical_issue", "complaint")) == (
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
    )


def test_multi_label_decode_preserves_taxonomy_order() -> None:
    assert encoder().decode((0, 1, 0, 1, 0, 0, 0, 0)) == (
        "price_objection",
        "technical_issue",
    )


def test_no_action_is_removed_when_actionable_label_is_predicted() -> None:
    subject = encoder()
    assert subject.decode((0, 0, 0, 1, 0, 0, 0, 1)) == ("technical_issue",)


def test_per_label_thresholds_are_applied() -> None:
    subject = encoder()
    thresholds = {label: 0.5 for label in subject.label_order}
    thresholds["technical_issue"] = 0.8
    vector = subject.threshold_probabilities(
        (0.1, 0.6, 0.1, 0.79, 0.1, 0.1, 0.1, 0.9),
        thresholds,
    )
    assert subject.decode(vector) == ("price_objection",)


def test_taxonomy_thresholds_follow_label_order() -> None:
    taxonomy = load_classification_taxonomy("config/classification_taxonomy.json")
    assert tuple(taxonomy_thresholds(taxonomy)) == taxonomy.label_ids


@pytest.mark.parametrize(
    "labels",
    [
        ("unknown",),
        ("no_action", "complaint"),
    ],
)
def test_invalid_labels_are_rejected(labels: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        encoder().encode(labels)
