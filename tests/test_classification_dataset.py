import json
from pathlib import Path

import pytest

from app.classification.dataset import (
    load_classification_dataset,
    load_classification_taxonomy,
    normalize_example_text,
)

TAXONOMY_PATH = Path("config/classification_taxonomy.json")
SEED_PATH = Path("data/synthetic/classification_seed.jsonl")


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )


def record(example_id: str, text: str, **changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "example_id": example_id,
        "text": text,
        "labels": ["no_action"],
        "split": "train",
        "source": "synthetic",
    }
    values.update(changes)
    return values


def test_seed_dataset_is_valid_balanced_and_ordered() -> None:
    taxonomy = load_classification_taxonomy(TAXONOMY_PATH)
    dataset = load_classification_dataset(SEED_PATH, taxonomy)
    assert taxonomy.label_ids == (
        "product_information",
        "price_objection",
        "cancellation_request",
        "technical_issue",
        "complaint",
        "renewal_interest",
        "churn_risk",
        "no_action",
    )
    assert dataset.total_examples >= 48
    assert all(dataset.label_counts[label] >= 6 for label in taxonomy.label_ids)
    assert set(dataset.split_counts) == {"train", "validation", "test"}
    assert isinstance(dataset.examples, tuple)
    assert dataset.examples[0].example_id == "synthetic_pi_001"
    assert any(len(item.labels) > 1 for item in dataset.examples)


def test_normalization_handles_unicode_whitespace_and_boundary_punctuation() -> None:
    assert normalize_example_text("  !!!İPTAL,\t talebi??? ") == "i̇ptal, talebi"


def test_duplicate_id_is_rejected_without_text(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    secret = "BU METİN HATA MESAJINDA OLMAMALI"
    write_jsonl(path, [record("same", "Birinci"), record("same", secret)])
    with pytest.raises(ValueError, match="Duplicate example_id") as captured:
        load_classification_dataset(path, load_classification_taxonomy(TAXONOMY_PATH))
    assert secret not in str(captured.value)


def test_duplicate_normalized_text_and_split_leakage_are_rejected(
    tmp_path: Path,
) -> None:
    taxonomy = load_classification_taxonomy(TAXONOMY_PATH)
    duplicate = tmp_path / "duplicate.jsonl"
    write_jsonl(
        duplicate,
        [record("one", "Merhaba dünya"), record("two", "  !!!MERHABA   DÜNYA...")],
    )
    with pytest.raises(ValueError, match="Duplicate normalized text"):
        load_classification_dataset(duplicate, taxonomy)

    leakage = tmp_path / "leakage.jsonl"
    write_jsonl(
        leakage,
        [
            record("one", "Tek anlam", split="train"),
            record("two", "TEK ANLAM!", split="test"),
        ],
    )
    with pytest.raises(ValueError, match="multiple splits"):
        load_classification_dataset(leakage, taxonomy)


def test_malformed_json_reports_line_without_content(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(record("one", "Güvenli")) + "\n{özel içerik",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Malformed JSON at line 2") as captured:
        load_classification_dataset(path, load_classification_taxonomy(TAXONOMY_PATH))
    assert "özel içerik" not in str(captured.value)


def test_unknown_label_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unknown.jsonl"
    write_jsonl(path, [record("one", "Sentetik örnek", labels=["unknown"])])
    with pytest.raises(ValueError, match="Unknown label"):
        load_classification_dataset(path, load_classification_taxonomy(TAXONOMY_PATH))
