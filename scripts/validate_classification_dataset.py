"""Validate a synthetic classification JSONL dataset and print safe counts."""

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.classification.dataset import (  # noqa: E402
    load_classification_dataset,
    load_classification_taxonomy,
    validate_required_label_counts,
)

DEFAULT_TAXONOMY_PATH = Path("config/classification_taxonomy.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", type=Path)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY_PATH,
    )
    arguments = parser.parse_args()

    try:
        taxonomy = load_classification_taxonomy(arguments.taxonomy)
        dataset = load_classification_dataset(arguments.jsonl_path, taxonomy)
        validate_required_label_counts(dataset, taxonomy)
    except ValueError as error:
        print(f"validation status: failed ({error})")
        return 1

    print(f"total examples: {dataset.total_examples}")
    print(f"split counts: {_format_counts(dataset.split_counts)}")
    print(f"label counts: {_format_counts(dataset.label_counts)}")
    for label, counts in dataset.label_split_counts.items():
        print(f"label split counts ({label}): {_format_counts(counts)}")
    print("validation status: valid")
    return 0


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in counts.items())


if __name__ == "__main__":
    raise SystemExit(main())
