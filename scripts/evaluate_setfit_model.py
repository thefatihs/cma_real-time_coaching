"""Evaluate a saved SetFit model without retraining."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.classification.artifacts import (  # noqa: E402
    load_training_metadata,
    save_evaluation_report,
    sha256_file,
)
from app.classification.dataset import (  # noqa: E402
    load_classification_dataset,
    load_classification_taxonomy,
)
from app.classification.encoding import (  # noqa: E402
    MultiLabelEncoder,
    taxonomy_thresholds,
)
from app.classification.evaluation import evaluate_model  # noqa: E402
from app.classification.models import DatasetSplit  # noqa: E402
from app.classification.training import examples_for_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()

    from setfit import SetFitModel

    taxonomy = load_classification_taxonomy(arguments.taxonomy)
    dataset = load_classification_dataset(arguments.dataset, taxonomy)
    metadata = load_training_metadata(arguments.model_dir)
    if metadata.taxonomy_checksum != sha256_file(arguments.taxonomy):
        raise ValueError("model taxonomy checksum does not match evaluation taxonomy")
    if metadata.dataset_checksum != sha256_file(arguments.dataset):
        raise ValueError("model dataset checksum does not match evaluation dataset")
    encoder = MultiLabelEncoder.from_taxonomy(taxonomy)
    if encoder.label_order != metadata.label_order:
        raise ValueError("model label order does not match taxonomy")
    split = DatasetSplit(arguments.split)
    examples = examples_for_split(dataset, split)
    model = SetFitModel.from_pretrained(arguments.model_dir, device="cpu")
    metrics = evaluate_model(
        model,
        examples,
        encoder,
        taxonomy_thresholds(taxonomy),
    )
    save_evaluation_report(
        arguments.report,
        metadata=metadata,
        split=arguments.split,
        thresholds=taxonomy_thresholds(taxonomy),
        metrics=metrics,
    )
    print(f"evaluation split: {arguments.split}")
    print(f"evaluated examples: {metrics.example_count}")
    print("evaluation status: completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
