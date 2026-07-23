"""Train the general Turkish multi-label SetFit baseline."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.classification.artifacts import DEFAULT_BACKBONE, MODEL_ID  # noqa: E402
from app.classification.dataset import (  # noqa: E402
    load_classification_dataset,
    load_classification_taxonomy,
)
from app.classification.encoding import MultiLabelEncoder  # noqa: E402
from app.classification.training import (  # noqa: E402
    TrainingParameters,
    train_setfit_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-iterations", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    arguments = parser.parse_args()

    taxonomy = load_classification_taxonomy(arguments.taxonomy)
    dataset = load_classification_dataset(arguments.dataset, taxonomy)
    parameters = TrainingParameters(
        seed=arguments.seed,
        num_epochs=arguments.num_epochs,
        batch_size=arguments.batch_size,
        num_iterations=arguments.num_iterations,
        learning_rate=arguments.learning_rate,
    )
    print(f"model_id: {MODEL_ID}")
    print(f"split counts: {_format_counts(dict(dataset.split_counts))}")
    print(f"label counts: {_format_counts(dict(dataset.label_counts))}")
    print("training status: started")
    train_setfit_baseline(
        dataset=dataset,
        encoder=MultiLabelEncoder.from_taxonomy(taxonomy),
        taxonomy_path=arguments.taxonomy,
        dataset_path=arguments.dataset,
        output_dir=arguments.output_dir,
        backbone=arguments.backbone,
        parameters=parameters,
    )
    print("training status: completed")
    return 0


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
