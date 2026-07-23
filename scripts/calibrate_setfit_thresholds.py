"""Calibrate SetFit thresholds using only the validation split."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.classification.artifacts import (  # noqa: E402
    load_training_metadata,
    sha256_file,
)
from app.classification.calibration import (  # noqa: E402
    CalibrationConfiguration,
    calibrate_validation_model,
    save_calibration_report,
    sha256_directory,
)
from app.classification.dataset import (  # noqa: E402
    load_classification_dataset,
    load_classification_taxonomy,
)
from app.classification.encoding import MultiLabelEncoder  # noqa: E402
from app.classification.models import DatasetSplit  # noqa: E402
from app.classification.training import examples_for_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-threshold", type=float, default=0.30)
    parser.add_argument("--maximum-threshold", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.05)
    arguments = parser.parse_args()

    from setfit import SetFitModel

    taxonomy = load_classification_taxonomy(arguments.taxonomy)
    dataset = load_classification_dataset(arguments.dataset, taxonomy)
    metadata = load_training_metadata(arguments.model_dir)
    taxonomy_checksum = sha256_file(arguments.taxonomy)
    dataset_checksum = sha256_file(arguments.dataset)
    if metadata.taxonomy_checksum != taxonomy_checksum:
        raise ValueError("model taxonomy checksum does not match calibration taxonomy")
    if metadata.dataset_checksum != dataset_checksum:
        raise ValueError("model dataset checksum does not match calibration dataset")
    encoder = MultiLabelEncoder.from_taxonomy(taxonomy)
    if metadata.label_order != encoder.label_order:
        raise ValueError("model label order does not match calibration taxonomy")

    configuration = CalibrationConfiguration(
        minimum_threshold=arguments.minimum_threshold,
        maximum_threshold=arguments.maximum_threshold,
        step=arguments.step,
    )
    model = SetFitModel.from_pretrained(arguments.model_dir, device="cpu")
    result = calibrate_validation_model(model, dataset, taxonomy, configuration)
    save_calibration_report(
        arguments.output,
        metadata=metadata,
        result=result,
        configuration=configuration,
        model_checksum=sha256_directory(arguments.model_dir),
        taxonomy_checksum=taxonomy_checksum,
        dataset_checksum=dataset_checksum,
    )
    validation_count = len(examples_for_split(dataset, DatasetSplit.VALIDATION))
    print(f"model_id: {metadata.model_id}")
    print("calibration split: validation")
    print(f"validation examples: {validation_count}")
    print("calibration status: completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
