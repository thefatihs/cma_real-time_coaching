"""SetFit baseline training orchestration with strict split isolation."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import random
from typing import Protocol

import numpy as np

from app.classification.artifacts import (
    MODEL_ID,
    TrainingArtifactMetadata,
    resolved_package_versions,
    save_training_artifacts,
    sha256_file,
)
from app.classification.dataset import ClassificationDataset
from app.classification.encoding import MultiLabelEncoder
from app.classification.models import ClassificationExample, DatasetSplit


class TrainableModel(Protocol):
    def save_pretrained(self, save_directory: str | Path) -> object: ...


class TrainerLike(Protocol):
    def train(self) -> object: ...


@dataclass(frozen=True, slots=True)
class TrainingParameters:
    seed: int = 42
    num_epochs: int = 1
    batch_size: int = 8
    num_iterations: int = 20
    learning_rate: float = 2e-5

    def __post_init__(self) -> None:
        if self.num_epochs <= 0 or self.batch_size <= 0 or self.num_iterations <= 0:
            raise ValueError("epoch, batch, and iteration values must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "seed": self.seed,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "num_iterations": self.num_iterations,
            "learning_rate": self.learning_rate,
            "multi_label_strategy": "one-vs-rest",
        }


def examples_for_split(
    dataset: ClassificationDataset, split: DatasetSplit
) -> tuple[ClassificationExample, ...]:
    return tuple(example for example in dataset.examples if example.split is split)


def build_split_payload(
    examples: tuple[ClassificationExample, ...], encoder: MultiLabelEncoder
) -> dict[str, list[object]]:
    return {
        "text": [example.text for example in examples],
        "label": [list(encoder.encode(example.labels)) for example in examples],
    }


def train_setfit_baseline(
    *,
    dataset: ClassificationDataset,
    encoder: MultiLabelEncoder,
    taxonomy_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    backbone: str,
    parameters: TrainingParameters,
    components_factory: Callable[
        [
            str,
            tuple[str, ...],
            dict[str, list[object]],
            dict[str, list[object]],
            TrainingParameters,
            Path,
        ],
        tuple[TrainableModel, TrainerLike],
    ]
    | None = None,
    timestamp_factory: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> TrainingArtifactMetadata:
    train_examples = examples_for_split(dataset, DatasetSplit.TRAIN)
    validation_examples = examples_for_split(dataset, DatasetSplit.VALIDATION)
    if not train_examples:
        raise ValueError("training split cannot be empty")
    if not validation_examples:
        raise ValueError("validation split cannot be empty")
    set_deterministic_seeds(parameters.seed)
    factory = components_factory or _setfit_components
    destination = Path(output_dir)
    model, trainer = factory(
        backbone,
        encoder.label_order,
        build_split_payload(train_examples, encoder),
        build_split_payload(validation_examples, encoder),
        parameters,
        destination,
    )
    trainer.train()
    model.save_pretrained(destination)
    metadata = TrainingArtifactMetadata(
        model_id=MODEL_ID,
        backbone=backbone,
        label_order=encoder.label_order,
        taxonomy_checksum=sha256_file(taxonomy_path),
        dataset_checksum=sha256_file(dataset_path),
        training_parameters=parameters.as_dict(),
        training_timestamp=timestamp_factory(),
        split_counts=dict(dataset.split_counts),
        package_versions=resolved_package_versions(),
    )
    save_training_artifacts(destination, metadata)
    return metadata


def set_deterministic_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        manual_seed = getattr(torch, "manual_seed", None)
        if callable(manual_seed):
            manual_seed(seed)
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.manual_seed_all(seed)
    except ImportError:
        return


def _setfit_components(
    backbone: str,
    label_order: tuple[str, ...],
    train_payload: dict[str, list[object]],
    validation_payload: dict[str, list[object]],
    parameters: TrainingParameters,
    output_dir: Path,
) -> tuple[TrainableModel, TrainerLike]:
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    model = SetFitModel.from_pretrained(
        backbone,
        labels=list(label_order),
        multi_target_strategy="one-vs-rest",
        device="cpu",
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        batch_size=parameters.batch_size,
        num_epochs=parameters.num_epochs,
        num_iterations=parameters.num_iterations,
        body_learning_rate=parameters.learning_rate,
        seed=parameters.seed,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=Dataset.from_dict(train_payload),
        eval_dataset=Dataset.from_dict(validation_payload),
    )
    return model, trainer
