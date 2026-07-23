from app.classification.evaluation import evaluate_model, evaluate_probabilities
from app.classification.encoding import MultiLabelEncoder
from app.classification.models import ClassificationExample, DatasetSplit


def encoder() -> MultiLabelEncoder:
    return MultiLabelEncoder(("a", "b", "no_action"))


def test_metrics_include_micro_macro_exact_match_and_hamming_loss() -> None:
    metrics = evaluate_probabilities(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0.9, 0.2, 0.1), (0.8, 0.9, 0.1), (0.1, 0.2, 0.9)),
        encoder(),
        {"a": 0.5, "b": 0.5, "no_action": 0.5},
        elapsed_seconds=0.03,
    )
    assert metrics.micro_precision == 0.75
    assert metrics.micro_recall == 1.0
    assert round(metrics.micro_f1, 3) == 0.857
    assert metrics.exact_match_ratio == 2 / 3
    assert metrics.hamming_loss == 1 / 9
    assert metrics.average_inference_time_ms == 10.0
    assert metrics.per_label["a"].support == 1
    assert metrics.per_label["a"].minimum_probability == 0.1
    assert metrics.per_label["a"].mean_probability == 0.6
    assert metrics.per_label["a"].maximum_probability == 0.9
    assert metrics.per_label["a"].predictions_above_threshold == 2
    assert metrics.per_label["a"].true_positives == 1
    assert metrics.per_label["a"].false_positives == 1
    assert metrics.per_label["a"].false_negatives == 0
    assert metrics.no_predicted_labels == 0
    assert metrics.no_action_diagnostics.no_business_label_above_threshold == 1
    assert metrics.no_action_diagnostics.no_action_above_threshold == 1
    assert metrics.no_action_diagnostics.pre_exclusivity_conflicts == 0


def test_zero_division_labels_are_safe() -> None:
    metrics = evaluate_probabilities(
        ((1, 0, 0),),
        ((0.9, 0.1, 0.1),),
        encoder(),
        {"a": 0.5, "b": 0.5, "no_action": 0.5},
    )
    assert metrics.per_label["b"].precision == 0.0
    assert metrics.per_label["b"].recall == 0.0
    assert metrics.per_label["b"].f1 == 0.0
    assert metrics.per_label["b"].support == 0


def test_model_evaluation_uses_fake_probabilities_and_timer() -> None:
    class FakeModel:
        def predict_proba(self, inputs: list[str]) -> list[list[float]]:
            assert len(inputs) == 1
            return [[0.1, 0.1, 0.9]]

    moments = iter((10.0, 10.025))
    examples = (
        ClassificationExample(
            example_id="synthetic_eval",
            text="Sentetik nötr ifade.",
            labels=("no_action",),
            split=DatasetSplit.TEST,
        ),
    )
    metrics = evaluate_model(
        FakeModel(),
        examples,
        encoder(),
        {"a": 0.5, "b": 0.5, "no_action": 0.5},
        timer=lambda: next(moments),
    )
    assert metrics.exact_match_ratio == 1.0
    assert round(metrics.average_inference_time_ms, 3) == 25.0


def test_no_action_conflict_and_empty_prediction_diagnostics() -> None:
    metrics = evaluate_probabilities(
        ((1, 0, 0), (0, 0, 1)),
        ((0.8, 0.1, 0.9), (0.1, 0.2, 0.3)),
        encoder(),
        {"a": 0.5, "b": 0.5, "no_action": 0.5},
    )
    assert metrics.no_action_diagnostics.pre_exclusivity_conflicts == 1
    assert metrics.no_action_diagnostics.no_business_label_above_threshold == 1
    assert metrics.no_action_diagnostics.no_action_above_threshold == 1
    assert metrics.no_predicted_labels == 1
    assert metrics.per_label["no_action"].predictions_above_threshold == 1
