from datetime import UTC, datetime

import pytest

from app.classification.postprocessing import apply_classification_contrast_guards
from app.events.models import (
    ClassificationLabel,
    ClassificationResultEvent,
    CoachingAction,
)


NOW = datetime(2026, 7, 24, tzinfo=UTC)


def result(*labels: str) -> ClassificationResultEvent:
    return ClassificationResultEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        transcript_event_id="stable_1",
        labels=[
            ClassificationLabel(
                name=label,
                score=0.949 if label == "product_information" else 0.940,
            )
            for label in labels
        ],
        action=CoachingAction.TEMPLATE_ACTION,
        model_id="common_turkish_setfit_v2",
        probabilities={
            "product_information": 0.949,
            "price_objection": 0.940,
        },
        thresholds={
            "product_information": 0.90,
            "price_objection": 0.35,
        },
        created_at_utc=NOW,
    )


@pytest.mark.parametrize(
    "text",
    [
        "Paketin aylık fiyatına kadar ücret seçeneklerini öğrenmek istiyorum.",
        "Paketin aylık fiyatı ne kadar?",
        "Paket ücretleri ve fiyat bilgisi nedir?",
        "Tarife fiyatı kaç lira?",
        "Paketin aylik fiyati ve uçret seçeneklerini öğrenebilir miyim",
    ],
)
def test_neutral_price_queries_suppress_final_price_objection_label(
    text: str,
) -> None:
    raw = result("product_information", "price_objection")
    guarded, metadata = apply_classification_contrast_guards(text, raw)
    assert [label.name for label in guarded.labels] == ["product_information"]
    assert guarded.probabilities["price_objection"] == 0.940
    assert raw.labels != guarded.labels
    assert metadata.suppressed_labels == ("price_objection",)


@pytest.mark.parametrize(
    "text",
    [
        "Bu fiyat çok yüksek.",
        "Bu ücret fazla, bütçemi aşıyor.",
        "Ödeyemem, indirim yapılmazsa devam etmem.",
    ],
)
def test_true_price_objection_is_preserved(text: str) -> None:
    guarded, metadata = apply_classification_contrast_guards(
        text, result("price_objection")
    )
    assert [label.name for label in guarded.labels] == ["price_objection"]
    assert metadata.suppressed_labels == ()


def test_price_query_and_objection_preserve_multilabel_output() -> None:
    guarded, _ = apply_classification_contrast_guards(
        "Fiyatı ne kadar, çünkü mevcut ücretiniz çok pahalı.",
        result("product_information", "price_objection"),
    )
    assert {label.name for label in guarded.labels} == {
        "product_information",
        "price_objection",
    }
