import pytest
from pydantic import ValidationError

from app.tenancy.models import (
    TenantASRConfig,
    TenantClassificationConfig,
    TenantCoachingConfig,
    TenantConfig,
    TenantContext,
    TenantRAGConfig,
)


def test_valid_full_tenant_configuration() -> None:
    config = TenantConfig(
        context=TenantContext(
            tenant_id="tenant_alpha",
            tenant_name="Alpha",
            user_id="user_001",
            roles=["agent", "supervisor"],
        ),
        asr=TenantASRConfig(),
        classification=TenantClassificationConfig(
            model_id="setfit-alpha",
            labels=["şikayet", "satış"],
            thresholds={"şikayet": 0.8},
        ),
        rag=TenantRAGConfig(knowledge_base_id="kb-alpha"),
        coaching=TenantCoachingConfig(allowed_actions=["template", "escalate"]),
    )

    assert config.context.tenant_id == "tenant_alpha"
    assert config.asr.model_name == "large-v3"
    assert config.model_dump_json()


def test_empty_tenant_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="tenant_id cannot be empty"):
        TenantContext(tenant_id="  ", tenant_name="Alpha")


def test_duplicate_roles_are_removed_in_original_order() -> None:
    context = TenantContext(
        tenant_id="tenant_alpha",
        tenant_name="Alpha",
        roles=["agent", "supervisor", "agent"],
    )

    assert context.roles == ["agent", "supervisor"]


@pytest.mark.parametrize(
    "changes",
    [
        {"stable_region_seconds": 20.0},
        {"rolling_window_seconds": 0.0},
    ],
)
def test_invalid_asr_timing_is_rejected(changes: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        TenantASRConfig.model_validate(changes)


def test_chunk_larger_than_rolling_window_is_rejected() -> None:
    with pytest.raises(ValidationError, match="chunk_duration_seconds"):
        TenantASRConfig(rolling_window_seconds=10, chunk_duration_seconds=11)


def test_whitespace_prompt_becomes_none() -> None:
    assert TenantASRConfig(initial_prompt="  ").initial_prompt is None


def test_invalid_classification_threshold_is_rejected() -> None:
    with pytest.raises(ValidationError, match="between 0 and 1"):
        TenantClassificationConfig(
            model_id="model", labels=["satış"], thresholds={"satış": 1.1}
        )


def test_unknown_threshold_label_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not configured"):
        TenantClassificationConfig(
            model_id="model", labels=["satış"], thresholds={"şikayet": 0.8}
        )


def test_threshold_for_uses_specific_and_default_values() -> None:
    config = TenantClassificationConfig(
        model_id="model",
        labels=["satış", "şikayet"],
        thresholds={"şikayet": 0.85},
        default_threshold=0.7,
    )

    assert config.threshold_for("şikayet") == 0.85
    assert config.threshold_for("satış") == 0.7
    with pytest.raises(ValueError, match="Unknown"):
        config.threshold_for("bilinmeyen")


def test_enabled_rag_requires_knowledge_base() -> None:
    with pytest.raises(ValidationError, match="knowledge_base_id"):
        TenantRAGConfig(enabled=True, knowledge_base_id="  ")

    assert (
        TenantRAGConfig(enabled=False, knowledge_base_id=" ").knowledge_base_id is None
    )


def test_duplicate_coaching_actions_are_removed() -> None:
    config = TenantCoachingConfig(allowed_actions=["template", "escalate", "template"])

    assert config.allowed_actions == ["template", "escalate"]
