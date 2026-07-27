import logging

import pytest
from pydantic import ValidationError

from app.calls.models import CallState
from app.coaching.llm_decision_gate import (
    LLMCoachingDecision,
    LLMCoachingDecisionGate,
    LLMCoachingDecisionReason,
)

TENANT_ID = "tenant_alpha"
CALL_ID = "call_001"
REVISION = 7


def decide(
    *,
    current: tuple[str, ...] = (),
    newly: tuple[str, ...] = (),
    enabled: tuple[str, ...] = ("product_information", "technical_issue"),
    rag_enabled: bool = True,
    llm_enabled: bool = True,
    tenant_id: str = TENANT_ID,
    call_id: str = CALL_ID,
    revision: int = REVISION,
):
    return LLMCoachingDecisionGate().decide(
        tenant_id=tenant_id,
        call_id=call_id,
        revision=revision,
        current_labels=current,
        newly_detected_labels=newly,
        rag_llm_enabled_labels=enabled,
        rag_enabled=rag_enabled,
        llm_enabled=llm_enabled,
    )


@pytest.mark.parametrize("label", ["product_information", "technical_issue"])
def test_new_enabled_label_requests_rag_llm(label: str) -> None:
    result = decide(current=(label,), newly=(label,))

    assert result.decision is LLMCoachingDecision.REQUEST_RAG_LLM
    assert result.reason is LLMCoachingDecisionReason.RAG_LLM_REQUESTED
    assert result.trigger_labels == (label,)


def test_new_non_enabled_label_keeps_local_coaching_ready() -> None:
    result = decide(current=("complaint",), newly=("complaint",))

    assert result.decision is LLMCoachingDecision.READY_COACHING_ONLY
    assert result.reason is LLMCoachingDecisionReason.LOCAL_COACHING_SUFFICIENT
    assert result.trigger_labels == ("complaint",)


def test_no_new_labels_skips_even_when_current_label_is_enabled() -> None:
    result = decide(current=("product_information",), newly=())

    assert result.decision is LLMCoachingDecision.SKIP
    assert result.reason is LLMCoachingDecisionReason.NO_NEW_LABELS
    assert result.trigger_labels == ()


@pytest.mark.parametrize(
    ("rag_enabled", "llm_enabled", "reason"),
    [
        (False, True, LLMCoachingDecisionReason.RAG_DISABLED),
        (True, False, LLMCoachingDecisionReason.LLM_DISABLED),
    ],
)
def test_disabled_service_preserves_ready_local_coaching(
    rag_enabled: bool,
    llm_enabled: bool,
    reason: LLMCoachingDecisionReason,
) -> None:
    result = decide(
        current=("product_information",),
        newly=("product_information",),
        rag_enabled=rag_enabled,
        llm_enabled=llm_enabled,
    )

    assert result.decision is LLMCoachingDecision.READY_COACHING_ONLY
    assert result.reason is reason


def test_any_enabled_new_label_requests_and_trigger_order_is_deterministic() -> None:
    result = decide(
        current=("technical_issue", "complaint", "product_information"),
        newly=("technical_issue", "complaint", "product_information"),
    )

    assert result.decision is LLMCoachingDecision.REQUEST_RAG_LLM
    assert result.trigger_labels == ("product_information", "technical_issue")


@pytest.mark.parametrize(
    ("current", "newly", "enabled"),
    [
        (("unknown",), (), ()),
        ((), ("unknown",), ()),
        ((), (), ("unknown",)),
        (("urun_bilgisi",), ("urun_bilgisi",), ("urun_bilgisi",)),
    ],
)
def test_unknown_or_noncanonical_label_is_rejected(
    current: tuple[str, ...],
    newly: tuple[str, ...],
    enabled: tuple[str, ...],
) -> None:
    result = decide(current=current, newly=newly, enabled=enabled)

    assert result.decision is LLMCoachingDecision.REJECTED
    assert result.reason is LLMCoachingDecisionReason.INVALID_LABEL


@pytest.mark.parametrize(
    ("current", "newly"),
    [
        (("no_action", "complaint"), ("complaint",)),
        (("no_action", "complaint"), ("no_action", "complaint")),
        (("no_action",), ("no_action", "complaint")),
    ],
)
def test_no_action_with_business_label_is_rejected(
    current: tuple[str, ...],
    newly: tuple[str, ...],
) -> None:
    result = decide(current=current, newly=newly)

    assert result.decision is LLMCoachingDecision.REJECTED
    assert result.reason is LLMCoachingDecisionReason.NO_ACTION_CONFLICT


def test_new_label_missing_from_current_state_is_rejected() -> None:
    result = decide(
        current=("complaint",),
        newly=("product_information",),
    )

    assert result.decision is LLMCoachingDecision.REJECTED
    assert result.reason is LLMCoachingDecisionReason.INCONSISTENT_LABEL_STATE


@pytest.mark.parametrize(
    ("tenant_id", "call_id"),
    [("", CALL_ID), (" ", CALL_ID), (TENANT_ID, ""), (TENANT_ID, " ")],
)
def test_empty_scope_is_rejected(tenant_id: str, call_id: str) -> None:
    result = decide(tenant_id=tenant_id, call_id=call_id)

    assert result.decision is LLMCoachingDecision.REJECTED
    assert result.reason is LLMCoachingDecisionReason.INVALID_SCOPE
    assert result.tenant_id == tenant_id
    assert result.call_id == call_id


def test_negative_revision_is_rejected() -> None:
    result = decide(revision=-1)

    assert result.decision is LLMCoachingDecision.REJECTED
    assert result.reason is LLMCoachingDecisionReason.INVALID_REVISION
    assert result.revision == -1


def test_result_is_immutable_and_deterministic() -> None:
    first = decide(
        current=("technical_issue", "product_information"),
        newly=("technical_issue", "product_information"),
    )
    second = decide(
        current=("product_information", "technical_issue"),
        newly=("product_information", "technical_issue"),
    )

    assert first == second
    with pytest.raises(ValidationError):
        first.decision = LLMCoachingDecision.SKIP


def test_gate_has_no_call_state_or_logging_side_effects(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = CallState(tenant_id=TENANT_ID, call_id=CALL_ID)
    before = state.model_dump()

    with caplog.at_level(logging.DEBUG):
        result = decide(
            current=("product_information",),
            newly=("product_information",),
        )

    assert result.decision is LLMCoachingDecision.REQUEST_RAG_LLM
    assert state.model_dump() == before
    assert "PRIVATE_TRANSCRIPT_SENTINEL" not in caplog.text
    assert "PRIVATE_PROMPT_SENTINEL" not in caplog.text
    assert "PRIVATE_PATH_SENTINEL" not in caplog.text
