"""Pure policy gate for deciding whether current coaching needs RAG and LLM."""

from collections.abc import Collection
from enum import Enum

from pydantic import BaseModel, ConfigDict

from app.events.labels import CANONICAL_LABELS


class LLMCoachingDecision(str, Enum):
    REQUEST_RAG_LLM = "request_rag_llm"
    READY_COACHING_ONLY = "ready_coaching_only"
    SKIP = "skip"
    REJECTED = "rejected"


class LLMCoachingDecisionReason(str, Enum):
    RAG_LLM_REQUESTED = "rag_llm_requested"
    NO_NEW_LABELS = "no_new_labels"
    LOCAL_COACHING_SUFFICIENT = "local_coaching_sufficient"
    RAG_DISABLED = "rag_disabled"
    LLM_DISABLED = "llm_disabled"
    INVALID_SCOPE = "invalid_scope"
    INVALID_REVISION = "invalid_revision"
    INVALID_LABEL = "invalid_label"
    INCONSISTENT_LABEL_STATE = "inconsistent_label_state"
    NO_ACTION_CONFLICT = "no_action_conflict"


class LLMCoachingDecisionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    call_id: str
    revision: int
    decision: LLMCoachingDecision
    trigger_labels: tuple[str, ...] = ()
    reason: LLMCoachingDecisionReason


class LLMCoachingDecisionGate:
    def decide(
        self,
        *,
        tenant_id: str,
        call_id: str,
        revision: int,
        current_labels: Collection[str],
        newly_detected_labels: Collection[str],
        rag_llm_enabled_labels: Collection[str],
        rag_enabled: bool,
        llm_enabled: bool,
    ) -> LLMCoachingDecisionResult:
        if not tenant_id.strip() or not call_id.strip():
            return _result(
                tenant_id,
                call_id,
                revision,
                LLMCoachingDecision.REJECTED,
                LLMCoachingDecisionReason.INVALID_SCOPE,
            )
        if revision < 0:
            return _result(
                tenant_id,
                call_id,
                revision,
                LLMCoachingDecision.REJECTED,
                LLMCoachingDecisionReason.INVALID_REVISION,
            )

        current = set(current_labels)
        newly_detected = set(newly_detected_labels)
        enabled = set(rag_llm_enabled_labels)
        if any(
            label not in CANONICAL_LABELS
            for labels in (current, newly_detected, enabled)
            for label in labels
        ):
            return _result(
                tenant_id,
                call_id,
                revision,
                LLMCoachingDecision.REJECTED,
                LLMCoachingDecisionReason.INVALID_LABEL,
            )
        if any(_has_no_action_conflict(labels) for labels in (current, newly_detected)):
            return _result(
                tenant_id,
                call_id,
                revision,
                LLMCoachingDecision.REJECTED,
                LLMCoachingDecisionReason.NO_ACTION_CONFLICT,
            )
        if not newly_detected.issubset(current):
            return _result(
                tenant_id,
                call_id,
                revision,
                LLMCoachingDecision.REJECTED,
                LLMCoachingDecisionReason.INCONSISTENT_LABEL_STATE,
            )
        if not newly_detected:
            return _result(
                tenant_id,
                call_id,
                revision,
                LLMCoachingDecision.SKIP,
                LLMCoachingDecisionReason.NO_NEW_LABELS,
            )

        business_new = newly_detected - {"no_action"}
        enabled_new = business_new.intersection(enabled)
        sorted_new = tuple(sorted(newly_detected, key=str.casefold))
        if not enabled_new:
            return _result(
                tenant_id,
                call_id,
                revision,
                LLMCoachingDecision.READY_COACHING_ONLY,
                LLMCoachingDecisionReason.LOCAL_COACHING_SUFFICIENT,
                sorted_new,
            )
        if not rag_enabled:
            return _result(
                tenant_id,
                call_id,
                revision,
                LLMCoachingDecision.READY_COACHING_ONLY,
                LLMCoachingDecisionReason.RAG_DISABLED,
                sorted_new,
            )
        if not llm_enabled:
            return _result(
                tenant_id,
                call_id,
                revision,
                LLMCoachingDecision.READY_COACHING_ONLY,
                LLMCoachingDecisionReason.LLM_DISABLED,
                sorted_new,
            )
        return _result(
            tenant_id,
            call_id,
            revision,
            LLMCoachingDecision.REQUEST_RAG_LLM,
            LLMCoachingDecisionReason.RAG_LLM_REQUESTED,
            tuple(sorted(enabled_new, key=str.casefold)),
        )


def _has_no_action_conflict(labels: set[str]) -> bool:
    return "no_action" in labels and len(labels) > 1


def _result(
    tenant_id: str,
    call_id: str,
    revision: int,
    decision: LLMCoachingDecision,
    reason: LLMCoachingDecisionReason,
    trigger_labels: tuple[str, ...] = (),
) -> LLMCoachingDecisionResult:
    return LLMCoachingDecisionResult(
        tenant_id=tenant_id,
        call_id=call_id,
        revision=revision,
        decision=decision,
        trigger_labels=trigger_labels,
        reason=reason,
    )
