import json
import logging

import pytest
from pydantic import ValidationError

from app.calls.models import CallState
from app.coaching.llm_result_gate import (
    MAX_JSON_DEPTH,
    MAX_RAW_OUTPUT_CHARACTERS,
    LLMCitationReference,
    LLMCoachingGateStatus,
    LLMCoachingRejectionReason,
    LLMCoachingResultGate,
)
from app.events.models import (
    CoachingAction,
    CoachingSuggestionSource,
    SuggestionPriority,
)

TENANT_ID = "tenant_alpha"
CALL_ID = "call_001"
REVISION = 7
ALLOWED = {("guide_a", "chunk_1"), ("guide_b", "chunk_2")}


def suggestion_payload(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "decision": "suggestion",
        "tenant_id": TENANT_ID,
        "call_id": CALL_ID,
        "revision": REVISION,
        "action": "RAG_ACTION",
        "title": "Synthetic guidance",
        "suggestion": "Use the approved synthetic guidance.",
        "priority": "HIGH",
        "citations": [{"document_id": "guide_a", "chunk_id": "chunk_1"}],
        "source": "llm",
    }
    values.update(changes)
    return values


def no_suggestion_payload(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "decision": "no_suggestion",
        "tenant_id": TENANT_ID,
        "call_id": CALL_ID,
        "revision": REVISION,
    }
    values.update(changes)
    return values


def evaluate(
    payload: object,
    *,
    allowed: set[tuple[str, str]] = ALLOWED,
):
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    return LLMCoachingResultGate().evaluate(
        tenant_id=TENANT_ID,
        call_id=CALL_ID,
        revision=REVISION,
        raw_output=raw,
        allowed_citations=allowed,
    )


def test_valid_suggestion_is_grounded_and_uses_existing_coaching_enums() -> None:
    result = evaluate(suggestion_payload())

    assert result.status is LLMCoachingGateStatus.VALID_SUGGESTION
    assert result.rejection_reason is None
    assert result.suggestion is not None
    assert result.suggestion.source is CoachingSuggestionSource.LLM
    assert result.suggestion.action is CoachingAction.RAG_ACTION
    assert result.suggestion.priority is SuggestionPriority.HIGH
    assert result.suggestion.citations == (
        LLMCitationReference(document_id="guide_a", chunk_id="chunk_1"),
    )
    assert (result.tenant_id, result.call_id, result.revision) == (
        TENANT_ID,
        CALL_ID,
        REVISION,
    )


def test_valid_no_suggestion_has_no_content_or_citations() -> None:
    result = evaluate(no_suggestion_payload(), allowed=set())

    assert result.status is LLMCoachingGateStatus.VALID_NO_SUGGESTION
    assert result.suggestion is None
    assert result.rejection_reason is None


def test_result_is_immutable_and_deterministic() -> None:
    payload = suggestion_payload()
    first = evaluate(payload)
    second = evaluate(payload)

    assert first == second
    with pytest.raises(ValidationError):
        first.status = LLMCoachingGateStatus.REJECTED
    assert first.suggestion is not None
    with pytest.raises(ValidationError):
        first.suggestion.title = "Changed"


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        " ",
        "{",
        "```json\n{}\n```",
        'explanation {"decision":"no_suggestion"}',
        '{"decision":"no_suggestion"} explanation',
        "{} {}",
        "[]",
        '{"decision":"suggestion","revision":NaN}',
        '{"decision":"suggestion","revision":Infinity}',
    ],
)
def test_invalid_json_forms_are_rejected(raw: str | None) -> None:
    result = LLMCoachingResultGate().evaluate(
        tenant_id=TENANT_ID,
        call_id=CALL_ID,
        revision=REVISION,
        raw_output=raw,
        allowed_citations=ALLOWED,
    )

    assert result.status is LLMCoachingGateStatus.REJECTED
    assert result.rejection_reason is LLMCoachingRejectionReason.INVALID_JSON


def test_duplicate_json_keys_are_rejected_at_any_level() -> None:
    raw = (
        '{"decision":"suggestion","tenant_id":"tenant_alpha",'
        '"call_id":"call_001","revision":7,"action":"RAG_ACTION",'
        '"title":"Synthetic","suggestion":"Synthetic","priority":"HIGH",'
        '"source":"llm","citations":[{"document_id":"guide_a",'
        '"document_id":"guide_b","chunk_id":"chunk_1"}]}'
    )

    result = evaluate(raw)

    assert result.rejection_reason is LLMCoachingRejectionReason.DUPLICATE_KEY


@pytest.mark.parametrize(
    "changes",
    [
        {"extra": "not allowed"},
        {"revision": "7"},
        {"citations": "guide_a:chunk_1"},
        {"priority": 1},
        {"source": "rule"},
        {"action": "UNKNOWN"},
    ],
)
def test_extra_and_wrong_type_fields_fail_schema(
    changes: dict[str, object],
) -> None:
    result = evaluate(suggestion_payload(**changes))

    assert result.rejection_reason is (
        LLMCoachingRejectionReason.SCHEMA_VALIDATION_FAILED
    )


def test_unsupported_decision_has_specific_reason() -> None:
    result = evaluate(no_suggestion_payload(decision="other"))

    assert result.rejection_reason is (LLMCoachingRejectionReason.UNSUPPORTED_DECISION)


def test_payload_character_limit_is_enforced_before_parsing() -> None:
    result = evaluate("x" * (MAX_RAW_OUTPUT_CHARACTERS + 1))

    assert result.rejection_reason is LLMCoachingRejectionReason.PAYLOAD_TOO_LARGE


def test_payload_depth_limit_is_enforced() -> None:
    nested: object = "leaf"
    for _ in range(MAX_JSON_DEPTH + 1):
        nested = {"nested": nested}

    result = evaluate(nested)

    assert result.rejection_reason is LLMCoachingRejectionReason.PAYLOAD_TOO_DEEP


@pytest.mark.parametrize(
    "changes",
    [
        {"tenant_id": "tenant_beta"},
        {"call_id": "call_002"},
        {"revision": REVISION + 1},
    ],
)
def test_scope_mismatch_is_rejected(changes: dict[str, object]) -> None:
    result = evaluate(suggestion_payload(**changes))

    assert result.rejection_reason is LLMCoachingRejectionReason.SCOPE_MISMATCH
    assert (result.tenant_id, result.call_id, result.revision) == (
        TENANT_ID,
        CALL_ID,
        REVISION,
    )


@pytest.mark.parametrize("missing", ["tenant_id", "call_id", "revision"])
def test_missing_scope_fails_schema(missing: str) -> None:
    payload = suggestion_payload()
    del payload[missing]

    result = evaluate(payload)

    assert result.rejection_reason is (
        LLMCoachingRejectionReason.SCHEMA_VALIDATION_FAILED
    )


def test_nested_scope_field_in_citation_is_rejected_as_extra() -> None:
    citation = {
        "document_id": "guide_a",
        "chunk_id": "chunk_1",
        "tenant_id": "tenant_beta",
    }

    result = evaluate(suggestion_payload(citations=[citation]))

    assert result.rejection_reason is (
        LLMCoachingRejectionReason.SCHEMA_VALIDATION_FAILED
    )


@pytest.mark.parametrize(
    ("citation", "allowed"),
    [
        ({"document_id": "invented", "chunk_id": "chunk_1"}, ALLOWED),
        ({"document_id": "guide_a", "chunk_id": "invented"}, ALLOWED),
        (
            {"document_id": "guide_a", "chunk_id": "chunk_2"},
            ALLOWED,
        ),
        ({"document_id": "guide_a", "chunk_id": "chunk_1"}, set()),
    ],
)
def test_ungrounded_citations_are_rejected(
    citation: dict[str, str],
    allowed: set[tuple[str, str]],
) -> None:
    result = evaluate(suggestion_payload(citations=[citation]), allowed=allowed)

    assert result.rejection_reason is (LLMCoachingRejectionReason.CITATION_NOT_ALLOWED)


def test_duplicate_citation_is_rejected() -> None:
    citation = {"document_id": "guide_a", "chunk_id": "chunk_1"}
    result = evaluate(suggestion_payload(citations=[citation, citation]))

    assert result.rejection_reason is (LLMCoachingRejectionReason.DUPLICATE_CITATION)


def test_suggestion_without_citation_fails_schema() -> None:
    result = evaluate(suggestion_payload(citations=[]))

    assert result.rejection_reason is (
        LLMCoachingRejectionReason.SCHEMA_VALIDATION_FAILED
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"citations": [{"document_id": "guide_a", "chunk_id": "chunk_1"}]},
        {"suggestion": "Unexpected"},
        {"title": "Unexpected"},
        {"source": "llm"},
    ],
)
def test_no_suggestion_rejects_suggestion_content(
    extra: dict[str, object],
) -> None:
    result = evaluate(no_suggestion_payload(**extra))

    assert result.rejection_reason is (
        LLMCoachingRejectionReason.SCHEMA_VALIDATION_FAILED
    )


def test_gate_has_no_call_state_or_logging_side_effects(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = CallState(tenant_id=TENANT_ID, call_id=CALL_ID)
    before = state.model_dump()
    private_output = (
        '{"decision":"suggestion","tenant_id":"tenant_alpha",'
        '"call_id":"call_001","revision":7,"suggestion":'
        '"PRIVATE_SUGGESTION_SENTINEL","private_path":'
        '"PRIVATE_PATH_SENTINEL"}'
    )

    with caplog.at_level(logging.DEBUG):
        result = evaluate(private_output)

    assert result.status is LLMCoachingGateStatus.REJECTED
    assert state.model_dump() == before
    assert "PRIVATE_SUGGESTION_SENTINEL" not in caplog.text
    assert "PRIVATE_PATH_SENTINEL" not in caplog.text
    assert private_output not in caplog.text
