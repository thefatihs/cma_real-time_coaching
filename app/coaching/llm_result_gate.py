"""Pure validation and grounding gate for untrusted LLM coaching JSON."""

from collections.abc import Collection
from enum import Enum
import json
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from app.events.models import (
    CoachingAction,
    CoachingSuggestionSource,
    SuggestionPriority,
)

MAX_RAW_OUTPUT_CHARACTERS = 8_192
MAX_JSON_DEPTH = 8
MAX_CITATIONS = 20


def coaching_wire_json_schema() -> dict[str, object]:
    """Return a fresh flattened schema for the untrusted coaching wire payload."""
    citation = {
        "type": "object",
        "properties": {
            "document_id": {"type": "string", "minLength": 1},
            "chunk_id": {"type": "string", "minLength": 1},
        },
        "required": ["document_id", "chunk_id"],
        "additionalProperties": False,
    }
    suggestion = {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["suggestion"]},
            "tenant_id": {"type": "string", "minLength": 1},
            "call_id": {"type": "string", "minLength": 1},
            "revision": {"type": "integer", "minimum": 0},
            "action": {
                "type": "string",
                "enum": [item.value for item in CoachingAction],
            },
            "title": {"type": "string", "minLength": 1, "maxLength": 120},
            "suggestion": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
            },
            "priority": {
                "type": "string",
                "enum": [item.value for item in SuggestionPriority],
            },
            "citations": {
                "type": "array",
                "items": citation,
                "minItems": 1,
                "maxItems": MAX_CITATIONS,
            },
            "source": {"type": "string", "enum": ["llm"]},
        },
        "required": [
            "decision",
            "tenant_id",
            "call_id",
            "revision",
            "action",
            "title",
            "suggestion",
            "priority",
            "citations",
            "source",
        ],
        "additionalProperties": False,
    }
    no_suggestion = {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["no_suggestion"]},
            "tenant_id": {"type": "string", "minLength": 1},
            "call_id": {"type": "string", "minLength": 1},
            "revision": {"type": "integer", "minimum": 0},
        },
        "required": ["decision", "tenant_id", "call_id", "revision"],
        "additionalProperties": False,
    }
    return {"oneOf": [suggestion, no_suggestion]}


class LLMCoachingGateStatus(str, Enum):
    VALID_SUGGESTION = "valid_suggestion"
    VALID_NO_SUGGESTION = "valid_no_suggestion"
    REJECTED = "rejected"


class LLMCoachingRejectionReason(str, Enum):
    INVALID_JSON = "invalid_json"
    DUPLICATE_KEY = "duplicate_key"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    PAYLOAD_TOO_DEEP = "payload_too_deep"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    SCOPE_MISMATCH = "scope_mismatch"
    UNSUPPORTED_DECISION = "unsupported_decision"
    CITATION_NOT_ALLOWED = "citation_not_allowed"
    DUPLICATE_CITATION = "duplicate_citation"


class LLMCitationReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: StrictStr
    chunk_id: StrictStr

    @field_validator("document_id", "chunk_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("citation identifiers cannot be empty")
        return cleaned

    @property
    def identity(self) -> tuple[str, str]:
        return (self.document_id, self.chunk_id)


class ValidatedLLMCoachingSuggestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    call_id: str
    revision: int
    action: CoachingAction
    title: str
    suggestion: str
    priority: SuggestionPriority
    citations: tuple[LLMCitationReference, ...]
    source: CoachingSuggestionSource = CoachingSuggestionSource.LLM

    @model_validator(mode="after")
    def validate_source_and_citations(self) -> Self:
        if self.source is not CoachingSuggestionSource.LLM:
            raise ValueError("source must be llm")
        if not self.citations:
            raise ValueError("suggestion requires at least one citation")
        return self


class LLMCoachingGateResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: LLMCoachingGateStatus
    tenant_id: str
    call_id: str
    revision: int
    suggestion: ValidatedLLMCoachingSuggestion | None = None
    rejection_reason: LLMCoachingRejectionReason | None = None

    @model_validator(mode="after")
    def validate_result_shape(self) -> Self:
        if self.status is LLMCoachingGateStatus.VALID_SUGGESTION:
            if self.suggestion is None or self.rejection_reason is not None:
                raise ValueError("valid suggestion result shape is invalid")
        elif self.status is LLMCoachingGateStatus.VALID_NO_SUGGESTION:
            if self.suggestion is not None or self.rejection_reason is not None:
                raise ValueError("valid no-suggestion result shape is invalid")
        elif self.suggestion is not None or self.rejection_reason is None:
            raise ValueError("rejected result shape is invalid")
        return self


class _SuggestionPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: Literal["suggestion"]
    tenant_id: StrictStr
    call_id: StrictStr
    revision: StrictInt
    action: CoachingAction
    title: StrictStr = Field(max_length=120)
    suggestion: StrictStr = Field(max_length=500)
    priority: SuggestionPriority
    citations: tuple[LLMCitationReference, ...] = Field(
        min_length=1,
        max_length=MAX_CITATIONS,
    )
    source: Literal["llm"]

    @field_validator("tenant_id", "call_id", "title", "suggestion")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("required text cannot be empty")
        return cleaned

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: int) -> int:
        if value < 0:
            raise ValueError("revision cannot be negative")
        return value


class _NoSuggestionPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: Literal["no_suggestion"]
    tenant_id: StrictStr
    call_id: StrictStr
    revision: StrictInt

    @field_validator("tenant_id", "call_id")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("required text cannot be empty")
        return cleaned

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: int) -> int:
        if value < 0:
            raise ValueError("revision cannot be negative")
        return value


class _DuplicateKeyError(ValueError):
    pass


class _NonFiniteNumberError(ValueError):
    pass


class LLMCoachingResultGate:
    def evaluate(
        self,
        *,
        tenant_id: str,
        call_id: str,
        revision: int,
        raw_output: str | None,
        allowed_citations: Collection[tuple[str, str]],
    ) -> LLMCoachingGateResult:
        trusted_tenant = _required_scope(tenant_id, "tenant_id")
        trusted_call = _required_scope(call_id, "call_id")
        if revision < 0:
            raise ValueError("revision cannot be negative")

        if raw_output is None or not raw_output.strip():
            return _rejected(
                trusted_tenant,
                trusted_call,
                revision,
                LLMCoachingRejectionReason.INVALID_JSON,
            )
        if len(raw_output) > MAX_RAW_OUTPUT_CHARACTERS:
            return _rejected(
                trusted_tenant,
                trusted_call,
                revision,
                LLMCoachingRejectionReason.PAYLOAD_TOO_LARGE,
            )

        try:
            payload = json.loads(
                raw_output,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_non_finite,
            )
        except _DuplicateKeyError:
            return _rejected(
                trusted_tenant,
                trusted_call,
                revision,
                LLMCoachingRejectionReason.DUPLICATE_KEY,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return _rejected(
                trusted_tenant,
                trusted_call,
                revision,
                LLMCoachingRejectionReason.INVALID_JSON,
            )

        if not isinstance(payload, dict):
            return _rejected(
                trusted_tenant,
                trusted_call,
                revision,
                LLMCoachingRejectionReason.INVALID_JSON,
            )
        if _json_depth(payload) > MAX_JSON_DEPTH:
            return _rejected(
                trusted_tenant,
                trusted_call,
                revision,
                LLMCoachingRejectionReason.PAYLOAD_TOO_DEEP,
            )

        decision = payload.get("decision")
        if decision not in {"suggestion", "no_suggestion"}:
            return _rejected(
                trusted_tenant,
                trusted_call,
                revision,
                LLMCoachingRejectionReason.UNSUPPORTED_DECISION,
            )
        model = _SuggestionPayload if decision == "suggestion" else _NoSuggestionPayload
        try:
            validated = model.model_validate(payload)
        except ValidationError:
            return _rejected(
                trusted_tenant,
                trusted_call,
                revision,
                LLMCoachingRejectionReason.SCHEMA_VALIDATION_FAILED,
            )

        if (
            validated.tenant_id != trusted_tenant
            or validated.call_id != trusted_call
            or validated.revision != revision
        ):
            return _rejected(
                trusted_tenant,
                trusted_call,
                revision,
                LLMCoachingRejectionReason.SCOPE_MISMATCH,
            )
        if isinstance(validated, _NoSuggestionPayload):
            return LLMCoachingGateResult(
                status=LLMCoachingGateStatus.VALID_NO_SUGGESTION,
                tenant_id=trusted_tenant,
                call_id=trusted_call,
                revision=revision,
            )

        identities = tuple(citation.identity for citation in validated.citations)
        if len(identities) != len(set(identities)):
            return _rejected(
                trusted_tenant,
                trusted_call,
                revision,
                LLMCoachingRejectionReason.DUPLICATE_CITATION,
            )
        allowed = set(allowed_citations)
        if any(identity not in allowed for identity in identities):
            return _rejected(
                trusted_tenant,
                trusted_call,
                revision,
                LLMCoachingRejectionReason.CITATION_NOT_ALLOWED,
            )

        suggestion = ValidatedLLMCoachingSuggestion(
            tenant_id=trusted_tenant,
            call_id=trusted_call,
            revision=revision,
            action=validated.action,
            title=validated.title,
            suggestion=validated.suggestion,
            priority=validated.priority,
            citations=validated.citations,
        )
        return LLMCoachingGateResult(
            status=LLMCoachingGateStatus.VALID_SUGGESTION,
            tenant_id=trusted_tenant,
            call_id=trusted_call,
            revision=revision,
            suggestion=suggestion,
        )


def _required_scope(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _reject_non_finite(value: str) -> object:
    del value
    raise _NonFiniteNumberError


def _json_depth(value: object) -> int:
    if isinstance(value, dict):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 0


def _rejected(
    tenant_id: str,
    call_id: str,
    revision: int,
    reason: LLMCoachingRejectionReason,
) -> LLMCoachingGateResult:
    return LLMCoachingGateResult(
        status=LLMCoachingGateStatus.REJECTED,
        tenant_id=tenant_id,
        call_id=call_id,
        revision=revision,
        rejection_reason=reason,
    )
