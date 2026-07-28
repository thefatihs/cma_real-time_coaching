"""Conservative deterministic speaker-role resolution."""

from enum import Enum
from math import isfinite
from typing import Protocol, Self
from unicodedata import combining, normalize

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.diarization.models import SpeakerRole


DEFAULT_AGENT_PHRASES = (
    "merhaba ben",
    "size nasil yardimci olabilirim",
    "kontrol ediyorum",
    "isleminizi gerceklestiriyorum",
)
DEFAULT_CUSTOMER_PHRASES = (
    "yardim istiyorum",
    "sorun yasiyorum",
    "iptal etmek istiyorum",
    "sikayetim var",
)


class RoleEvidenceCode(str, Enum):
    STRONG_AGENT = "strong_agent"
    STRONG_CUSTOMER = "strong_customer"
    WEAK_POSITIONAL = "weak_positional"
    CONFLICTING = "conflicting"
    INFERRED_OPPOSITE = "inferred_opposite"
    INSUFFICIENT = "insufficient"
    MISSING_GLOBAL_ID = "missing_global_id"
    OVERLAP_IGNORED = "overlap_ignored"


class SpeakerRoleResolutionErrorCategory(str, Enum):
    INVALID_SCOPE = "invalid_scope"
    INVALID_REVISION = "invalid_revision"
    INVALID_RANGE = "invalid_range"
    SCOPE_MISMATCH = "scope_mismatch"
    REVISION_MISMATCH = "revision_mismatch"
    DUPLICATE_OR_CONFLICTING_SPAN = "duplicate_or_conflicting_span"
    TEXT_LIMIT_EXCEEDED = "text_limit_exceeded"


class SpeakerRoleResolutionError(ValueError):
    """Privacy-safe role resolver boundary error."""

    def __init__(self, category: SpeakerRoleResolutionErrorCategory) -> None:
        self.category = category
        super().__init__(category.value)


class SpeakerAttributedTextSpan(BaseModel):
    """Immutable bounded text evidence attributed to global speakers."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    tenant_id: str
    call_id: str
    transcript_revision: int
    start_seconds: float
    end_seconds: float
    global_speaker_ids: tuple[str, ...] = ()
    role: SpeakerRole = SpeakerRole.UNKNOWN
    text: str = Field(repr=False)

    @field_validator("tenant_id", "call_id")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("invalid_scope")
        return cleaned

    @field_validator("transcript_revision")
    @classmethod
    def validate_revision(cls, value: int) -> int:
        if value < 0:
            raise ValueError("invalid_revision")
        return value

    @field_validator("start_seconds", "end_seconds")
    @classmethod
    def validate_timestamp(cls, value: float) -> float:
        if not isfinite(value) or value < 0:
            raise ValueError("invalid_timestamp")
        return value

    @field_validator("global_speaker_ids")
    @classmethod
    def validate_global_speakers(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("invalid_global_speaker_ids")
        return cleaned

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("invalid_text")
        return value

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("invalid_range")
        if self.role is SpeakerRole.OVERLAP:
            if len(self.global_speaker_ids) < 2:
                raise ValueError("overlap_requires_multiple_speakers")
        elif len(self.global_speaker_ids) > 1:
            raise ValueError("non_overlap_requires_at_most_one_speaker")
        return self


class SpeakerRoleResolutionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    call_id: str
    transcript_revision: int
    spans: tuple[SpeakerAttributedTextSpan, ...]

    @field_validator("tenant_id", "call_id")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("invalid_scope")
        return cleaned

    @field_validator("transcript_revision")
    @classmethod
    def validate_revision(cls, value: int) -> int:
        if value < 0:
            raise ValueError("invalid_revision")
        return value


class SpeakerRoleAssignment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    global_speaker_id: str | None
    role: SpeakerRole
    confidence: float | None = None
    evidence: RoleEvidenceCode


class SpeakerRoleResolutionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    call_id: str
    transcript_revision: int
    assignments: tuple[SpeakerRoleAssignment, ...]
    ignored_evidence: tuple[RoleEvidenceCode, ...] = ()


class SpeakerRoleResolverProtocol(Protocol):
    def resolve(
        self,
        request: SpeakerRoleResolutionRequest,
    ) -> SpeakerRoleResolutionResult: ...


class RuleBasedSpeakerRoleResolver:
    """Resolve roles from bounded normalized phrases without retaining text."""

    def __init__(
        self,
        *,
        agent_phrases: tuple[str, ...] = DEFAULT_AGENT_PHRASES,
        customer_phrases: tuple[str, ...] = DEFAULT_CUSTOMER_PHRASES,
        agent_threshold: int = 1,
        customer_threshold: int = 1,
        max_text_characters: int = 4096,
        opening_seconds: float = 8.0,
        inference_min_confidence: float = 0.9,
    ) -> None:
        if (
            agent_threshold <= 0
            or customer_threshold <= 0
            or max_text_characters <= 0
            or not isfinite(opening_seconds)
            or opening_seconds < 0
            or not 0.0 <= inference_min_confidence <= 1.0
        ):
            raise ValueError("invalid_role_resolver_configuration")
        self._agent_phrases = _normalize_phrases(agent_phrases)
        self._customer_phrases = _normalize_phrases(customer_phrases)
        self._agent_threshold = agent_threshold
        self._customer_threshold = customer_threshold
        self._max_text_characters = max_text_characters
        self._opening_seconds = opening_seconds
        self._inference_min_confidence = inference_min_confidence

    def resolve(
        self,
        request: SpeakerRoleResolutionRequest,
    ) -> SpeakerRoleResolutionResult:
        spans = self._validate_and_order(request)
        ignored: set[RoleEvidenceCode] = {
            RoleEvidenceCode.OVERLAP_IGNORED
            for span in spans
            if span.role is SpeakerRole.OVERLAP
        }
        speaker_ids = sorted(
            {
                speaker_id
                for span in spans
                if span.role is not SpeakerRole.OVERLAP
                for speaker_id in span.global_speaker_ids
            }
        )
        if any(not span.global_speaker_ids for span in spans):
            speaker_ids_with_missing: list[str | None] = [*speaker_ids, None]
        else:
            speaker_ids_with_missing = [*speaker_ids]

        scores: dict[str, tuple[int, int, bool]] = {}
        earliest_start = min((span.start_seconds for span in spans), default=0.0)
        for speaker_id in speaker_ids:
            agent_score = 0
            customer_score = 0
            weak_position = False
            for span in spans:
                if span.role is SpeakerRole.OVERLAP:
                    continue
                if span.global_speaker_ids != (speaker_id,):
                    continue
                normalized_text = _normalize_text(span.text)
                agent_score += sum(
                    phrase in normalized_text for phrase in self._agent_phrases
                )
                customer_score += sum(
                    phrase in normalized_text for phrase in self._customer_phrases
                )
                if span.start_seconds <= earliest_start + self._opening_seconds:
                    weak_position = True
            scores[speaker_id] = (agent_score, customer_score, weak_position)

        assignments: list[SpeakerRoleAssignment] = []
        for speaker_id in speaker_ids_with_missing:
            score = None if speaker_id is None else scores[speaker_id]
            assignments.append(self._direct_assignment(speaker_id, score))
        assignments = self._infer_opposite(assignments, scores)
        return SpeakerRoleResolutionResult(
            tenant_id=request.tenant_id,
            call_id=request.call_id,
            transcript_revision=request.transcript_revision,
            assignments=tuple(assignments),
            ignored_evidence=tuple(sorted(ignored, key=lambda item: item.value)),
        )

    def _validate_and_order(
        self,
        request: SpeakerRoleResolutionRequest,
    ) -> tuple[SpeakerAttributedTextSpan, ...]:
        total_characters = 0
        seen: dict[tuple[float, float, tuple[str, ...]], SpeakerAttributedTextSpan] = {}
        for span in request.spans:
            if span.tenant_id != request.tenant_id or span.call_id != request.call_id:
                raise SpeakerRoleResolutionError(
                    SpeakerRoleResolutionErrorCategory.SCOPE_MISMATCH
                )
            if span.transcript_revision != request.transcript_revision:
                raise SpeakerRoleResolutionError(
                    SpeakerRoleResolutionErrorCategory.REVISION_MISMATCH
                )
            total_characters += len(span.text)
            key = (span.start_seconds, span.end_seconds, span.global_speaker_ids)
            previous = seen.get(key)
            if previous is not None:
                raise SpeakerRoleResolutionError(
                    SpeakerRoleResolutionErrorCategory.DUPLICATE_OR_CONFLICTING_SPAN
                )
            seen[key] = span
        if total_characters > self._max_text_characters:
            raise SpeakerRoleResolutionError(
                SpeakerRoleResolutionErrorCategory.TEXT_LIMIT_EXCEEDED
            )
        return tuple(
            sorted(
                request.spans,
                key=lambda span: (
                    span.start_seconds,
                    span.end_seconds,
                    span.global_speaker_ids,
                ),
            )
        )

    def _direct_assignment(
        self,
        speaker_id: str | None,
        score: tuple[int, int, bool] | None,
    ) -> SpeakerRoleAssignment:
        if speaker_id is None:
            return SpeakerRoleAssignment(
                global_speaker_id=None,
                role=SpeakerRole.UNKNOWN,
                evidence=RoleEvidenceCode.MISSING_GLOBAL_ID,
            )
        assert score is not None
        agent_score, customer_score, weak_position = score
        if agent_score > 0 and customer_score > 0:
            return SpeakerRoleAssignment(
                global_speaker_id=speaker_id,
                role=SpeakerRole.UNKNOWN,
                evidence=RoleEvidenceCode.CONFLICTING,
            )
        if agent_score >= self._agent_threshold:
            return SpeakerRoleAssignment(
                global_speaker_id=speaker_id,
                role=SpeakerRole.AGENT,
                confidence=min(1.0, agent_score / self._agent_threshold),
                evidence=RoleEvidenceCode.STRONG_AGENT,
            )
        if customer_score >= self._customer_threshold:
            return SpeakerRoleAssignment(
                global_speaker_id=speaker_id,
                role=SpeakerRole.CUSTOMER,
                confidence=min(1.0, customer_score / self._customer_threshold),
                evidence=RoleEvidenceCode.STRONG_CUSTOMER,
            )
        return SpeakerRoleAssignment(
            global_speaker_id=speaker_id,
            role=SpeakerRole.UNKNOWN,
            evidence=(
                RoleEvidenceCode.WEAK_POSITIONAL
                if weak_position
                else RoleEvidenceCode.INSUFFICIENT
            ),
        )

    def _infer_opposite(
        self,
        assignments: list[SpeakerRoleAssignment],
        scores: dict[str, tuple[int, int, bool]],
    ) -> list[SpeakerRoleAssignment]:
        identified = [
            assignment
            for assignment in assignments
            if assignment.global_speaker_id is not None
        ]
        if len(identified) != 2:
            return assignments
        resolved = [
            assignment
            for assignment in identified
            if assignment.role in (SpeakerRole.AGENT, SpeakerRole.CUSTOMER)
            and (assignment.confidence or 0.0) >= self._inference_min_confidence
        ]
        unknown = [
            assignment
            for assignment in identified
            if assignment.role is SpeakerRole.UNKNOWN
        ]
        if len(resolved) != 1 or len(unknown) != 1:
            return assignments
        other = unknown[0]
        assert other.global_speaker_id is not None
        agent_score, customer_score, _ = scores[other.global_speaker_id]
        inferred_role = (
            SpeakerRole.CUSTOMER
            if resolved[0].role is SpeakerRole.AGENT
            else SpeakerRole.AGENT
        )
        conflicting_score = (
            agent_score if inferred_role is SpeakerRole.CUSTOMER else customer_score
        )
        if conflicting_score > 0 or other.evidence is RoleEvidenceCode.CONFLICTING:
            return assignments
        replacement = SpeakerRoleAssignment(
            global_speaker_id=other.global_speaker_id,
            role=inferred_role,
            confidence=self._inference_min_confidence,
            evidence=RoleEvidenceCode.INFERRED_OPPOSITE,
        )
        return [
            replacement if assignment is other else assignment
            for assignment in assignments
        ]


def _normalize_phrases(phrases: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(_normalize_text(phrase) for phrase in phrases)
    if not normalized or any(not phrase for phrase in normalized):
        raise ValueError("invalid_role_evidence_phrases")
    return tuple(dict.fromkeys(normalized))


def _normalize_text(text: str) -> str:
    translation = str.maketrans(
        {
            "ç": "c",
            "ğ": "g",
            "ı": "i",
            "ö": "o",
            "ş": "s",
            "ü": "u",
        }
    )
    normalized = "".join(
        character
        for character in normalize("NFKD", text.casefold())
        if not combining(character)
    ).translate(translation)
    return " ".join(
        "".join(
            character if character.isalnum() else " " for character in normalized
        ).split()
    )
