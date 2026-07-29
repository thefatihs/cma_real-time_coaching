"""Pure role application and customer-speech projection."""

from enum import Enum
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field

from app.diarization.models import DiarizedTranscriptEvent, DiarizedWord, SpeakerRole
from app.diarization.role_resolver import (
    RoleEvidenceCode,
    SpeakerRoleAssignment,
    SpeakerRoleResolutionResult,
)


class CustomerProjectionStatus(str, Enum):
    READY = "ready"
    EMPTY = "empty"


class CustomerProjectionReason(str, Enum):
    TRUSTED_CUSTOMER_SPEECH = "trusted_customer_speech"
    NO_TRUSTED_CUSTOMER_SPEECH = "no_trusted_customer_speech"


class ProjectionExclusionReason(str, Enum):
    AGENT = "agent"
    UNKNOWN_ROLE = "unknown_role"
    OVERLAP = "overlap"
    LOW_CONFIDENCE = "low_confidence"
    MISSING_SPEAKER = "missing_speaker"


class ProjectionExclusionCount(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: ProjectionExclusionReason
    count: int = Field(ge=0)


class DiarizationRoutingErrorCategory(str, Enum):
    INVALID_SCOPE = "invalid_scope"
    INVALID_REVISION = "invalid_revision"
    INVALID_PARENT_RANGE = "invalid_parent_range"
    SCOPE_MISMATCH = "scope_mismatch"
    REVISION_MISMATCH = "revision_mismatch"
    WORD_OUTSIDE_PARENT = "word_outside_parent"
    MALFORMED_WORD = "malformed_word"
    DUPLICATE_OR_CONFLICTING_WORD = "duplicate_or_conflicting_word"
    UNKNOWN_GLOBAL_SPEAKER = "unknown_global_speaker"
    CONFLICTING_ROLE_MAPPING = "conflicting_role_mapping"
    MALFORMED_ROLE_ASSIGNMENT = "malformed_role_assignment"


class DiarizationRoutingError(ValueError):
    """Privacy-safe routing boundary error."""

    def __init__(self, category: DiarizationRoutingErrorCategory) -> None:
        self.category = category
        super().__init__(category.value)


class RoleTaggedWord(BaseModel):
    """Immutable word with an applied trusted role assignment."""

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
    text: str = Field(repr=False)
    local_speaker_ids: tuple[str, ...]
    global_speaker_id: str | None = None
    global_speaker_ids: tuple[str, ...] = ()
    speaker_confidence: float | None = None
    role: SpeakerRole
    role_confidence: float | None = None
    role_evidence: RoleEvidenceCode | None = None


class CustomerSpeechProjection(BaseModel):
    """Immutable trigger-ready projection containing trusted customer speech."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    tenant_id: str
    call_id: str
    transcript_revision: int
    customer_words: tuple[RoleTaggedWord, ...]
    customer_text: str = Field(repr=False)
    customer_start_seconds: float | None = None
    customer_end_seconds: float | None = None
    excluded_agent_word_count: int
    excluded_unknown_word_count: int
    excluded_overlap_word_count: int
    excluded_below_confidence_word_count: int
    status: CustomerProjectionStatus
    reason: CustomerProjectionReason
    exclusion_counts: tuple[ProjectionExclusionCount, ...] = ()


class CustomerSpeechProjectionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    call_id: str
    transcript_revision: int
    event: DiarizedTranscriptEvent
    role_resolution: SpeakerRoleResolutionResult


class CustomerSpeechProjector:
    """Apply resolved roles without performing role inference."""

    def __init__(self, *, trusted_customer_confidence: float = 0.9) -> None:
        if (
            not isfinite(trusted_customer_confidence)
            or not 0.0 <= trusted_customer_confidence <= 1.0
        ):
            raise ValueError("invalid_customer_confidence_threshold")
        self._trusted_customer_confidence = trusted_customer_confidence

    def project(
        self,
        request: CustomerSpeechProjectionRequest,
    ) -> CustomerSpeechProjection:
        tagged_words = self.apply_roles(request)

        customer_words: list[RoleTaggedWord] = []
        excluded_agent = 0
        excluded_unknown = 0
        excluded_overlap = 0
        excluded_below_confidence = 0
        diagnostic_counts = {reason: 0 for reason in ProjectionExclusionReason}
        for word in tagged_words:
            if word.role is SpeakerRole.OVERLAP:
                excluded_overlap += 1
                diagnostic_counts[ProjectionExclusionReason.OVERLAP] += 1
            elif word.role is SpeakerRole.AGENT:
                excluded_agent += 1
                diagnostic_counts[ProjectionExclusionReason.AGENT] += 1
            elif word.role is SpeakerRole.CUSTOMER:
                if (
                    word.global_speaker_id is not None
                    and word.role_confidence is not None
                    and word.role_confidence >= self._trusted_customer_confidence
                ):
                    customer_words.append(word)
                else:
                    excluded_below_confidence += 1
                    diagnostic_counts[ProjectionExclusionReason.LOW_CONFIDENCE] += 1
            else:
                excluded_unknown += 1
                diagnostic_counts[
                    (
                        ProjectionExclusionReason.MISSING_SPEAKER
                        if word.global_speaker_id is None
                        else ProjectionExclusionReason.UNKNOWN_ROLE
                    )
                ] += 1

        has_customer_speech = bool(customer_words)
        return CustomerSpeechProjection(
            tenant_id=request.tenant_id,
            call_id=request.call_id,
            transcript_revision=request.transcript_revision,
            customer_words=tuple(customer_words),
            customer_text=" ".join(word.text for word in customer_words),
            customer_start_seconds=(
                customer_words[0].start_seconds if has_customer_speech else None
            ),
            customer_end_seconds=(
                customer_words[-1].end_seconds if has_customer_speech else None
            ),
            excluded_agent_word_count=excluded_agent,
            excluded_unknown_word_count=excluded_unknown,
            excluded_overlap_word_count=excluded_overlap,
            excluded_below_confidence_word_count=excluded_below_confidence,
            exclusion_counts=tuple(
                ProjectionExclusionCount(
                    reason=reason,
                    count=diagnostic_counts[reason],
                )
                for reason in ProjectionExclusionReason
            ),
            status=(
                CustomerProjectionStatus.READY
                if has_customer_speech
                else CustomerProjectionStatus.EMPTY
            ),
            reason=(
                CustomerProjectionReason.TRUSTED_CUSTOMER_SPEECH
                if has_customer_speech
                else CustomerProjectionReason.NO_TRUSTED_CUSTOMER_SPEECH
            ),
        )

    def apply_roles(
        self,
        request: CustomerSpeechProjectionRequest,
    ) -> tuple[RoleTaggedWord, ...]:
        self._validate_scope(request)
        words = self._validate_and_order_words(request)
        role_map = self._validate_role_map(request.role_resolution, words)
        return tuple(self._apply_role(word, role_map) for word in words)

    @staticmethod
    def _validate_scope(request: CustomerSpeechProjectionRequest) -> None:
        if not request.tenant_id.strip() or not request.call_id.strip():
            raise DiarizationRoutingError(DiarizationRoutingErrorCategory.INVALID_SCOPE)
        if request.transcript_revision < 0:
            raise DiarizationRoutingError(
                DiarizationRoutingErrorCategory.INVALID_REVISION
            )
        event = request.event
        resolution = request.role_resolution
        if (
            event.tenant_id != request.tenant_id
            or event.call_id != request.call_id
            or resolution.tenant_id != request.tenant_id
            or resolution.call_id != request.call_id
        ):
            raise DiarizationRoutingError(
                DiarizationRoutingErrorCategory.SCOPE_MISMATCH
            )
        if (
            event.transcript_revision != request.transcript_revision
            or resolution.transcript_revision != request.transcript_revision
        ):
            raise DiarizationRoutingError(
                DiarizationRoutingErrorCategory.REVISION_MISMATCH
            )
        if (
            not isfinite(event.start_seconds)
            or not isfinite(event.end_seconds)
            or event.start_seconds < 0
            or event.end_seconds <= event.start_seconds
        ):
            raise DiarizationRoutingError(
                DiarizationRoutingErrorCategory.INVALID_PARENT_RANGE
            )

    @staticmethod
    def _validate_and_order_words(
        request: CustomerSpeechProjectionRequest,
    ) -> tuple[DiarizedWord, ...]:
        event = request.event
        seen: set[tuple[float, float, tuple[str, ...], tuple[str, ...]]] = set()
        for word in event.words:
            if (
                word.tenant_id != request.tenant_id
                or word.call_id != request.call_id
                or word.transcript_revision != request.transcript_revision
            ):
                raise DiarizationRoutingError(
                    DiarizationRoutingErrorCategory.SCOPE_MISMATCH
                    if (
                        word.tenant_id != request.tenant_id
                        or word.call_id != request.call_id
                    )
                    else DiarizationRoutingErrorCategory.REVISION_MISMATCH
                )
            if (
                not isfinite(word.start_seconds)
                or not isfinite(word.end_seconds)
                or word.start_seconds < 0
                or word.end_seconds <= word.start_seconds
                or word.speaker_confidence is not None
                and (
                    not isfinite(word.speaker_confidence)
                    or not 0.0 <= word.speaker_confidence <= 1.0
                )
            ):
                raise DiarizationRoutingError(
                    DiarizationRoutingErrorCategory.MALFORMED_WORD
                )
            if (
                word.start_seconds < event.start_seconds
                or word.end_seconds > event.end_seconds
            ):
                raise DiarizationRoutingError(
                    DiarizationRoutingErrorCategory.WORD_OUTSIDE_PARENT
                )
            key = (
                word.start_seconds,
                word.end_seconds,
                word.local_speaker_ids,
                word.global_speaker_ids,
            )
            if key in seen:
                raise DiarizationRoutingError(
                    DiarizationRoutingErrorCategory.DUPLICATE_OR_CONFLICTING_WORD
                )
            seen.add(key)
        return tuple(
            sorted(
                event.words,
                key=lambda word: (
                    word.start_seconds,
                    word.end_seconds,
                    word.local_speaker_ids,
                    word.global_speaker_ids,
                ),
            )
        )

    @staticmethod
    def _validate_role_map(
        resolution: SpeakerRoleResolutionResult,
        words: tuple[DiarizedWord, ...],
    ) -> dict[str, SpeakerRoleAssignment]:
        known_speakers = {
            speaker_id
            for word in words
            for speaker_id in (
                word.global_speaker_ids
                or ((word.global_speaker_id,) if word.global_speaker_id else ())
            )
        }
        role_map: dict[str, SpeakerRoleAssignment] = {}
        for assignment in resolution.assignments:
            speaker_id = assignment.global_speaker_id
            if speaker_id is None:
                continue
            if speaker_id not in known_speakers:
                raise DiarizationRoutingError(
                    DiarizationRoutingErrorCategory.UNKNOWN_GLOBAL_SPEAKER
                )
            if assignment.confidence is not None and (
                not isfinite(assignment.confidence)
                or not 0.0 <= assignment.confidence <= 1.0
            ):
                raise DiarizationRoutingError(
                    DiarizationRoutingErrorCategory.MALFORMED_ROLE_ASSIGNMENT
                )
            previous = role_map.get(speaker_id)
            if previous is not None:
                raise DiarizationRoutingError(
                    DiarizationRoutingErrorCategory.CONFLICTING_ROLE_MAPPING
                )
            role_map[speaker_id] = assignment
        return role_map

    @staticmethod
    def _apply_role(
        word: DiarizedWord,
        role_map: dict[str, SpeakerRoleAssignment],
    ) -> RoleTaggedWord:
        if word.role is SpeakerRole.OVERLAP or len(word.global_speaker_ids) > 1:
            role = SpeakerRole.OVERLAP
            confidence = None
            evidence = None
        elif word.global_speaker_id is None:
            role = SpeakerRole.UNKNOWN
            confidence = None
            evidence = None
        else:
            assignment = role_map.get(word.global_speaker_id)
            role = assignment.role if assignment is not None else SpeakerRole.UNKNOWN
            confidence = assignment.confidence if assignment is not None else None
            evidence = assignment.evidence if assignment is not None else None
        return RoleTaggedWord(
            tenant_id=word.tenant_id,
            call_id=word.call_id,
            transcript_revision=word.transcript_revision,
            start_seconds=word.start_seconds,
            end_seconds=word.end_seconds,
            text=word.text,
            local_speaker_ids=word.local_speaker_ids,
            global_speaker_id=word.global_speaker_id,
            global_speaker_ids=word.global_speaker_ids,
            speaker_confidence=word.speaker_confidence,
            role=role,
            role_confidence=confidence,
            role_evidence=evidence,
        )
