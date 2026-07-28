"""Safe offline composition of injectable diarization components."""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Protocol

from app.asr.models import ASRWordTimestamp
from app.diarization.alignment import WordAlignmentRequest, align_words_to_speakers
from app.diarization.identity_tracker import (
    SpeakerIdentityTracker,
    SpeakerIdentityTrackingRequest,
)
from app.diarization.models import (
    DiarizationRequest,
    DiarizationResult,
    DiarizationTurn,
    DiarizedTranscriptEvent,
    DiarizedWord,
)
from app.diarization.protocols import SpeakerDiarizerProtocol
from app.diarization.role_resolver import (
    SpeakerAttributedTextSpan,
    SpeakerRoleResolutionRequest,
    SpeakerRoleResolutionResult,
    SpeakerRoleResolverProtocol,
)
from app.diarization.routing import (
    CustomerSpeechProjection,
    CustomerSpeechProjectionRequest,
    RoleTaggedWord,
)


class CustomerSpeechProjectorProtocol(Protocol):
    def apply_roles(
        self,
        request: CustomerSpeechProjectionRequest,
    ) -> tuple[RoleTaggedWord, ...]: ...

    def project(
        self,
        request: CustomerSpeechProjectionRequest,
    ) -> CustomerSpeechProjection: ...


class DiarizationCompositionStatus(str, Enum):
    COMPLETED = "completed"
    EMPTY = "empty"
    REJECTED = "rejected"
    FAILED_SAFE = "failed_safe"


class DiarizationCompositionReason(str, Enum):
    COMPOSED = "composed"
    NO_CUSTOMER_SPEECH = "no_customer_speech"
    INVALID_REQUEST_SCOPE = "invalid_request_scope"
    INVALID_REQUEST_REVISION = "invalid_request_revision"
    INVALID_REQUEST_WINDOW = "invalid_request_window"
    INVALID_INPUT_WORD = "invalid_input_word"
    DIARIZER_FAILED = "diarizer_failed"
    DIARIZER_OUTPUT_INVALID = "diarizer_output_invalid"
    TRACKING_FAILED = "tracking_failed"
    ALIGNMENT_FAILED = "alignment_failed"
    ROLE_RESOLUTION_FAILED = "role_resolution_failed"
    ROLE_RESOLUTION_OUTPUT_INVALID = "role_resolution_output_invalid"
    PROJECTION_FAILED = "projection_failed"
    PROJECTION_OUTPUT_INVALID = "projection_output_invalid"


@dataclass(frozen=True, slots=True)
class DiarizationCompositionRequest:
    tenant_id: str
    call_id: str
    transcript_revision: int
    diarization_request: DiarizationRequest
    words: tuple[ASRWordTimestamp, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class DiarizationCompositionOutcome:
    status: DiarizationCompositionStatus
    reason: DiarizationCompositionReason
    tenant_id: str
    call_id: str
    transcript_revision: int
    tracked_turns: tuple[DiarizationTurn, ...] = ()
    diarized_words: tuple[DiarizedWord, ...] = field(default=(), repr=False)
    role_resolution: SpeakerRoleResolutionResult | None = None
    role_tagged_words: tuple[RoleTaggedWord, ...] = field(default=(), repr=False)
    customer_projection: CustomerSpeechProjection | None = None


class _CompositionBoundaryError(ValueError):
    def __init__(self, reason: DiarizationCompositionReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


class OfflineDiarizationComposer:
    """Compose existing stages transactionally without runtime side effects."""

    def __init__(
        self,
        *,
        diarizer: SpeakerDiarizerProtocol,
        identity_tracker: SpeakerIdentityTracker,
        role_resolver: SpeakerRoleResolverProtocol,
        customer_projector: CustomerSpeechProjectorProtocol,
    ) -> None:
        self._diarizer = diarizer
        self._identity_tracker = identity_tracker
        self._role_resolver = role_resolver
        self._customer_projector = customer_projector

    def compose(
        self,
        request: DiarizationCompositionRequest,
    ) -> DiarizationCompositionOutcome:
        rejection = self._validate_request(request)
        if rejection is not None:
            return self._outcome(
                request,
                DiarizationCompositionStatus.REJECTED,
                rejection,
            )
        checkpoint = self._identity_tracker.checkpoint(
            tenant_id=request.tenant_id,
            call_id=request.call_id,
        )
        step = DiarizationCompositionReason.DIARIZER_FAILED
        try:
            diarization_result = self._diarizer.diarize(request.diarization_request)
            self._validate_diarization_result(request, diarization_result)

            step = DiarizationCompositionReason.TRACKING_FAILED
            tracked_turns = self._identity_tracker.track(
                SpeakerIdentityTrackingRequest(
                    tenant_id=request.tenant_id,
                    call_id=request.call_id,
                    window_start_seconds=(
                        request.diarization_request.window_start_seconds
                    ),
                    window_end_seconds=request.diarization_request.window_end_seconds,
                    turns=diarization_result.turns,
                )
            )
            self._validate_tracked_turns(request, tracked_turns)

            step = DiarizationCompositionReason.ALIGNMENT_FAILED
            diarized_words = align_words_to_speakers(
                WordAlignmentRequest(
                    tenant_id=request.tenant_id,
                    call_id=request.call_id,
                    transcript_revision=request.transcript_revision,
                    parent_start_seconds=(
                        request.diarization_request.window_start_seconds
                    ),
                    parent_end_seconds=(request.diarization_request.window_end_seconds),
                    words=request.words,
                    turns=tracked_turns,
                )
            )
            self._validate_diarized_words(request, diarized_words)

            step = DiarizationCompositionReason.ROLE_RESOLUTION_FAILED
            role_resolution = self._role_resolver.resolve(
                SpeakerRoleResolutionRequest(
                    tenant_id=request.tenant_id,
                    call_id=request.call_id,
                    transcript_revision=request.transcript_revision,
                    spans=tuple(
                        SpeakerAttributedTextSpan(
                            tenant_id=word.tenant_id,
                            call_id=word.call_id,
                            transcript_revision=word.transcript_revision,
                            start_seconds=word.start_seconds,
                            end_seconds=word.end_seconds,
                            global_speaker_ids=(
                                word.global_speaker_ids
                                or (
                                    (word.global_speaker_id,)
                                    if word.global_speaker_id is not None
                                    else ()
                                )
                            ),
                            role=word.role,
                            text=word.text,
                        )
                        for word in diarized_words
                    ),
                )
            )
            self._validate_role_resolution(request, role_resolution)

            event = DiarizedTranscriptEvent(
                tenant_id=request.tenant_id,
                call_id=request.call_id,
                transcript_event_id=(
                    f"offline-diarization-{request.transcript_revision}"
                ),
                transcript_revision=request.transcript_revision,
                start_seconds=request.diarization_request.window_start_seconds,
                end_seconds=request.diarization_request.window_end_seconds,
                turns=tracked_turns,
                words=diarized_words,
            )
            projection_request = CustomerSpeechProjectionRequest(
                tenant_id=request.tenant_id,
                call_id=request.call_id,
                transcript_revision=request.transcript_revision,
                event=event,
                role_resolution=role_resolution,
            )
            step = DiarizationCompositionReason.PROJECTION_FAILED
            role_tagged_words = self._customer_projector.apply_roles(projection_request)
            customer_projection = self._customer_projector.project(projection_request)
            self._validate_projection(request, customer_projection)
        except _CompositionBoundaryError as error:
            self._identity_tracker.restore(checkpoint)
            return self._outcome(
                request,
                DiarizationCompositionStatus.FAILED_SAFE,
                error.reason,
            )
        except Exception:
            self._identity_tracker.restore(checkpoint)
            return self._outcome(
                request,
                DiarizationCompositionStatus.FAILED_SAFE,
                step,
            )

        has_customer_speech = bool(customer_projection.customer_words)
        return DiarizationCompositionOutcome(
            status=(
                DiarizationCompositionStatus.COMPLETED
                if has_customer_speech
                else DiarizationCompositionStatus.EMPTY
            ),
            reason=(
                DiarizationCompositionReason.COMPOSED
                if has_customer_speech
                else DiarizationCompositionReason.NO_CUSTOMER_SPEECH
            ),
            tenant_id=request.tenant_id,
            call_id=request.call_id,
            transcript_revision=request.transcript_revision,
            tracked_turns=tracked_turns,
            diarized_words=diarized_words,
            role_resolution=role_resolution,
            role_tagged_words=role_tagged_words,
            customer_projection=customer_projection,
        )

    @staticmethod
    def _validate_request(
        request: DiarizationCompositionRequest,
    ) -> DiarizationCompositionReason | None:
        source = request.diarization_request
        if (
            not request.tenant_id.strip()
            or not request.call_id.strip()
            or source.tenant_id != request.tenant_id
            or source.call_id != request.call_id
        ):
            return DiarizationCompositionReason.INVALID_REQUEST_SCOPE
        if request.transcript_revision < 0:
            return DiarizationCompositionReason.INVALID_REQUEST_REVISION
        if (
            not isfinite(source.window_start_seconds)
            or not isfinite(source.window_end_seconds)
            or source.window_start_seconds < 0
            or source.window_end_seconds <= source.window_start_seconds
        ):
            return DiarizationCompositionReason.INVALID_REQUEST_WINDOW
        for word in request.words:
            if (
                not isfinite(word.start_seconds)
                or not isfinite(word.end_seconds)
                or word.start_seconds < source.window_start_seconds
                or word.end_seconds > source.window_end_seconds
                or word.end_seconds <= word.start_seconds
            ):
                return DiarizationCompositionReason.INVALID_INPUT_WORD
        return None

    @staticmethod
    def _validate_diarization_result(
        request: DiarizationCompositionRequest,
        result: DiarizationResult,
    ) -> None:
        source = request.diarization_request
        if (
            result.tenant_id != request.tenant_id
            or result.call_id != request.call_id
            or result.window_start_seconds != source.window_start_seconds
            or result.window_end_seconds != source.window_end_seconds
        ):
            raise _CompositionBoundaryError(
                DiarizationCompositionReason.DIARIZER_OUTPUT_INVALID
            )
        for turn in result.turns:
            if (
                turn.tenant_id != request.tenant_id
                or turn.call_id != request.call_id
                or turn.start_seconds < source.window_start_seconds
                or turn.end_seconds > source.window_end_seconds
            ):
                raise _CompositionBoundaryError(
                    DiarizationCompositionReason.DIARIZER_OUTPUT_INVALID
                )

    @staticmethod
    def _validate_tracked_turns(
        request: DiarizationCompositionRequest,
        turns: tuple[DiarizationTurn, ...],
    ) -> None:
        if any(
            turn.tenant_id != request.tenant_id
            or turn.call_id != request.call_id
            or len(turn.global_speaker_ids) != len(turn.local_speaker_ids)
            for turn in turns
        ):
            raise _CompositionBoundaryError(
                DiarizationCompositionReason.TRACKING_FAILED
            )

    @staticmethod
    def _validate_diarized_words(
        request: DiarizationCompositionRequest,
        words: tuple[DiarizedWord, ...],
    ) -> None:
        if any(
            word.tenant_id != request.tenant_id
            or word.call_id != request.call_id
            or word.transcript_revision != request.transcript_revision
            for word in words
        ):
            raise _CompositionBoundaryError(
                DiarizationCompositionReason.ALIGNMENT_FAILED
            )

    @staticmethod
    def _validate_role_resolution(
        request: DiarizationCompositionRequest,
        result: SpeakerRoleResolutionResult,
    ) -> None:
        if (
            result.tenant_id != request.tenant_id
            or result.call_id != request.call_id
            or result.transcript_revision != request.transcript_revision
        ):
            raise _CompositionBoundaryError(
                DiarizationCompositionReason.ROLE_RESOLUTION_OUTPUT_INVALID
            )

    @staticmethod
    def _validate_projection(
        request: DiarizationCompositionRequest,
        projection: CustomerSpeechProjection,
    ) -> None:
        if (
            projection.tenant_id != request.tenant_id
            or projection.call_id != request.call_id
            or projection.transcript_revision != request.transcript_revision
        ):
            raise _CompositionBoundaryError(
                DiarizationCompositionReason.PROJECTION_OUTPUT_INVALID
            )

    @staticmethod
    def _outcome(
        request: DiarizationCompositionRequest,
        status: DiarizationCompositionStatus,
        reason: DiarizationCompositionReason,
    ) -> DiarizationCompositionOutcome:
        return DiarizationCompositionOutcome(
            status=status,
            reason=reason,
            tenant_id=request.tenant_id,
            call_id=request.call_id,
            transcript_revision=request.transcript_revision,
        )
