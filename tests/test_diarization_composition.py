from copy import deepcopy

from app.asr.models import ASRWordTimestamp
from app.diarization.composition import (
    DiarizationCompositionReason,
    DiarizationCompositionRequest,
    DiarizationCompositionStatus,
    OfflineDiarizationComposer,
)
from app.diarization.identity_tracker import SpeakerIdentityTracker
from app.diarization.models import (
    DiarizationRequest,
    DiarizationResult,
    DiarizationTurn,
    SpeakerRole,
)
from app.diarization.role_resolver import (
    RoleEvidenceCode,
    SpeakerRoleAssignment,
    SpeakerRoleResolutionRequest,
    SpeakerRoleResolutionResult,
)
from app.diarization.routing import (
    CustomerSpeechProjection,
    CustomerSpeechProjectionRequest,
    CustomerSpeechProjector,
    RoleTaggedWord,
)


def _audio_request(
    *,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    start: float = 0,
    end: float = 10,
) -> DiarizationRequest:
    return DiarizationRequest(
        tenant_id=tenant_id,
        call_id=call_id,
        window_start_seconds=start,
        window_end_seconds=end,
        sample_rate_hz=16_000,
        mono_audio=(0.0, 0.1, -0.1),
    )


def _turn(
    start: float,
    end: float,
    *local_ids: str,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    role: SpeakerRole = SpeakerRole.UNKNOWN,
) -> DiarizationTurn:
    return DiarizationTurn(
        tenant_id=tenant_id,
        call_id=call_id,
        start_seconds=start,
        end_seconds=end,
        local_speaker_ids=local_ids,
        role=role,
    )


def _result(
    *turns: DiarizationTurn,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    start: float = 0,
    end: float = 10,
) -> DiarizationResult:
    return DiarizationResult(
        tenant_id=tenant_id,
        call_id=call_id,
        window_start_seconds=start,
        window_end_seconds=end,
        turns=turns,
    )


def _word(text: str, start: float, end: float) -> ASRWordTimestamp:
    return ASRWordTimestamp(
        text=text,
        start_seconds=start,
        end_seconds=end,
        probability=0.9,
    )


def _request(
    *words: ASRWordTimestamp,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    revision: int = 1,
    start: float = 0,
    end: float = 10,
) -> DiarizationCompositionRequest:
    return DiarizationCompositionRequest(
        tenant_id=tenant_id,
        call_id=call_id,
        transcript_revision=revision,
        diarization_request=_audio_request(
            tenant_id=tenant_id,
            call_id=call_id,
            start=start,
            end=end,
        ),
        words=words,
    )


class SequenceDiarizer:
    def __init__(self, *results: DiarizationResult, fail: bool = False) -> None:
        self.results = results
        self.fail = fail
        self.calls = 0

    def diarize(self, request: DiarizationRequest) -> DiarizationResult:
        if self.fail:
            raise RuntimeError("private diarizer failure")
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


class FixedRoleResolver:
    def __init__(
        self,
        roles: dict[str, SpeakerRole],
        *,
        fail: bool = False,
        wrong_scope: bool = False,
    ) -> None:
        self.roles = roles
        self.fail = fail
        self.wrong_scope = wrong_scope

    def resolve(
        self,
        request: SpeakerRoleResolutionRequest,
    ) -> SpeakerRoleResolutionResult:
        if self.fail:
            raise RuntimeError("private resolver failure")
        speaker_ids = sorted(
            {
                speaker_id
                for span in request.spans
                if span.role is not SpeakerRole.OVERLAP
                for speaker_id in span.global_speaker_ids
            }
        )
        assignments = tuple(
            SpeakerRoleAssignment(
                global_speaker_id=speaker_id,
                role=self.roles.get(speaker_id, SpeakerRole.UNKNOWN),
                confidence=(
                    1.0
                    if self.roles.get(speaker_id, SpeakerRole.UNKNOWN)
                    is not SpeakerRole.UNKNOWN
                    else None
                ),
                evidence=(
                    RoleEvidenceCode.STRONG_AGENT
                    if self.roles.get(speaker_id) is SpeakerRole.AGENT
                    else (
                        RoleEvidenceCode.STRONG_CUSTOMER
                        if self.roles.get(speaker_id) is SpeakerRole.CUSTOMER
                        else RoleEvidenceCode.INSUFFICIENT
                    )
                ),
            )
            for speaker_id in speaker_ids
        )
        return SpeakerRoleResolutionResult(
            tenant_id="wrong" if self.wrong_scope else request.tenant_id,
            call_id=request.call_id,
            transcript_revision=request.transcript_revision,
            assignments=assignments,
        )


class FailingProjector:
    def apply_roles(
        self,
        request: CustomerSpeechProjectionRequest,
    ) -> tuple[RoleTaggedWord, ...]:
        raise RuntimeError("private projector failure")

    def project(
        self,
        request: CustomerSpeechProjectionRequest,
    ) -> CustomerSpeechProjection:
        raise RuntimeError("private projector failure")


def _composer(
    diarizer: SequenceDiarizer,
    tracker: SpeakerIdentityTracker | None = None,
    resolver: FixedRoleResolver | None = None,
    projector: CustomerSpeechProjector | FailingProjector | None = None,
) -> OfflineDiarizationComposer:
    return OfflineDiarizationComposer(
        diarizer=diarizer,
        identity_tracker=tracker or SpeakerIdentityTracker(),
        role_resolver=resolver
        or FixedRoleResolver(
            {
                "CALL_SPEAKER_0001": SpeakerRole.AGENT,
                "CALL_SPEAKER_0002": SpeakerRole.CUSTOMER,
            }
        ),
        customer_projector=projector or CustomerSpeechProjector(),
    )


def test_complete_two_speaker_composition_projects_only_customer_text() -> None:
    composer = _composer(
        SequenceDiarizer(_result(_turn(0, 5, "local-a"), _turn(5, 10, "local-b")))
    )

    outcome = composer.compose(_request(_word("Merhaba", 1, 2), _word("İptal", 6, 7)))

    assert outcome.status is DiarizationCompositionStatus.COMPLETED
    assert [turn.global_speaker_id for turn in outcome.tracked_turns] == [
        "CALL_SPEAKER_0001",
        "CALL_SPEAKER_0002",
    ]
    assert outcome.role_resolution is not None
    assert outcome.customer_projection is not None
    assert outcome.customer_projection.customer_text == "İptal"
    assert [word.role for word in outcome.role_tagged_words] == [
        SpeakerRole.AGENT,
        SpeakerRole.CUSTOMER,
    ]


def test_local_ids_can_swap_across_overlapping_windows() -> None:
    tracker = SpeakerIdentityTracker()
    composer = _composer(
        SequenceDiarizer(
            _result(_turn(0, 5, "a"), _turn(5, 10, "b")),
            _result(
                _turn(4, 5, "b"),
                _turn(5, 10, "a"),
                start=4,
                end=14,
            ),
        ),
        tracker,
    )
    first = composer.compose(_request(_word("Bir", 6, 7)))
    second = composer.compose(
        _request(
            _word("İki", 6, 7),
            revision=2,
            start=4,
            end=14,
        )
    )

    assert first.diarized_words[0].global_speaker_id == (
        second.diarized_words[0].global_speaker_id
    )
    assert tracker.retained_window_count(tenant_id="tenant-a", call_id="call-a") == 2


def test_unknown_overlap_and_empty_inputs_produce_safe_empty_projection() -> None:
    overlap_composer = _composer(
        SequenceDiarizer(
            _result(
                _turn(
                    0,
                    5,
                    "a",
                    "b",
                    role=SpeakerRole.OVERLAP,
                )
            )
        )
    )
    overlap = overlap_composer.compose(_request(_word("Çakışma", 1, 2)))
    assert overlap.status is DiarizationCompositionStatus.EMPTY
    assert overlap.customer_projection is not None
    assert overlap.customer_projection.excluded_overlap_word_count == 1

    empty = _composer(SequenceDiarizer(_result())).compose(_request())
    assert empty.status is DiarizationCompositionStatus.EMPTY
    assert empty.tracked_turns == ()
    assert empty.diarized_words == ()
    assert empty.customer_projection is not None
    assert empty.customer_projection.customer_words == ()


def test_words_without_turn_overlap_remain_unknown_and_excluded() -> None:
    outcome = _composer(SequenceDiarizer(_result(_turn(0, 2, "speaker")))).compose(
        _request(_word("Belirsiz", 5, 6))
    )

    assert outcome.status is DiarizationCompositionStatus.EMPTY
    assert outcome.customer_projection is not None
    assert outcome.customer_projection.excluded_unknown_word_count == 1


def test_wrong_scope_revision_and_word_range_are_rejected() -> None:
    composer = _composer(SequenceDiarizer(_result()))
    wrong_scope = DiarizationCompositionRequest(
        tenant_id="tenant-b",
        call_id="call-a",
        transcript_revision=1,
        diarization_request=_audio_request(),
        words=(),
    )

    assert composer.compose(wrong_scope).reason is (
        DiarizationCompositionReason.INVALID_REQUEST_SCOPE
    )
    assert composer.compose(_request(revision=-1)).reason is (
        DiarizationCompositionReason.INVALID_REQUEST_REVISION
    )
    outside = _word("Dışarıda", 11, 12)
    assert composer.compose(_request(outside)).reason is (
        DiarizationCompositionReason.INVALID_INPUT_WORD
    )


def test_malformed_diarizer_and_role_outputs_fail_closed() -> None:
    malformed = DiarizationResult.model_construct(
        tenant_id="wrong",
        call_id="call-a",
        window_start_seconds=0.0,
        window_end_seconds=10.0,
        turns=(),
    )
    assert _composer(SequenceDiarizer(malformed)).compose(_request()).reason is (
        DiarizationCompositionReason.DIARIZER_OUTPUT_INVALID
    )
    wrong_role = _composer(
        SequenceDiarizer(_result(_turn(0, 5, "speaker"))),
        resolver=FixedRoleResolver({}, wrong_scope=True),
    ).compose(_request(_word("Söz", 1, 2)))
    assert wrong_role.reason is (
        DiarizationCompositionReason.ROLE_RESOLUTION_OUTPUT_INVALID
    )


def test_component_exceptions_use_fixed_safe_failure_reasons() -> None:
    diarizer_failure = _composer(SequenceDiarizer(_result(), fail=True)).compose(
        _request()
    )
    assert diarizer_failure.reason is DiarizationCompositionReason.DIARIZER_FAILED

    resolver_failure = _composer(
        SequenceDiarizer(_result(_turn(0, 5, "speaker"))),
        resolver=FixedRoleResolver({}, fail=True),
    ).compose(_request(_word("Söz", 1, 2)))
    assert resolver_failure.reason is (
        DiarizationCompositionReason.ROLE_RESOLUTION_FAILED
    )

    projector_failure = _composer(
        SequenceDiarizer(_result(_turn(0, 5, "speaker"))),
        projector=FailingProjector(),
    ).compose(_request(_word("Söz", 1, 2)))
    assert projector_failure.reason is DiarizationCompositionReason.PROJECTION_FAILED


def test_tracker_rolls_back_after_downstream_failure() -> None:
    tracker = SpeakerIdentityTracker()
    result = _result(_turn(0, 5, "speaker"))
    failed = _composer(
        SequenceDiarizer(result),
        tracker,
        resolver=FixedRoleResolver({}, fail=True),
    ).compose(_request(_word("Söz", 1, 2)))

    assert failed.status is DiarizationCompositionStatus.FAILED_SAFE
    assert tracker.retained_window_count(tenant_id="tenant-a", call_id="call-a") == 0
    recovered = _composer(SequenceDiarizer(result), tracker).compose(
        _request(_word("Söz", 1, 2))
    )
    assert recovered.tracked_turns[0].global_speaker_id == "CALL_SPEAKER_0001"


def test_exact_request_is_idempotent_without_duplicate_tracker_history() -> None:
    tracker = SpeakerIdentityTracker()
    request = _request(_word("İptal", 6, 7))
    composer = _composer(
        SequenceDiarizer(_result(_turn(0, 5, "a"), _turn(5, 10, "b"))),
        tracker,
    )

    first = composer.compose(request)
    second = composer.compose(request)

    assert first == second
    assert tracker.retained_window_count(tenant_id="tenant-a", call_id="call-a") == 1


def test_tenant_call_state_is_isolated_and_inputs_are_not_mutated() -> None:
    request = _request(_word("İptal", 6, 7))
    snapshot = deepcopy(request)
    tracker = SpeakerIdentityTracker()
    first = _composer(
        SequenceDiarizer(_result(_turn(5, 10, "speaker"))),
        tracker,
    ).compose(request)
    other_request = _request(
        _word("Talep", 6, 7),
        tenant_id="tenant-b",
        call_id="call-b",
    )
    other_result = _result(
        _turn(5, 10, "speaker", tenant_id="tenant-b", call_id="call-b"),
        tenant_id="tenant-b",
        call_id="call-b",
    )
    second = _composer(SequenceDiarizer(other_result), tracker).compose(other_request)

    assert first.tracked_turns[0].global_speaker_id == "CALL_SPEAKER_0001"
    assert second.tracked_turns[0].global_speaker_id == "CALL_SPEAKER_0001"
    assert request == snapshot


def test_outcome_and_failures_do_not_expose_sensitive_text() -> None:
    marker = "özel müşteri konuşması"
    request = _request(_word(marker, 1, 2))
    outcome = _composer(
        SequenceDiarizer(_result(_turn(0, 5, "speaker"))),
        resolver=FixedRoleResolver({}, fail=True),
    ).compose(request)

    assert marker not in repr(request)
    assert marker not in repr(outcome)
    assert outcome.reason is DiarizationCompositionReason.ROLE_RESOLUTION_FAILED
