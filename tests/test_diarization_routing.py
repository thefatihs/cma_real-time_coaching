from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.diarization.models import (
    DiarizedTranscriptEvent,
    DiarizedWord,
    SpeakerRole,
)
from app.diarization.role_resolver import (
    RoleEvidenceCode,
    SpeakerRoleAssignment,
    SpeakerRoleResolutionResult,
)
from app.diarization.routing import (
    CustomerProjectionReason,
    CustomerProjectionStatus,
    CustomerSpeechProjectionRequest,
    CustomerSpeechProjector,
    DiarizationRoutingError,
    DiarizationRoutingErrorCategory,
)


def _word(
    text: str,
    start: float,
    end: float,
    speaker_id: str | None,
    *,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    revision: int = 6,
    role: SpeakerRole = SpeakerRole.UNKNOWN,
    second_speaker_id: str | None = None,
) -> DiarizedWord:
    global_ids = () if speaker_id is None else (speaker_id,)
    local_ids = ("local-a",)
    if second_speaker_id is not None:
        global_ids += (second_speaker_id,)
        local_ids += ("local-b",)
    return DiarizedWord(
        tenant_id=tenant_id,
        call_id=call_id,
        transcript_revision=revision,
        start_seconds=start,
        end_seconds=end,
        text=text,
        local_speaker_ids=local_ids,
        global_speaker_id=speaker_id if second_speaker_id is None else None,
        global_speaker_ids=global_ids,
        role=role,
        speaker_confidence=0.8,
    )


def _event(
    *words: DiarizedWord,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    revision: int = 6,
) -> DiarizedTranscriptEvent:
    return DiarizedTranscriptEvent(
        tenant_id=tenant_id,
        call_id=call_id,
        transcript_event_id="event-1",
        transcript_revision=revision,
        start_seconds=0,
        end_seconds=20,
        words=words,
    )


def _assignment(
    speaker_id: str,
    role: SpeakerRole,
    *,
    confidence: float = 1.0,
) -> SpeakerRoleAssignment:
    return SpeakerRoleAssignment(
        global_speaker_id=speaker_id,
        role=role,
        confidence=confidence,
        evidence=(
            RoleEvidenceCode.STRONG_CUSTOMER
            if role is SpeakerRole.CUSTOMER
            else RoleEvidenceCode.STRONG_AGENT
        ),
    )


def _resolution(
    *assignments: SpeakerRoleAssignment,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    revision: int = 6,
) -> SpeakerRoleResolutionResult:
    return SpeakerRoleResolutionResult(
        tenant_id=tenant_id,
        call_id=call_id,
        transcript_revision=revision,
        assignments=assignments,
    )


def _request(
    event: DiarizedTranscriptEvent,
    resolution: SpeakerRoleResolutionResult,
    *,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    revision: int = 6,
) -> CustomerSpeechProjectionRequest:
    return CustomerSpeechProjectionRequest(
        tenant_id=tenant_id,
        call_id=call_id,
        transcript_revision=revision,
        event=event,
        role_resolution=resolution,
    )


def test_resolved_customer_words_enter_projection_with_metadata() -> None:
    word = _word("Yardım", 1, 2, "customer")
    projection = CustomerSpeechProjector().project(
        _request(
            _event(word),
            _resolution(_assignment("customer", SpeakerRole.CUSTOMER)),
        )
    )

    assert projection.customer_text == "Yardım"
    assert projection.customer_start_seconds == 1
    assert projection.customer_end_seconds == 2
    assert projection.status is CustomerProjectionStatus.READY
    assert projection.reason is CustomerProjectionReason.TRUSTED_CUSTOMER_SPEECH
    tagged = projection.customer_words[0]
    assert tagged.local_speaker_ids == word.local_speaker_ids
    assert tagged.global_speaker_id == "customer"
    assert tagged.speaker_confidence == 0.8
    assert tagged.role is SpeakerRole.CUSTOMER
    assert tagged.role_confidence == 1.0
    assert tagged.role_evidence is RoleEvidenceCode.STRONG_CUSTOMER


def test_agent_unknown_overlap_and_missing_identity_are_excluded() -> None:
    words = (
        _word("Temsilci", 0, 1, "agent"),
        _word("Belirsiz", 1, 2, "unknown"),
        _word(
            "Çakışma",
            2,
            3,
            "agent",
            role=SpeakerRole.OVERLAP,
            second_speaker_id="customer",
        ),
        _word("Kimliksiz", 3, 4, None),
    )
    projection = CustomerSpeechProjector().project(
        _request(
            _event(*words),
            _resolution(
                _assignment("agent", SpeakerRole.AGENT),
                _assignment("unknown", SpeakerRole.UNKNOWN),
                _assignment("customer", SpeakerRole.CUSTOMER),
            ),
        )
    )

    assert projection.customer_words == ()
    assert projection.customer_text == ""
    assert projection.excluded_agent_word_count == 1
    assert projection.excluded_unknown_word_count == 2
    assert projection.excluded_overlap_word_count == 1
    assert projection.status is CustomerProjectionStatus.EMPTY
    assert projection.reason is CustomerProjectionReason.NO_TRUSTED_CUSTOMER_SPEECH


def test_mixed_conversation_preserves_only_chronological_customer_word_order() -> None:
    words = (
        _word("Merhaba", 0, 1, "agent"),
        _word("İptal", 1, 2, "customer"),
        _word("etmek", 2, 3, "customer"),
        _word("istiyorum", 3, 4, "customer"),
    )
    projection = CustomerSpeechProjector().project(
        _request(
            _event(*words),
            _resolution(
                _assignment("agent", SpeakerRole.AGENT),
                _assignment("customer", SpeakerRole.CUSTOMER),
            ),
        )
    )

    assert projection.customer_text == "İptal etmek istiyorum"
    assert [word.text for word in projection.customer_words] == [
        "İptal",
        "etmek",
        "istiyorum",
    ]


def test_below_threshold_customer_is_excluded() -> None:
    projection = CustomerSpeechProjector(trusted_customer_confidence=0.9).project(
        _request(
            _event(_word("Talep", 0, 1, "customer")),
            _resolution(
                _assignment(
                    "customer",
                    SpeakerRole.CUSTOMER,
                    confidence=0.5,
                )
            ),
        )
    )

    assert projection.customer_words == ()
    assert projection.excluded_below_confidence_word_count == 1


@pytest.mark.parametrize(
    ("event", "resolution", "category"),
    [
        (
            _event(
                _word("Söz", 0, 1, "speaker", tenant_id="tenant-b"),
                tenant_id="tenant-b",
            ),
            _resolution(_assignment("speaker", SpeakerRole.CUSTOMER)),
            DiarizationRoutingErrorCategory.SCOPE_MISMATCH,
        ),
        (
            _event(_word("Söz", 0, 1, "speaker")),
            _resolution(
                _assignment("speaker", SpeakerRole.CUSTOMER),
                call_id="call-b",
            ),
            DiarizationRoutingErrorCategory.SCOPE_MISMATCH,
        ),
        (
            _event(_word("Söz", 0, 1, "speaker")),
            _resolution(
                _assignment("speaker", SpeakerRole.CUSTOMER),
                revision=7,
            ),
            DiarizationRoutingErrorCategory.REVISION_MISMATCH,
        ),
    ],
)
def test_scope_and_revision_mismatch_rejected(
    event: DiarizedTranscriptEvent,
    resolution: SpeakerRoleResolutionResult,
    category: DiarizationRoutingErrorCategory,
) -> None:
    with pytest.raises(DiarizationRoutingError) as error:
        CustomerSpeechProjector().project(_request(event, resolution))

    assert error.value.category is category


def test_unknown_global_speaker_and_conflicting_role_mapping_rejected() -> None:
    event = _event(_word("Söz", 0, 1, "known"))
    projector = CustomerSpeechProjector()

    with pytest.raises(DiarizationRoutingError) as unknown:
        projector.project(
            _request(
                event,
                _resolution(_assignment("other", SpeakerRole.CUSTOMER)),
            )
        )
    assert (
        unknown.value.category is DiarizationRoutingErrorCategory.UNKNOWN_GLOBAL_SPEAKER
    )

    with pytest.raises(DiarizationRoutingError) as conflicting:
        projector.project(
            _request(
                event,
                _resolution(
                    _assignment("known", SpeakerRole.CUSTOMER),
                    _assignment("known", SpeakerRole.AGENT),
                ),
            )
        )
    assert (
        conflicting.value.category
        is DiarizationRoutingErrorCategory.CONFLICTING_ROLE_MAPPING
    )


def test_duplicate_or_conflicting_words_rejected() -> None:
    word = _word("Tekrar", 0, 1, "speaker")
    event = DiarizedTranscriptEvent.model_construct(
        tenant_id="tenant-a",
        call_id="call-a",
        transcript_event_id="event-1",
        transcript_revision=6,
        start_seconds=0.0,
        end_seconds=20.0,
        turns=(),
        words=(word, word.model_copy(update={"text": "Farklı"})),
    )

    with pytest.raises(DiarizationRoutingError) as error:
        CustomerSpeechProjector().project(
            _request(
                event,
                _resolution(_assignment("speaker", SpeakerRole.CUSTOMER)),
            )
        )

    assert (
        error.value.category
        is DiarizationRoutingErrorCategory.DUPLICATE_OR_CONFLICTING_WORD
    )


def test_projection_is_deterministic_and_inputs_are_not_mutated() -> None:
    words = (
        _word("Bir", 0, 1, "customer"),
        _word("talep", 1, 2, "customer"),
    )
    event = _event(*words)
    resolution = _resolution(_assignment("customer", SpeakerRole.CUSTOMER))
    request = _request(event, resolution)
    snapshot = deepcopy(request)
    projector = CustomerSpeechProjector()

    first = projector.project(request)
    second = projector.project(request)

    assert first == second
    assert request == snapshot
    with pytest.raises(ValidationError):
        first.customer_text = "değişti"  # type: ignore[misc]


def test_sensitive_text_is_absent_from_repr_errors_and_diagnostics() -> None:
    private_text = "özel müşteri konuşması"
    word = _word(private_text, 0, 1, "customer")
    projection = CustomerSpeechProjector().project(
        _request(
            _event(word),
            _resolution(_assignment("customer", SpeakerRole.CUSTOMER)),
        )
    )

    assert private_text not in repr(projection)
    malformed = word.model_copy(update={"end_seconds": float("nan")})
    event = DiarizedTranscriptEvent.model_construct(
        tenant_id="tenant-a",
        call_id="call-a",
        transcript_event_id="event-1",
        transcript_revision=6,
        start_seconds=0.0,
        end_seconds=20.0,
        turns=(),
        words=(malformed,),
    )
    with pytest.raises(DiarizationRoutingError) as error:
        CustomerSpeechProjector().project(
            _request(
                event,
                _resolution(_assignment("customer", SpeakerRole.CUSTOMER)),
            )
        )
    assert str(error.value) == "malformed_word"
    assert private_text not in repr(error.value)
