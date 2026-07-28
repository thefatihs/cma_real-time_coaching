from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.diarization.models import SpeakerRole
from app.diarization.role_resolver import (
    RoleEvidenceCode,
    RuleBasedSpeakerRoleResolver,
    SpeakerAttributedTextSpan,
    SpeakerRoleAssignment,
    SpeakerRoleResolutionError,
    SpeakerRoleResolutionErrorCategory,
    SpeakerRoleResolutionRequest,
    SpeakerRoleResolutionResult,
    SpeakerRoleResolverProtocol,
)


def _span(
    text: str,
    speaker_id: str | None,
    *,
    start: float = 0.0,
    end: float = 1.0,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    revision: int = 4,
    role: SpeakerRole = SpeakerRole.UNKNOWN,
    extra_speaker_id: str | None = None,
) -> SpeakerAttributedTextSpan:
    speaker_ids = () if speaker_id is None else (speaker_id,)
    if extra_speaker_id is not None:
        speaker_ids += (extra_speaker_id,)
    return SpeakerAttributedTextSpan(
        tenant_id=tenant_id,
        call_id=call_id,
        transcript_revision=revision,
        start_seconds=start,
        end_seconds=end,
        global_speaker_ids=speaker_ids,
        role=role,
        text=text,
    )


def _request(
    *spans: SpeakerAttributedTextSpan,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    revision: int = 4,
) -> SpeakerRoleResolutionRequest:
    return SpeakerRoleResolutionRequest(
        tenant_id=tenant_id,
        call_id=call_id,
        transcript_revision=revision,
        spans=spans,
    )


def _assignment(
    result: SpeakerRoleResolutionResult,
    speaker_id: str | None,
) -> SpeakerRoleAssignment:
    return next(
        assignment
        for assignment in result.assignments
        if assignment.global_speaker_id == speaker_id
    )


def test_strong_greeting_resolves_agent() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(_span("Merhaba, ben Ayşe.", "CALL_SPEAKER_0001"))
    )

    assignment = _assignment(result, "CALL_SPEAKER_0001")
    assert assignment.role is SpeakerRole.AGENT
    assert assignment.evidence is RoleEvidenceCode.STRONG_AGENT
    assert assignment.confidence == 1.0


def test_strong_customer_request_resolves_customer() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(_span("Bir sorun yaşıyorum.", "CALL_SPEAKER_0001"))
    )

    assignment = _assignment(result, "CALL_SPEAKER_0001")
    assert assignment.role is SpeakerRole.CUSTOMER
    assert assignment.evidence is RoleEvidenceCode.STRONG_CUSTOMER


def test_first_speaker_position_alone_remains_unknown() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(_span("Günaydın.", "CALL_SPEAKER_0001", start=0, end=1))
    )

    assignment = _assignment(result, "CALL_SPEAKER_0001")
    assert assignment.role is SpeakerRole.UNKNOWN
    assert assignment.evidence is RoleEvidenceCode.WEAK_POSITIONAL
    assert assignment.confidence is None


def test_high_confidence_agent_allows_safe_opposite_customer_inference() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(
            _span("Size nasıl yardımcı olabilirim?", "CALL_SPEAKER_0001"),
            _span(
                "Teşekkür ederim.",
                "CALL_SPEAKER_0002",
                start=1,
                end=2,
            ),
        )
    )

    inferred = _assignment(result, "CALL_SPEAKER_0002")
    assert inferred.role is SpeakerRole.CUSTOMER
    assert inferred.evidence is RoleEvidenceCode.INFERRED_OPPOSITE


def test_conflicting_evidence_remains_unknown_and_blocks_inference() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(
            _span(
                "Merhaba ben arıyorum ve sorun yaşıyorum.",
                "CALL_SPEAKER_0001",
            ),
            _span("Merhaba.", "CALL_SPEAKER_0002", start=1, end=2),
        )
    )

    conflicted = _assignment(result, "CALL_SPEAKER_0001")
    assert conflicted.role is SpeakerRole.UNKNOWN
    assert conflicted.evidence is RoleEvidenceCode.CONFLICTING
    assert _assignment(result, "CALL_SPEAKER_0002").role is SpeakerRole.UNKNOWN


def test_insufficient_evidence_and_more_than_two_speakers_remain_unknown() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(
            _span("Tamam.", "CALL_SPEAKER_0001", start=10, end=11),
            _span("Peki.", "CALL_SPEAKER_0002", start=11, end=12),
            _span("Anladım.", "CALL_SPEAKER_0003", start=12, end=13),
        )
    )

    assert {assignment.role for assignment in result.assignments} == {
        SpeakerRole.UNKNOWN
    }


def test_overlap_text_is_ignored_as_independent_evidence() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(
            _span(
                "Merhaba ben müşteri temsilcisiyim.",
                "CALL_SPEAKER_0001",
                extra_speaker_id="CALL_SPEAKER_0002",
                role=SpeakerRole.OVERLAP,
            )
        )
    )

    assert result.assignments == ()
    assert result.ignored_evidence == (RoleEvidenceCode.OVERLAP_IGNORED,)


def test_two_speakers_receive_isolated_direct_evidence() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(
            _span("İşleminizi gerçekleştiriyorum.", "CALL_SPEAKER_0001"),
            _span(
                "İptal etmek istiyorum.",
                "CALL_SPEAKER_0002",
                start=1,
                end=2,
            ),
        )
    )

    assert _assignment(result, "CALL_SPEAKER_0001").role is SpeakerRole.AGENT
    assert _assignment(result, "CALL_SPEAKER_0002").role is SpeakerRole.CUSTOMER


@pytest.mark.parametrize(
    ("span", "category"),
    [
        (
            _span("Merhaba.", "speaker", tenant_id="tenant-b"),
            SpeakerRoleResolutionErrorCategory.SCOPE_MISMATCH,
        ),
        (
            _span("Merhaba.", "speaker", call_id="call-b"),
            SpeakerRoleResolutionErrorCategory.SCOPE_MISMATCH,
        ),
        (
            _span("Merhaba.", "speaker", revision=5),
            SpeakerRoleResolutionErrorCategory.REVISION_MISMATCH,
        ),
    ],
)
def test_scope_and_revision_mismatch_rejected(
    span: SpeakerAttributedTextSpan,
    category: SpeakerRoleResolutionErrorCategory,
) -> None:
    with pytest.raises(SpeakerRoleResolutionError) as error:
        RuleBasedSpeakerRoleResolver().resolve(_request(span))

    assert error.value.category is category


def test_resolution_is_deterministic_and_input_is_not_mutated() -> None:
    resolver: SpeakerRoleResolverProtocol = RuleBasedSpeakerRoleResolver()
    spans = (
        _span("Sorun yaşıyorum.", "CALL_SPEAKER_0002", start=2, end=3),
        _span("Merhaba ben Deniz.", "CALL_SPEAKER_0001", start=0, end=1),
    )
    snapshot = deepcopy(spans)
    request = _request(*spans)

    first = resolver.resolve(request)
    second = resolver.resolve(request)

    assert first == second
    assert spans == snapshot
    with pytest.raises(ValidationError):
        spans[0].text = "değişti"  # type: ignore[misc]


def test_text_processing_is_bounded_fail_closed() -> None:
    resolver = RuleBasedSpeakerRoleResolver(max_text_characters=8)

    with pytest.raises(SpeakerRoleResolutionError) as error:
        resolver.resolve(_request(_span("uzun gizli metin", "speaker")))

    assert (
        error.value.category is SpeakerRoleResolutionErrorCategory.TEXT_LIMIT_EXCEEDED
    )


def test_missing_global_identity_returns_unknown() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(_span("Sorun yaşıyorum.", None))
    )

    assignment = _assignment(result, None)
    assert assignment.role is SpeakerRole.UNKNOWN
    assert assignment.evidence is RoleEvidenceCode.MISSING_GLOBAL_ID


def test_sensitive_text_is_absent_from_result_repr_and_safe_errors() -> None:
    private_text = "özel müşteri metni ve dosya yolu"
    span = _span(private_text, "speaker")
    result = RuleBasedSpeakerRoleResolver().resolve(_request(span))

    assert private_text not in repr(span)
    assert private_text not in repr(result)
    with pytest.raises(SpeakerRoleResolutionError) as error:
        RuleBasedSpeakerRoleResolver(max_text_characters=2).resolve(_request(span))
    assert str(error.value) == "text_limit_exceeded"
    assert private_text not in repr(error.value)
