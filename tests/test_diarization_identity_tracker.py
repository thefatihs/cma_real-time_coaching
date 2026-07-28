from dataclasses import FrozenInstanceError

import pytest

from app.diarization.identity_tracker import (
    SpeakerIdentityTracker,
    SpeakerIdentityTrackingError,
    SpeakerIdentityTrackingErrorCategory,
    SpeakerIdentityTrackingRequest,
)
from app.diarization.models import DiarizationTurn, SpeakerRole


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


def _request(
    start: float,
    end: float,
    *turns: DiarizationTurn,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
) -> SpeakerIdentityTrackingRequest:
    return SpeakerIdentityTrackingRequest(
        tenant_id=tenant_id,
        call_id=call_id,
        window_start_seconds=start,
        window_end_seconds=end,
        turns=turns,
    )


def test_local_names_can_swap_while_global_speakers_remain_stable() -> None:
    tracker = SpeakerIdentityTracker()
    first = tracker.track(
        _request(0, 10, _turn(0, 6, "SPEAKER_00"), _turn(6, 10, "SPEAKER_01"))
    )

    second = tracker.track(
        _request(5, 15, _turn(5, 6, "SPEAKER_01"), _turn(6, 10, "SPEAKER_00"))
    )

    assert first[0].global_speaker_id == second[0].global_speaker_id
    assert first[1].global_speaker_id == second[1].global_speaker_id


def test_two_speakers_remain_distinct_and_new_speaker_is_allocated() -> None:
    tracker = SpeakerIdentityTracker(max_local_speakers=3)
    first = tracker.track(_request(0, 10, _turn(0, 5, "a"), _turn(5, 10, "b")))
    second = tracker.track(
        _request(
            4,
            14,
            _turn(4, 5, "x"),
            _turn(5, 10, "y"),
            _turn(10, 14, "z"),
        )
    )

    assert first[0].global_speaker_id != first[1].global_speaker_id
    assert second[0].global_speaker_id == first[0].global_speaker_id
    assert second[1].global_speaker_id == first[1].global_speaker_id
    assert second[2].global_speaker_id == "CALL_SPEAKER_0003"


def test_one_to_one_matching_is_deterministic_for_equal_overlap() -> None:
    tracker = SpeakerIdentityTracker()
    first = tracker.track(_request(0, 4, _turn(0, 2, "first"), _turn(2, 4, "second")))

    matched = tracker.track(_request(1, 5, _turn(1, 3, "alpha"), _turn(1, 3, "beta")))

    assert matched[0].global_speaker_id == first[0].global_speaker_id
    assert matched[1].global_speaker_id == first[1].global_speaker_id


def test_overlap_turn_preserves_all_distinct_global_identities() -> None:
    tracker = SpeakerIdentityTracker()
    first = tracker.track(_request(0, 6, _turn(0, 3, "a"), _turn(3, 6, "b")))

    overlap = tracker.track(
        _request(
            2,
            7,
            _turn(2, 6, "left", "right", role=SpeakerRole.OVERLAP),
        )
    )[0]

    assert overlap.global_speaker_id is None
    assert set(overlap.global_speaker_ids) == {
        first[0].global_speaker_id,
        first[1].global_speaker_id,
    }
    assert len(overlap.global_speaker_ids) == len(overlap.local_speaker_ids)
    assert overlap.role is SpeakerRole.OVERLAP


def test_reprocessing_exact_window_is_idempotent() -> None:
    tracker = SpeakerIdentityTracker()
    request = _request(0, 5, _turn(0, 5, "speaker"))

    first = tracker.track(request)
    second = tracker.track(request)

    assert first is second
    assert tracker.retained_window_count(tenant_id="tenant-a", call_id="call-a") == 1


def test_conflicting_reprocess_and_out_of_order_window_fail_closed() -> None:
    tracker = SpeakerIdentityTracker()
    tracker.track(_request(2, 6, _turn(2, 6, "speaker")))

    with pytest.raises(SpeakerIdentityTrackingError) as conflict:
        tracker.track(_request(2, 6, _turn(2, 5, "speaker")))
    assert (
        conflict.value.category
        is SpeakerIdentityTrackingErrorCategory.CONFLICTING_REPROCESS
    )

    with pytest.raises(SpeakerIdentityTrackingError) as out_of_order:
        tracker.track(_request(1, 5, _turn(1, 5, "speaker")))
    assert (
        out_of_order.value.category
        is SpeakerIdentityTrackingErrorCategory.NON_MONOTONIC_WINDOW
    )


def test_tenant_and_call_state_are_isolated() -> None:
    tracker = SpeakerIdentityTracker()
    tenant_a = tracker.track(_request(0, 2, _turn(0, 2, "speaker")))[0]
    tenant_b = tracker.track(
        _request(
            0,
            2,
            _turn(0, 2, "speaker", tenant_id="tenant-b"),
            tenant_id="tenant-b",
        )
    )[0]

    assert tenant_a.global_speaker_id == "CALL_SPEAKER_0001"
    assert tenant_b.global_speaker_id == "CALL_SPEAKER_0001"
    with pytest.raises(SpeakerIdentityTrackingError) as mismatch:
        tracker.track(
            _request(
                2,
                4,
                _turn(2, 4, "speaker", call_id="call-b"),
            )
        )
    assert (
        mismatch.value.category is SpeakerIdentityTrackingErrorCategory.SCOPE_MISMATCH
    )


def test_reset_removes_only_exact_call_state() -> None:
    tracker = SpeakerIdentityTracker()
    tracker.track(_request(0, 2, _turn(0, 2, "a")))
    tracker.track(
        _request(
            0,
            2,
            _turn(0, 2, "b", call_id="call-b"),
            call_id="call-b",
        )
    )

    assert tracker.reset(tenant_id="tenant-a", call_id="call-a")
    assert not tracker.reset(tenant_id="tenant-a", call_id="call-a")
    assert tracker.retained_window_count(tenant_id="tenant-a", call_id="call-a") == 0
    assert tracker.retained_window_count(tenant_id="tenant-a", call_id="call-b") == 1


def test_history_is_bounded_without_reusing_allocated_ordinals() -> None:
    tracker = SpeakerIdentityTracker(history_window_limit=2)
    for index in range(4):
        start = float(index * 2)
        tracker.track(_request(start, start + 2, _turn(start, start + 2, "speaker")))

    assert tracker.retained_window_count(tenant_id="tenant-a", call_id="call-a") == 2
    latest = tracker.track(_request(8, 10, _turn(8, 10, "new")))[0]
    assert latest.global_speaker_id == "CALL_SPEAKER_0005"


def test_inputs_and_outputs_are_immutable_and_input_is_not_mutated() -> None:
    tracker = SpeakerIdentityTracker()
    original = _turn(0, 2, "speaker")
    request = _request(0, 2, original)

    output = tracker.track(request)

    assert original.global_speaker_id is None
    assert original.global_speaker_ids == ()
    with pytest.raises(FrozenInstanceError):
        request.call_id = "other"  # type: ignore[misc]
    with pytest.raises(Exception):
        output[0].global_speaker_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("tracking_request", "category"),
    [
        (
            _request(0, 2, _turn(0, 2, "a", tenant_id="other")),
            SpeakerIdentityTrackingErrorCategory.SCOPE_MISMATCH,
        ),
        (
            _request(0, 2, _turn(0, 3, "a")),
            SpeakerIdentityTrackingErrorCategory.TURN_OUTSIDE_WINDOW,
        ),
        (
            _request(0, 2, _turn(0, 2, "a"), _turn(0, 2, "a")),
            SpeakerIdentityTrackingErrorCategory.DUPLICATE_OR_CONFLICTING_TURN,
        ),
        (
            _request(
                0,
                2,
                _turn(0, 1, "a"),
                _turn(1, 2, "b"),
                _turn(0.5, 1.5, "c"),
            ),
            SpeakerIdentityTrackingErrorCategory.TOO_MANY_LOCAL_SPEAKERS,
        ),
    ],
)
def test_invalid_turn_inputs_fail_closed(
    tracking_request: SpeakerIdentityTrackingRequest,
    category: SpeakerIdentityTrackingErrorCategory,
) -> None:
    tracker = SpeakerIdentityTracker()

    with pytest.raises(SpeakerIdentityTrackingError) as error:
        tracker.track(tracking_request)

    assert error.value.category is category


def test_error_diagnostics_are_fixed_and_privacy_safe() -> None:
    private_marker = "private-customer-name"
    tracker = SpeakerIdentityTracker()
    request = _request(
        0,
        2,
        _turn(0, 2, "speaker", tenant_id=private_marker),
    )

    with pytest.raises(SpeakerIdentityTrackingError) as error:
        tracker.track(request)

    assert str(error.value) == "scope_mismatch"
    assert private_marker not in str(error.value)
