from pathlib import Path
import socket

import pytest

from app.diarization import (
    DiarizationRequest,
    DiarizationResult,
    DiarizationTurn,
    FakeSpeakerDiarizer,
    SpeakerDiarizerProtocol,
    SpeakerRole,
)


def request(**changes: object) -> DiarizationRequest:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "window_start_seconds": 4.0,
        "window_end_seconds": 6.0,
        "sample_rate_hz": 16_000,
        "mono_audio": (0.0, 0.1, -0.1),
    }
    values.update(changes)
    return DiarizationRequest.model_validate(values)


def result(**changes: object) -> DiarizationResult:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "window_start_seconds": 4.0,
        "window_end_seconds": 6.0,
        "turns": (
            DiarizationTurn(
                tenant_id="tenant_alpha",
                call_id="call_001",
                start_seconds=4.2,
                end_seconds=5.8,
                local_speaker_ids=("speaker_0",),
                role=SpeakerRole.UNKNOWN,
            ),
        ),
    }
    values.update(changes)
    return DiarizationResult.model_validate(values)


def diarize(
    subject: SpeakerDiarizerProtocol,
    source_request: DiarizationRequest,
) -> DiarizationResult:
    return subject.diarize(source_request)


def test_fake_backend_returns_preconfigured_immutable_result() -> None:
    expected = result()
    subject = FakeSpeakerDiarizer(expected)

    first = diarize(subject, request())
    second = diarize(subject, request())

    assert first is expected
    assert second is expected
    assert first.turns == expected.turns
    assert subject.__dict__ == {"_result": expected}


@pytest.mark.parametrize(
    "changes",
    [
        {"tenant_id": "tenant_beta"},
        {"call_id": "call_002"},
        {"window_start_seconds": 3.0},
        {"window_end_seconds": 7.0},
    ],
)
def test_fake_backend_rejects_wrong_scope_without_audio_diagnostics(
    changes: dict[str, object],
) -> None:
    subject = FakeSpeakerDiarizer(result())
    marker = 0.314159265

    with pytest.raises(ValueError) as error:
        subject.diarize(request(mono_audio=(marker,), **changes))

    assert str(marker) not in str(error.value)
    assert "PRIVATE" not in str(error.value)


def test_fake_backend_performs_no_filesystem_or_network_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("external access is forbidden")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    subject = FakeSpeakerDiarizer(result())

    assert subject.diarize(request()) == result()


def test_fake_backend_repr_does_not_contain_audio_or_private_paths() -> None:
    subject = FakeSpeakerDiarizer(result())
    source_request = request(mono_audio=(0.2718281828,))

    assert "0.2718281828" not in repr(source_request)
    assert "CallMetricPrivate" not in repr(source_request)
    assert "CallMetricPrivate" not in repr(subject)
