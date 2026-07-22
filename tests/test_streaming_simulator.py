import json
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.events.models import AudioChunkEvent
from app.streaming.simulator import StreamStep, simulate_audio_stream
from scripts.simulate_audio_stream import main


def audio_event(sequence: int, duration: float = 2.0) -> AudioChunkEvent:
    return AudioChunkEvent(
        tenant_id="tenant-a",
        call_id="call-a",
        sequence_number=sequence,
        received_at_utc=datetime(2026, 7, 22, tzinfo=UTC),
        chunk_start_seconds=sequence * 2.0,
        chunk_duration_seconds=duration,
        sample_rate_hz=8_000,
        channel_count=1,
        codec_name="pcm_s16",
        audio_bytes=b"synthetic",
    )


def synthetic_generator(events: list[AudioChunkEvent]):
    def generate(
        path: Path, tenant_id: str, call_id: str, duration: float
    ) -> Iterator[AudioChunkEvent]:
        assert path == Path("synthetic.wav")
        assert tenant_id == "tenant-a"
        assert call_id == "call-a"
        assert duration == 2.0
        return iter(events)

    return generate


def test_fast_simulation_buffers_ordered_chunks_without_sleeping() -> None:
    sleeps: list[float] = []
    events = [audio_event(index) for index in range(3)]

    steps = list(
        simulate_audio_stream(
            Path("synthetic.wav"),
            "tenant-a",
            "call-a",
            window_seconds=3.0,
            sleep_function=sleeps.append,
            chunk_generator=synthetic_generator(events),
        )
    )

    assert sleeps == []
    assert [step.sequence_number for step in steps] == [0, 1, 2]
    assert steps[-1].buffer_start_seconds == 2.0
    assert steps[-1].buffer_end_seconds == 6.0
    assert steps[-1].buffer_duration_seconds == 4.0
    assert steps[-1].buffer_chunk_count == 2
    assert steps[-1].first_buffer_sequence == 1
    assert steps[-1].last_buffer_sequence == 2


def test_realtime_uses_each_actual_chunk_duration() -> None:
    sleeps: list[float] = []
    events = [audio_event(0), audio_event(1, 0.25)]

    list(
        simulate_audio_stream(
            Path("synthetic.wav"),
            "tenant-a",
            "call-a",
            realtime=True,
            sleep_function=sleeps.append,
            chunk_generator=synthetic_generator(events),
        )
    )

    assert sleeps == [2.0, 0.25]


def test_stream_step_is_immutable_and_contains_no_audio() -> None:
    step = next(
        simulate_audio_stream(
            Path("synthetic.wav"),
            "tenant-a",
            "call-a",
            chunk_generator=synthetic_generator([audio_event(0)]),
        )
    )

    assert "audio" not in vars(StreamStep)["__dataclass_fields__"]
    with pytest.raises(FrozenInstanceError):
        step.sequence_number = 9  # type: ignore[misc]


@pytest.mark.parametrize("field", ["tenant", "call"])
def test_rejects_invalid_identifiers_before_generating(field: str) -> None:
    values = {"tenant": "tenant-a", "call": "call-a"}
    values[field] = "  "

    with pytest.raises(ValueError, match=f"{field}_id"):
        simulate_audio_stream(
            Path("synthetic.wav"),
            values["tenant"],
            values["call"],
            chunk_generator=synthetic_generator([]),
        )


def test_cli_prints_safe_json_steps_and_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    step = StreamStep("tenant-a", "call-a", 0, 0.0, 1.25, 0.0, 1.25, 1.25, 1, 0, 0)

    assert (
        main(
            ["synthetic.wav", "--tenant-id", "tenant-a", "--call-id", "call-a"],
            simulator=lambda *args, **kwargs: iter([step]),
        )
        == 0
    )

    lines = capsys.readouterr().out.splitlines()
    chunk_output = json.loads(lines[0])
    summary = json.loads(lines[1])
    assert chunk_output["sequence_number"] == 0
    assert "audio_bytes" not in chunk_output
    assert summary == {
        "type": "summary",
        "total_chunk_count": 1,
        "audio_duration_seconds": 1.25,
    }


@pytest.mark.parametrize(
    "error", [FileNotFoundError("missing"), ValueError("invalid duration")]
)
def test_cli_reports_safe_validation_errors(
    error: Exception, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_simulator(*args: object, **kwargs: object) -> Iterator[StreamStep]:
        raise error

    assert (
        main(
            ["audio.wav", "--tenant-id", "tenant-a", "--call-id", "call-a"],
            simulator=failing_simulator,
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
