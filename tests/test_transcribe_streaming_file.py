from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.events.models import TranscriptEvent, TranscriptKind
from app.streaming.pipeline import StreamingASRResult, StreamingASRStep
from app.streaming.window_transcriber import WindowTranscriber
from app.tenancy.models import TenantASRConfig, TenantContext
from scripts.transcribe_streaming_file import build_parser, main, write_transcript


NOW = datetime(2026, 7, 22, tzinfo=UTC)


def event(kind: TranscriptKind) -> TranscriptEvent:
    return TranscriptEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        event_id=f"event-{kind.value}",
        kind=kind,
        text="synthetic text",
        start_seconds=0.0,
        end_seconds=1.0,
        revision=1,
        created_at_utc=NOW,
    )


def make_result() -> StreamingASRResult:
    step = StreamingASRStep(
        tenant_id="tenant_alpha",
        call_id="call_001",
        sequence_number=0,
        chunk_start_seconds=0.0,
        chunk_end_seconds=2.0,
        window_start_seconds=0.0,
        window_end_seconds=2.0,
        window_duration_seconds=2.0,
        raw_window_text="synthetic draft",
        transcript_events=(event(TranscriptKind.PARTIAL),),
        stable_transcript="stable",
        partial_transcript="draft",
        transcription_time_seconds=0.25,
    )
    return StreamingASRResult(
        tenant_id="tenant_alpha",
        call_id="call_001",
        steps=(step,),
        final_event=event(TranscriptKind.FINAL),
        stable_transcript="stable final",
        partial_transcript="",
        total_chunks=1,
        audio_duration_seconds=2.0,
    )


class FakePipeline:
    def __init__(
        self, result: StreamingASRResult | None = None, error: Exception | None = None
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[Path, str]] = []

    def run(self, audio_path: Path, call_id: str) -> StreamingASRResult:
        self.calls.append((audio_path, call_id))
        if self.error:
            raise self.error
        assert self.result is not None
        return self.result


def base_args(path: Path) -> list[str]:
    return [str(path), "--tenant-id", "tenant_alpha", "--call-id", "call_001"]


def test_valid_argument_parsing() -> None:
    args = build_parser().parse_args(
        ["audio.wav", "--tenant-id", "tenant", "--call-id", "call"]
    )
    assert (args.model, args.language, args.beam_size, args.cpu_threads) == (
        "large-v3",
        "tr",
        5,
        4,
    )
    assert (args.chunk_duration, args.window_seconds, args.stable_region_seconds) == (
        2.0,
        20.0,
        5.0,
    )


@pytest.mark.parametrize("missing", ["tenant", "call"])
def test_tenant_and_call_ids_are_required(missing: str) -> None:
    args = (
        ["audio.wav", "--call-id", "call"]
        if missing == "tenant"
        else ["audio.wav", "--tenant-id", "tenant"]
    )
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(args)
    assert error.value.code == 2


def test_successful_execution_safe_configuration_and_final_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = tmp_path / "synthetic.wav"
    audio.touch()
    fake = FakePipeline(make_result())
    engine_settings: dict[str, object] = {}
    pipeline_inputs: list[object] = []

    def engine_factory(**settings: object) -> object:
        engine_settings.update(settings)
        return object()

    def pipeline_factory(
        tenant: TenantContext,
        settings: TenantASRConfig,
        transcriber: WindowTranscriber,
    ) -> FakePipeline:
        pipeline_inputs.extend((tenant, settings, transcriber))
        return fake

    code = main(
        base_args(audio),
        engine_factory=engine_factory,
        pipeline_factory=pipeline_factory,
        clock=iter([10.0, 11.0]).__next__,
    )
    output = capsys.readouterr().out
    assert code == 0
    assert fake.calls == [(audio, "call_001")]
    assert (
        engine_settings["device"] == "cpu" and engine_settings["compute_type"] == "int8"
    )
    assert getattr(pipeline_inputs[0], "tenant_id") == "tenant_alpha"
    assert getattr(pipeline_inputs[1], "rolling_window_seconds") == 20.0
    assert "Streaming ASR configuration:" in output
    assert "Initial prompt configured: False" in output
    assert str(audio) not in output
    assert "Final transcript:\nstable final" in output
    assert "Total chunks: 1" in output
    assert "Approximate overall RTF: 0.500" in output


def test_show_steps_enabled_and_disabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = tmp_path / "synthetic.wav"
    audio.touch()
    fake = FakePipeline(make_result())

    def factory(
        tenant: TenantContext,
        settings: TenantASRConfig,
        transcriber: WindowTranscriber,
    ) -> FakePipeline:
        return fake

    assert (
        main(
            base_args(audio),
            engine_factory=lambda **kwargs: object(),
            pipeline_factory=factory,
            clock=iter([0.0, 1.0]).__next__,
        )
        == 0
    )
    assert "Chunk 0:" not in capsys.readouterr().out
    assert (
        main(
            [*base_args(audio), "--show-steps"],
            engine_factory=lambda **kwargs: object(),
            pipeline_factory=factory,
            clock=iter([0.0, 1.0]).__next__,
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Chunk 0: window=0.00-2.00s processing=0.25s events=PARTIAL" in output
    assert "stable='stable' partial='draft'" in output


def test_output_text_creates_parents_and_writes_only_final_transcript(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "synthetic.wav"
    audio.touch()
    output = tmp_path / "nested" / "transcript.txt"
    code = main(
        [*base_args(audio), "--output-text", str(output)],
        engine_factory=lambda **kwargs: object(),
        pipeline_factory=lambda *args: FakePipeline(make_result()),
        clock=iter([0.0, 1.0]).__next__,
    )
    assert code == 0
    assert output.read_text(encoding="utf-8") == "stable final"


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_invalid_audio_path_is_rejected_before_construction(
    tmp_path: Path, kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "missing.wav" if kind == "missing" else tmp_path
    constructed: list[object] = []
    code = main(
        base_args(path), engine_factory=lambda **kwargs: constructed.append(kwargs)
    )
    assert code == 1
    assert (
        "not found" if kind == "missing" else "directory"
    ) in capsys.readouterr().err
    assert constructed == []


@pytest.mark.parametrize(
    "option", ["--chunk-duration", "--window-seconds", "--stable-region-seconds"]
)
def test_invalid_durations_are_rejected(option: str) -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(
            ["audio.wav", "--tenant-id", "tenant", "--call-id", "call", option, "0"]
        )
    assert error.value.code == 2


def test_pipeline_failure_returns_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = tmp_path / "synthetic.wav"
    audio.touch()
    fake = FakePipeline(error=RuntimeError("synthetic pipeline failure"))
    code = main(
        base_args(audio),
        engine_factory=lambda **kwargs: object(),
        pipeline_factory=lambda *args: fake,
    )
    assert code == 1
    assert (
        "Streaming transcription failed: synthetic pipeline failure"
        in capsys.readouterr().err
    )


def test_no_binary_audio_appears_in_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = tmp_path / "synthetic.wav"
    audio.touch()
    main(
        base_args(audio),
        engine_factory=lambda **kwargs: object(),
        pipeline_factory=lambda *args: FakePipeline(make_result()),
        clock=iter([0.0, 1.0]).__next__,
    )
    output = capsys.readouterr().out.encode()
    assert b"audio_bytes" not in output
    assert bytes([1, 0]) not in output


def test_writer_uses_utf8(tmp_path: Path) -> None:
    output = tmp_path / "new" / "transcript.txt"
    write_transcript(output, "synthetic")
    assert output.read_text(encoding="utf-8") == "synthetic"
