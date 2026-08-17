from __future__ import annotations

from io import BytesIO, StringIO
from threading import Event, get_ident
from types import SimpleNamespace
from datetime import UTC, datetime, timedelta
from typing import Any, BinaryIO

import pytest

from scripts.run_gpu_stdin_microphone_demo import (
    _render_result,
    _render_step,
    iter_stdin_audio_events,
    iter_stdin_pcm_chunks,
)
from scripts.run_ssh_stdin_microphone_client import (
    MAX_CAPTURE_BLOCK_BYTES,
    NativeSSHStdinCapture,
    build_ssh_command,
)


class RetainedSink(BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.closed_by_client = 0
        self.write_threads: list[int] = []

    def write(self, data: Any) -> int:
        self.write_threads.append(get_ident())
        return super().write(data)

    def close(self) -> None:
        self.closed_by_client += 1


class FakeProcess:
    def __init__(self, sink: BinaryIO | None = None) -> None:
        self.stdin = sink
        self.terminated = 0
        self.waited = 0

    def poll(self) -> int | None:
        return None if not self.terminated else 1

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.waited += 1
        return int(bool(self.terminated))

    def terminate(self) -> None:
        self.terminated += 1


class FakeStream:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def close(self) -> None:
        self.closed += 1


def test_ssh_command_is_argument_safe_and_contains_no_transport_protocol() -> None:
    command = build_ssh_command(
        ssh_executable="ssh",
        ssh_host="gpu-demo",
        remote_repo="/srv/callmetric demo",
        tenant_key="tenant_alpha",
        call_id="internship demo",
    )

    assert command[:2] == ("ssh", "gpu-demo")
    assert "run_gpu_stdin_microphone_demo.py" in command[2]
    assert "'/srv/callmetric demo'" in command[2]
    assert "'internship demo'" in command[2]
    assert "START" not in command[2]
    assert "token" not in command[2].lower()


@pytest.mark.parametrize("host", ("-oProxyCommand=x", "gpu demo", ""))
def test_ssh_host_rejects_option_or_whitespace_injection(host: str) -> None:
    with pytest.raises(ValueError, match="invalid_ssh_host"):
        build_ssh_command(
            ssh_executable="ssh",
            ssh_host=host,
            remote_repo="/srv/callmetric",
            tenant_key="tenant_alpha",
            call_id="demo",
        )


def test_native_callback_writes_pcm_only_from_worker_and_closes_cleanly() -> None:
    sink = RetainedSink()
    process = FakeProcess(sink)
    stream = FakeStream()
    callbacks: list[object] = []
    capture = NativeSSHStdinCapture(
        ("ssh", "gpu-demo", "command"),
        stream_factory=lambda callback: callbacks.append(callback) or stream,
        process_factory=lambda _command: process,
    )

    assert capture.start()
    callback_thread = get_ident()
    callback = callbacks[0]
    assert callable(callback)
    callback(b"\1\0" * 800, 800, object(), object())
    for _attempt in range(1_000):
        if sink.write_threads:
            break
        Event().wait(0.001)
    assert sink.getvalue() == b"\1\0" * 800
    assert callback_thread not in sink.write_threads

    assert capture.close() == 0
    assert stream.started == stream.stopped == stream.closed == 1
    assert sink.closed_by_client == 1


def test_native_callback_rejects_odd_or_oversized_payload_without_writing() -> None:
    for payload in (b"\1", b"\0\0" * (MAX_CAPTURE_BLOCK_BYTES // 2 + 1)):
        sink = RetainedSink()
        process = FakeProcess(sink)
        callbacks: list[object] = []
        capture = NativeSSHStdinCapture(
            ("ssh", "gpu-demo", "command"),
            stream_factory=lambda callback: callbacks.append(callback) or FakeStream(),
            process_factory=lambda _command: process,
        )
        capture.start()
        callback = callbacks[0]
        assert callable(callback)
        callback(payload, 1, object(), object())
        assert capture.failed
        assert capture.close() == 0
        assert sink.getvalue() == b""


def test_stdin_reader_aggregates_full_chunk_and_valid_even_tail() -> None:
    payload = b"\1\0" * 40_000
    chunks = tuple(iter_stdin_pcm_chunks(BytesIO(payload), chunk_bytes=64_000))

    assert tuple(map(len, chunks)) == (64_000, 16_000)
    assert b"".join(chunks) == payload


def test_stdin_reader_rejects_odd_terminal_payload() -> None:
    with pytest.raises(ValueError, match="odd_pcm_stdin_payload"):
        tuple(iter_stdin_pcm_chunks(BytesIO(b"\1\0\1"), chunk_bytes=64_000))


class CountingStream(BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.read_count = 0

    def read(self, size: int | None = -1) -> bytes:
        self.read_count += 1
        return super().read(size)


def test_gpu_events_are_lazy_ordered_scoped_and_timestamped_through_eof() -> None:
    stream = CountingStream(b"\2\0" * 33_000)
    times = iter(
        (
            datetime(2026, 8, 17, tzinfo=UTC),
            datetime(2026, 8, 17, tzinfo=UTC) + timedelta(seconds=2),
        )
    )
    events = iter_stdin_audio_events(
        stream,
        tenant_id="tenant-a",
        call_id="call-a",
        utc_now=lambda: next(times),
    )

    assert stream.read_count == 0
    first = next(events)
    assert stream.read_count == 1
    assert (first.tenant_id, first.call_id, first.sequence_number) == (
        "tenant-a",
        "call-a",
        0,
    )
    assert (first.chunk_start_seconds, first.chunk_duration_seconds) == (0.0, 2.0)
    second = next(events)
    assert (second.sequence_number, second.chunk_start_seconds) == (1, 2.0)
    assert second.chunk_duration_seconds == pytest.approx(0.0625)
    assert second.received_at_utc > first.received_at_utc
    assert tuple(events) == ()
    assert stream.read_count == 3


def test_demo_output_renders_transcript_setfit_and_coaching() -> None:
    label = SimpleNamespace(name="price_objection", score=0.91)
    classification = SimpleNamespace(
        classification_event=SimpleNamespace(labels=(label,))
    )
    suggestion = SimpleNamespace(
        title="Fiyat itirazı", suggestion="Seçenekleri açıklayın"
    )
    coaching = SimpleNamespace(
        result=SimpleNamespace(displayed_suggestions=(suggestion,))
    )
    step = SimpleNamespace(
        stable_transcript="Bu fiyat pahalı.",
        partial_transcript="",
        classification_outcomes=(classification,),
        coaching_outcomes=(coaching,),
    )
    result = SimpleNamespace(
        stable_transcript="Bu fiyat pahalı.",
        partial_transcript="",
        classification_outcomes=(classification,),
        coaching_outcomes=(coaching,),
    )
    output = StringIO()

    _render_step(step, output)  # type: ignore[arg-type]
    _render_result(result, output)  # type: ignore[arg-type]

    rendered = output.getvalue()
    assert "TRANSCRIPT: Bu fiyat pahalı." in rendered
    assert "SETFIT: price_objection (0.91)" in rendered
    assert "COACHING: Fiyat itirazı | Seçenekleri açıklayın" in rendered
