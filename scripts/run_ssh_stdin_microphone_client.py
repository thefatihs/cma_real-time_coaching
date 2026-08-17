"""Stream exact native Windows PCM to a GPU command over SSH stdin."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from queue import Empty, Full, Queue
import re
import shlex
import subprocess
import sys
from threading import Event, Lock, Thread, current_thread
from typing import Any, BinaryIO, Final, Protocol, cast


SAMPLE_RATE_HZ: Final = 16_000
CHANNEL_COUNT: Final = 1
SAMPLE_FORMAT: Final = "int16"
MAX_CAPTURE_BLOCK_BYTES: Final = 64_000
CAPTURE_QUEUE_DEPTH: Final = 8
SHUTDOWN_TIMEOUT_SECONDS: Final = 5.0
_SSH_HOST_PATTERN: Final = re.compile(r"[A-Za-z0-9_.@:-]+")


class NativeInputStream(Protocol):
    def start(self) -> object: ...
    def stop(self) -> object: ...
    def close(self) -> object: ...


class SSHProcess(Protocol):
    stdin: BinaryIO | None

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...


StreamFactory = Callable[
    [Callable[[object, int, object, object], None]], NativeInputStream
]
ProcessFactory = Callable[[Sequence[str]], SSHProcess]


def _sounddevice_stream_factory(
    callback: Callable[[object, int, object, object], None],
) -> NativeInputStream:
    import sounddevice  # type: ignore[import-untyped]

    device = sounddevice.query_devices(kind="input")
    if not device or int(device.get("max_input_channels", 0)) < CHANNEL_COUNT:
        raise LookupError("native_microphone_unavailable")
    return cast(
        NativeInputStream,
        sounddevice.RawInputStream(
            samplerate=SAMPLE_RATE_HZ,
            channels=CHANNEL_COUNT,
            dtype=SAMPLE_FORMAT,
            callback=callback,
        ),
    )


def _ssh_process_factory(command: Sequence[str]) -> SSHProcess:
    return cast(
        SSHProcess,
        subprocess.Popen(list(command), stdin=subprocess.PIPE),
    )


def build_ssh_command(
    *,
    ssh_executable: str,
    ssh_host: str,
    remote_repo: str,
    tenant_key: str,
    call_id: str,
) -> tuple[str, ...]:
    """Build one argument-safe local SSH command with a quoted remote command."""
    if not ssh_executable.strip():
        raise ValueError("invalid_ssh_executable")
    if not _SSH_HOST_PATTERN.fullmatch(ssh_host) or ssh_host.startswith("-"):
        raise ValueError("invalid_ssh_host")
    values = (remote_repo, tenant_key, call_id)
    if any(not value.strip() for value in values):
        raise ValueError("invalid_remote_argument")
    remote_command = " ".join(
        (
            "cd",
            shlex.quote(remote_repo),
            "&&",
            "exec",
            ".venv/bin/python",
            "scripts/run_gpu_stdin_microphone_demo.py",
            "--tenant-key",
            shlex.quote(tenant_key),
            "--call-id",
            shlex.quote(call_id),
        )
    )
    return ssh_executable, ssh_host, remote_command


class NativeSSHStdinCapture:
    """Keep PortAudio callback work bounded and write PCM on one worker."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        stream_factory: StreamFactory = _sounddevice_stream_factory,
        process_factory: ProcessFactory = _ssh_process_factory,
        queue_depth: int = CAPTURE_QUEUE_DEPTH,
    ) -> None:
        if queue_depth <= 0:
            raise ValueError("invalid_capture_queue_depth")
        self._command = tuple(command)
        self._stream_factory = stream_factory
        self._process_factory = process_factory
        self._queue: Queue[bytes] = Queue(maxsize=queue_depth)
        self._admitting = Event()
        self._stop = Event()
        self._drained = Event()
        self._drained.set()
        self._failed = Event()
        self._lock = Lock()
        self._stream: NativeInputStream | None = None
        self._process: SSHProcess | None = None
        self._worker: Thread | None = None

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    @property
    def process_running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    def wait(self, timeout_seconds: float) -> None:
        self._stop.wait(timeout=timeout_seconds)

    def start(self) -> bool:
        with self._lock:
            if self._worker is not None:
                return False
            process = self._process_factory(self._command)
            try:
                if process.stdin is None:
                    raise RuntimeError("ssh_stdin_unavailable")
                stream = self._stream_factory(self.audio_callback)
            except Exception:
                if process.stdin is not None:
                    process.stdin.close()
                process.terminate()
                raise
            worker = Thread(
                target=self._write_worker,
                name="ssh-stdin-microphone-writer",
                daemon=True,
            )
            self._process = process
            self._stream = stream
            self._worker = worker
            worker.start()
            try:
                stream.start()
            except Exception:
                self._stop.set()
                process.stdin.close()
                process.terminate()
                stream.close()
                raise RuntimeError("native_microphone_open_failed") from None
            self._admitting.set()
            return True

    def audio_callback(
        self,
        input_data: object,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        del frames, time_info, status
        if not self._admitting.is_set():
            return
        pcm = bytes(cast(Any, input_data))
        if not pcm or len(pcm) > MAX_CAPTURE_BLOCK_BYTES or len(pcm) % 2:
            self._failed.set()
            self._admitting.clear()
            return
        self._drained.clear()
        try:
            self._queue.put_nowait(pcm)
        except Full:
            self._failed.set()
            self._admitting.clear()

    def close(self) -> int:
        self._admitting.clear()
        self._stop.set()
        with self._lock:
            stream, self._stream = self._stream, None
            process = self._process
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                self._failed.set()
            try:
                stream.close()
            except Exception:
                self._failed.set()
        if not self._drained.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS):
            self._failed.set()
        worker = self._worker
        if worker is not None and worker is not current_thread():
            worker.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        if process is None:
            return 1 if self.failed else 0
        stdin = process.stdin
        if stdin is not None and not stdin.closed:
            stdin.close()
        try:
            return process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.terminate()
            return process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)

    def _write_worker(self) -> None:
        process = self._process
        assert process is not None and process.stdin is not None
        stdin = process.stdin
        while not self._stop.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                stdin.write(payload)
                stdin.flush()
            except (BrokenPipeError, OSError):
                self._failed.set()
                self._admitting.clear()
                self._stop.set()
            finally:
                self._queue.task_done()
                if self._queue.empty():
                    self._drained.set()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Windows mikrofonunu SSH stdin ile GPU demosuna aktar."
    )
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--remote-repo", required=True)
    parser.add_argument("--tenant-key", default="tenant_alpha")
    parser.add_argument("--call-id", default="internship-demo")
    parser.add_argument("--ssh-executable", default="ssh")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        command = build_ssh_command(
            ssh_executable=args.ssh_executable,
            ssh_host=args.ssh_host,
            remote_repo=args.remote_repo,
            tenant_key=args.tenant_key,
            call_id=args.call_id,
        )
        capture = NativeSSHStdinCapture(command)
        capture.start()
    except (LookupError, OSError, RuntimeError, ValueError):
        print("Mikrofon/SSH fallback başlatılamadı.", file=sys.stderr)
        return 1
    print("Mikrofon akışı başladı. Durdurmak için Ctrl+C.")
    try:
        while not capture.failed:
            if not capture.process_running:
                break
            capture.wait(0.25)
    except KeyboardInterrupt:
        pass
    result = capture.close()
    return result if result != 0 else int(capture.failed)


if __name__ == "__main__":
    raise SystemExit(main())
