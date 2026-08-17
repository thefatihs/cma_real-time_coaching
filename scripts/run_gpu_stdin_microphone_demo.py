"""Run the existing live CallMetric pipeline from exact PCM16 stdin."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
import sys
from threading import Event
from typing import BinaryIO, Final, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.asr.faster_whisper_engine import FasterWhisperEngine  # noqa: E402
from app.audio_ingress.local_microphone import (  # noqa: E402
    LOCAL_MIC_CHUNK_BYTES,
    LOCAL_MIC_GATE_ENVIRONMENT_KEY,
    LOCAL_MIC_CHANNEL_COUNT,
    LOCAL_MIC_SAMPLE_RATE_HZ,
    LocalMicTestCapability,
    create_local_mic_test_capability,
)
from app.classification.runtime import RuntimeSetFitClassifier  # noqa: E402
from app.events.models import AudioChunkEvent  # noqa: E402
from app.streaming.pipeline import (  # noqa: E402
    StreamingASRPipeline,
    StreamingASRResult,
    StreamingASRStep,
)
from app.streaming.window_transcriber import WindowTranscriber  # noqa: E402
from live_dashboard.demo_data import tenant_demos  # noqa: E402
from live_dashboard.runtime_wiring import (  # noqa: E402
    ArtifactAvailability,
    DashboardServiceSelection,
    build_live_pipeline,
    inspect_default_artifacts,
)
from live_dashboard.view_models import create_local_execution  # noqa: E402


STDIN_PCM_CODEC: Final = "pcm_s16le"


def iter_stdin_pcm_chunks(
    stream: BinaryIO,
    *,
    chunk_bytes: int = LOCAL_MIC_CHUNK_BYTES,
) -> Iterator[bytes]:
    """Aggregate arbitrary stdin reads into bounded even-length PCM chunks."""
    if chunk_bytes <= 0 or chunk_bytes % 2:
        raise ValueError("invalid_pcm_chunk_size")
    buffer = bytearray()
    while True:
        data = stream.read(chunk_bytes - len(buffer))
        if not data:
            break
        buffer.extend(data)
        if len(buffer) == chunk_bytes:
            yield bytes(buffer)
            buffer.clear()
    if len(buffer) % 2:
        raise ValueError("odd_pcm_stdin_payload")
    if buffer:
        yield bytes(buffer)


def iter_stdin_audio_events(
    stream: BinaryIO,
    *,
    tenant_id: str,
    call_id: str,
    chunk_bytes: int = LOCAL_MIC_CHUNK_BYTES,
    utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Iterator[AudioChunkEvent]:
    """Read PCM lazily as downstream requests correctly scoped audio events."""
    emitted_samples = 0
    for sequence_number, payload in enumerate(
        iter_stdin_pcm_chunks(stream, chunk_bytes=chunk_bytes)
    ):
        sample_count = len(payload) // 2
        yield AudioChunkEvent(
            tenant_id=tenant_id,
            call_id=call_id,
            sequence_number=sequence_number,
            received_at_utc=utc_now(),
            chunk_start_seconds=emitted_samples / LOCAL_MIC_SAMPLE_RATE_HZ,
            chunk_duration_seconds=sample_count / LOCAL_MIC_SAMPLE_RATE_HZ,
            sample_rate_hz=LOCAL_MIC_SAMPLE_RATE_HZ,
            channel_count=LOCAL_MIC_CHANNEL_COUNT,
            codec_name=STDIN_PCM_CODEC,
            audio_bytes=payload,
        )
        emitted_samples += sample_count


def _render_step(step: StreamingASRStep, output: TextIO) -> None:
    transcript = step.stable_transcript or step.partial_transcript
    if transcript:
        print(f"TRANSCRIPT: {transcript}", file=output, flush=True)
    for outcome in step.classification_outcomes:
        event = outcome.classification_event
        if event is not None:
            labels = ", ".join(
                f"{label.name} ({label.score:.2f})" for label in event.labels
            )
            print(f"SETFIT: {labels or '-'}", file=output, flush=True)
    for outcome in step.coaching_outcomes:
        if outcome.result is None:
            continue
        for suggestion in outcome.result.displayed_suggestions:
            print(
                f"COACHING: {suggestion.title} | {suggestion.suggestion}",
                file=output,
                flush=True,
            )


def _render_result(result: StreamingASRResult, output: TextIO) -> None:
    transcript = result.stable_transcript or result.partial_transcript
    print(f"FINAL TRANSCRIPT: {transcript or '-'}", file=output, flush=True)
    for outcome in result.classification_outcomes:
        event = outcome.classification_event
        if event is not None:
            labels = ", ".join(label.name for label in event.labels)
            print(f"FINAL SETFIT: {labels or '-'}", file=output, flush=True)
    for outcome in result.coaching_outcomes:
        if outcome.result is None:
            continue
        for suggestion in outcome.result.displayed_suggestions:
            print(
                f"FINAL COACHING: {suggestion.title} | {suggestion.suggestion}",
                file=output,
                flush=True,
            )


def build_pipeline(
    tenant_key: str,
    call_id: str,
) -> tuple[
    StreamingASRPipeline,
    FasterWhisperEngine,
    LocalMicTestCapability,
    object,
]:
    demos = tenant_demos()
    tenant = demos.get(tenant_key)
    if tenant is None:
        raise ValueError("unknown_tenant_key")
    availability = inspect_default_artifacts()
    if not availability.compatible:
        raise RuntimeError("setfit_artifacts_unavailable")
    state = create_local_execution(tenant, call_id)
    engine = FasterWhisperEngine(
        model_size="large-v3",
        device="cuda",
        compute_type="float16",
        language=tenant.config.asr.language,
        beam_size=tenant.config.asr.beam_size,
        vad_filter=tenant.config.asr.vad_filter,
        condition_on_previous_text=(tenant.config.asr.condition_on_previous_text),
        initial_prompt=tenant.config.asr.initial_prompt,
    )
    engine.prepare()
    pipeline = build_live_pipeline(
        state.runtime,
        WindowTranscriber(engine),
        selection=DashboardServiceSelection(True, True),
        availability=ArtifactAvailability(True),
        classifier_provider=RuntimeSetFitClassifier,
        integration=None,
    )
    resource = object()
    capability = create_local_mic_test_capability(
        tenant_id=tenant.config.context.tenant_id,
        call_id=call_id,
        resource=resource,
        server_address="127.0.0.1",
        environment={LOCAL_MIC_GATE_ENVIRONMENT_KEY: "1"},
    )
    return pipeline, engine, capability, resource


def run_demo(
    *,
    tenant_key: str,
    call_id: str,
    stdin: BinaryIO,
    output: TextIO,
    pipeline_builder: Callable[
        [str, str],
        tuple[
            StreamingASRPipeline,
            FasterWhisperEngine,
            LocalMicTestCapability,
            object,
        ],
    ] = build_pipeline,
) -> int:
    pipeline, engine, capability, resource = pipeline_builder(tenant_key, call_id)
    cancellation = Event()
    try:
        chunks = iter_stdin_audio_events(
            stdin,
            tenant_id=capability.tenant_id,
            call_id=call_id,
        )
        result = pipeline.run_live(
            chunks,
            call_id,
            capability=capability,
            execution_resource=resource,
            cancellation=cancellation,
            step_callback=lambda step: _render_step(step, output),
            retain_history=False,
        )
        _render_result(result, output)
        return 0
    except KeyboardInterrupt:
        cancellation.set()
        return 130
    except Exception:
        cancellation.set()
        return 1
    finally:
        capability.revoke()
        engine.release_model()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PCM16 stdin akışını mevcut GPU pipeline'ında işle."
    )
    parser.add_argument("--tenant-key", default="tenant_alpha")
    parser.add_argument("--call-id", default="internship-demo")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_demo(
            tenant_key=args.tenant_key,
            call_id=args.call_id,
            stdin=sys.stdin.buffer,
            output=sys.stdout,
        )
    except (OSError, RuntimeError, ValueError):
        print("GPU stdin mikrofon demosu güvenli biçimde sonlandı.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
