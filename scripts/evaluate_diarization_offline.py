import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from collections.abc import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.asr.faster_whisper_engine import FasterWhisperEngine  # noqa: E402
from app.diarization.composition import OfflineDiarizationComposer  # noqa: E402
from app.diarization.identity_tracker import SpeakerIdentityTracker  # noqa: E402
from app.diarization.offline_evaluation import (  # noqa: E402
    OfflineDiarizationEvaluator,
    OfflineEvaluationReason,
    OfflineEvaluationRequest,
    OfflineEvaluationStatus,
    OfflineEvaluationSummary,
    OfflineMonoAudioLoader,
)
from app.diarization.pyannote_backend import PyannoteSpeakerDiarizer  # noqa: E402
from app.diarization.role_resolver import RuleBasedSpeakerRoleResolver  # noqa: E402
from app.diarization.routing import CustomerSpeechProjector  # noqa: E402


EvaluationRunner = Callable[[OfflineEvaluationRequest], OfflineEvaluationSummary]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one privacy-safe local offline diarization evaluation."
    )
    parser.add_argument("tenant_id")
    parser.add_argument("call_id")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--expected-speakers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_default_runner(
    request: OfflineEvaluationRequest,
) -> OfflineDiarizationEvaluator:
    diarizer = PyannoteSpeakerDiarizer(
        tenant_id=request.tenant_id,
        call_id=request.call_id,
        fixed_two_speakers=request.expected_speaker_count == 2,
        max_speakers=request.expected_speaker_count,
    )
    composer = OfflineDiarizationComposer(
        diarizer=diarizer,
        identity_tracker=SpeakerIdentityTracker(
            max_local_speakers=request.expected_speaker_count
        ),
        role_resolver=RuleBasedSpeakerRoleResolver(),
        customer_projector=CustomerSpeechProjector(),
    )
    return OfflineDiarizationEvaluator(
        audio_loader=OfflineMonoAudioLoader(),
        asr_engine=FasterWhisperEngine(word_timestamps=True),
        composition_processor=composer,
    )


def safe_summary_json(summary: OfflineEvaluationSummary) -> str:
    payload = {
        **asdict(summary),
        "status": summary.status.value,
        "reason": summary.reason.value,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def main(
    argv: Sequence[str] | None = None,
    runner_factory: Callable[
        [OfflineEvaluationRequest], EvaluationRunner | OfflineDiarizationEvaluator
    ] = build_default_runner,
) -> int:
    args = build_parser().parse_args(argv)
    request = OfflineEvaluationRequest(
        tenant_id=args.tenant_id,
        call_id=args.call_id,
        audio_path=args.audio_path,
        output_directory=args.output_directory,
        expected_speaker_count=args.expected_speakers,
        overwrite=args.overwrite,
    )
    try:
        runner = runner_factory(request)
        summary = (
            runner.evaluate(request)
            if isinstance(runner, OfflineDiarizationEvaluator)
            else runner(request)
        )
    except Exception:
        summary = OfflineEvaluationSummary(
            status=OfflineEvaluationStatus.FAILED,
            reason=OfflineEvaluationReason.INVALID_INPUT,
            audio_duration_seconds=None,
            asr_time_seconds=None,
            asr_real_time_factor=None,
            diarization_time_seconds=None,
            diarization_real_time_factor=None,
            total_processing_time_seconds=None,
            total_real_time_factor=None,
            diarization_turn_count=0,
            global_speaker_count=0,
            agent_role_count=0,
            customer_role_count=0,
            unknown_role_count=0,
            projected_customer_word_count=0,
            excluded_agent_word_count=0,
            excluded_unknown_word_count=0,
            excluded_overlap_word_count=0,
            excluded_below_confidence_word_count=0,
            skipped_zero_duration_word_count=0,
            transcript_revision=0,
        )
    print(safe_summary_json(summary))
    return 0 if summary.status is OfflineEvaluationStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
