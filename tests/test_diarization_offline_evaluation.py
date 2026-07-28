from copy import deepcopy
import json
from pathlib import Path

import pytest

from app.asr.models import (
    ASRWordTimestamp,
    TranscriptionResult,
    TranscriptionSegment,
)
from app.diarization.composition import (
    DiarizationCompositionOutcome,
    DiarizationCompositionReason,
    DiarizationCompositionRequest,
    DiarizationCompositionStatus,
)
from app.diarization.models import DiarizationTurn, SpeakerRole
from app.diarization.offline_evaluation import (
    REPORT_FILENAME,
    DecodedMonoAudio,
    OfflineDiarizationEvaluator,
    OfflineEvaluationReason,
    OfflineEvaluationRequest,
    OfflineEvaluationStatus,
)
from app.diarization.role_resolver import (
    RoleEvidenceCode,
    SpeakerRoleAssignment,
    SpeakerRoleResolutionResult,
)
from app.diarization.routing import (
    CustomerProjectionReason,
    CustomerProjectionStatus,
    CustomerSpeechProjection,
    RoleTaggedWord,
)
from scripts.evaluate_diarization_offline import main


PRIVATE_TEXT = "özel müşteri konuşması"


def _audio(
    *,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    samples: tuple[float, ...] = (0.0,) * 32_000,
) -> DecodedMonoAudio:
    return DecodedMonoAudio(
        tenant_id=tenant_id,
        call_id=call_id,
        sample_rate_hz=16_000,
        samples=samples,
    )


def _transcription() -> TranscriptionResult:
    words = (
        ASRWordTimestamp("Merhaba", 0.0, 0.5, 0.9),
        ASRWordTimestamp(PRIVATE_TEXT, 1.0, 1.5, 0.9),
    )
    return TranscriptionResult(
        text=f"Merhaba {PRIVATE_TEXT}",
        language="tr",
        language_probability=0.9,
        duration_seconds=2.0,
        processing_time_seconds=1.0,
        segments=[
            TranscriptionSegment(
                start_seconds=0.0,
                end_seconds=2.0,
                text=f"Merhaba {PRIVATE_TEXT}",
                words=words,
            )
        ],
    )


def _composition() -> DiarizationCompositionOutcome:
    agent_turn = DiarizationTurn(
        tenant_id="tenant-a",
        call_id="call-a",
        start_seconds=0,
        end_seconds=1,
        local_speaker_ids=("local-a",),
        global_speaker_id="CALL_SPEAKER_0001",
        global_speaker_ids=("CALL_SPEAKER_0001",),
    )
    customer_turn = DiarizationTurn(
        tenant_id="tenant-a",
        call_id="call-a",
        start_seconds=1,
        end_seconds=2,
        local_speaker_ids=("local-b",),
        global_speaker_id="CALL_SPEAKER_0002",
        global_speaker_ids=("CALL_SPEAKER_0002",),
    )
    customer_word = RoleTaggedWord(
        tenant_id="tenant-a",
        call_id="call-a",
        transcript_revision=1,
        start_seconds=1,
        end_seconds=1.5,
        text=PRIVATE_TEXT,
        local_speaker_ids=("local-b",),
        global_speaker_id="CALL_SPEAKER_0002",
        global_speaker_ids=("CALL_SPEAKER_0002",),
        role=SpeakerRole.CUSTOMER,
        role_confidence=1.0,
        role_evidence=RoleEvidenceCode.STRONG_CUSTOMER,
    )
    agent_word = RoleTaggedWord(
        tenant_id="tenant-a",
        call_id="call-a",
        transcript_revision=1,
        start_seconds=0,
        end_seconds=0.5,
        text="Merhaba",
        local_speaker_ids=("local-a",),
        global_speaker_id="CALL_SPEAKER_0001",
        global_speaker_ids=("CALL_SPEAKER_0001",),
        role=SpeakerRole.AGENT,
        role_confidence=1.0,
        role_evidence=RoleEvidenceCode.STRONG_AGENT,
    )
    role_resolution = SpeakerRoleResolutionResult(
        tenant_id="tenant-a",
        call_id="call-a",
        transcript_revision=1,
        assignments=(
            SpeakerRoleAssignment(
                global_speaker_id="CALL_SPEAKER_0001",
                role=SpeakerRole.AGENT,
                confidence=1.0,
                evidence=RoleEvidenceCode.STRONG_AGENT,
            ),
            SpeakerRoleAssignment(
                global_speaker_id="CALL_SPEAKER_0002",
                role=SpeakerRole.CUSTOMER,
                confidence=1.0,
                evidence=RoleEvidenceCode.STRONG_CUSTOMER,
            ),
        ),
    )
    projection = CustomerSpeechProjection(
        tenant_id="tenant-a",
        call_id="call-a",
        transcript_revision=1,
        customer_words=(customer_word,),
        customer_text=PRIVATE_TEXT,
        customer_start_seconds=1,
        customer_end_seconds=1.5,
        excluded_agent_word_count=1,
        excluded_unknown_word_count=0,
        excluded_overlap_word_count=0,
        excluded_below_confidence_word_count=0,
        status=CustomerProjectionStatus.READY,
        reason=CustomerProjectionReason.TRUSTED_CUSTOMER_SPEECH,
    )
    return DiarizationCompositionOutcome(
        status=DiarizationCompositionStatus.COMPLETED,
        reason=DiarizationCompositionReason.COMPOSED,
        tenant_id="tenant-a",
        call_id="call-a",
        transcript_revision=1,
        tracked_turns=(agent_turn, customer_turn),
        role_resolution=role_resolution,
        role_tagged_words=(agent_word, customer_word),
        customer_projection=projection,
    )


class FakeLoader:
    def __init__(
        self,
        audio: DecodedMonoAudio | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.audio = audio or _audio()
        self.fail = fail
        self.calls: list[tuple[Path, str, str]] = []

    def load(
        self,
        audio_path: Path,
        *,
        tenant_id: str,
        call_id: str,
    ) -> DecodedMonoAudio:
        self.calls.append((audio_path, tenant_id, call_id))
        if self.fail:
            raise ValueError("private decoder detail")
        return self.audio


class FakeASR:
    def __init__(
        self,
        result: TranscriptionResult | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.result = result or _transcription()
        self.fail = fail
        self.calls = 0

    def transcribe_audio(self, audio: object) -> TranscriptionResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("private ASR detail")
        return self.result


class FakeComposition:
    def __init__(
        self,
        outcome: DiarizationCompositionOutcome | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.outcome = outcome or _composition()
        self.fail = fail
        self.requests: list[DiarizationCompositionRequest] = []

    def compose(
        self,
        request: DiarizationCompositionRequest,
    ) -> DiarizationCompositionOutcome:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("private composition detail")
        return self.outcome


class RepeatingClock:
    def __init__(self) -> None:
        self.values = iter((10.0, 12.0, 15.0, 10.0, 12.0, 15.0))

    def __call__(self) -> float:
        return next(self.values)


def _request(
    audio_path: Path,
    *,
    output_directory: Path | None = None,
    overwrite: bool = False,
) -> OfflineEvaluationRequest:
    return OfflineEvaluationRequest(
        tenant_id="tenant-a",
        call_id="call-a",
        audio_path=audio_path,
        output_directory=output_directory,
        expected_speaker_count=2,
        overwrite=overwrite,
    )


def _evaluator(
    *,
    loader: FakeLoader | None = None,
    asr: FakeASR | None = None,
    composition: FakeComposition | None = None,
    clock: RepeatingClock | None = None,
) -> OfflineDiarizationEvaluator:
    return OfflineDiarizationEvaluator(
        audio_loader=loader or FakeLoader(),
        asr_engine=asr or FakeASR(),
        composition_processor=composition or FakeComposition(),
        clock=clock or RepeatingClock(),
    )


def test_valid_synthetic_mono_evaluation_has_safe_metrics(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "synthetic.wav"
    audio_path.write_bytes(b"synthetic")
    composition = FakeComposition()

    summary = _evaluator(composition=composition).evaluate(_request(audio_path))

    assert summary.status is OfflineEvaluationStatus.COMPLETED
    assert summary.audio_duration_seconds == 2.0
    assert summary.asr_time_seconds == 2.0
    assert summary.asr_real_time_factor == 1.0
    assert summary.diarization_time_seconds == 3.0
    assert summary.diarization_real_time_factor == 1.5
    assert summary.total_processing_time_seconds == 5.0
    assert summary.total_real_time_factor == 2.5
    assert summary.diarization_turn_count == 2
    assert summary.global_speaker_count == 2
    assert (summary.agent_role_count, summary.customer_role_count) == (1, 1)
    assert summary.projected_customer_word_count == 1
    assert composition.requests[0].diarization_request.mono_audio == _audio().samples


@pytest.mark.parametrize(
    "loader",
    [
        FakeLoader(fail=True),
        FakeLoader(_audio(samples=())),
        FakeLoader(_audio(samples=(0.0, float("nan")))),
    ],
)
def test_non_mono_empty_and_non_finite_audio_fail_safely(
    tmp_path: Path,
    loader: FakeLoader,
) -> None:
    audio_path = tmp_path / "synthetic.wav"
    audio_path.write_bytes(b"synthetic")

    summary = _evaluator(loader=loader).evaluate(_request(audio_path))

    assert summary.status is OfflineEvaluationStatus.FAILED
    assert summary.reason is OfflineEvaluationReason.INVALID_INPUT


def test_wrong_tenant_or_call_scope_is_rejected(tmp_path: Path) -> None:
    audio_path = tmp_path / "synthetic.wav"
    audio_path.write_bytes(b"synthetic")

    summary = _evaluator(
        loader=FakeLoader(_audio(tenant_id="other", call_id="other"))
    ).evaluate(_request(audio_path))

    assert summary.reason is OfflineEvaluationReason.INVALID_INPUT


def test_unsupported_missing_and_unsafe_output_paths_are_rejected(
    tmp_path: Path,
) -> None:
    unsupported = tmp_path / "synthetic.txt"
    unsupported.write_bytes(b"synthetic")
    assert _evaluator().evaluate(_request(unsupported)).reason is (
        OfflineEvaluationReason.UNSUPPORTED_AUDIO
    )
    missing = tmp_path / "missing.wav"
    assert _evaluator().evaluate(_request(missing)).reason is (
        OfflineEvaluationReason.INVALID_INPUT
    )
    audio_path = tmp_path / "synthetic.wav"
    audio_path.write_bytes(b"synthetic")
    assert (
        _evaluator()
        .evaluate(_request(audio_path, output_directory=tmp_path / "missing-output"))
        .reason
        is OfflineEvaluationReason.INVALID_INPUT
    )


def test_component_failures_use_fixed_categories(tmp_path: Path) -> None:
    audio_path = tmp_path / "synthetic.wav"
    audio_path.write_bytes(b"synthetic")

    assert (
        _evaluator(asr=FakeASR(fail=True)).evaluate(_request(audio_path)).reason
        is OfflineEvaluationReason.ASR_FAILED
    )
    assert (
        _evaluator(composition=FakeComposition(fail=True))
        .evaluate(_request(audio_path))
        .reason
        is OfflineEvaluationReason.COMPOSITION_FAILED
    )
    failed_outcome = _composition().__class__(
        status=DiarizationCompositionStatus.FAILED_SAFE,
        reason=DiarizationCompositionReason.DIARIZER_FAILED,
        tenant_id="tenant-a",
        call_id="call-a",
        transcript_revision=1,
    )
    assert (
        _evaluator(composition=FakeComposition(failed_outcome))
        .evaluate(_request(audio_path))
        .reason
        is OfflineEvaluationReason.DIARIZATION_FAILED
    )


def test_optional_json_excludes_text_and_source_path(tmp_path: Path) -> None:
    audio_path = tmp_path / "private-name.wav"
    audio_path.write_bytes(b"synthetic")
    output = tmp_path / "reports"
    output.mkdir()

    summary = _evaluator().evaluate(_request(audio_path, output_directory=output))
    report = (output / REPORT_FILENAME).read_text(encoding="utf-8")
    payload = json.loads(report)

    assert summary.status is OfflineEvaluationStatus.COMPLETED
    assert payload["summary"]["status"] == "completed"
    assert PRIVATE_TEXT not in report
    assert str(audio_path) not in report
    assert audio_path.name not in report
    assert "text" not in payload["words"][0]


def test_atomic_no_overwrite_and_explicit_overwrite(tmp_path: Path) -> None:
    audio_path = tmp_path / "synthetic.wav"
    audio_path.write_bytes(b"synthetic")
    output = tmp_path / "reports"
    output.mkdir()
    evaluator = _evaluator(clock=RepeatingClock())

    first = evaluator.evaluate(_request(audio_path, output_directory=output))
    original = (output / REPORT_FILENAME).read_bytes()
    second = evaluator.evaluate(_request(audio_path, output_directory=output))

    assert first.status is OfflineEvaluationStatus.COMPLETED
    assert second.reason is OfflineEvaluationReason.OUTPUT_FAILED
    assert (output / REPORT_FILENAME).read_bytes() == original

    overwrite_evaluator = _evaluator()
    replaced = overwrite_evaluator.evaluate(
        _request(audio_path, output_directory=output, overwrite=True)
    )
    assert replaced.status is OfflineEvaluationStatus.COMPLETED
    assert not list(output.glob("*.tmp"))


def test_repeated_evaluation_is_deterministic_and_does_not_mutate_inputs(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "synthetic.wav"
    audio_path.write_bytes(b"synthetic")
    audio = _audio()
    loader = FakeLoader(audio)
    transcription = _transcription()
    asr = FakeASR(transcription)
    composition = _composition()
    processor = FakeComposition(composition)
    request = _request(audio_path)
    snapshots = deepcopy((audio, transcription, composition, request))
    evaluator = _evaluator(
        loader=loader,
        asr=asr,
        composition=processor,
        clock=RepeatingClock(),
    )

    first = evaluator.evaluate(request)
    second = evaluator.evaluate(request)

    assert first == second
    assert (audio, transcription, composition, request) == snapshots


def test_summary_repr_and_cli_output_never_expose_text_or_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_path = tmp_path / "private-name.wav"
    audio_path.write_bytes(b"synthetic")
    summary = _evaluator().evaluate(_request(audio_path))

    def runner(request: OfflineEvaluationRequest):
        return summary

    exit_code = main(
        ["tenant-a", "call-a", str(audio_path)],
        runner_factory=lambda request: runner,
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert PRIVATE_TEXT not in repr(summary)
    assert PRIVATE_TEXT not in captured.out
    assert str(audio_path) not in captured.out
    assert audio_path.name not in captured.out
    assert captured.err == ""
