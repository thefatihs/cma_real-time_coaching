"""Optional, in-memory pyannote speaker diarization backend."""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from math import isfinite
from threading import Lock
from typing import Any, cast

from app.diarization.models import (
    DiarizationRequest,
    DiarizationResult,
    DiarizationTurn,
    SpeakerRole,
)

DEFAULT_PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"
MAX_TERMINAL_END_OVERRUN_SECONDS = 0.050


class PyannoteDiarizationErrorCategory(str, Enum):
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    MODEL_LOAD_FAILED = "model_load_failed"
    INFERENCE_FAILED = "inference_failed"
    MALFORMED_OUTPUT = "malformed_output"
    SCOPE_MISMATCH = "scope_mismatch"


class PyannoteDiarizationError(RuntimeError):
    """Safe backend failure containing no provider exception details."""

    def __init__(self, category: PyannoteDiarizationErrorCategory) -> None:
        self.category = category
        super().__init__(category.value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(category={self.category.value!r})"


class PyannoteSpeakerDiarizer:
    """Lazily run one pyannote pipeline for a trusted call scope."""

    _telemetry_lock = Lock()

    def __init__(
        self,
        *,
        tenant_id: str,
        call_id: str,
        model_id: str = DEFAULT_PYANNOTE_MODEL_ID,
        device: str = "cpu",
        fixed_two_speakers: bool = True,
        max_speakers: int = 2,
    ) -> None:
        self._tenant_id = _required_text(tenant_id)
        self._call_id = _required_text(call_id)
        self._model_id = _required_text(model_id)
        self._device = _required_text(device)
        if max_speakers <= 0:
            raise ValueError("max_speakers must be positive")
        self._fixed_two_speakers = fixed_two_speakers
        self._max_speakers = max_speakers
        self._pipeline: Any | None = None

    def diarize(self, request: DiarizationRequest) -> DiarizationResult:
        if request.tenant_id != self._tenant_id or request.call_id != self._call_id:
            raise PyannoteDiarizationError(
                PyannoteDiarizationErrorCategory.SCOPE_MISMATCH
            )

        pipeline, torch = self._get_pipeline()
        try:
            waveform = torch.tensor(
                request.mono_audio,
                dtype=torch.float32,
                device="cpu",
            ).reshape(1, -1)
            audio = {
                "waveform": waveform,
                "sample_rate": request.sample_rate_hz,
            }
            with self._telemetry_disabled():
                if self._fixed_two_speakers:
                    raw_output = pipeline(audio, num_speakers=2)
                else:
                    raw_output = pipeline(audio)
            turns = self._convert_output(raw_output, request)
        except PyannoteDiarizationError:
            raise
        except Exception:
            raise PyannoteDiarizationError(
                PyannoteDiarizationErrorCategory.INFERENCE_FAILED
            ) from None

        return DiarizationResult(
            tenant_id=request.tenant_id,
            call_id=request.call_id,
            window_start_seconds=request.window_start_seconds,
            window_end_seconds=request.window_end_seconds,
            turns=turns,
        )

    def _get_pipeline(self) -> tuple[Any, Any]:
        if self._pipeline is not None:
            try:
                torch = importlib.import_module("torch")
            except Exception:
                raise PyannoteDiarizationError(
                    PyannoteDiarizationErrorCategory.DEPENDENCY_UNAVAILABLE
                ) from None
            return self._pipeline, torch

        try:
            torch = importlib.import_module("torch")
            pyannote_audio = importlib.import_module("pyannote.audio")
            pipeline_type = pyannote_audio.Pipeline
        except Exception:
            raise PyannoteDiarizationError(
                PyannoteDiarizationErrorCategory.DEPENDENCY_UNAVAILABLE
            ) from None

        try:
            with self._telemetry_disabled():
                pipeline = pipeline_type.from_pretrained(self._model_id)
                if pipeline is None:
                    raise TypeError
                pipeline.to(torch.device(self._device))
        except Exception:
            raise PyannoteDiarizationError(
                PyannoteDiarizationErrorCategory.MODEL_LOAD_FAILED
            ) from None

        self._pipeline = pipeline
        return pipeline, torch

    def _convert_output(
        self,
        raw_output: object,
        request: DiarizationRequest,
    ) -> tuple[DiarizationTurn, ...]:
        annotation = getattr(raw_output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(raw_output, "speaker_diarization", None)
        if annotation is None and callable(getattr(raw_output, "itertracks", None)):
            annotation = raw_output
        itertracks = getattr(annotation, "itertracks", None)
        if not callable(itertracks):
            raise PyannoteDiarizationError(
                PyannoteDiarizationErrorCategory.MALFORMED_OUTPUT
            )

        duration = request.window_end_seconds - request.window_start_seconds
        try:
            tracks = cast(Iterable[object], itertracks(yield_label=True))
            parsed_tracks: list[tuple[float, float, str]] = []
            for track in tracks:
                if not isinstance(track, tuple) or len(track) != 3:
                    raise TypeError
                segment, _, raw_speaker = track
                relative_start = float(segment.start)
                relative_end = float(segment.end)
                speaker = _required_text(str(raw_speaker))
                if (
                    not isfinite(relative_start)
                    or not isfinite(relative_end)
                    or relative_start < 0.0
                    or relative_end <= relative_start
                ):
                    raise ValueError
                parsed_tracks.append((relative_start, relative_end, speaker))

            terminal_end = max(
                (relative_end for _, relative_end, _ in parsed_tracks),
                default=duration,
            )
            converted: list[DiarizationTurn] = []
            speakers: set[str] = set()
            for relative_start, relative_end, speaker in parsed_tracks:
                if relative_end > duration:
                    overrun = relative_end - duration
                    if (
                        relative_end != terminal_end
                        or overrun > MAX_TERMINAL_END_OVERRUN_SECONDS
                        or duration <= relative_start
                    ):
                        raise ValueError
                    relative_end = duration
                speakers.add(speaker)
                if len(speakers) > self._max_speakers:
                    raise PyannoteDiarizationError(
                        PyannoteDiarizationErrorCategory.MALFORMED_OUTPUT
                    )
                converted.append(
                    DiarizationTurn(
                        tenant_id=request.tenant_id,
                        call_id=request.call_id,
                        start_seconds=request.window_start_seconds + relative_start,
                        end_seconds=request.window_start_seconds + relative_end,
                        local_speaker_ids=(speaker,),
                        role=SpeakerRole.UNKNOWN,
                    )
                )
        except PyannoteDiarizationError:
            raise
        except Exception:
            raise PyannoteDiarizationError(
                PyannoteDiarizationErrorCategory.MALFORMED_OUTPUT
            ) from None

        return tuple(
            sorted(
                converted,
                key=lambda turn: (
                    turn.start_seconds,
                    turn.end_seconds,
                    turn.local_speaker_ids,
                ),
            )
        )

    @classmethod
    @contextmanager
    def _telemetry_disabled(cls) -> Iterator[None]:
        names = ("PYANNOTE_METRICS_ENABLED", "HF_HUB_DISABLE_TELEMETRY")
        with cls._telemetry_lock:
            previous: Mapping[str, str | None] = {
                name: os.environ.get(name) for name in names
            }
            os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            try:
                yield
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value


def _required_text(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("configuration value cannot be empty")
    return cleaned
