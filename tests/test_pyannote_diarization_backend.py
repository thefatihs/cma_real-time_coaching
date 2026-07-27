import builtins
from collections.abc import Sequence
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from app.diarization import (
    DiarizationRequest,
    PyannoteDiarizationError,
    PyannoteDiarizationErrorCategory,
    PyannoteSpeakerDiarizer,
    SpeakerDiarizerProtocol,
    SpeakerRole,
)


class FakeTensor:
    def __init__(self, values: object, *, dtype: object, device: str) -> None:
        self.values = values
        self.dtype = dtype
        self.device = device
        self.shape: tuple[int, ...] = ()

    def reshape(self, *shape: int) -> "FakeTensor":
        self.shape = shape
        return self


class FakeTorch(ModuleType):
    float32 = object()

    def __init__(self) -> None:
        super().__init__("torch")
        self.device_calls: list[str] = []

    def tensor(self, values: object, *, dtype: object, device: str) -> FakeTensor:
        return FakeTensor(values, dtype=dtype, device=device)

    def device(self, value: str) -> str:
        self.device_calls.append(value)
        return value


class FakeSegment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class FakeAnnotation:
    def __init__(
        self,
        tracks: Sequence[tuple[FakeSegment, object, object]],
    ) -> None:
        self.tracks = list(tracks)

    def itertracks(
        self, *, yield_label: bool
    ) -> list[tuple[FakeSegment, object, object]]:
        assert yield_label is True
        return self.tracks


class FakePipeline:
    load_count = 0
    loaded_model_ids: list[str] = []
    instance: "FakePipeline | None" = None

    def __init__(self, output: object) -> None:
        self.output = output
        self.to_calls: list[object] = []
        self.calls: list[tuple[dict[str, object], dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, model_id: str) -> "FakePipeline":
        cls.load_count += 1
        cls.loaded_model_ids.append(model_id)
        assert cls.instance is not None
        return cls.instance

    def to(self, device: object) -> None:
        self.to_calls.append(device)

    def __call__(self, audio: dict[str, object], **kwargs: object) -> object:
        self.calls.append((audio, kwargs))
        return self.output


def make_request(**changes: object) -> DiarizationRequest:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "window_start_seconds": 10.0,
        "window_end_seconds": 14.0,
        "sample_rate_hz": 16_000,
        "mono_audio": (0.0, 0.25, -0.25, 0.0),
    }
    values.update(changes)
    return DiarizationRequest.model_validate(values)


def install_fake_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    output: object,
) -> tuple[FakeTorch, FakePipeline]:
    fake_torch = FakeTorch()
    fake_pipeline = FakePipeline(output)
    FakePipeline.load_count = 0
    FakePipeline.loaded_model_ids = []
    FakePipeline.instance = fake_pipeline
    fake_pyannote = ModuleType("pyannote")
    fake_audio = ModuleType("pyannote.audio")
    fake_audio.Pipeline = FakePipeline  # type: ignore[attr-defined]
    fake_pyannote.audio = fake_audio  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)
    return fake_torch, fake_pipeline


def make_backend(**changes: object) -> PyannoteSpeakerDiarizer:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
    }
    values.update(changes)
    return PyannoteSpeakerDiarizer(**values)  # type: ignore[arg-type]


def test_package_import_does_not_import_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def guarded_import(name: str, package: str | None = None) -> ModuleType:
        if name in {"torch", "pyannote.audio"}:
            raise AssertionError("optional dependency imported eagerly")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)

    assert importlib.import_module("app.diarization").PyannoteSpeakerDiarizer


def test_lazy_loads_pipeline_once_and_implements_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = SimpleNamespace(
        exclusive_speaker_diarization=FakeAnnotation(
            [(FakeSegment(0.0, 1.0), None, "speaker_0")]
        )
    )
    _, pipeline = install_fake_dependencies(monkeypatch, output)
    backend = make_backend()
    subject: SpeakerDiarizerProtocol = backend

    subject.diarize(make_request())
    subject.diarize(make_request())

    assert FakePipeline.load_count == 1
    assert len(pipeline.calls) == 2


def test_passes_only_mono_float32_audio_and_two_speaker_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation = FakeAnnotation([(FakeSegment(0.0, 1.0), None, "speaker_0")])
    fake_torch, pipeline = install_fake_dependencies(
        monkeypatch,
        SimpleNamespace(exclusive_speaker_diarization=annotation),
    )

    make_backend().diarize(make_request())

    audio, kwargs = pipeline.calls[0]
    assert set(audio) == {"waveform", "sample_rate"}
    waveform = audio["waveform"]
    assert isinstance(waveform, FakeTensor)
    assert waveform.shape == (1, -1)
    assert waveform.dtype is fake_torch.float32
    assert waveform.device == "cpu"
    assert audio["sample_rate"] == 16_000
    assert kwargs == {"num_speakers": 2}
    assert pipeline.to_calls == ["cpu"]


def test_exclusive_output_is_preferred_and_regular_is_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exclusive = FakeAnnotation([(FakeSegment(1.0, 2.0), None, "exclusive")])
    regular = FakeAnnotation([(FakeSegment(0.0, 1.0), None, "regular")])
    install_fake_dependencies(
        monkeypatch,
        SimpleNamespace(
            exclusive_speaker_diarization=exclusive,
            speaker_diarization=regular,
        ),
    )
    assert make_backend().diarize(make_request()).turns[0].local_speaker_ids == (
        "exclusive",
    )

    install_fake_dependencies(
        monkeypatch,
        SimpleNamespace(
            exclusive_speaker_diarization=None,
            speaker_diarization=regular,
        ),
    )
    assert make_backend().diarize(make_request()).turns[0].local_speaker_ids == (
        "regular",
    )


def test_converts_absolute_timestamps_and_orders_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracks = [
        (FakeSegment(2.0, 3.0), None, "speaker_1"),
        (FakeSegment(0.25, 1.0), None, "speaker_0"),
    ]
    install_fake_dependencies(
        monkeypatch,
        SimpleNamespace(exclusive_speaker_diarization=FakeAnnotation(tracks)),
    )

    result = make_backend().diarize(make_request())

    assert [(turn.start_seconds, turn.end_seconds) for turn in result.turns] == [
        (10.25, 11.0),
        (12.0, 13.0),
    ]
    assert all(turn.role is SpeakerRole.UNKNOWN for turn in result.turns)
    assert all(turn.global_speaker_id is None for turn in result.turns)
    assert all(turn.speaker_confidence is None for turn in result.turns)


@pytest.mark.parametrize(
    "tracks",
    [
        [(FakeSegment(float("nan"), 1.0), None, "speaker_0")],
        [(FakeSegment(0.0, float("inf")), None, "speaker_0")],
        [(FakeSegment(-0.1, 1.0), None, "speaker_0")],
        [(FakeSegment(1.0, 1.0), None, "speaker_0")],
        [(FakeSegment(0.0, 4.1), None, "speaker_0")],
        [("not-a-segment",)],
    ],
)
def test_rejects_malformed_or_out_of_window_output(
    monkeypatch: pytest.MonkeyPatch,
    tracks: list[tuple[Any, ...]],
) -> None:
    install_fake_dependencies(
        monkeypatch,
        SimpleNamespace(exclusive_speaker_diarization=FakeAnnotation(tracks)),  # type: ignore[arg-type]
    )

    with pytest.raises(PyannoteDiarizationError) as error:
        make_backend().diarize(make_request())

    assert error.value.category is PyannoteDiarizationErrorCategory.MALFORMED_OUTPUT


def test_rejects_more_than_configured_speaker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracks = [
        (FakeSegment(0.0, 1.0), None, "speaker_0"),
        (FakeSegment(1.0, 2.0), None, "speaker_1"),
        (FakeSegment(2.0, 3.0), None, "speaker_2"),
    ]
    install_fake_dependencies(
        monkeypatch,
        SimpleNamespace(exclusive_speaker_diarization=FakeAnnotation(tracks)),
    )

    with pytest.raises(PyannoteDiarizationError) as error:
        make_backend().diarize(make_request())

    assert error.value.category is PyannoteDiarizationErrorCategory.MALFORMED_OUTPUT


@pytest.mark.parametrize(
    ("tenant_id", "call_id"),
    [("tenant_beta", "call_001"), ("tenant_alpha", "call_002")],
)
def test_rejects_wrong_scope_before_loading(
    tenant_id: str,
    call_id: str,
) -> None:
    backend = make_backend()

    with pytest.raises(PyannoteDiarizationError) as error:
        backend.diarize(make_request(tenant_id=tenant_id, call_id=call_id))

    assert error.value.category is PyannoteDiarizationErrorCategory.SCOPE_MISMATCH
    assert backend.__dict__["_pipeline"] is None


def test_safe_dependency_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "pyannote.audio", raising=False)
    real_import = importlib.import_module

    def missing(name: str, package: str | None = None) -> ModuleType:
        if name == "pyannote.audio":
            raise ModuleNotFoundError("PRIVATE token path")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)

    with pytest.raises(PyannoteDiarizationError) as error:
        make_backend().diarize(make_request())

    assert (
        error.value.category is PyannoteDiarizationErrorCategory.DEPENDENCY_UNAVAILABLE
    )
    assert "PRIVATE" not in str(error.value)


def test_safe_model_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_dependencies(monkeypatch, object())

    def load_failure(model_id: str) -> FakePipeline:
        raise OSError(f"PRIVATE/{model_id}/token")

    monkeypatch.setattr(FakePipeline, "from_pretrained", load_failure)
    with pytest.raises(PyannoteDiarizationError) as load_error:
        make_backend().diarize(make_request())
    assert (
        load_error.value.category is PyannoteDiarizationErrorCategory.MODEL_LOAD_FAILED
    )
    assert "PRIVATE" not in repr(load_error.value)


def test_safe_inference_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_dependencies(monkeypatch, object())

    def inference_failure(
        self: FakePipeline,
        audio: object,
        **kwargs: object,
    ) -> object:
        raise RuntimeError("PRIVATE raw audio token")

    monkeypatch.setattr(FakePipeline, "__call__", inference_failure)
    with pytest.raises(PyannoteDiarizationError) as inference_error:
        make_backend().diarize(make_request())
    assert (
        inference_error.value.category
        is PyannoteDiarizationErrorCategory.INFERENCE_FAILED
    )
    assert "PRIVATE" not in repr(inference_error.value)


def test_never_opens_files_and_restores_telemetry_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation = FakeAnnotation([(FakeSegment(0.0, 1.0), None, "speaker_0")])
    install_fake_dependencies(
        monkeypatch,
        SimpleNamespace(exclusive_speaker_diarization=annotation),
    )

    def forbidden_open(*args: object, **kwargs: object) -> object:
        raise AssertionError("file access is forbidden")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    monkeypatch.setattr(Path, "open", forbidden_open)
    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "original")
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)

    make_backend().diarize(make_request())

    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "original"
    assert "HF_HUB_DISABLE_TELEMETRY" not in os.environ


def test_optional_two_speaker_hint_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation = FakeAnnotation([(FakeSegment(0.0, 1.0), None, "speaker_0")])
    _, pipeline = install_fake_dependencies(
        monkeypatch,
        SimpleNamespace(exclusive_speaker_diarization=annotation),
    )

    make_backend(fixed_two_speakers=False).diarize(make_request())

    assert pipeline.calls[0][1] == {}
