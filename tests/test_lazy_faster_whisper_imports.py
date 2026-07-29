from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
from typing import Any

import pytest

from app.asr.faster_whisper_engine import FasterWhisperEngine


NATIVE_PROVIDER_MODULES = ("faster_whisper", "ctranslate2", "torch")
IMPORT_TARGETS = (
    "app.asr.faster_whisper_engine",
    "app.diarization.offline_evaluation",
    "scripts.transcribe_file",
    "scripts.transcribe_streaming_file",
    "scripts.evaluate_diarization_offline",
)


def test_engine_cli_and_offline_modules_do_not_import_native_providers() -> None:
    script = """
import builtins
import importlib
import sys

blocked_roots = {"faster_whisper", "ctranslate2", "torch"}
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", maxsplit=1)[0] in blocked_roots:
        raise AssertionError(f"blocked native import attempted: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
for target in sys.argv[1:]:
    importlib.import_module(target)
loaded_roots = {name.split(".", maxsplit=1)[0] for name in sys.modules}
unexpected = blocked_roots & loaded_roots
if unexpected:
    raise AssertionError(f"native modules loaded: {sorted(unexpected)}")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, *IMPORT_TARGETS],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_engine_construction_does_not_load_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_loader() -> object:
        raise AssertionError("provider loader must remain deferred")

    monkeypatch.setattr(
        "app.asr.faster_whisper_engine._load_whisper_model_constructor",
        unexpected_loader,
    )

    FasterWhisperEngine()


def test_invalid_input_fails_before_provider_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_loader() -> object:
        raise AssertionError("provider loader must not run for invalid input")

    monkeypatch.setattr(
        "app.asr.faster_whisper_engine._load_whisper_model_constructor",
        unexpected_loader,
    )

    with pytest.raises(FileNotFoundError):
        FasterWhisperEngine().transcribe_file(tmp_path / "missing.wav")


def test_first_valid_transcription_loads_once_and_cached_model_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader_calls: list[None] = []
    constructor_calls: list[tuple[object, ...]] = []
    transcribe_calls: list[tuple[object, dict[str, Any]]] = []

    class SyntheticModel:
        def transcribe(
            self, audio: object, **settings: Any
        ) -> tuple[tuple[object, ...], SimpleNamespace]:
            transcribe_calls.append((audio, settings))
            return (), SimpleNamespace(
                duration=0.0,
                language="tr",
                language_probability=1.0,
            )

    def constructor(*args: object, **kwargs: object) -> SyntheticModel:
        constructor_calls.append((*args, kwargs))
        return SyntheticModel()

    def loader() -> object:
        loader_calls.append(None)
        return constructor

    monkeypatch.setattr(
        "app.asr.faster_whisper_engine._load_whisper_model_constructor",
        loader,
    )
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    engine = FasterWhisperEngine()

    engine.transcribe_file(audio_path)
    engine.transcribe_file(audio_path)

    assert loader_calls == [None]
    assert len(constructor_calls) == 1
    assert len(transcribe_calls) == 2
