"""Provider-neutral, tenant-safe speaker diarization contracts."""

from app.diarization.fake_backend import FakeSpeakerDiarizer
from app.diarization.models import (
    DiarizationRequest,
    DiarizationResult,
    DiarizationTurn,
    DiarizedTranscriptEvent,
    DiarizedWord,
    SpeakerRole,
)
from app.diarization.protocols import SpeakerDiarizerProtocol

__all__ = [
    "DiarizationRequest",
    "DiarizationResult",
    "DiarizationTurn",
    "DiarizedTranscriptEvent",
    "DiarizedWord",
    "FakeSpeakerDiarizer",
    "SpeakerDiarizerProtocol",
    "SpeakerRole",
]
