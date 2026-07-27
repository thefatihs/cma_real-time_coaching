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
from app.diarization.pyannote_backend import (
    DEFAULT_PYANNOTE_MODEL_ID,
    PyannoteDiarizationError,
    PyannoteDiarizationErrorCategory,
    PyannoteSpeakerDiarizer,
)

__all__ = [
    "DiarizationRequest",
    "DiarizationResult",
    "DiarizationTurn",
    "DiarizedTranscriptEvent",
    "DiarizedWord",
    "DEFAULT_PYANNOTE_MODEL_ID",
    "FakeSpeakerDiarizer",
    "PyannoteDiarizationError",
    "PyannoteDiarizationErrorCategory",
    "PyannoteSpeakerDiarizer",
    "SpeakerDiarizerProtocol",
    "SpeakerRole",
]
