"""Provider-neutral, tenant-safe speaker diarization contracts."""

from app.diarization.alignment import (
    UNKNOWN_LOCAL_SPEAKER_ID,
    WordAlignmentError,
    WordAlignmentErrorCategory,
    WordAlignmentRequest,
    align_words_to_speakers,
)
from app.diarization.fake_backend import FakeSpeakerDiarizer
from app.diarization.identity_tracker import (
    SpeakerIdentityTracker,
    SpeakerIdentityTrackingError,
    SpeakerIdentityTrackingErrorCategory,
    SpeakerIdentityTrackingRequest,
)
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
from app.diarization.role_resolver import (
    RoleEvidenceCode,
    RuleBasedSpeakerRoleResolver,
    SpeakerAttributedTextSpan,
    SpeakerRoleAssignment,
    SpeakerRoleResolutionError,
    SpeakerRoleResolutionErrorCategory,
    SpeakerRoleResolutionRequest,
    SpeakerRoleResolutionResult,
    SpeakerRoleResolverProtocol,
)
from app.diarization.routing import (
    CustomerProjectionReason,
    CustomerProjectionStatus,
    CustomerSpeechProjection,
    CustomerSpeechProjectionRequest,
    CustomerSpeechProjector,
    DiarizationRoutingError,
    DiarizationRoutingErrorCategory,
    RoleTaggedWord,
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
    "SpeakerIdentityTracker",
    "SpeakerIdentityTrackingError",
    "SpeakerIdentityTrackingErrorCategory",
    "SpeakerIdentityTrackingRequest",
    "SpeakerRole",
    "SpeakerAttributedTextSpan",
    "SpeakerRoleAssignment",
    "SpeakerRoleResolutionError",
    "SpeakerRoleResolutionErrorCategory",
    "SpeakerRoleResolutionRequest",
    "SpeakerRoleResolutionResult",
    "SpeakerRoleResolverProtocol",
    "RoleEvidenceCode",
    "RuleBasedSpeakerRoleResolver",
    "CustomerProjectionReason",
    "CustomerProjectionStatus",
    "CustomerSpeechProjection",
    "CustomerSpeechProjectionRequest",
    "CustomerSpeechProjector",
    "DiarizationRoutingError",
    "DiarizationRoutingErrorCategory",
    "RoleTaggedWord",
    "UNKNOWN_LOCAL_SPEAKER_ID",
    "WordAlignmentError",
    "WordAlignmentErrorCategory",
    "WordAlignmentRequest",
    "align_words_to_speakers",
]
