"""Deterministic, in-memory speaker diarizer for tests and composition."""

from app.diarization.models import DiarizationRequest, DiarizationResult


class FakeSpeakerDiarizer:
    """Return one prevalidated synthetic result without retaining request audio."""

    def __init__(self, result: DiarizationResult) -> None:
        self._result = result

    def diarize(self, request: DiarizationRequest) -> DiarizationResult:
        if request.tenant_id != self._result.tenant_id:
            raise ValueError("diarization request tenant_id does not match result")
        if request.call_id != self._result.call_id:
            raise ValueError("diarization request call_id does not match result")
        if (
            request.window_start_seconds != self._result.window_start_seconds
            or request.window_end_seconds != self._result.window_end_seconds
        ):
            raise ValueError("diarization request window does not match result")
        return self._result
