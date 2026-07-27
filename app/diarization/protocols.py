"""Synchronous speaker diarization protocol."""

from typing import Protocol

from app.diarization.models import DiarizationRequest, DiarizationResult


class SpeakerDiarizerProtocol(Protocol):
    def diarize(self, request: DiarizationRequest) -> DiarizationResult: ...
