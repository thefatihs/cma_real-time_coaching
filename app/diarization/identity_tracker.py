"""Deterministic call-scoped speaker identity tracking."""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite

from app.diarization.models import DiarizationTurn


class SpeakerIdentityTrackingErrorCategory(str, Enum):
    INVALID_SCOPE = "invalid_scope"
    INVALID_WINDOW = "invalid_window"
    SCOPE_MISMATCH = "scope_mismatch"
    TURN_OUTSIDE_WINDOW = "turn_outside_window"
    DUPLICATE_OR_CONFLICTING_TURN = "duplicate_or_conflicting_turn"
    TOO_MANY_LOCAL_SPEAKERS = "too_many_local_speakers"
    NON_MONOTONIC_WINDOW = "non_monotonic_window"
    CONFLICTING_REPROCESS = "conflicting_reprocess"


class SpeakerIdentityTrackingError(ValueError):
    """Privacy-safe tracker boundary error."""

    def __init__(self, category: SpeakerIdentityTrackingErrorCategory) -> None:
        self.category = category
        super().__init__(category.value)


@dataclass(frozen=True, slots=True)
class SpeakerIdentityTrackingRequest:
    tenant_id: str
    call_id: str
    window_start_seconds: float
    window_end_seconds: float
    turns: tuple[DiarizationTurn, ...]

    def __post_init__(self) -> None:
        if not self.tenant_id.strip() or not self.call_id.strip():
            raise SpeakerIdentityTrackingError(
                SpeakerIdentityTrackingErrorCategory.INVALID_SCOPE
            )
        if (
            not isfinite(self.window_start_seconds)
            or not isfinite(self.window_end_seconds)
            or self.window_start_seconds < 0
            or self.window_end_seconds <= self.window_start_seconds
        ):
            raise SpeakerIdentityTrackingError(
                SpeakerIdentityTrackingErrorCategory.INVALID_WINDOW
            )


@dataclass(frozen=True, slots=True)
class _TrackedInterval:
    global_speaker_id: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, slots=True)
class _ProcessedWindow:
    start_seconds: float
    end_seconds: float
    input_turns: tuple[DiarizationTurn, ...]
    output_turns: tuple[DiarizationTurn, ...]
    intervals: tuple[_TrackedInterval, ...]


@dataclass(slots=True)
class _CallTrackingState:
    next_global_ordinal: int = 1
    creation_order: dict[str, int] = field(default_factory=dict)
    windows: list[_ProcessedWindow] = field(default_factory=list)


class SpeakerIdentityTracker:
    """Track stable speaker IDs independently for each exact call scope."""

    def __init__(
        self,
        *,
        max_local_speakers: int = 2,
        history_window_limit: int = 3,
    ) -> None:
        if max_local_speakers <= 0 or history_window_limit <= 0:
            raise ValueError("invalid_tracker_configuration")
        self._max_local_speakers = max_local_speakers
        self._history_window_limit = history_window_limit
        self._states: dict[tuple[str, str], _CallTrackingState] = {}

    def track(
        self,
        request: SpeakerIdentityTrackingRequest,
    ) -> tuple[DiarizationTurn, ...]:
        normalized_turns = self._validate_turns(request)
        scope = (request.tenant_id, request.call_id)
        state = self._states.get(scope)
        if state is None:
            state = _CallTrackingState()
            self._states[scope] = state

        if state.windows:
            last = state.windows[-1]
            same_range = (
                request.window_start_seconds == last.start_seconds
                and request.window_end_seconds == last.end_seconds
            )
            if same_range:
                if normalized_turns == last.input_turns:
                    return last.output_turns
                raise SpeakerIdentityTrackingError(
                    SpeakerIdentityTrackingErrorCategory.CONFLICTING_REPROCESS
                )
            if request.window_start_seconds <= last.start_seconds:
                raise SpeakerIdentityTrackingError(
                    SpeakerIdentityTrackingErrorCategory.NON_MONOTONIC_WINDOW
                )

        local_ids = sorted(
            {
                local_id
                for turn in normalized_turns
                for local_id in turn.local_speaker_ids
            }
        )
        mapping = self._match_speakers(local_ids, normalized_turns, state)
        for local_id in local_ids:
            if local_id not in mapping:
                mapping[local_id] = self._allocate_global_id(state)

        output = tuple(
            turn.model_copy(
                update={
                    "global_speaker_id": (
                        mapping[turn.local_speaker_ids[0]]
                        if len(turn.local_speaker_ids) == 1
                        else None
                    ),
                    "global_speaker_ids": tuple(
                        mapping[local_id] for local_id in turn.local_speaker_ids
                    ),
                }
            )
            for turn in normalized_turns
        )
        intervals = tuple(
            _TrackedInterval(
                global_speaker_id=mapping[local_id],
                start_seconds=turn.start_seconds,
                end_seconds=turn.end_seconds,
            )
            for turn in normalized_turns
            for local_id in turn.local_speaker_ids
        )
        state.windows.append(
            _ProcessedWindow(
                start_seconds=request.window_start_seconds,
                end_seconds=request.window_end_seconds,
                input_turns=normalized_turns,
                output_turns=output,
                intervals=intervals,
            )
        )
        if len(state.windows) > self._history_window_limit:
            del state.windows[: -self._history_window_limit]
        return output

    def reset(self, *, tenant_id: str, call_id: str) -> bool:
        if not tenant_id.strip() or not call_id.strip():
            raise SpeakerIdentityTrackingError(
                SpeakerIdentityTrackingErrorCategory.INVALID_SCOPE
            )
        return self._states.pop((tenant_id, call_id), None) is not None

    def retained_window_count(self, *, tenant_id: str, call_id: str) -> int:
        state = self._states.get((tenant_id, call_id))
        return 0 if state is None else len(state.windows)

    def _validate_turns(
        self,
        request: SpeakerIdentityTrackingRequest,
    ) -> tuple[DiarizationTurn, ...]:
        seen: set[tuple[float, float, tuple[str, ...]]] = set()
        local_ids: set[str] = set()
        for turn in request.turns:
            if turn.tenant_id != request.tenant_id or turn.call_id != request.call_id:
                raise SpeakerIdentityTrackingError(
                    SpeakerIdentityTrackingErrorCategory.SCOPE_MISMATCH
                )
            if (
                turn.start_seconds < request.window_start_seconds
                or turn.end_seconds > request.window_end_seconds
            ):
                raise SpeakerIdentityTrackingError(
                    SpeakerIdentityTrackingErrorCategory.TURN_OUTSIDE_WINDOW
                )
            key = (
                turn.start_seconds,
                turn.end_seconds,
                tuple(sorted(turn.local_speaker_ids)),
            )
            if key in seen:
                raise SpeakerIdentityTrackingError(
                    SpeakerIdentityTrackingErrorCategory.DUPLICATE_OR_CONFLICTING_TURN
                )
            seen.add(key)
            local_ids.update(turn.local_speaker_ids)
        if len(local_ids) > self._max_local_speakers:
            raise SpeakerIdentityTrackingError(
                SpeakerIdentityTrackingErrorCategory.TOO_MANY_LOCAL_SPEAKERS
            )
        return tuple(
            sorted(
                request.turns,
                key=lambda turn: (
                    turn.start_seconds,
                    turn.end_seconds,
                    turn.local_speaker_ids,
                ),
            )
        )

    def _match_speakers(
        self,
        local_ids: list[str],
        turns: tuple[DiarizationTurn, ...],
        state: _CallTrackingState,
    ) -> dict[str, str]:
        totals: dict[tuple[str, str], float] = {}
        for local_id in local_ids:
            current_intervals = [
                (turn.start_seconds, turn.end_seconds)
                for turn in turns
                if local_id in turn.local_speaker_ids
            ]
            for previous in (
                interval for window in state.windows for interval in window.intervals
            ):
                overlap = sum(
                    max(
                        0.0,
                        min(end, previous.end_seconds)
                        - max(start, previous.start_seconds),
                    )
                    for start, end in current_intervals
                )
                if overlap > 0:
                    key = (local_id, previous.global_speaker_id)
                    totals[key] = totals.get(key, 0.0) + overlap

        candidates = sorted(
            (
                (-overlap, state.creation_order[global_id], local_id, global_id)
                for (local_id, global_id), overlap in totals.items()
            )
        )
        mapping: dict[str, str] = {}
        used_globals: set[str] = set()
        for _, _, local_id, global_id in candidates:
            if local_id not in mapping and global_id not in used_globals:
                mapping[local_id] = global_id
                used_globals.add(global_id)
        return mapping

    @staticmethod
    def _allocate_global_id(state: _CallTrackingState) -> str:
        ordinal = state.next_global_ordinal
        global_id = f"CALL_SPEAKER_{ordinal:04d}"
        state.next_global_ordinal += 1
        state.creation_order[global_id] = ordinal
        return global_id
