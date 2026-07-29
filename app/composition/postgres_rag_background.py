"""Bounded background execution for prepared PostgreSQL RAG orchestration."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from threading import Event, Lock

from app.composition.postgres_rag_runtime import (
    ProfileVerifiedPostgreSQLRAGRunner,
)
from app.orchestration.models import OrchestrationRequest, OrchestrationResult


class RAGOrchestrationSubmissionStatus(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    STALE = "stale"
    CAPACITY_REJECTED = "capacity_rejected"
    NOT_STARTED = "not_started"
    CLOSED = "closed"


class RAGOrchestrationCompletionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    EMPTY = "empty"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RAGOrchestrationIdentity:
    tenant_id: str
    call_id: str
    transcript_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _required_text(self.tenant_id, "tenant_id"),
        )
        object.__setattr__(
            self,
            "call_id",
            _required_text(self.call_id, "call_id"),
        )
        if type(self.transcript_revision) is not int:
            raise ValueError("transcript_revision must be an integer")
        if self.transcript_revision < 0:
            raise ValueError("transcript_revision cannot be negative")

    @classmethod
    def from_request(
        cls,
        request: OrchestrationRequest,
    ) -> RAGOrchestrationIdentity:
        return cls(
            tenant_id=request.tenant_id,
            call_id=request.call_id,
            transcript_revision=request.transcript_revision,
        )


@dataclass(frozen=True, slots=True)
class RAGOrchestrationSubmission:
    identity: RAGOrchestrationIdentity
    status: RAGOrchestrationSubmissionStatus


@dataclass(frozen=True, slots=True)
class RAGOrchestrationCompletion:
    identity: RAGOrchestrationIdentity
    status: RAGOrchestrationCompletionStatus
    result: OrchestrationResult | None = field(default=None, repr=False)
    error: Exception | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.status is RAGOrchestrationCompletionStatus.SUCCEEDED:
            if self.result is None or self.error is not None:
                raise ValueError("successful completion shape is invalid")
        elif self.status is RAGOrchestrationCompletionStatus.EMPTY:
            if self.result is not None or self.error is not None:
                raise ValueError("empty completion shape is invalid")
        elif self.result is not None or self.error is None:
            raise ValueError("failed completion shape is invalid")


@dataclass(slots=True)
class _StartAttempt:
    finished: Event = field(default_factory=Event)
    error: BaseException | None = None


class BoundedPostgreSQLRAGManager:
    """Own bounded background execution for one prepared RAG runner."""

    def __init__(
        self,
        *,
        runner: ProfileVerifiedPostgreSQLRAGRunner,
        max_workers: int,
        capacity: int,
    ) -> None:
        if not isinstance(runner, ProfileVerifiedPostgreSQLRAGRunner):
            raise ValueError("runner must be ProfileVerifiedPostgreSQLRAGRunner")
        self._max_workers = _strict_positive_integer(max_workers, "max_workers")
        self._capacity = _strict_positive_integer(capacity, "capacity")
        if self._capacity < self._max_workers:
            raise ValueError("capacity must be greater than or equal to max_workers")
        self._runner = runner
        self._lock = Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._start_attempt: _StartAttempt | None = None
        self._closed = False
        self._current_revisions: dict[tuple[str, str], int] = {}
        self._submitted_identities: set[RAGOrchestrationIdentity] = set()
        self._futures: dict[
            RAGOrchestrationIdentity,
            Future[OrchestrationResult | None],
        ] = {}
        self._completions: dict[
            RAGOrchestrationIdentity,
            RAGOrchestrationCompletion,
        ] = {}
        self._reservations = 0

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("background manager is closed")
            if self._executor is not None:
                return
            attempt = self._start_attempt
            leader = attempt is None
            if leader:
                attempt = _StartAttempt()
                self._start_attempt = attempt
        if not leader:
            if attempt is None:
                raise RuntimeError("background manager start invariant failed")
            attempt.finished.wait()
            if attempt.error is not None:
                raise attempt.error
            return

        if attempt is None:
            raise RuntimeError("background manager start invariant failed")
        try:
            self._runner.prepare()
            executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="callmetric-rag",
            )
        except BaseException as error:
            attempt.error = error
            attempt.finished.set()
            with self._lock:
                if self._start_attempt is attempt:
                    self._start_attempt = None
            raise

        with self._lock:
            if self._closed:
                executor.shutdown(wait=False, cancel_futures=True)
                error = RuntimeError("background manager is closed")
                attempt.error = error
                attempt.finished.set()
                if self._start_attempt is attempt:
                    self._start_attempt = None
                raise error
            self._executor = executor
            attempt.finished.set()
            if self._start_attempt is attempt:
                self._start_attempt = None

    def announce_current_revision(
        self,
        *,
        tenant_id: str,
        call_id: str,
        transcript_revision: int,
    ) -> None:
        identity = RAGOrchestrationIdentity(
            tenant_id=tenant_id,
            call_id=call_id,
            transcript_revision=transcript_revision,
        )
        scope = _scope(identity)
        to_cancel: list[Future[OrchestrationResult | None]] = []
        with self._lock:
            if self._closed:
                raise RuntimeError("background manager is closed")
            current = self._current_revisions.get(scope)
            if current is not None and identity.transcript_revision < current:
                raise ValueError("transcript_revision cannot decrease")
            if current == identity.transcript_revision:
                return
            self._current_revisions[scope] = identity.transcript_revision
            if current is None:
                return
            stale_completions = tuple(
                candidate
                for candidate in self._completions
                if _scope(candidate) == scope
                and candidate.transcript_revision < identity.transcript_revision
            )
            for candidate in stale_completions:
                self._completions.pop(candidate)
                self._release_reservation()
            self._submitted_identities = {
                candidate
                for candidate in self._submitted_identities
                if _scope(candidate) != scope
                or candidate.transcript_revision >= identity.transcript_revision
            }
            to_cancel = [
                future
                for candidate, future in self._futures.items()
                if _scope(candidate) == scope
                and candidate.transcript_revision < identity.transcript_revision
            ]
        for future in to_cancel:
            future.cancel()

    def submit(
        self,
        request: OrchestrationRequest,
    ) -> RAGOrchestrationSubmission:
        if not isinstance(request, OrchestrationRequest):
            raise ValueError("request must be OrchestrationRequest")
        identity = RAGOrchestrationIdentity.from_request(request)
        with self._lock:
            if self._closed:
                return _submission(identity, RAGOrchestrationSubmissionStatus.CLOSED)
            executor = self._executor
            if executor is None:
                return _submission(
                    identity,
                    RAGOrchestrationSubmissionStatus.NOT_STARTED,
                )
            current = self._current_revisions.get(_scope(identity))
            if current != identity.transcript_revision:
                return _submission(identity, RAGOrchestrationSubmissionStatus.STALE)
            if identity in self._submitted_identities:
                return _submission(
                    identity,
                    RAGOrchestrationSubmissionStatus.DUPLICATE,
                )
            if self._reservations >= self._capacity:
                return _submission(
                    identity,
                    RAGOrchestrationSubmissionStatus.CAPACITY_REJECTED,
                )
            self._reservations += 1
            self._submitted_identities.add(identity)
            try:
                future = executor.submit(self._runner.run, request)
            except BaseException:
                self._submitted_identities.remove(identity)
                self._release_reservation()
                raise
            self._futures[identity] = future
        future.add_done_callback(
            lambda completed, selected=identity: self._complete(
                selected,
                completed,
            )
        )
        return _submission(identity, RAGOrchestrationSubmissionStatus.ACCEPTED)

    def poll(
        self,
        identity: RAGOrchestrationIdentity,
    ) -> RAGOrchestrationCompletion | None:
        if not isinstance(identity, RAGOrchestrationIdentity):
            raise ValueError("identity must be RAGOrchestrationIdentity")
        with self._lock:
            if self._closed:
                return None
            if self._current_revisions.get(_scope(identity)) != (
                identity.transcript_revision
            ):
                return None
            completion = self._completions.pop(identity, None)
            if completion is not None:
                self._release_reservation()
            return completion

    def close(self, *, wait: bool = False) -> None:
        if type(wait) is not bool:
            raise ValueError("wait must be a boolean")
        with self._lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
            self._executor = None
            futures = tuple(self._futures.values())
            self._futures.clear()
            self._completions.clear()
            self._submitted_identities.clear()
            self._current_revisions.clear()
            self._reservations = 0
        for future in futures:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)

    def _complete(
        self,
        identity: RAGOrchestrationIdentity,
        future: Future[OrchestrationResult | None],
    ) -> None:
        if future.cancelled():
            completion: RAGOrchestrationCompletion | None = None
        else:
            try:
                result = future.result()
            except Exception as error:
                completion = RAGOrchestrationCompletion(
                    identity=identity,
                    status=RAGOrchestrationCompletionStatus.FAILED,
                    error=error,
                )
            else:
                completion = _validated_completion(identity, result)
        with self._lock:
            if self._closed:
                return
            self._futures.pop(identity, None)
            if (
                completion is None
                or self._current_revisions.get(_scope(identity))
                != identity.transcript_revision
            ):
                self._release_reservation()
                return
            self._completions[identity] = completion

    def _release_reservation(self) -> None:
        if self._reservations <= 0:
            raise RuntimeError("background manager capacity invariant failed")
        self._reservations -= 1


def _validated_completion(
    identity: RAGOrchestrationIdentity,
    result: OrchestrationResult | None,
) -> RAGOrchestrationCompletion:
    if result is None:
        return RAGOrchestrationCompletion(
            identity=identity,
            status=RAGOrchestrationCompletionStatus.EMPTY,
        )
    if not isinstance(result, OrchestrationResult):
        return RAGOrchestrationCompletion(
            identity=identity,
            status=RAGOrchestrationCompletionStatus.FAILED,
            error=ValueError("orchestration runner returned an invalid result"),
        )
    if (
        result.tenant_id != identity.tenant_id
        or result.call_id != identity.call_id
        or result.transcript_revision != identity.transcript_revision
    ):
        return RAGOrchestrationCompletion(
            identity=identity,
            status=RAGOrchestrationCompletionStatus.FAILED,
            error=ValueError("orchestration result identity does not match request"),
        )
    return RAGOrchestrationCompletion(
        identity=identity,
        status=RAGOrchestrationCompletionStatus.SUCCEEDED,
        result=result,
    )


def _submission(
    identity: RAGOrchestrationIdentity,
    status: RAGOrchestrationSubmissionStatus,
) -> RAGOrchestrationSubmission:
    return RAGOrchestrationSubmission(identity=identity, status=status)


def _scope(identity: RAGOrchestrationIdentity) -> tuple[str, str]:
    return (identity.tenant_id, identity.call_id)


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def _strict_positive_integer(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value
