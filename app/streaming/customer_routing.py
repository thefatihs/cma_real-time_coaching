"""Feature-flagged customer-only classification routing."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from app.calls.models import CallState
from app.diarization.routing import (
    CustomerProjectionStatus,
    CustomerSpeechProjection,
)
from app.events.models import TranscriptEvent


class CustomerProjectionProviderProtocol(Protocol):
    def get_projection(
        self,
        *,
        tenant_id: str,
        call_id: str,
        transcript_revision: int,
    ) -> CustomerSpeechProjection | None: ...


class CustomerRoutingStatus(str, Enum):
    LEGACY_PATH = "legacy_path"
    CUSTOMER_PROCESSED = "customer_processed"
    NO_CUSTOMER_SPEECH = "no_customer_speech"
    ALREADY_PROCESSED = "already_processed"
    REJECTED = "rejected"
    FAILED_SAFE = "failed_safe"


class CustomerRoutingReason(str, Enum):
    FEATURE_DISABLED = "feature_disabled"
    TRUSTED_CUSTOMER_TEXT = "trusted_customer_text"
    EMPTY_PROJECTION = "empty_projection"
    DUPLICATE_REVISION = "duplicate_revision"
    OUT_OF_ORDER_REVISION = "out_of_order_revision"
    INVALID_SCOPE = "invalid_scope"
    INVALID_REVISION = "invalid_revision"
    MISSING_PROJECTION = "missing_projection"
    MALFORMED_PROJECTION = "malformed_projection"
    PROVIDER_FAILURE = "provider_failure"


@dataclass(frozen=True, slots=True)
class CustomerRoutingOutcome:
    status: CustomerRoutingStatus
    reason: CustomerRoutingReason
    tenant_id: str
    call_id: str
    transcript_revision: int


@dataclass(frozen=True, slots=True)
class CustomerRoutingDecision:
    outcome: CustomerRoutingOutcome
    routed_event: TranscriptEvent | None = field(default=None, repr=False)


class CustomerOnlyClassificationRouter:
    """Select trusted classification text without retaining any text."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        projection_provider: CustomerProjectionProviderProtocol | None = None,
    ) -> None:
        self._enabled = enabled
        self._projection_provider = projection_provider
        self._processed_revisions: dict[tuple[str, str], int] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def prepare(
        self,
        event: TranscriptEvent,
        call_state: CallState,
    ) -> CustomerRoutingDecision:
        if not self._enabled:
            return self._decision(
                event,
                CustomerRoutingStatus.LEGACY_PATH,
                CustomerRoutingReason.FEATURE_DISABLED,
                routed_event=event,
            )
        if (
            event.tenant_id != call_state.tenant_id
            or event.call_id != call_state.call_id
        ):
            return self._decision(
                event,
                CustomerRoutingStatus.REJECTED,
                CustomerRoutingReason.INVALID_SCOPE,
            )
        if event.revision != call_state.transcript_revision:
            return self._decision(
                event,
                CustomerRoutingStatus.REJECTED,
                CustomerRoutingReason.INVALID_REVISION,
            )

        scope = (call_state.tenant_id, call_state.call_id)
        last_revision = self._processed_revisions.get(scope)
        if last_revision is not None:
            if event.revision == last_revision:
                return self._decision(
                    event,
                    CustomerRoutingStatus.ALREADY_PROCESSED,
                    CustomerRoutingReason.DUPLICATE_REVISION,
                )
            if event.revision < last_revision:
                return self._decision(
                    event,
                    CustomerRoutingStatus.REJECTED,
                    CustomerRoutingReason.OUT_OF_ORDER_REVISION,
                )
        if self._projection_provider is None:
            return self._decision(
                event,
                CustomerRoutingStatus.REJECTED,
                CustomerRoutingReason.MISSING_PROJECTION,
            )

        try:
            projection = self._projection_provider.get_projection(
                tenant_id=call_state.tenant_id,
                call_id=call_state.call_id,
                transcript_revision=event.revision,
            )
        except Exception:
            return self._decision(
                event,
                CustomerRoutingStatus.FAILED_SAFE,
                CustomerRoutingReason.PROVIDER_FAILURE,
            )
        if projection is None:
            return self._decision(
                event,
                CustomerRoutingStatus.REJECTED,
                CustomerRoutingReason.MISSING_PROJECTION,
            )
        try:
            valid_projection = self._valid_projection(projection, event)
        except Exception:
            valid_projection = False
        if not valid_projection:
            return self._decision(
                event,
                CustomerRoutingStatus.REJECTED,
                CustomerRoutingReason.MALFORMED_PROJECTION,
            )

        self._processed_revisions[scope] = event.revision
        if projection.status is CustomerProjectionStatus.EMPTY:
            return self._decision(
                event,
                CustomerRoutingStatus.NO_CUSTOMER_SPEECH,
                CustomerRoutingReason.EMPTY_PROJECTION,
            )
        routed_event = event.model_copy(update={"text": projection.customer_text})
        return self._decision(
            event,
            CustomerRoutingStatus.CUSTOMER_PROCESSED,
            CustomerRoutingReason.TRUSTED_CUSTOMER_TEXT,
            routed_event=routed_event,
        )

    @staticmethod
    def _valid_projection(
        projection: CustomerSpeechProjection,
        event: TranscriptEvent,
    ) -> bool:
        if (
            projection.tenant_id != event.tenant_id
            or projection.call_id != event.call_id
            or projection.transcript_revision != event.revision
        ):
            return False
        customer_text = projection.customer_text.strip()
        if projection.status is CustomerProjectionStatus.EMPTY:
            return (
                not customer_text
                and not projection.customer_words
                and projection.customer_start_seconds is None
                and projection.customer_end_seconds is None
            )
        if not customer_text or not projection.customer_words:
            return False
        if " ".join(word.text for word in projection.customer_words) != customer_text:
            return False
        return all(
            word.tenant_id == event.tenant_id
            and word.call_id == event.call_id
            and word.transcript_revision == event.revision
            for word in projection.customer_words
        )

    @staticmethod
    def _decision(
        event: TranscriptEvent,
        status: CustomerRoutingStatus,
        reason: CustomerRoutingReason,
        *,
        routed_event: TranscriptEvent | None = None,
    ) -> CustomerRoutingDecision:
        return CustomerRoutingDecision(
            outcome=CustomerRoutingOutcome(
                status=status,
                reason=reason,
                tenant_id=event.tenant_id,
                call_id=event.call_id,
                transcript_revision=event.revision,
            ),
            routed_event=routed_event,
        )
