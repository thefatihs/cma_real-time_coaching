"""Core file-based streaming ASR pipeline orchestration."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Protocol

from app.audio_ingress.local_microphone import LocalMicTestCapability
from app.calls.models import CallClassificationMetadata, CallState
from app.classification.streaming import (
    ClassificationProcessingStatus,
    ProvisionalClassificationPolicy,
    RuntimeClassifierProtocol,
    StableClassificationOutcome,
    StableTranscriptClassificationStage,
)
from app.coaching.coordinator import StableCoachingOutcome
from app.events.models import (
    AudioChunkEvent,
    ClassificationResultEvent,
    TranscriptEvent,
)
from app.integration.rag_coaching import CoachingCompletionPumpProtocol
from app.streaming.audio_window import ASRAudioWindow, AudioWindowBuilder
from app.streaming.chunk_generator import generate_audio_chunks
from app.streaming.customer_routing import (
    CustomerOnlyClassificationRouter,
    CustomerProjectionProviderProtocol,
    CustomerRoutingOutcome,
    CustomerRoutingStatus,
)
from app.streaming.rolling_buffer import RollingAudioBuffer
from app.streaming.transcript_reconciler import TranscriptReconciler
from app.streaming.window_transcriber import WindowTranscriptionResult
from app.tenancy.models import TenantASRConfig, TenantContext


@dataclass(frozen=True, slots=True)
class StreamingASRStep:
    tenant_id: str
    call_id: str
    sequence_number: int
    chunk_start_seconds: float
    chunk_end_seconds: float
    window_start_seconds: float
    window_end_seconds: float
    window_duration_seconds: float
    raw_window_text: str
    transcript_events: tuple[TranscriptEvent, ...]
    stable_transcript: str
    partial_transcript: str
    transcription_time_seconds: float
    classification_outcomes: tuple[StableClassificationOutcome, ...] = ()
    coaching_outcomes: tuple[StableCoachingOutcome, ...] = ()
    customer_routing_outcomes: tuple[CustomerRoutingOutcome, ...] = ()
    audio_preparation_time_seconds: float = 0.0
    asr_segment_count: int = 0


@dataclass(frozen=True, slots=True)
class StreamingASRResult:
    tenant_id: str
    call_id: str
    steps: tuple[StreamingASRStep, ...]
    final_event: TranscriptEvent | None
    stable_transcript: str
    partial_transcript: str
    total_chunks: int
    audio_duration_seconds: float
    classification_outcomes: tuple[StableClassificationOutcome, ...] = ()
    classification_metadata: CallClassificationMetadata = CallClassificationMetadata()
    coaching_outcomes: tuple[StableCoachingOutcome, ...] = ()
    customer_routing_outcomes: tuple[CustomerRoutingOutcome, ...] = ()


@dataclass(frozen=True, slots=True)
class StreamingASRPlan:
    tenant_id: str
    call_id: str
    total_chunks: int
    audio_duration_seconds: float


class WindowTranscriberProtocol(Protocol):
    def transcribe(self, window: ASRAudioWindow) -> WindowTranscriptionResult: ...


class CoachingProcessorProtocol(Protocol):
    def process_safely(
        self,
        event: TranscriptEvent,
        current_seconds: float,
        *,
        classification_event: ClassificationResultEvent | None = None,
        active_labels: tuple[str, ...] | None = None,
    ) -> StableCoachingOutcome: ...


ChunkGenerator = Callable[[Path, str, str, float], Iterable[AudioChunkEvent]]
StepCallback = Callable[[StreamingASRStep], None]
PlanCallback = Callable[[StreamingASRPlan], None]
CoachingCoordinatorFactory = Callable[[CallState], CoachingProcessorProtocol]


class StreamingASRPipeline:
    def __init__(
        self,
        tenant_context: TenantContext,
        asr_config: TenantASRConfig,
        window_transcriber: WindowTranscriberProtocol,
        *,
        chunk_generator: ChunkGenerator = generate_audio_chunks,
        runtime_classifier: RuntimeClassifierProtocol | None = None,
        coaching_coordinator_factory: CoachingCoordinatorFactory | None = None,
        customer_only_classification_enabled: bool = False,
        customer_projection_provider: CustomerProjectionProviderProtocol | None = None,
    ) -> None:
        self._tenant_context = tenant_context
        self._asr_config = asr_config
        self._window_transcriber = window_transcriber
        self._chunk_generator = chunk_generator
        self._classification_stage = StableTranscriptClassificationStage(
            runtime_classifier
        )
        self._coaching_coordinator_factory = coaching_coordinator_factory
        self._customer_router = CustomerOnlyClassificationRouter(
            enabled=customer_only_classification_enabled,
            projection_provider=customer_projection_provider,
        )

    def configure_provisional_coaching(
        self,
        policy: ProvisionalClassificationPolicy,
        *,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        """Explicitly opt this pipeline into bounded PARTIAL classification."""
        self._classification_stage.configure_provisional_policy(
            policy,
            monotonic_clock=monotonic_clock,
        )

    def run(
        self,
        audio_path: Path,
        call_id: str,
        *,
        step_callback: StepCallback | None = None,
        plan_callback: PlanCallback | None = None,
        retain_history: bool = True,
    ) -> StreamingASRResult:
        audio_duration_seconds = 0.0
        planning_chunks = self._chunk_generator(
            audio_path,
            self._tenant_context.tenant_id,
            call_id,
            self._asr_config.chunk_duration_seconds,
        )
        total_chunks = 0
        for chunk in planning_chunks:
            total_chunks += 1
            audio_duration_seconds = max(
                audio_duration_seconds,
                chunk.chunk_start_seconds + chunk.chunk_duration_seconds,
            )
        if total_chunks == 0:
            raise ValueError("Audio file generated no audio chunks")
        if plan_callback is not None:
            plan_callback(
                StreamingASRPlan(
                    tenant_id=self._tenant_context.tenant_id,
                    call_id=call_id,
                    total_chunks=total_chunks,
                    audio_duration_seconds=audio_duration_seconds,
                )
            )
        chunks = self._chunk_generator(
            audio_path,
            self._tenant_context.tenant_id,
            call_id,
            self._asr_config.chunk_duration_seconds,
        )
        return self._execute_chunks(
            chunks,
            call_id=call_id,
            initial_audio_duration_seconds=audio_duration_seconds,
            step_callback=step_callback,
            retain_history=retain_history,
        )

    def run_live(
        self,
        chunks: Iterable[AudioChunkEvent],
        call_id: str,
        *,
        capability: LocalMicTestCapability,
        execution_resource: object,
        cancellation: Event,
        step_callback: StepCallback | None = None,
        retain_history: bool = False,
    ) -> StreamingASRResult:
        """Process an authorized bounded live chunk stream without file planning."""
        if not capability.authorizes(
            tenant_id=self._tenant_context.tenant_id,
            call_id=call_id,
            resource=execution_resource,
        ):
            raise PermissionError("invalid_local_microphone_capability")
        return self._execute_chunks(
            chunks,
            call_id=call_id,
            initial_audio_duration_seconds=0.0,
            step_callback=step_callback,
            retain_history=retain_history,
            cancellation=cancellation,
        )

    def _execute_chunks(
        self,
        chunks: Iterable[AudioChunkEvent],
        *,
        call_id: str,
        initial_audio_duration_seconds: float,
        step_callback: StepCallback | None,
        retain_history: bool,
        cancellation: Event | None = None,
    ) -> StreamingASRResult:
        call_state = CallState(
            tenant_id=self._tenant_context.tenant_id,
            call_id=call_id,
        )
        buffer = RollingAudioBuffer(self._asr_config.rolling_window_seconds)
        window_builder = AudioWindowBuilder()
        reconciler = TranscriptReconciler(
            stable_region_seconds=self._asr_config.stable_region_seconds
        )
        steps: list[StreamingASRStep] = []
        all_classification_outcomes: list[StableClassificationOutcome] = []
        all_coaching_outcomes: list[StableCoachingOutcome] = []
        all_customer_routing_outcomes: list[CustomerRoutingOutcome] = []
        coaching_coordinator = (
            self._coaching_coordinator_factory(call_state)
            if self._coaching_coordinator_factory is not None
            else None
        )
        audio_duration_seconds = initial_audio_duration_seconds
        processed_chunks = 0
        for chunk in chunks:
            if cancellation is not None and cancellation.is_set():
                break
            processed_chunks += 1
            call_state.apply_audio_chunk(chunk)
            buffer.append(chunk)
            window = window_builder.build(buffer)
            transcription = self._window_transcriber.transcribe(window)
            self._validate_transcription_scope(transcription, call_id)
            transcript_events = reconciler.ingest(transcription)
            step_classification_outcomes: list[StableClassificationOutcome] = []
            step_coaching_outcomes: list[StableCoachingOutcome] = []
            step_customer_routing_outcomes: list[CustomerRoutingOutcome] = []
            for event in transcript_events:
                previous_stable = call_state.stable_transcript
                call_state.apply_transcript(event)
                stable_changed = call_state.stable_transcript != previous_stable
                outcome, coaching_outcome, routing_outcome = (
                    self._process_classification_and_coaching(
                        event=event,
                        call_state=call_state,
                        previous_stable=previous_stable,
                        stable_changed=stable_changed,
                        coordinator=coaching_coordinator,
                    )
                )
                step_classification_outcomes.append(outcome)
                if retain_history:
                    all_classification_outcomes.append(outcome)
                if routing_outcome is not None:
                    step_customer_routing_outcomes.append(routing_outcome)
                    if retain_history:
                        all_customer_routing_outcomes.append(routing_outcome)
                if coaching_outcome is not None:
                    step_coaching_outcomes.append(coaching_outcome)
                    if retain_history:
                        all_coaching_outcomes.append(coaching_outcome)

            chunk_end_seconds = chunk.chunk_start_seconds + chunk.chunk_duration_seconds
            audio_duration_seconds = max(audio_duration_seconds, chunk_end_seconds)
            completed_coaching = self._drain_completed_coaching(
                coordinator=coaching_coordinator,
                current_seconds=chunk_end_seconds,
            )
            step_coaching_outcomes.extend(completed_coaching)
            if retain_history:
                all_coaching_outcomes.extend(completed_coaching)
            step = StreamingASRStep(
                tenant_id=chunk.tenant_id,
                call_id=chunk.call_id,
                sequence_number=chunk.sequence_number,
                chunk_start_seconds=chunk.chunk_start_seconds,
                chunk_end_seconds=chunk_end_seconds,
                window_start_seconds=window.start_seconds,
                window_end_seconds=window.end_seconds,
                window_duration_seconds=window.duration_seconds,
                raw_window_text=transcription.text,
                transcript_events=transcript_events,
                classification_outcomes=tuple(step_classification_outcomes),
                coaching_outcomes=tuple(step_coaching_outcomes),
                customer_routing_outcomes=tuple(step_customer_routing_outcomes),
                stable_transcript=reconciler.stable_transcript,
                partial_transcript=reconciler.partial_transcript,
                transcription_time_seconds=(transcription.processing_time_seconds),
                audio_preparation_time_seconds=(
                    transcription.audio_preparation_time_seconds
                ),
                asr_segment_count=len(transcription.segments),
            )
            if retain_history:
                steps.append(step)
            if step_callback is not None:
                step_callback(step)

        if processed_chunks == 0 and cancellation is None:
            raise ValueError("Audio file generated no audio chunks")
        final_event = reconciler.finalize()
        if final_event is not None:
            previous_stable = call_state.stable_transcript
            call_state.apply_transcript(final_event)
            stable_changed = call_state.stable_transcript != previous_stable
            final_classification, final_coaching, final_routing = (
                self._process_classification_and_coaching(
                    event=final_event,
                    call_state=call_state,
                    previous_stable=previous_stable,
                    stable_changed=stable_changed,
                    coordinator=coaching_coordinator,
                )
            )
            all_classification_outcomes.append(final_classification)
            if final_routing is not None:
                all_customer_routing_outcomes.append(final_routing)
            if final_coaching is not None:
                all_coaching_outcomes.append(final_coaching)
        final_completed_coaching = self._drain_completed_coaching(
            coordinator=coaching_coordinator,
            current_seconds=audio_duration_seconds,
        )
        all_coaching_outcomes.extend(final_completed_coaching)

        return StreamingASRResult(
            tenant_id=call_state.tenant_id,
            call_id=call_state.call_id,
            steps=tuple(steps),
            final_event=final_event,
            classification_outcomes=tuple(all_classification_outcomes),
            classification_metadata=call_state.classification_metadata(),
            coaching_outcomes=tuple(all_coaching_outcomes),
            customer_routing_outcomes=tuple(all_customer_routing_outcomes),
            stable_transcript=reconciler.stable_transcript,
            partial_transcript=reconciler.partial_transcript,
            total_chunks=processed_chunks,
            audio_duration_seconds=audio_duration_seconds,
        )

    def _process_classification_and_coaching(
        self,
        *,
        event: TranscriptEvent,
        call_state: CallState,
        previous_stable: str,
        stable_changed: bool,
        coordinator: CoachingProcessorProtocol | None,
    ) -> tuple[
        StableClassificationOutcome,
        StableCoachingOutcome | None,
        CustomerRoutingOutcome | None,
    ]:
        if event.kind.value == "PARTIAL" or not stable_changed:
            classification = self._classification_stage.process(
                event,
                cumulative_stable_transcript=call_state.stable_transcript,
                stable_changed=stable_changed,
                call_state=call_state,
                stable_delta=event.text,
                preceding_stable_transcript=previous_stable,
                allow_provisional=not self._customer_router.enabled,
            )
            coaching = (
                self._process_coaching(
                    coordinator=coordinator,
                    event=event,
                    stable_changed=stable_changed,
                    classification_outcome=classification,
                )
                if event.kind.value == "PARTIAL"
                and classification.classification_event is not None
                and classification.classification_event.labels
                else None
            )
            return classification, coaching, None

        decision = self._customer_router.prepare(event, call_state)
        if (
            self._customer_router.enabled
            and decision.outcome.status is not CustomerRoutingStatus.CUSTOMER_PROCESSED
        ):
            classification = StableClassificationOutcome(
                status=(
                    ClassificationProcessingStatus.DUPLICATE_REVISION_SKIPPED
                    if decision.outcome.status
                    is CustomerRoutingStatus.ALREADY_PROCESSED
                    else ClassificationProcessingStatus.EMPTY_SKIPPED
                ),
                transcript_revision=event.revision,
                source_sequence=event.source_chunk_sequence,
            )
            return classification, None, decision.outcome

        routed_event = decision.routed_event
        if routed_event is None:
            raise RuntimeError("customer routing invariant failed")
        customer_only = self._customer_router.enabled
        classification = self._classification_stage.process(
            routed_event,
            cumulative_stable_transcript=(
                routed_event.text if customer_only else call_state.stable_transcript
            ),
            stable_changed=stable_changed,
            call_state=call_state,
            stable_delta=routed_event.text,
            preceding_stable_transcript=("" if customer_only else previous_stable),
        )
        coaching = self._process_coaching(
            coordinator=coordinator,
            event=routed_event,
            stable_changed=stable_changed,
            classification_outcome=classification,
        )
        return classification, coaching, decision.outcome

    @staticmethod
    def _drain_completed_coaching(
        *,
        coordinator: CoachingProcessorProtocol | None,
        current_seconds: float,
    ) -> tuple[StableCoachingOutcome, ...]:
        if not isinstance(coordinator, CoachingCompletionPumpProtocol):
            return ()
        return coordinator.drain_completed(current_seconds=current_seconds)

    @staticmethod
    def _process_coaching(
        *,
        coordinator: CoachingProcessorProtocol | None,
        event: TranscriptEvent,
        stable_changed: bool,
        classification_outcome: StableClassificationOutcome,
    ) -> StableCoachingOutcome | None:
        if (
            coordinator is None
            or (
                event.kind.value == "PARTIAL" and not classification_outcome.provisional
            )
            or (event.kind.value != "PARTIAL" and not stable_changed)
            or not event.text.strip()
        ):
            return None
        classification_event = classification_outcome.classification_event
        active_labels = (
            tuple(label.name for label in classification_event.labels)
            if classification_event is not None
            else ()
        )
        return coordinator.process_safely(
            event,
            event.end_seconds,
            classification_event=classification_event,
            active_labels=active_labels,
        )

    def _validate_transcription_scope(
        self, transcription: WindowTranscriptionResult, call_id: str
    ) -> None:
        if transcription.tenant_id != self._tenant_context.tenant_id:
            raise ValueError("Window transcription tenant_id does not match pipeline")
        if transcription.call_id != call_id:
            raise ValueError("Window transcription call_id does not match pipeline")
