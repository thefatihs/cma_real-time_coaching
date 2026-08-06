# Fatih Progress

This log uses local chronological numbering and records only Fatih-owned work.

## F-001 — ASR Audio Foundations

- Added exact PCM audio-window contracts, in-memory Whisper normalization,
  sample-rate handling and safe file transcription utilities.

## F-002 — Streaming ASR

- Added tenant-aware rolling buffers, deterministic transcript reconciliation,
  file-based streaming simulation and the streaming ASR pipeline/CLI.

## F-003 — Classification

- Added the multi-label SetFit taxonomy, dataset, calibration, threshold
  profiles and tenant-aware stable-transcript classification runtime.

## F-004 — Deterministic Coaching

- Added rule/classification evidence aggregation, deterministic coaching
  coordination and priority-aware active/history suggestion handling.

## F-005 — Coaching Lifecycle and Safe Admission

- Corrected cooldown ordering and replacement semantics, then added the
  fail-closed external LLM candidate admission contract without direct state
  mutation.

## F-006 — Live Dashboard

- Added runtime classification/coaching views, active/history view models,
  responsive representative-view foundations and privacy-safe operational
  state isolation.

## F-007 — Diarization Stage 1

- Added immutable two-speaker diarization models, protocol, deterministic fake
  backend and optional lazy in-memory pyannote CPU backend.

## F-008 — Diarization Stage 2

- Added optional Faster-Whisper word timestamps and deterministic
  greatest-overlap word-to-speaker alignment.

## F-009 — Diarization Stage 3

- Added bounded call-scoped global speaker identity tracking across overlapping
  windows with deterministic one-to-one matching.

## F-010 — Diarization Stage 4

- Added conservative deterministic speaker-role resolution with bounded
  Turkish evidence and fail-closed UNKNOWN behavior.

## F-011 — Diarization Stages 5–7

- Added trusted customer-speech projection, feature-flagged customer-only
  routing and transactional offline composition of the diarization stages.

## F-012 — Diarization Stage 8A

- Added a privacy-safe offline mono-call evaluation harness with injectable
  components, aggregate-only diagnostics and atomic optional JSON output.

## F-013 — Zero-Duration ASR Artifact Policy

- Added a bounded policy that skips at most one valid exact zero-duration
  Faster-Whisper word artifact without changing timestamps or segment text,
  and propagates only an aggregate warning count.

## F-014 — Progress Ownership and Conflict-Marker Protection

- Added the independent Fatih progress log and repository instructions that
  keep contributor progress files isolated unless a shared milestone is
  explicitly assigned.
- Added a tracked-text conflict-marker scanner with privacy-safe diagnostics
  and synthetic coverage for all marker types, allowed text, ignored binary
  and cache paths, and the current repository.
- Changed files: `AGENTS.md`, `docs/progress/fatih.md`,
  `scripts/check_conflict_markers.py`, `tests/test_conflict_markers.py`.
- Tests: focused 5 passed; full 1262 passed (1 dependency warning). Ruff and
  Pyright passed.
- Next planned step: use the scanner as part of normal repository validation;
  no runtime integration is required.

## F-015 â€” Pyannote Terminal Boundary Normalization

- Added a fail-closed 50 ms tolerance that clamps only the selected output's
  terminal end boundary to the call window.
- Changed files: `app/diarization/pyannote_backend.py`,
  `tests/test_pyannote_diarization_backend.py`, `docs/progress/fatih.md`.
- Tests: focused 47 passed; full 1555 passed and 10 opt-in integration tests
  skipped (1 dependency warning). Ruff and Pyright passed.
- Next planned step: validate the bounded policy in the Linux CPU runtime.

## F-016 â€” Privacy-Safe Role Observability

- Added immutable ID-free role evidence, inference, decision and projection
  exclusion diagnostics without changing resolution or routing behavior.
- Changed files: `app/diarization/__init__.py`,
  `app/diarization/role_resolver.py`, `app/diarization/routing.py`,
  `tests/test_diarization_role_resolver.py`,
  `tests/test_diarization_routing.py`, `docs/progress/fatih.md`.
- Tests: focused 53 passed; full 1938 passed and 12 opt-in integration tests
  skipped (1 dependency warning). Ruff and Pyright passed.
- Next planned step: use the aggregate contracts in privacy-safe operator
  diagnostics.

## F-017 — Speaker-Aware Dashboard

- Added a scope-checked, bounded dashboard projection for existing diarization
  role diagnostics and aggregate customer-projection counts without retaining
  speaker IDs, transcript text, timestamps, or evidence content.
- Changed files: `live_dashboard/app.py`, `live_dashboard/view_models.py`,
  `tests/test_live_dashboard_rendering.py`,
  `tests/test_live_dashboard_speaker_view.py`, `docs/progress/fatih.md`.
- Tests: focused dashboard tests 80 passed; full suite 1986 passed and 12
  opt-in integration tests skipped (1 existing dependency warning). Ruff and
  Pyright passed.
- Next planned step: wire the aggregate view into an explicitly diarization-
  enabled live runtime while preserving the legacy dashboard path.

## F-018 — Provider-Neutral Mono Live-Audio Ingress

- Added immutable bounded START/AUDIO_CHUNK/END contracts and a non-blocking,
  call-scoped mono ingress lifecycle with strict ordering, idempotent exact
  duplicates, fixed overload outcomes, immediate END/cancellation cleanup and
  privacy-safe latency/queue counters.
- Added bounded in-memory adaptation to the existing `AudioChunkEvent`
  contract without selecting a provider transport or invoking downstream work.
- Changed files: `app/audio_ingress/__init__.py`,
  `app/audio_ingress/contracts.py`, `app/audio_ingress/boundary.py`,
  `tests/test_live_audio_ingress.py`, `docs/progress/fatih.md`.
- Tests: focused ingress/legacy audio tests 62 passed; full suite 2019 passed
  and 12 opt-in integration tests skipped (1 existing dependency warning).
  Ruff and Pyright passed.
- Next planned step: bind a reviewed provider transport only after its codec,
  authentication, framing and lifecycle contracts are available.

## F-019 — Secret-Safe RAG Runtime Status Presentation

- Added deterministic visible READY, DISABLED and UNAVAILABLE presentation
  from the existing immutable dashboard RAG runtime state, with malformed
  values mapped to the fixed safe UNAVAILABLE text.
- Rendering retains only the enum state across reruns and does not invoke
  provider, manager, readiness, profile or completion lifecycle operations.
- Changed files: `live_dashboard/app.py`, `live_dashboard/presentation.py`,
  `tests/test_live_dashboard_rag_runtime_status.py`,
  `docs/progress/fatih.md`.
- Tests: focused D3 14 passed, speaker dashboard 5 passed, dashboard rendering
  12 passed; full suite 2090 passed and 13 skipped (1 existing dependency
  warning). Ruff and Pyright passed.
- Next planned step: review PR50-D3 for integration without adding provider
  lifecycle or dashboard state contracts.

## F-020 — Incremental Uploaded-Audio Dashboard

- Added a call-scoped single worker with cancellation, an immutable
  latest-value snapshot mailbox, monotonic per-chunk revisions and bounded
  privacy-safe diagnostics on the existing dashboard execution resource.
- Uploaded-audio processing now publishes transcript, intent/risk, coaching,
  progress and latency presentation state before END; rendering only polls
  retained snapshots. Real-time pacing uses an injectable clock/wait seam.
- Added an opt-in pipeline mode that omits accumulated step history while
  preserving callbacks and the default result contract for existing callers.
- Changed files: `live_dashboard/app.py`, `live_dashboard/runtime_wiring.py`,
  `live_dashboard/view_models.py`, `app/streaming/pipeline.py`, the four
  focused test modules, and `docs/progress/fatih.md`.
- Tests: focused dashboard/pipeline tests 146 passed; full suite 2099 passed
  and 13 opt-in tests skipped (1 existing dependency warning). Ruff and
  Pyright passed.
- Next planned step: review incremental polling and exactly-once cleanup
  behavior before integration.
- Manual acceptance follow-up: made the async Start handoff deterministic with
  one post-submit rerun and lifecycle-driven polling; added model-free fast and
  real-time upload/Start regression coverage.
- Added immutable execution mode/stage metadata and immediate Turkish
  start, preparation, engine, chunk-progress and terminal feedback without
  render-side processing; duplicate starts remain disabled while RUNNING.
  Focused tests: 154 passed.

## F-021 — Provisional PARTIAL Coaching

- Added an explicit, dashboard-opt-in PROVISIONAL/CONFIRMED/WITHDRAWN coaching
  lifecycle with bounded per-chunk PARTIAL gating, a one-second injectable
  cadence, fixed approved labels and stricter confidence thresholds.
- Meaningful PARTIAL text can publish deterministic template coaching before
  finalization; matching committed results promote the same card, while changed
  or unsupported results replace or withdraw it without consuming committed
  classification/coaching revision state.
- Changed files: the Fatih-owned classification, coaching, call/event,
  streaming and dashboard contracts plus four focused test modules.
- Tests: focused suites 197 passed; full suite 2121 passed and 13 opt-in tests
  skipped (1 existing dependency warning). Ruff and Pyright passed.
- Next planned step: complete manual dashboard acceptance before committing.

## F-022 — Local Browser Microphone Test

- Added a default-off, loopback-only `LOCAL_MIC_TEST` capability with
  exact tenant/call/resource scope, bounded mono PCM16LE 16 kHz normalization,
  two-second ingress chunks and deterministic terminal revocation.
- Added audio-only WebRTC capture with no ICE servers; its callback performs
  only in-memory normalization and bounded enqueueing. Existing live ingress,
  streaming ASR and provisional coaching publish snapshots before END.
- Changed files: `pyproject.toml`, `uv.lock`, the Fatih-owned audio-ingress,
  streaming and dashboard files, four focused test modules, and this progress
  file.
- Tests: focused microphone/dashboard tests 194 passed; PyAV media tests 52
  passed; full suite 2218 passed and 15 opt-in tests skipped (1 existing
  dependency warning). Ruff and Pyright passed.
- Manual follow-up keeps the WebRTC component in the same polling fragment,
  preserves a stable per-session component identity and drains a bounded final
  chunk before disconnect END so the first processed snapshot is not lost.
- Readiness follow-up blocks browser capture until the retained worker has
  loaded and warmed one local-only `tiny` CPU/int8 ASR engine, and records
  separate bounded load, warm-up, audio-preparation and inference timings.
- Focused readiness/dashboard/pipeline tests: 232 passed; PyAV/media tests:
  76 passed; full suite: 2231 passed and 15 opt-in tests skipped. Ruff,
  formatting and Pyright passed.
- Next planned step: run manual localhost microphone acceptance with the
  explicit local gate before committing.

## F-023 — Persistent Local Microphone Capture

- Separated the retained call/ASR worker lifecycle from each ephemeral browser
  capture, adding manual PAUSING/PAUSED resume and transient RECONNECTING
  behavior without emitting call END or clearing transcript/coaching state.
- Manual pause flushes bounded audio once, drains accepted chunks, revokes the
  old exact-scope capability, and resume issues a fresh capability/component
  generation while preserving call scope, revisions, metrics, and the warmed
  ASR pipeline.
- Applied the final live pipeline result to the retained dashboard state so
  reconciler FINAL transcript/classification/coaching outcomes are not lost;
  added bounded content-free ASR/event acceptance diagnostics.
- Made microphone Start, call Finish, and system Reset edge-triggered. Finish
  retains the completed report without recreating WebRTC; Reset removes the
  resource and performs a full rerun into a clean idle state.
- Changed files: Fatih-owned local audio ingress and dashboard presentation,
  focused microphone/dashboard/streaming tests, and this progress file.
- Tests: focused microphone/dashboard/streaming/ASR 231 passed; PyAV/media 78
  passed; full Windows suite 2477 passed, 17 skipped, with only the 19
  documented Ubuntu/POSIX vLLM controller tests failing. Ruff and formatting
  passed; Pyright remains limited to the three documented Ubuntu-only
  `os.getuid`/`os.O_NOFOLLOW` Windows findings.
- Next planned step: manually verify live transcript finalization and
  edge-triggered finish/reset behavior on localhost.
- Empty-ASR follow-up confirmed that PyAV normalization is unchanged from
  `102a6da3` and preserves synthetic packed/planar float and signed amplitude
  through mono 16 kHz PCM16 and the in-memory Whisper adapter.
- Bound each WebRTC callback to its capture generation so callbacks retained
  from a revoked capture cannot feed a resumed capability. Added bounded,
  content-free pre/post-resample energy, PCM, ASR-window/segment, and fixed
  rejection diagnostics with distinct dashboard states.
- Tests: focused microphone/audio/streaming/dashboard 240 passed; media 124
  passed; full Windows suite 2488 passed, 17 skipped, with only the 19
  documented Ubuntu/POSIX vLLM controller failures. Ruff, formatting, lock,
  dependency consistency, conflict and diff checks passed; Pyright remains
  limited to the three documented Ubuntu-only portability findings.
- Next planned step: rerun the localhost microphone acceptance test and use the
  new aggregate energy fields to distinguish silent capture from an empty ASR
  model result without exposing audio or transcript content.

## F-024 — Explicit Local Microphone ASR Profiles

- Preserved the default `cpu-tiny` profile and added an explicit
  `gpu-large-v3` profile selected only through
  `CALLMETRIC_LOCAL_MIC_ASR_PROFILE`.
- The GPU profile requires visible CTranslate2 CUDA, float16 support, exact
  `large-v3` CUDA/float16 model loading, and a successful warm-up before
  microphone capture. Unknown profiles and readiness failures remain
  fail-closed without CPU or model fallback.
- Added bounded model/device/profile/timing diagnostics and Turkish readiness
  failures. Pause/resume retains the warmed engine; completion/reset releases
  the loaded model through one idempotent cleanup path.
- Changed files: Fatih-owned ASR engine, local microphone ingress/dashboard,
  focused ASR/dashboard tests, and this progress file.
- Tests: focused ASR/microphone/streaming/dashboard 261 passed; PyAV/media 124
  passed; full Windows suite 2497 passed and 17 skipped, with only the 19
  documented Ubuntu/POSIX vLLM controller failures.
- Next planned step: run manual `gpu-large-v3` acceptance on a CUDA machine and
  verify displayed load, warm-up, inference, and real-time-factor diagnostics.

## Uploaded-audio GPU ASR profile

- Added independent `CALLMETRIC_UPLOADED_ASR_PROFILE` selection with the
  unchanged `cpu-large-v3` default and an exact, no-fallback
  `gpu-large-v3` CUDA/float16 option.
- GPU upload processing now verifies CTranslate2 CUDA/float16 support, loads
  `large-v3`, and completes a synthetic warm-up before audio chunk generation.
  Immutable snapshots expose only bounded runtime settings and timing metadata.
- Changed files: Fatih-owned dashboard app/view models, focused dashboard
  tests, and this progress file.
- Tests: focused upload/ASR/dashboard/streaming 229 passed; PyAV/media 128
  passed; full Windows suite 2508 passed and 17 skipped, with only the 19
  documented POSIX vLLM controller failures.
- Ruff, format, lock, dependency, conflict-marker, and diff checks passed.
  Pyright remains limited to the three documented POSIX portability findings.
- Next planned step: manually verify both upload and microphone GPU profiles on
  the CUDA target while confirming displayed preparation and inference timing.

## Uploaded-audio PARTIAL cadence

- Replaced accelerated upload PARTIAL gating with monotonic processed-audio
  progress while preserving the existing wall-clock cadence for live
  microphone capture and callers without media progress.
- Added fail-closed handling for invalid, repeated, or regressing media
  timestamps and source-scoped cadence reset without changing transcript
  reconciliation, confidence, coaching deduplication, or bounded-card policy.
- Changed files: Fatih-owned classification and streaming pipeline code,
  focused streaming tests, and this progress file.
- Tests: focused streaming/coaching/uploaded-dashboard 262 passed; media/audio
  ingress 148 passed; full Windows suite 2518 passed and 17 skipped, with only
  the 19 documented POSIX-only vLLM controller failures.
- Next planned step: manually verify several distinct pre-END risk updates
  during accelerated `gpu-large-v3` uploaded-audio processing.

## Provider-independent mono RTP ingress

- Added immutable, tenant/call/source-generation-scoped START, PCMA/PCMU packet,
  and END contracts plus a bounded in-memory RTP ordering adapter.
- The adapter deterministically decodes 8 kHz mono G.711 to 16 kHz PCM16,
  emits existing two-second `AudioChunkEvent` values, bounds jitter/output
  state, tracks fixed privacy-safe counters, and releases audio on reset or
  source replacement.
- Changed files: Fatih-owned audio-ingress RTP module/exports, synthetic focused
  tests, and this progress file.
- Tests: focused RTP/audio-ingress/streaming 186 passed; additional PyAV
  execution-resource 26 passed; full Windows suite 2537 passed and 17 skipped,
  with only the 19 documented POSIX-only vLLM controller failures.
- Ruff and format passed. New RTP code has 0 Pyright errors; repository Pyright
  remains limited to the three documented POSIX portability findings.
- Next planned step: review the provider packet mapping and lifecycle contract
  before any real transport or SIP integration is specified.

## Rule-only provisional coaching fallback

- Added cadence-gated deterministic PARTIAL classification when optional SetFit
  artifacts are unavailable, using only existing tenant rule-engine evidence.
- Rule-only outcomes carry fixed internal provenance without probabilities;
  SetFit thresholds, media-time/wall-clock cadence selection, isolation,
  deduplication, capacity, promotion, withdrawal, and FINAL behavior are
  unchanged.
- Changed files: Fatih-owned classification, coaching, streaming, dashboard
  runtime wiring, focused synthetic tests, and this progress file.
- Tests: focused core 221 passed; dashboard/worker/microphone 111 passed; full
  Windows suite 2543 passed and 17 skipped, with only the 19 documented
  POSIX-only vLLM controller failures.
- Ruff, format, conflict-marker, and diff checks passed. Pyright remains
  limited to the three documented POSIX portability findings.
- Next planned step: manually verify a pre-END rule-only provisional card in
  accelerated uploaded-audio execution without SetFit artifacts.
