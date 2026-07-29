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
