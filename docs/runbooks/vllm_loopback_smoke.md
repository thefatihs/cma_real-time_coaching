# vLLM Loopback Smoke Runbook

This is an opt-in, synthetic-only GPU smoke environment for the PR53 Linux
host. It is not a public or production deployment. The host publishes native
vLLM HTTPS only on `127.0.0.1:8001`; no private-network, public-network,
firewall, or AWS security-group exposure is part of this runbook. PR54 will
separately design Windows-to-VM connectivity and certificate trust.

## Immutable artifacts and rationale

The single-container Compose project pins
`vllm/vllm-openai:v0.26.0-ubuntu2404` to Linux/amd64 manifest
`sha256:1161da8a5edbdff239ab1812784d7fe5d28775c675809a8420e8a0a05d0e56d1`.
The controller also verifies multi-architecture index
`sha256:ef7bfc14df9233e3e5d41e733e3be0afa6abbe5ae5f14ee0758110030f6cd53e`
before pulling.

The model is the official Apache-2.0, non-thinking instruct checkpoint
`Qwen/Qwen2.5-7B-Instruct-AWQ` at immutable revision
`b25037543e9394b818fdfca67ab2a00ecc7dd641`, served as
`callmetric-qwen25-7b-awq`. AWQ INT4 is selected because it fits the L40S
comfortably while preserving substantially more disk and VRAM headroom than
larger FP8 candidates. Raw `/v1/completions` remains prompt-dependent and does
not apply the model's chat template.

## Security boundaries

vLLM terminates TLS itself. Each invocation generates a private ephemeral CA
and server certificate outside the repository. The server certificate SAN is
exactly `DNS:localhost,IP:127.0.0.1`; the smoke verifies the private CA,
hostname matching, missing-CA rejection, and deliberate hostname-mismatch
rejection. A new random bearer token is passed through process environment and
is never printed or placed in command arguments.

Requests, prompts, responses, tokens, keys, certificates, cache contents, and
temporary paths must not be logged. Only the committed deterministic Turkish
synthetic prompt may be submitted. Never use customer data, production
endpoints, real secrets, or private certificate material.

## Persistent cache and capacity

Set `CALLMETRIC_VLLM_SMOKE_CACHE_DIR` to an existing absolute directory outside
the repository, owned by the invoking user and not group/world writable. The
runner mounts it as the Hugging Face cache and never deletes it. Do not inspect
or manually publish cache contents.

Before execution, the root filesystem must have at least 56.556 GiB free:
39.556 GiB conservative transient usage plus a 15 GiB final reserve and 2 GiB
additional margin. The guard includes compressed layers, a 2.5x expanded-image
estimate, the complete model repository, and one full-model temporary
allowance.

After the reviewed artifacts are committed, explicitly pin that clean commit
for the runner's immutable HEAD check:

```bash
export CALLMETRIC_VLLM_SMOKE_EXPECTED_HEAD="$(git rev-parse HEAD)"
```

## Future smoke execution

Do not run this command during the repository-artifact phase. After review and
explicit approval for image/model downloads, certificate generation, and one
GPU container lifecycle:

```bash
uv run python scripts/run_vllm_loopback_smoke.py
```

The controller verifies repository identity, host, clean Git state, Docker
Compose, immutable registry and model metadata, GPU 0, disk capacity, the four
protected containers, and free port 8001 before pulling. Pull and startup have
bounded timeouts. Runtime limits are 8192 context tokens, 0.80 GPU-memory
utilization, two sequences, and 256 output tokens.

## Health and smoke acceptance

Success requires trusted HTTPS health; missing-CA and hostname-mismatch
failures; rejection of absent and incorrect bearer tokens; exactly
`callmetric-qwen25-7b-awq` from `/v1/models`; one non-empty raw completion from
the synthetic prompt; and increased GPU 0 memory activity. Completion content
is never displayed or logged.

## Verified GPU smoke result

The real loopback GPU smoke passed exactly once in 6m20s. The immutable image
pull took approximately 3m; model preparation, startup, and verification took
approximately 3m20s. The execution used
`vllm/vllm-openai:v0.26.0-ubuntu2404`, index
`sha256:ef7bfc14df9233e3e5d41e733e3be0afa6abbe5ae5f14ee0758110030f6cd53e`,
Linux/amd64 manifest
`sha256:1161da8a5edbdff239ab1812784d7fe5d28775c675809a8420e8a0a05d0e56d1`,
and `Qwen/Qwen2.5-7B-Instruct-AWQ` revision
`b25037543e9394b818fdfca67ab2a00ecc7dd641`.

Trusted TLS, missing-CA rejection, hostname-mismatch rejection, absent and
incorrect bearer-token rejection, exact `/v1/models` identity, and a non-empty
synthetic `/v1/completions` result bounded to 256 output tokens all passed.
GPU activity reached 6,057 MiB and returned to 0 MiB. Cleanup removed the PR53
container, project network, and ephemeral TLS/token material while preserving
the immutable image and persistent model cache. Port 8001 was free afterward,
approximately 34.13 GiB disk remained, and all four protected containers were
still running with unchanged identities.

Verification recorded 42 passing runner tests and 87 passing lightweight LLM
tests. Ruff and focused Pyright passed. Full Pyright was not considered
authoritative because unrelated optional ML/runtime dependencies were
intentionally not installed.

## Kullanılan Teknolojiler

- Python 3.12.12; uv 0.12.0; pytest, Ruff, and Pyright
- Docker Engine and Docker Compose
- NVIDIA L40S, NVIDIA Container Toolkit, and CUDA runtime
- vLLM v0.26.0 and Qwen2.5-7B-Instruct-AWQ with AWQ INT4 quantization
- OpenAI-compatible `/v1/completions`
- HTTPS/TLS with ephemeral CA/server certificates and bearer-token authentication
- Linux/AWS Ubuntu 24.04
- Persistent Hugging Face model cache
- Synthetic Turkish coaching smoke data

The service remains loopback-only. PR54 will separately design Windows
dashboard connectivity and certificate trust.

## Cleanup and rollback

On success or failure, the controller runs `down` only for its randomized PR53
Compose project. This removes its one container and project-scoped network,
then the temporary-directory context deletes only that run's certificates.
The persistent model cache and all unrelated containers, images, volumes, and
networks remain untouched. Never use Docker prune, broad stop/remove commands,
or an existing Compose environment. Rollback is simply omission of this
opt-in project; no application or networking files are changed.
