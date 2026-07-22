# CallMetric Live ASR — Codex Instructions

## Project

- Python 3.12 project managed with uv.
- Use complete type annotations.
- Keep the architecture modular and tenant-aware.
- Every call-related event must preserve tenant_id and call_id.
- Accuracy is currently more important than local inference latency.
- Production latency will later be optimized on AWS GPU.

## Privacy and Safety

- Never access, inspect, copy or modify C:\CallMetricPrivate.
- Never add real recordings, transcripts, customer data or credentials to Git.
- Never print or log raw audio bytes.
- Use only synthetic data in automated tests.
- Do not run or download ASR models unless the task explicitly requests it.
- Do not connect to AWS or external services unless explicitly requested.

## Scope

- Inspect only files and directories relevant to the current task.
- Make the smallest maintainable change.
- Do not implement unrelated future features.
- Do not install packages unless explicitly required.
- Do not create Git commits.
- Ask before destructive commands.

## Testing

During development, run focused tests for the changed module.

At the end, run once:

- uv run pytest
- uv run ruff check .
- uv run ruff format --check .
- uv run pyright

If formatting is required, run uv run ruff format . and repeat the checks.

## Final Response

Keep the final report under 15 lines and include only:

- changed files;
- focused and full test results;
- Ruff result;
- Pyright result;
- remaining warnings;
- final Git status.
## Project Progress Log

- After every completed development task, update `PROJECT_PROGRESS.md`.
- Add only a short factual summary of the completed feature.
- Include changed files, test results and the next planned step.
- Never include private audio, transcript content, customer information or
  absolute private file paths.
- Updating `PROJECT_PROGRESS.md` is part of the task's definition of done.