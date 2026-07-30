from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from live_dashboard.demo_data import TenantDemo, tenant_demos
from live_dashboard.smoke_tenant_override import (
    DashboardSmokeTenantOverrideError,
    SMOKE_TENANT_OVERRIDE_ENVIRONMENT_VARIABLE,
    apply_smoke_tenant_override,
)

_VALID_OVERRIDE = {
    "enabled": True,
    "tenant_id": "tenant_alpha",
    "knowledge_base_id": "kb_smoke",
    "top_k": 3,
    "minimum_score": 0.6,
    "enable_llm": True,
}
_ERROR = "dashboard smoke tenant override is invalid"


def _write_override(
    tmp_path: Path,
    payload: object = _VALID_OVERRIDE,
    *,
    name: str = "override.json",
) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _environment(path: Path) -> dict[str, str]:
    return {SMOKE_TENANT_OVERRIDE_ENVIRONMENT_VARIABLE: str(path)}


def _default_demos(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, TenantDemo]:
    monkeypatch.delenv(SMOKE_TENANT_OVERRIDE_ENVIRONMENT_VARIABLE, raising=False)
    return tenant_demos()


def _assert_invalid(
    demos: dict[str, TenantDemo],
    environment: dict[str, str],
) -> DashboardSmokeTenantOverrideError:
    with pytest.raises(DashboardSmokeTenantOverrideError) as raised:
        apply_smoke_tenant_override(demos, environment=environment)
    assert str(raised.value) == _ERROR
    assert raised.value.__cause__ is None
    return raised.value


def test_module_import_performs_no_file_or_provider_activity() -> None:
    script = """
import builtins
import importlib
import sys

blocked = {"psycopg", "sentence_transformers", "streamlit", "vllm"}
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", maxsplit=1)[0] in blocked:
        raise AssertionError(f"blocked provider import attempted: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
importlib.import_module("live_dashboard.smoke_tenant_override")
loaded = {name.split(".", maxsplit=1)[0] for name in sys.modules}
if unexpected := blocked & loaded:
    raise AssertionError(f"provider modules loaded: {sorted(unexpected)}")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_absent_environment_returns_exact_existing_collection_without_file_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demos = _default_demos(monkeypatch)
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: pytest.fail("override file was accessed"),
    )

    result = apply_smoke_tenant_override(demos, environment={})

    assert result is demos
    assert all(not demo.config.rag.enabled for demo in result.values())
    assert all(not demo.config.coaching.enable_llm for demo in result.values())


def test_valid_override_activates_only_an_independent_selected_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demos = _default_demos(monkeypatch)
    alpha_before = demos["tenant_alpha"]
    beta_before = demos["tenant_beta"]

    result = apply_smoke_tenant_override(
        demos,
        environment=_environment(_write_override(tmp_path)),
    )

    activated = result["tenant_alpha"]
    assert activated is not alpha_before
    assert activated.config is not alpha_before.config
    assert activated.config.context == alpha_before.config.context
    assert activated.config.asr == alpha_before.config.asr
    assert activated.config.classification == alpha_before.config.classification
    assert activated.config.rag.enabled
    assert activated.config.rag.knowledge_base_id == "kb_smoke"
    assert activated.config.rag.top_k == 3
    assert activated.config.rag.minimum_score == 0.6
    assert activated.config.coaching.enable_llm
    assert result["tenant_beta"] is beta_before
    assert result["tenant_beta"] == beta_before
    assert not alpha_before.config.rag.enabled
    assert not alpha_before.config.coaching.enable_llm


def test_repeated_application_is_equal_but_independently_copied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demos = _default_demos(monkeypatch)
    environment = _environment(_write_override(tmp_path))

    first = apply_smoke_tenant_override(demos, environment=environment)
    second = apply_smoke_tenant_override(demos, environment=environment)

    assert first == second
    assert first is not second
    assert first["tenant_alpha"] is not second["tenant_alpha"]
    assert first["tenant_alpha"].config is not second["tenant_alpha"].config


@pytest.mark.parametrize("raw_path", ["", " ", "override.json"])
def test_path_must_be_canonical_and_absolute(
    raw_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demos = _default_demos(monkeypatch)
    _assert_invalid(
        demos,
        {SMOKE_TENANT_OVERRIDE_ENVIRONMENT_VARIABLE: raw_path},
    )


def test_explicit_traversal_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demos = _default_demos(monkeypatch)
    raw_path = str(tmp_path / ".." / "override.json")
    _assert_invalid(
        demos,
        {SMOKE_TENANT_OVERRIDE_ENVIRONMENT_VARIABLE: raw_path},
    )


def test_missing_file_and_directory_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demos = _default_demos(monkeypatch)
    for path in (tmp_path / "missing.json", tmp_path):
        _assert_invalid(demos, _environment(path))


def test_symlink_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demos = _default_demos(monkeypatch)
    target = _write_override(tmp_path)
    link = tmp_path / "override-link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    _assert_invalid(demos, _environment(link))


def test_file_size_bound_accepts_exact_limit_and_rejects_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demos = _default_demos(monkeypatch)
    compact = json.dumps(_VALID_OVERRIDE).encode()
    exact = tmp_path / "exact.json"
    exact.write_bytes(compact + b" " * (65_536 - len(compact)))
    accepted = apply_smoke_tenant_override(demos, environment=_environment(exact))
    assert accepted["tenant_alpha"].config.rag.enabled

    overflow = tmp_path / "overflow.json"
    overflow.write_bytes(exact.read_bytes() + b" ")
    _assert_invalid(demos, _environment(overflow))


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\xef\xbb\xbf{}",
        b'{"enabled":true}\0',
        b"\xff",
        b"{",
        b"[]",
    ],
)
def test_invalid_encoding_or_json_shape_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
) -> None:
    demos = _default_demos(monkeypatch)
    path = tmp_path / "invalid.json"
    path.write_bytes(content)
    _assert_invalid(demos, _environment(path))


@pytest.mark.parametrize(
    "content",
    [
        (
            '{"enabled":true,"tenant_id":"tenant_alpha",'
            '"tenant_id":"tenant_beta","knowledge_base_id":"kb_smoke",'
            '"top_k":3,"minimum_score":0.6,"enable_llm":true}'
        ),
        '{"enabled":true}',
        json.dumps({**_VALID_OVERRIDE, "unknown": "value"}),
        json.dumps(
            {
                **_VALID_OVERRIDE,
                "knowledge_base_id": {"Api_Token": "synthetic-secret"},
            }
        ),
    ],
)
def test_duplicate_missing_unknown_and_nested_secret_keys_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    demos = _default_demos(monkeypatch)
    path = tmp_path / "invalid-keys.json"
    path.write_text(content, encoding="utf-8")
    _assert_invalid(demos, _environment(path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enabled", False),
        ("enabled", 1),
        ("enable_llm", False),
        ("enable_llm", 1),
        ("tenant_id", ""),
        ("tenant_id", " tenant_alpha"),
        ("tenant_id", "tenant_unknown"),
        ("knowledge_base_id", ""),
        ("knowledge_base_id", " kb_smoke"),
        ("top_k", True),
        ("top_k", 0),
        ("top_k", -1),
        ("minimum_score", True),
        ("minimum_score", -0.1),
        ("minimum_score", 1.1),
        ("minimum_score", float("nan")),
        ("minimum_score", float("inf")),
    ],
)
def test_strict_field_validation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    demos = _default_demos(monkeypatch)
    path = _write_override(tmp_path, {**_VALID_OVERRIDE, field: value})
    _assert_invalid(demos, _environment(path))


@pytest.mark.parametrize("minimum_score", [0, 1])
def test_minimum_score_boundaries_are_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    minimum_score: int,
) -> None:
    demos = _default_demos(monkeypatch)
    path = _write_override(
        tmp_path,
        {**_VALID_OVERRIDE, "top_k": 1, "minimum_score": minimum_score},
    )

    result = apply_smoke_tenant_override(demos, environment=_environment(path))

    assert result["tenant_alpha"].config.rag.top_k == 1
    assert result["tenant_alpha"].config.rag.minimum_score == minimum_score


def test_error_is_fixed_and_contains_no_sensitive_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demos = _default_demos(monkeypatch)
    sensitive_id = "synthetic-sensitive-tenant"
    sensitive_kb = "synthetic-sensitive-kb"
    path = _write_override(
        tmp_path,
        {
            **_VALID_OVERRIDE,
            "tenant_id": sensitive_id,
            "knowledge_base_id": sensitive_kb,
        },
        name="synthetic-sensitive-path.json",
    )

    error = _assert_invalid(demos, _environment(path))
    rendered = str(error)

    assert rendered == _ERROR
    assert str(path) not in rendered
    assert sensitive_id not in rendered
    assert sensitive_kb not in rendered
    assert "synthetic-sensitive" not in rendered


def test_committed_example_has_exact_safe_schema() -> None:
    path = Path("docs/examples/dashboard-rag-smoke-tenant.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == _VALID_OVERRIDE
    rendered = path.read_text(encoding="utf-8").casefold()
    assert all(
        marker not in rendered
        for marker in (
            "dsn",
            "password",
            "api_token",
            "endpoint",
            "certificate",
            "transcript",
            "prompt",
        )
    )


def test_runbook_requires_verified_tls_and_does_not_claim_compose_tls() -> None:
    runbook = Path("docs/runbooks/postgres_rag_smoke.md").read_text(encoding="utf-8")

    assert "sslmode=verify-full" in runbook
    assert "trusted root CA" in runbook
    assert "hostname matches" in runbook
    assert "CALLMETRIC_VLLM_VERIFY_TLS=true" in runbook
    assert "does not configure" in runbook
    assert "or prove TLS" in runbook
    assert "compose.postgres-integration.yml" in runbook
