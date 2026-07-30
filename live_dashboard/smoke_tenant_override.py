"""Strict opt-in activation of one existing synthetic dashboard tenant."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, field_validator

from app.tenancy.models import TenantConfig

if TYPE_CHECKING:
    from live_dashboard.demo_data import TenantDemo

SMOKE_TENANT_OVERRIDE_ENVIRONMENT_VARIABLE = (
    "CALLMETRIC_DASHBOARD_SMOKE_TENANT_OVERRIDE_PATH"
)
_MAX_FILE_BYTES = 65_536
_EXPECTED_KEYS = frozenset(
    {
        "enabled",
        "tenant_id",
        "knowledge_base_id",
        "top_k",
        "minimum_score",
        "enable_llm",
    }
)
_SECRET_KEY_MARKERS = (
    "api_key",
    "base_url",
    "certificate",
    "credential",
    "dsn",
    "endpoint",
    "password",
    "private_key",
    "secret",
    "token",
)
_INVALID_OVERRIDE = "dashboard smoke tenant override is invalid"


class DashboardSmokeTenantOverrideError(ValueError):
    """Fixed secret-safe failure for an invalid opt-in smoke override."""


class _SmokeTenantOverride(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    tenant_id: str
    knowledge_base_id: str
    top_k: int
    minimum_score: float
    enable_llm: bool

    @field_validator("enabled", "enable_llm", mode="before")
    @classmethod
    def validate_enabled_flag(cls, value: object) -> bool:
        if type(value) is not bool or value is not True:
            raise ValueError("activation flags must be exactly true")
        return value

    @field_validator("tenant_id", "knowledge_base_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("identifier must be canonical and nonblank")
        return value

    @field_validator("top_k", mode="before")
    @classmethod
    def validate_top_k(cls, value: object) -> int:
        if type(value) is not int or value <= 0:
            raise ValueError("top_k must be a positive integer")
        return value

    @field_validator("minimum_score", mode="before")
    @classmethod
    def validate_minimum_score(cls, value: object) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ValueError("minimum_score must be finite and between zero and one")
        return float(value)


def apply_smoke_tenant_override(
    demos: dict[str, TenantDemo],
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, TenantDemo]:
    """Return defaults unchanged or activate one independently copied demo."""
    source = os.environ if environment is None else environment
    raw_path = source.get(SMOKE_TENANT_OVERRIDE_ENVIRONMENT_VARIABLE)
    if raw_path is None:
        return demos
    try:
        override = _load_override(raw_path)
        selected = demos.get(override.tenant_id)
        if selected is None or not isinstance(selected.config, TenantConfig):
            raise ValueError("unknown synthetic tenant")
        copied_config = selected.config.model_copy(deep=True)
        copied_config.rag = copied_config.rag.model_copy(
            update={
                "enabled": override.enabled,
                "knowledge_base_id": override.knowledge_base_id,
                "top_k": override.top_k,
                "minimum_score": override.minimum_score,
            }
        )
        copied_config.coaching = copied_config.coaching.model_copy(
            update={"enable_llm": override.enable_llm}
        )
        validated_config = TenantConfig.model_validate(copied_config.model_dump())
        updated = replace(selected, config=validated_config)
        return {
            tenant_id: updated if tenant_id == override.tenant_id else demo
            for tenant_id, demo in demos.items()
        }
    except Exception:
        raise DashboardSmokeTenantOverrideError(_INVALID_OVERRIDE) from None


def _load_override(raw_path: object) -> _SmokeTenantOverride:
    path = _validated_path(raw_path)
    file_status = path.stat()
    if not stat.S_ISREG(file_status.st_mode) or file_status.st_size > _MAX_FILE_BYTES:
        raise ValueError("invalid override file")
    with path.open("rb") as stream:
        opened_status = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(opened_status.st_mode)
            or opened_status.st_size > _MAX_FILE_BYTES
        ):
            raise ValueError("invalid override file")
        content = stream.read(_MAX_FILE_BYTES + 1)
    if (
        not content
        or len(content) > _MAX_FILE_BYTES
        or content.startswith(b"\xef\xbb\xbf")
        or b"\0" in content
    ):
        raise ValueError("invalid override content")
    payload = json.loads(
        content.decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError("override must be an object")
    _reject_secret_keys(payload)
    if set(payload) != _EXPECTED_KEYS:
        raise ValueError("override keys are invalid")
    return _SmokeTenantOverride.model_validate(payload)


def _validated_path(raw_path: object) -> Path:
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or raw_path != raw_path.strip()
        or "\0" in raw_path
    ):
        raise ValueError("invalid override path")
    path = Path(raw_path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("invalid override path")
    if path.is_symlink() or not path.is_file():
        raise ValueError("invalid override path")
    return path


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate override key")
        payload[key] = value
    return payload


def _reject_secret_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = key.casefold()
            if any(marker in normalized for marker in _SECRET_KEY_MARKERS):
                raise ValueError("secret-like key is prohibited")
            _reject_secret_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_secret_keys(nested)
