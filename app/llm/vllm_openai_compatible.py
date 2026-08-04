"""Production-safe gateway for an external vLLM OpenAI-compatible API."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
import re
import ssl
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.llm.models import LLMRequest, LLMResponse

_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_NUMERIC_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_INTEGER_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)$")
_INVALID_RESPONSE = "vLLM response is invalid"
_INVALID_TLS_CONFIGURATION = "vLLM TLS configuration is invalid"
_INVALID_STRUCTURED_OUTPUT = "vLLM structured output schema is invalid"


class VLLMOpenAICompatibleSettings(BaseSettings):
    """Immutable environment-backed settings for one external vLLM API."""

    model_config = SettingsConfigDict(
        env_prefix="CALLMETRIC_VLLM_",
        env_file=None,
        extra="forbid",
        frozen=True,
    )

    base_url: str
    model_id: str
    api_token: SecretStr | None = None
    ca_certificate_path: SecretStr | None = None
    connect_timeout_seconds: float
    read_timeout_seconds: float
    max_output_tokens: int
    temperature: float
    verify_tls: bool

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        if (
            value != value.strip()
            or not value.startswith("https://")
            or any(character.isspace() for character in value)
        ):
            raise ValueError("base_url must be a canonical HTTPS /v1 URL")
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            raise ValueError("base_url must be a canonical HTTPS /v1 URL") from None
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"/v1", "/v1/"}
            or port is not None
            and not 1 <= port <= 65535
        ):
            raise ValueError("base_url must be a canonical HTTPS /v1 URL")
        return urlunsplit(("https", parsed.netloc, "/v1", "", ""))

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        if value != value.strip() or not _MODEL_ID_PATTERN.fullmatch(value):
            raise ValueError("model_id must be canonical and nonblank")
        return value

    @field_validator("api_token")
    @classmethod
    def validate_api_token(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        secret = value.get_secret_value()
        if not secret or secret != secret.strip():
            raise ValueError("api_token must be canonical and nonblank")
        return value

    @field_validator(
        "connect_timeout_seconds",
        "read_timeout_seconds",
        "temperature",
        mode="before",
    )
    @classmethod
    def validate_numeric_setting(cls, value: object, info: object) -> float:
        numeric = _strict_numeric(value)
        field_name = getattr(info, "field_name", "")
        minimum, maximum = {
            "connect_timeout_seconds": (1.0, 60.0),
            "read_timeout_seconds": (1.0, 600.0),
            "temperature": (0.0, 2.0),
        }[field_name]
        if not minimum <= numeric <= maximum:
            raise ValueError(f"{field_name} is outside the allowed range")
        return numeric

    @field_validator("max_output_tokens", mode="before")
    @classmethod
    def validate_max_output_tokens(cls, value: object) -> int:
        if isinstance(value, str):
            if not _INTEGER_PATTERN.fullmatch(value):
                raise ValueError("max_output_tokens must be an integer")
            value = int(value)
        if type(value) is not int:
            raise ValueError("max_output_tokens must be an integer")
        if not 1 <= value <= 32768:
            raise ValueError("max_output_tokens is outside the allowed range")
        return value

    @field_validator("verify_tls", mode="before")
    @classmethod
    def validate_verify_tls(cls, value: object) -> bool:
        if value == "true":
            value = True
        if type(value) is not bool or value is not True:
            raise ValueError(_INVALID_TLS_CONFIGURATION)
        return value


class VLLMOpenAICompatibleGateway:
    """Synchronous non-streaming gateway for an external vLLM service."""

    def __init__(
        self,
        settings: VLLMOpenAICompatibleSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        structured_output_json_schema: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(settings, VLLMOpenAICompatibleSettings):
            raise ValueError("settings must be VLLMOpenAICompatibleSettings")
        if transport is not None and not isinstance(transport, httpx.BaseTransport):
            raise ValueError("transport must be an httpx BaseTransport")
        self._settings = settings
        self._transport = transport
        self._structured_output_json_schema = _validated_schema_copy(
            structured_output_json_schema
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        if not isinstance(request, LLMRequest):
            raise ValueError("request must be LLMRequest")
        settings = self._settings
        timeout = httpx.Timeout(
            connect=settings.connect_timeout_seconds,
            read=settings.read_timeout_seconds,
            write=settings.read_timeout_seconds,
            pool=settings.connect_timeout_seconds,
        )
        headers = {"Accept": "application/json"}
        if settings.api_token is not None:
            headers["Authorization"] = f"Bearer {settings.api_token.get_secret_value()}"
        request_payload: dict[str, object] = {
            "model": settings.model_id,
            "prompt": request.input_text,
            "max_tokens": settings.max_output_tokens,
            "temperature": settings.temperature,
            "stream": False,
        }
        if self._structured_output_json_schema is not None:
            request_payload["structured_outputs"] = {
                "json": deepcopy(self._structured_output_json_schema)
            }
        with httpx.Client(
            verify=_tls_verification(settings),
            timeout=timeout,
            transport=self._transport,
            headers=headers,
        ) as client:
            response = client.post(
                f"{settings.base_url}/completions",
                json=request_payload,
            )
            response.raise_for_status()
            try:
                payload = response.json()
            except json.JSONDecodeError:
                raise ValueError(_INVALID_RESPONSE) from None
        text = _response_text(payload)
        return LLMResponse(
            tenant_id=request.tenant_id,
            call_id=request.call_id,
            text=text,
        )


def _strict_numeric(value: object) -> float:
    if isinstance(value, str):
        if not _NUMERIC_PATTERN.fullmatch(value):
            raise ValueError("numeric setting must be canonical")
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numeric setting must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("numeric setting must be finite")
    return numeric


def _validated_schema_copy(
    value: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise ValueError(_INVALID_STRUCTURED_OUTPUT)
    try:
        copied = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ValueError(_INVALID_STRUCTURED_OUTPUT) from None
    if not isinstance(copied, dict) or not copied:
        raise ValueError(_INVALID_STRUCTURED_OUTPUT)
    return copied


def _tls_verification(
    settings: VLLMOpenAICompatibleSettings,
) -> bool | ssl.SSLContext:
    if settings.verify_tls is not True:
        raise ValueError(_INVALID_TLS_CONFIGURATION)
    configured_path = settings.ca_certificate_path
    if configured_path is None:
        return True
    raw_path = configured_path.get_secret_value()
    try:
        path = Path(raw_path)
        if (
            not raw_path
            or raw_path != raw_path.strip()
            or "\0" in raw_path
            or not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError(_INVALID_TLS_CONFIGURATION)
        return ssl.create_default_context(cafile=str(path))
    except (OSError, ValueError):
        raise ValueError(_INVALID_TLS_CONFIGURATION) from None


def _response_text(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ValueError(_INVALID_RESPONSE)
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError(_INVALID_RESPONSE)
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError(_INVALID_RESPONSE)
    if choice.get("finish_reason") == "length":
        raise ValueError(_INVALID_RESPONSE)
    text = choice.get("text")
    if not isinstance(text, str):
        raise ValueError(_INVALID_RESPONSE)
    cleaned = text.strip()
    if not cleaned:
        raise ValueError(_INVALID_RESPONSE)
    return cleaned
