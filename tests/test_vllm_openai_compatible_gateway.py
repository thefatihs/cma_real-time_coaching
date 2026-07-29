"""Deterministic tests for the external vLLM OpenAI-compatible gateway."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, cast

import httpx
import pytest
from pydantic import SecretStr, ValidationError

import app.llm as llm_exports
import app.llm.vllm_openai_compatible as provider
from app.composition.postgres_rag_orchestration import (
    LLMGatewayFactory,
    _DeferredLLMGateway,
)
from app.llm import (
    LLMGateway,
    LLMRequest,
    LLMResponse,
    VLLMOpenAICompatibleGateway,
    VLLMOpenAICompatibleSettings,
)

_TOKEN = "synthetic-provider-token"
_ENVIRONMENT = {
    "CALLMETRIC_VLLM_BASE_URL": "https://vllm.invalid/v1",
    "CALLMETRIC_VLLM_MODEL_ID": "synthetic/model-v1",
    "CALLMETRIC_VLLM_API_TOKEN": _TOKEN,
    "CALLMETRIC_VLLM_CONNECT_TIMEOUT_SECONDS": "7.5",
    "CALLMETRIC_VLLM_READ_TIMEOUT_SECONDS": "90",
    "CALLMETRIC_VLLM_MAX_OUTPUT_TOKENS": "512",
    "CALLMETRIC_VLLM_TEMPERATURE": "0.25",
    "CALLMETRIC_VLLM_VERIFY_TLS": "true",
}


def _settings(**updates: object) -> VLLMOpenAICompatibleSettings:
    values: dict[str, object] = {
        "base_url": "https://vllm.invalid/v1",
        "model_id": "synthetic/model-v1",
        "api_token": SecretStr(_TOKEN),
        "connect_timeout_seconds": 7.5,
        "read_timeout_seconds": 90,
        "max_output_tokens": 512,
        "temperature": 0.25,
        "verify_tls": True,
    }
    values.update(updates)
    return VLLMOpenAICompatibleSettings.model_validate(values)


def _request(**updates: object) -> LLMRequest:
    values: dict[str, object] = {
        "tenant_id": "tenant-synthetic",
        "call_id": "call-synthetic",
        "input_text": "Synthetic Unicode prompt: çözüm.",
    }
    values.update(updates)
    return LLMRequest.model_validate(values)


def _transport(
    handler: httpx.MockTransport | Any,
) -> httpx.MockTransport:
    if isinstance(handler, httpx.MockTransport):
        return handler
    return httpx.MockTransport(handler)


def _gateway(
    handler: Any,
    *,
    settings: VLLMOpenAICompatibleSettings | None = None,
) -> VLLMOpenAICompatibleGateway:
    return VLLMOpenAICompatibleGateway(
        settings if settings is not None else _settings(),
        transport=_transport(handler),
    )


def test_settings_load_exact_environment_are_frozen_and_use_no_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name, value in _ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    settings_factory = cast(
        Callable[[], VLLMOpenAICompatibleSettings],
        VLLMOpenAICompatibleSettings,
    )
    settings = settings_factory()

    assert settings.base_url == "https://vllm.invalid/v1"
    assert settings.model_id == "synthetic/model-v1"
    assert settings.api_token is not None
    assert settings.api_token.get_secret_value() == _TOKEN
    assert settings.connect_timeout_seconds == 7.5
    assert settings.read_timeout_seconds == 90.0
    assert settings.max_output_tokens == 512
    assert settings.temperature == 0.25
    assert settings.verify_tls is True
    with pytest.raises(ValidationError):
        settings.temperature = 1.0

    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(f"{key}={value}" for key, value in _ENVIRONMENT.items()),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError):
        settings_factory()


def test_all_fields_except_api_token_are_required() -> None:
    required = {
        "base_url",
        "model_id",
        "connect_timeout_seconds",
        "read_timeout_seconds",
        "max_output_tokens",
        "temperature",
        "verify_tls",
    }

    assert {
        name
        for name, field in VLLMOpenAICompatibleSettings.model_fields.items()
        if field.is_required()
    } == required
    assert _settings(api_token=None).api_token is None


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://vllm.invalid/v1", "https://vllm.invalid/v1"),
        ("https://vllm.invalid/v1/", "https://vllm.invalid/v1"),
    ],
)
def test_base_url_canonical_policy(base_url: str, expected: str) -> None:
    assert _settings(base_url=base_url).base_url == expected


@pytest.mark.parametrize(
    "base_url",
    [
        "http://vllm.invalid/v1",
        "HTTPS://vllm.invalid/v1",
        " https://vllm.invalid/v1",
        "https://vllm.invalid/v1 ",
        "https://user@vllm.invalid/v1",
        "https://user:password@vllm.invalid/v1",
        "https://vllm.invalid/v1?query=value",
        "https://vllm.invalid/v1#fragment",
        "https://vllm.invalid/",
        "https://vllm.invalid/v1/completions",
        "https://vllm.invalid/unsafe/v1",
        "https:///v1",
        "https://vllm.invalid:99999/v1",
    ],
)
def test_base_url_rejects_noncanonical_or_unsafe_values(base_url: str) -> None:
    with pytest.raises(ValidationError, match="canonical HTTPS /v1"):
        _settings(base_url=base_url)


@pytest.mark.parametrize(
    "model_id",
    ["", " ", " synthetic/model", "synthetic/model ", "synthetic model", "@model"],
)
def test_model_id_must_be_canonical(model_id: str) -> None:
    with pytest.raises(ValidationError, match="model_id"):
        _settings(model_id=model_id)


@pytest.mark.parametrize(
    ("field", "valid_values"),
    [
        ("connect_timeout_seconds", (1, 1.5, 60)),
        ("read_timeout_seconds", (1, 10.5, 600)),
        ("temperature", (0, 0.5, 2)),
    ],
)
def test_numeric_boundaries_are_accepted(
    field: str,
    valid_values: tuple[float, ...],
) -> None:
    for value in valid_values:
        assert getattr(_settings(**{field: value}), field) == float(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("connect_timeout_seconds", 0),
        ("connect_timeout_seconds", 60.1),
        ("connect_timeout_seconds", True),
        ("connect_timeout_seconds", float("nan")),
        ("read_timeout_seconds", 0),
        ("read_timeout_seconds", 601),
        ("read_timeout_seconds", False),
        ("read_timeout_seconds", float("inf")),
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("temperature", True),
        ("temperature", " 1"),
    ],
)
def test_numeric_settings_are_strict_and_bounded(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _settings(**{field: value})


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 1), (32768, 32768), ("512", 512)],
)
def test_max_output_token_boundaries(value: object, expected: int) -> None:
    assert _settings(max_output_tokens=value).max_output_tokens == expected


@pytest.mark.parametrize("value", [0, 32769, True, 1.5, "1.5", " 1"])
def test_max_output_tokens_are_strict_and_bounded(value: object) -> None:
    with pytest.raises(ValidationError):
        _settings(max_output_tokens=value)


@pytest.mark.parametrize("value", [False, 0, 1, "false", "True"])
def test_tls_verification_cannot_be_disabled_or_coerced(value: object) -> None:
    with pytest.raises(ValidationError, match="exactly true"):
        _settings(verify_tls=value)


def test_api_token_is_optional_canonical_and_secret_safe() -> None:
    settings = _settings()
    representations = (
        repr(settings),
        str(settings),
        repr(settings.model_dump()),
        settings.model_dump_json(),
    )

    assert all(_TOKEN not in representation for representation in representations)
    assert _settings(api_token=None).api_token is None
    with pytest.raises(ValidationError) as raised:
        VLLMOpenAICompatibleSettings.model_validate(
            {
                **_settings().model_dump(),
                "api_token": _TOKEN,
                "temperature": 3,
            }
        )
    assert _TOKEN not in str(raised.value)
    for invalid in ("", " ", " token", "token "):
        with pytest.raises(ValidationError):
            _settings(api_token=invalid)


def test_constructor_is_side_effect_free_retains_identity_and_is_structural() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"text": "unused"}]})

    settings = _settings()
    transport = httpx.MockTransport(handler)
    gateway = VLLMOpenAICompatibleGateway(settings, transport=transport)
    structural: LLMGateway = gateway

    assert structural is gateway
    assert gateway._settings is settings  # noqa: SLF001
    assert gateway._transport is transport  # noqa: SLF001
    assert calls == 0


@pytest.mark.parametrize(
    ("settings", "transport"),
    [
        (object(), None),
        (_settings(), object()),
    ],
)
def test_constructor_rejects_invalid_collaborators(
    settings: object,
    transport: object | None,
) -> None:
    with pytest.raises(ValueError):
        VLLMOpenAICompatibleGateway(
            cast(VLLMOpenAICompatibleSettings, settings),
            transport=cast(httpx.BaseTransport | None, transport),
        )


def test_exact_http_request_scope_and_response_mapping() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"text": "  Synthetic generated çözüm.  "}]},
        )

    response = _gateway(handler).generate(_request())

    assert response == LLMResponse(
        tenant_id="tenant-synthetic",
        call_id="call-synthetic",
        text="Synthetic generated çözüm.",
    )
    assert len(requests) == 1
    sent = requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == "https://vllm.invalid/v1/completions"
    assert sent.url.query == b""
    assert json.loads(sent.content) == {
        "model": "synthetic/model-v1",
        "prompt": "Synthetic Unicode prompt: çözüm.",
        "max_tokens": 512,
        "temperature": 0.25,
        "stream": False,
    }
    assert sent.headers["accept"] == "application/json"
    assert sent.headers["content-type"] == "application/json"
    assert sent.headers["authorization"] == f"Bearer {_TOKEN}"
    serialized = b"\n".join(
        (
            str(sent.url).encode(),
            sent.content,
            repr(dict(sent.headers)).encode(),
        )
    )
    assert b"tenant-synthetic" not in serialized
    assert b"call-synthetic" not in serialized
    assert b"knowledge_base" not in serialized


def test_authorization_header_is_absent_without_token() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"text": "Synthetic"}]})

    _gateway(handler, settings=_settings(api_token=None)).generate(_request())

    assert "authorization" not in requests[0].headers


def test_timeout_tls_and_transport_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    client_kwargs: list[dict[str, object]] = []
    original_client = httpx.Client
    transport = httpx.MockTransport(
        lambda request: (
            requests.append(request)
            or httpx.Response(200, json={"choices": [{"text": "Synthetic"}]})
        )
    )

    def client_factory(**kwargs: object) -> httpx.Client:
        client_kwargs.append(kwargs)
        return cast(Any, original_client)(**kwargs)

    monkeypatch.setattr(provider.httpx, "Client", client_factory)
    settings = _settings(connect_timeout_seconds=3.5, read_timeout_seconds=22)

    VLLMOpenAICompatibleGateway(settings, transport=transport).generate(_request())

    assert len(client_kwargs) == 1
    assert client_kwargs[0]["verify"] is True
    assert client_kwargs[0]["transport"] is transport
    timeout = cast(httpx.Timeout, client_kwargs[0]["timeout"])
    assert timeout.connect == 3.5
    assert timeout.pool == 3.5
    assert timeout.read == 22.0
    assert timeout.write == 22.0
    assert requests[0].extensions["timeout"] == {
        "connect": 3.5,
        "read": 22.0,
        "write": 22.0,
        "pool": 3.5,
    }


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"choices": None},
        {"choices": []},
        {"choices": [{"text": "first"}, {"text": "second"}]},
        {"choices": ["not-an-object"]},
        {"choices": [{}]},
        {"choices": [{"text": None}]},
        {"choices": [{"text": "   "}]},
    ],
)
def test_malformed_response_semantics_use_fixed_safe_error(payload: object) -> None:
    gateway = _gateway(
        lambda _request: httpx.Response(
            200,
            json=payload,
        )
    )

    with pytest.raises(ValueError) as raised:
        gateway.generate(_request())

    assert str(raised.value) == "vLLM response is invalid"
    for sensitive in (
        _TOKEN,
        "synthetic/model-v1",
        "vllm.invalid",
        "tenant-synthetic",
        "call-synthetic",
        "Synthetic Unicode prompt",
    ):
        assert sensitive not in str(raised.value)


def test_malformed_json_uses_fixed_safe_error() -> None:
    gateway = _gateway(
        lambda _request: httpx.Response(
            200,
            content=b"{",
            headers={"content-type": "application/json"},
        )
    )

    with pytest.raises(ValueError) as raised:
        gateway.generate(_request())

    assert str(raised.value) == "vLLM response is invalid"


@pytest.mark.parametrize("exception_type", [httpx.ReadTimeout, httpx.ConnectError])
def test_transport_exception_identity_propagates(
    exception_type: type[httpx.HTTPError],
) -> None:
    expected = exception_type("synthetic transport failure")

    def handler(_request: httpx.Request) -> httpx.Response:
        raise expected

    with pytest.raises(exception_type) as raised:
        _gateway(handler).generate(_request())

    assert raised.value is expected


def test_non_success_status_raises_native_http_status_error() -> None:
    request_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_seen.append(request)
        return httpx.Response(503, json={"detail": "synthetic unavailable"})

    with pytest.raises(httpx.HTTPStatusError) as raised:
        _gateway(handler).generate(_request())

    assert raised.value.request is request_seen[0]
    assert raised.value.response.status_code == 503


def test_client_construction_exception_identity_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = RuntimeError("synthetic client construction failure")

    def fail_client(**_kwargs: object) -> httpx.Client:
        raise expected

    monkeypatch.setattr(provider.httpx, "Client", fail_client)

    with pytest.raises(RuntimeError) as raised:
        _gateway(lambda _request: httpx.Response(200)).generate(_request())

    assert raised.value is expected


def test_short_lived_client_closes_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={"choices": [{"text": "Synthetic"}]},
        )
    )
    close_calls = 0
    original_close = transport.close

    def close() -> None:
        nonlocal close_calls
        close_calls += 1
        original_close()

    monkeypatch.setattr(transport, "close", close)

    VLLMOpenAICompatibleGateway(_settings(), transport=transport).generate(_request())

    assert close_calls == 1


def test_repeated_requests_are_deterministic() -> None:
    gateway = _gateway(
        lambda _request: httpx.Response(
            200,
            json={"choices": [{"text": "Synthetic generated response."}]},
        )
    )
    request = _request()

    assert gateway.generate(request) == gateway.generate(request)


@dataclass
class CountingFactory:
    gateway: VLLMOpenAICompatibleGateway
    calls: int = 0

    def __call__(self) -> VLLMOpenAICompatibleGateway:
        self.calls += 1
        return self.gateway


def test_pr46_a_deferred_factory_creates_gateway_once() -> None:
    gateway = _gateway(
        lambda _request: httpx.Response(
            200,
            json={"choices": [{"text": "Synthetic generated response."}]},
        )
    )
    factory: LLMGatewayFactory = CountingFactory(gateway)
    deferred = _DeferredLLMGateway(factory)

    first = deferred.generate(_request())
    second = deferred.generate(_request())

    assert first == second
    assert cast(CountingFactory, factory).calls == 1


def test_app_llm_exports_are_exact() -> None:
    assert llm_exports.__all__ == [
        "InMemoryLLMGateway",
        "LLMGateway",
        "LLMRequest",
        "LLMResponse",
        "VLLMOpenAICompatibleGateway",
        "VLLMOpenAICompatibleSettings",
    ]
    assert llm_exports.VLLMOpenAICompatibleGateway is VLLMOpenAICompatibleGateway
    assert llm_exports.VLLMOpenAICompatibleSettings is VLLMOpenAICompatibleSettings
