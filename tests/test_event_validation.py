from types import SimpleNamespace

import pytest

from app.events.validation import ensure_same_call, ensure_same_tenant


def test_matching_tenant_and_call_values_are_returned() -> None:
    first = SimpleNamespace(tenant_id="tenant_alpha", call_id="call_001")
    second = SimpleNamespace(tenant_id="tenant_alpha", call_id="call_001")

    assert ensure_same_tenant(first, second) == "tenant_alpha"
    assert ensure_same_call(first, second) == "call_001"


def test_tenant_and_call_mismatches_are_rejected() -> None:
    first = SimpleNamespace(tenant_id="tenant_alpha", call_id="call_001")

    with pytest.raises(ValueError, match="Mismatched tenant_id"):
        ensure_same_tenant(first, SimpleNamespace(tenant_id="tenant_beta"))
    with pytest.raises(ValueError, match="Mismatched call_id"):
        ensure_same_call(first, SimpleNamespace(call_id="call_002"))


def test_missing_attribute_and_empty_input_are_rejected() -> None:
    with pytest.raises(ValueError, match="lacks required tenant_id"):
        ensure_same_tenant(object())
    with pytest.raises(ValueError, match="At least one"):
        ensure_same_call()
