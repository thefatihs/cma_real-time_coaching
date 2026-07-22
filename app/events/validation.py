def ensure_same_tenant(*events_or_contexts: object) -> str:
    return _ensure_same_identifier("tenant_id", events_or_contexts)


def ensure_same_call(*events: object) -> str:
    return _ensure_same_identifier("call_id", events)


def _ensure_same_identifier(attribute: str, objects: tuple[object, ...]) -> str:
    if not objects:
        raise ValueError(f"At least one object is required to validate {attribute}")

    identifiers: list[str] = []
    for item in objects:
        if not hasattr(item, attribute):
            raise ValueError(f"Object lacks required {attribute} attribute")
        value = getattr(item, attribute)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Object has an invalid {attribute}")
        identifiers.append(value)

    if len(set(identifiers)) != 1:
        raise ValueError(f"Mismatched {attribute} values")
    return identifiers[0]
