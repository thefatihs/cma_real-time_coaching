"""Immutable trusted-text document source."""

from pydantic import BaseModel, ConfigDict, field_validator

from app.vector_store.models import Metadata


class TextDocumentSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    text: str
    metadata: Metadata = ()

    @field_validator("document_id", "text")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return _required_text(value, getattr(info, "field_name", "value"))

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: Metadata) -> Metadata:
        result: list[tuple[str, str]] = []
        keys: set[str] = set()
        for key, item in value:
            clean_key = _required_text(key, "metadata key")
            clean_item = _required_text(item, f"metadata value for {clean_key}")
            if clean_key in keys:
                raise ValueError("metadata keys must be unique")
            keys.add(clean_key)
            result.append((clean_key, clean_item))
        return tuple(result)


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
