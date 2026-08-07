from collections.abc import Callable
from datetime import UTC, datetime
import json

import pytest

import app.audio_ingress.tcp_microphone_relay as subject
from app.audio_ingress import (
    BoundedMicrophoneRelayProtocol,
    MicrophoneRelayProtocolError,
    MicrophoneRelayRecordParser,
    RelayAcceptanceStatus,
    RelayMessageType,
    RelayReason,
    RelaySessionState,
    encode_relay_record,
    relay_tokens_match,
)


TOKEN = "synthetic-ephemeral-token-0001"
OTHER_TOKEN = "synthetic-ephemeral-token-0002"
TENANT_ID = "tenant_alpha"
CALL_ID = "call_001"
STREAM_ID = "stream_001"
CAPTURED_AT = datetime(2026, 8, 7, 9, 0, tzinfo=UTC).isoformat()


def start_metadata(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "token": TOKEN,
        "tenant_id": TENANT_ID,
        "call_id": CALL_ID,
        "stream_id": STREAM_ID,
        "sequence_number": 0,
        "generation": 1,
        "codec_name": "pcm_s16le",
        "sample_rate_hz": 16_000,
        "channel_count": 1,
    }
    values.update(changes)
    return values


def audio_metadata(
    sequence: int,
    *,
    generation: int = 1,
    payload: bytes = b"\0\0",
    **changes: object,
) -> dict[str, object]:
    values: dict[str, object] = {
        "sequence_number": sequence,
        "generation": generation,
        "sample_count": len(payload) // 2,
        "captured_at_utc": CAPTURED_AT,
    }
    values.update(changes)
    return values


def control_metadata(sequence: int, generation: int = 1) -> dict[str, object]:
    return {"sequence_number": sequence, "generation": generation}


def end_metadata(
    sequence: int,
    generation: int = 1,
) -> dict[str, object]:
    return {
        "sequence_number": sequence,
        "generation": generation,
        "end_reason": "completed",
    }


def protocol(**changes: object) -> BoundedMicrophoneRelayProtocol:
    values: dict[str, object] = {
        "expected_token": TOKEN,
        "tenant_id": TENANT_ID,
        "call_id": CALL_ID,
        "stream_id": STREAM_ID,
    }
    values.update(changes)
    return BoundedMicrophoneRelayProtocol(**values)  # type: ignore[arg-type]


def encode(
    message_type: RelayMessageType,
    metadata: dict[str, object],
    payload: bytes = b"",
) -> bytes:
    return encode_relay_record(message_type, metadata, payload)


def raw_record(
    *,
    message_type: int,
    metadata_bytes: bytes,
    payload: bytes = b"",
    magic: bytes = subject.RELAY_MAGIC,
    version: int = subject.RELAY_VERSION,
    declared_metadata_length: int | None = None,
    declared_payload_length: int | None = None,
) -> bytes:
    return (
        subject.RELAY_HEADER.pack(
            magic,
            version,
            message_type,
            (
                len(metadata_bytes)
                if declared_metadata_length is None
                else declared_metadata_length
            ),
            len(payload)
            if declared_payload_length is None
            else declared_payload_length,
        )
        + metadata_bytes
        + payload
    )


def error_reason(
    raw: bytes,
    *,
    parser: MicrophoneRelayRecordParser | None = None,
) -> RelayReason:
    selected = parser or MicrophoneRelayRecordParser()
    with pytest.raises(MicrophoneRelayProtocolError) as raised:
        selected.feed(raw)
    return raised.value.reason


def test_fragmented_start_is_emitted_only_after_complete_record() -> None:
    selected = protocol()
    encoded = encode(RelayMessageType.START, start_metadata())
    acceptances = ()

    for byte in encoded:
        current = selected.feed(bytes((byte,)))
        if current:
            acceptances = current

    assert len(acceptances) == 1
    assert acceptances[0].reason is RelayReason.STARTED
    assert selected.state is RelaySessionState.STREAMING
    assert selected.buffered_bytes == 0


def test_coalesced_records_drive_complete_lifecycle() -> None:
    selected = protocol()
    payload = b"\x01\x00" * 8
    raw = b"".join(
        (
            encode(RelayMessageType.START, start_metadata()),
            encode(
                RelayMessageType.AUDIO,
                audio_metadata(1, payload=payload),
                payload,
            ),
            encode(RelayMessageType.END, end_metadata(2)),
        )
    )

    results = selected.feed(raw)

    assert [result.reason for result in results] == [
        RelayReason.STARTED,
        RelayReason.AUDIO_ACCEPTED,
        RelayReason.ENDED,
    ]
    assert selected.state is RelaySessionState.ENDED
    assert selected.buffered_bytes == 0


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (
            raw_record(
                message_type=RelayMessageType.START.value,
                metadata_bytes=b"{}",
                magic=b"FAIL",
            ),
            RelayReason.BAD_MAGIC,
        ),
        (
            raw_record(
                message_type=RelayMessageType.START.value,
                metadata_bytes=b"{}",
                version=subject.RELAY_VERSION + 1,
            ),
            RelayReason.UNSUPPORTED_VERSION,
        ),
        (
            raw_record(message_type=255, metadata_bytes=b"{}"),
            RelayReason.UNKNOWN_MESSAGE_TYPE,
        ),
    ],
)
def test_bad_header_values_fail_closed(raw: bytes, reason: RelayReason) -> None:
    parser = MicrophoneRelayRecordParser()

    assert error_reason(raw, parser=parser) is reason
    assert parser.failed
    assert parser.buffered_bytes == 0
    assert error_reason(b"", parser=parser) is RelayReason.TERMINAL_STATE


@pytest.mark.parametrize(
    ("message_type", "metadata_length", "payload_length", "reason"),
    [
        (
            RelayMessageType.START,
            subject.MAX_RELAY_METADATA_BYTES + 1,
            0,
            RelayReason.METADATA_TOO_LARGE,
        ),
        (
            RelayMessageType.AUDIO,
            1,
            subject.MAX_RELAY_AUDIO_PAYLOAD_BYTES + 1,
            RelayReason.PAYLOAD_TOO_LARGE,
        ),
    ],
)
def test_oversized_prefix_is_rejected_before_body_buffering(
    message_type: RelayMessageType,
    metadata_length: int,
    payload_length: int,
    reason: RelayReason,
) -> None:
    parser = MicrophoneRelayRecordParser()
    header = subject.RELAY_HEADER.pack(
        subject.RELAY_MAGIC,
        subject.RELAY_VERSION,
        message_type.value,
        metadata_length,
        payload_length,
    )

    assert error_reason(header, parser=parser) is reason
    assert parser.buffered_bytes == 0


@pytest.mark.parametrize(
    ("metadata_bytes", "reason"),
    [
        (b"{", RelayReason.MALFORMED_JSON),
        (b"\xff", RelayReason.INVALID_UTF8),
        (
            (
                b'{"token":"synthetic-ephemeral-token-0001",'
                b'"token":"synthetic-ephemeral-token-0002"}'
            ),
            RelayReason.DUPLICATE_JSON_KEY,
        ),
        (b'{"sequence_number":NaN}', RelayReason.NON_FINITE_NUMBER),
        (b'{"sequence_number":Infinity}', RelayReason.NON_FINITE_NUMBER),
    ],
)
def test_invalid_json_is_sanitized(
    metadata_bytes: bytes,
    reason: RelayReason,
) -> None:
    raw = raw_record(
        message_type=RelayMessageType.START.value,
        metadata_bytes=metadata_bytes,
    )

    assert error_reason(raw) is reason


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"unexpected": True}, RelayReason.UNKNOWN_METADATA_FIELD),
        ({"token": None}, RelayReason.INVALID_METADATA),
    ],
)
def test_unknown_and_invalid_start_fields_fail(
    changes: dict[str, object],
    reason: RelayReason,
) -> None:
    metadata = start_metadata()
    if "unexpected" in changes:
        metadata.update(changes)
    else:
        metadata.update(changes)

    with pytest.raises(MicrophoneRelayProtocolError) as raised:
        encode(RelayMessageType.START, metadata)

    assert raised.value.reason is reason


def test_missing_start_field_fails_with_fixed_reason() -> None:
    metadata = start_metadata()
    del metadata["stream_id"]

    with pytest.raises(MicrophoneRelayProtocolError) as raised:
        encode(RelayMessageType.START, metadata)

    assert raised.value.reason is RelayReason.MISSING_METADATA_FIELD


@pytest.mark.parametrize(
    "changes",
    [
        {"codec_name": "float32le"},
        {"sample_rate_hz": 8_000},
        {"channel_count": 2},
    ],
)
def test_start_requires_exact_pcm_format(changes: dict[str, object]) -> None:
    with pytest.raises(MicrophoneRelayProtocolError) as raised:
        encode(RelayMessageType.START, start_metadata(**changes))

    assert raised.value.reason is RelayReason.INVALID_FORMAT


@pytest.mark.parametrize(
    ("payload", "changes", "reason"),
    [
        (b"", {}, RelayReason.AUDIO_PAYLOAD_REQUIRED),
        (b"\0", {}, RelayReason.ODD_AUDIO_PAYLOAD),
        (b"\0\0", {"sample_count": 2}, RelayReason.INVALID_SAMPLE_COUNT),
    ],
)
def test_audio_payload_and_sample_count_are_exact(
    payload: bytes,
    changes: dict[str, object],
    reason: RelayReason,
) -> None:
    metadata = audio_metadata(1, payload=payload)
    metadata.update(changes)
    with pytest.raises(MicrophoneRelayProtocolError) as raised:
        encode(
            RelayMessageType.AUDIO,
            metadata,
            payload,
        )

    assert raised.value.reason is reason


def test_maximum_audio_payload_is_accepted() -> None:
    payload = b"\0\0" * (subject.MAX_RELAY_AUDIO_PAYLOAD_BYTES // 2)

    encoded = encode(
        RelayMessageType.AUDIO,
        audio_metadata(1, payload=payload),
        payload,
    )
    records = MicrophoneRelayRecordParser().feed(encoded)

    assert len(records) == 1
    assert len(records[0].payload) == subject.MAX_RELAY_AUDIO_PAYLOAD_BYTES


def test_control_messages_reject_payloads() -> None:
    with pytest.raises(MicrophoneRelayProtocolError) as raised:
        encode(
            RelayMessageType.PAUSE,
            control_metadata(1),
            b"\0\0",
        )

    assert raised.value.reason is RelayReason.UNEXPECTED_PAYLOAD


def test_ack_and_error_records_are_parseable_but_not_client_state_inputs() -> None:
    parser = MicrophoneRelayRecordParser()
    records = parser.feed(
        encode(
            RelayMessageType.ACK,
            {"sequence_number": 1, "reason": RelayReason.AUDIO_ACCEPTED.value},
        )
        + encode(
            RelayMessageType.ERROR,
            {"sequence_number": 1, "reason": RelayReason.INVALID_FORMAT.value},
        )
    )

    assert [record.message_type for record in records] == [
        RelayMessageType.ACK,
        RelayMessageType.ERROR,
    ]

    selected = protocol()
    selected.feed(encode(RelayMessageType.START, start_metadata()))
    result = selected.feed(
        encode(
            RelayMessageType.ACK,
            {"sequence_number": 1, "reason": RelayReason.AUDIO_ACCEPTED.value},
        )
    )
    assert result[0].reason is RelayReason.UNEXPECTED_MESSAGE_ORDER
    assert selected.state is RelaySessionState.FAILED


def test_start_audio_pause_resume_audio_end_lifecycle() -> None:
    selected = protocol()
    selected.feed(encode(RelayMessageType.START, start_metadata()))
    selected.feed(
        encode(
            RelayMessageType.AUDIO,
            audio_metadata(1),
            b"\0\0",
        )
    )
    paused = selected.feed(encode(RelayMessageType.PAUSE, control_metadata(2)))
    resumed = selected.feed(
        encode(RelayMessageType.RESUME, control_metadata(3, generation=2))
    )
    selected.feed(
        encode(
            RelayMessageType.AUDIO,
            audio_metadata(4, generation=2),
            b"\0\0",
        )
    )
    ended = selected.feed(encode(RelayMessageType.END, end_metadata(5, generation=2)))

    assert paused[0].state is RelaySessionState.PAUSED
    assert resumed[0].state is RelaySessionState.STREAMING
    assert resumed[0].reason is RelayReason.RESUMED
    assert ended[0].state is RelaySessionState.ENDED


def test_end_is_accepted_from_paused_state() -> None:
    selected = protocol()
    selected.feed(encode(RelayMessageType.START, start_metadata()))
    selected.feed(encode(RelayMessageType.PAUSE, control_metadata(1)))

    result = selected.feed(encode(RelayMessageType.END, end_metadata(2)))

    assert result[0].reason is RelayReason.ENDED
    assert selected.state is RelaySessionState.ENDED


@pytest.mark.parametrize(
    "record",
    [
        lambda: encode(
            RelayMessageType.AUDIO,
            audio_metadata(0),
            b"\0\0",
        ),
        lambda: encode(RelayMessageType.PAUSE, control_metadata(0)),
        lambda: encode(RelayMessageType.RESUME, control_metadata(0, generation=2)),
        lambda: encode(RelayMessageType.END, end_metadata(0)),
    ],
)
def test_start_is_required_for_all_lifecycle_messages(
    record: Callable[[], bytes],
) -> None:
    selected = protocol()

    result = selected.feed(record())

    assert result[0].reason is RelayReason.UNEXPECTED_MESSAGE_ORDER
    assert selected.state is RelaySessionState.FAILED


def test_audio_is_rejected_while_paused() -> None:
    selected = protocol()
    selected.feed(encode(RelayMessageType.START, start_metadata()))
    selected.feed(encode(RelayMessageType.PAUSE, control_metadata(1)))

    result = selected.feed(
        encode(
            RelayMessageType.AUDIO,
            audio_metadata(2),
            b"\0\0",
        )
    )

    assert result[0].reason is RelayReason.UNEXPECTED_MESSAGE_ORDER
    assert selected.state is RelaySessionState.FAILED


def test_resume_requires_exact_next_generation() -> None:
    selected = protocol()
    selected.feed(encode(RelayMessageType.START, start_metadata()))
    selected.feed(encode(RelayMessageType.PAUSE, control_metadata(1)))

    result = selected.feed(
        encode(RelayMessageType.RESUME, control_metadata(2, generation=3))
    )

    assert result[0].reason is RelayReason.GENERATION_MISMATCH
    assert selected.state is RelaySessionState.FAILED


def test_stale_generation_fails_closed_after_resume() -> None:
    selected = protocol()
    selected.feed(encode(RelayMessageType.START, start_metadata()))
    selected.feed(encode(RelayMessageType.PAUSE, control_metadata(1)))
    selected.feed(encode(RelayMessageType.RESUME, control_metadata(2, generation=2)))

    result = selected.feed(
        encode(
            RelayMessageType.AUDIO,
            audio_metadata(3, generation=1),
            b"\0\0",
        )
    )

    assert result[0].reason is RelayReason.STALE_GENERATION
    assert selected.state is RelaySessionState.FAILED


def test_sequence_gap_fails_closed() -> None:
    selected = protocol()
    selected.feed(encode(RelayMessageType.START, start_metadata()))

    result = selected.feed(
        encode(
            RelayMessageType.AUDIO,
            audio_metadata(2),
            b"\0\0",
        )
    )

    assert result[0].reason is RelayReason.SEQUENCE_GAP
    assert selected.state is RelaySessionState.FAILED


def test_identical_recent_duplicate_is_idempotent() -> None:
    selected = protocol()
    selected.feed(encode(RelayMessageType.START, start_metadata()))
    frame = encode(
        RelayMessageType.AUDIO,
        audio_metadata(1),
        b"\0\0",
    )
    selected.feed(frame)

    result = selected.feed(frame)

    assert result[0].status is RelayAcceptanceStatus.DUPLICATE
    assert result[0].reason is RelayReason.EXACT_DUPLICATE
    assert selected.state is RelaySessionState.STREAMING
    assert selected.duplicate_history_size == 2


def test_conflicting_duplicate_fails_closed() -> None:
    selected = protocol()
    selected.feed(encode(RelayMessageType.START, start_metadata()))
    selected.feed(
        encode(
            RelayMessageType.AUDIO,
            audio_metadata(1),
            b"\0\0",
        )
    )

    result = selected.feed(
        encode(
            RelayMessageType.AUDIO,
            audio_metadata(1),
            b"\1\0",
        )
    )

    assert result[0].reason is RelayReason.CONFLICTING_DUPLICATE
    assert selected.state is RelaySessionState.FAILED


def test_duplicate_history_is_bounded_and_evicted_sequences_are_stale() -> None:
    selected = protocol()
    selected.feed(encode(RelayMessageType.START, start_metadata()))
    first = encode(
        RelayMessageType.AUDIO,
        audio_metadata(1),
        b"\0\0",
    )
    selected.feed(first)
    for sequence in range(2, 70):
        selected.feed(
            encode(
                RelayMessageType.AUDIO,
                audio_metadata(sequence),
                bytes((sequence % 251, 0)),
            )
        )

    assert selected.duplicate_history_size == subject.MAX_RELAY_DUPLICATE_HISTORY
    result = selected.feed(first)
    assert result[0].reason is RelayReason.STALE_SEQUENCE
    assert selected.state is RelaySessionState.FAILED


def test_no_messages_are_accepted_after_end() -> None:
    selected = protocol()
    selected.feed(encode(RelayMessageType.START, start_metadata()))
    selected.feed(encode(RelayMessageType.END, end_metadata(1)))

    result = selected.feed(
        encode(
            RelayMessageType.AUDIO,
            audio_metadata(2),
            b"\0\0",
        )
    )

    assert result[0].status is RelayAcceptanceStatus.REJECTED
    assert result[0].reason is RelayReason.TERMINAL_STATE
    assert selected.state is RelaySessionState.ENDED


def test_failed_protocol_remains_terminal() -> None:
    selected = protocol()
    first = selected.feed(
        encode(
            RelayMessageType.AUDIO,
            audio_metadata(0),
            b"\0\0",
        )
    )
    second = selected.feed(encode(RelayMessageType.START, start_metadata()))

    assert first[0].state is RelaySessionState.FAILED
    assert second[0].reason is RelayReason.TERMINAL_STATE
    assert selected.state is RelaySessionState.FAILED


def test_authentication_and_scope_fail_closed() -> None:
    wrong_token = protocol()
    token_result = wrong_token.feed(
        encode(
            RelayMessageType.START,
            start_metadata(token=OTHER_TOKEN),
        )
    )
    wrong_scope = protocol()
    scope_result = wrong_scope.feed(
        encode(
            RelayMessageType.START,
            start_metadata(call_id="other_call"),
        )
    )

    assert token_result[0].reason is RelayReason.AUTHENTICATION_FAILED
    assert scope_result[0].reason is RelayReason.SCOPE_MISMATCH
    assert wrong_token.state is wrong_scope.state is RelaySessionState.FAILED


@pytest.mark.parametrize(
    "changes",
    [
        {"tenant_id": ""},
        {"call_id": "x" * (subject.MAX_RELAY_IDENTIFIER_LENGTH + 1)},
        {"stream_id": " "},
        {"token": "short"},
    ],
)
def test_start_scope_and_token_are_nonempty_and_bounded(
    changes: dict[str, object],
) -> None:
    with pytest.raises(MicrophoneRelayProtocolError) as raised:
        encode(RelayMessageType.START, start_metadata(**changes))

    assert raised.value.reason in {
        RelayReason.INVALID_IDENTIFIER,
        RelayReason.INVALID_TOKEN,
        RelayReason.INVALID_METADATA,
    }


def test_token_helper_uses_constant_time_comparison_without_repr_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def compare(left: bytes, right: bytes) -> bool:
        calls.append((len(left), len(right)))
        return left == right

    monkeypatch.setattr(subject.hmac, "compare_digest", compare)
    assert relay_tokens_match(TOKEN, TOKEN)
    assert not relay_tokens_match(TOKEN, OTHER_TOKEN)
    assert calls == [(len(TOKEN), len(TOKEN)), (len(TOKEN), len(OTHER_TOKEN))]

    parser = MicrophoneRelayRecordParser()
    record = parser.feed(encode(RelayMessageType.START, start_metadata()))[0]
    assert TOKEN not in repr(record)
    assert TOKEN not in repr(record.metadata)
    error = MicrophoneRelayProtocolError(RelayReason.AUTHENTICATION_FAILED)
    assert TOKEN not in str(error)
    assert TOKEN not in repr(error)


def test_parser_buffer_never_exceeds_one_maximum_record() -> None:
    payload = b"\0\0" * (subject.MAX_RELAY_AUDIO_PAYLOAD_BYTES // 2)
    encoded = encode(
        RelayMessageType.AUDIO,
        audio_metadata(1, payload=payload),
        payload,
    )
    parser = MicrophoneRelayRecordParser()
    maximum_observed = 0
    records = ()
    for offset in range(0, len(encoded), 127):
        current = parser.feed(encoded[offset : offset + 127])
        maximum_observed = max(maximum_observed, parser.buffered_bytes)
        if current:
            records = current

    assert len(records) == 1
    assert maximum_observed <= subject.MAX_RELAY_RECORD_BYTES
    assert parser.buffered_bytes == 0


def test_non_object_json_and_nested_nonfinite_values_fail_safely() -> None:
    assert (
        error_reason(
            raw_record(
                message_type=RelayMessageType.START.value,
                metadata_bytes=b"[]",
            )
        )
        is RelayReason.INVALID_METADATA
    )
    with pytest.raises(MicrophoneRelayProtocolError) as raised:
        encode(
            RelayMessageType.START,
            {**start_metadata(), "sample_rate_hz": float("inf")},
        )
    assert raised.value.reason is RelayReason.NON_FINITE_NUMBER


def test_metadata_encoder_is_deterministic_and_bounded() -> None:
    metadata = start_metadata()
    first = encode(RelayMessageType.START, metadata)
    second = encode(
        RelayMessageType.START,
        dict(reversed(tuple(metadata.items()))),
    )

    assert first == second
    _magic, _version, _kind, metadata_length, payload_length = (
        subject.RELAY_HEADER.unpack(first[: subject.RELAY_HEADER_BYTES])
    )
    assert metadata_length <= subject.MAX_RELAY_METADATA_BYTES
    assert payload_length == 0
    decoded = json.loads(
        first[subject.RELAY_HEADER_BYTES : subject.RELAY_HEADER_BYTES + metadata_length]
    )
    assert set(decoded) == set(metadata)
