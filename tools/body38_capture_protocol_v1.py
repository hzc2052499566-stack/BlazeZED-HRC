"""Control protocol for the BODY_38 vs BlazePose paired capture.

Isaac steps the timeline; the host consumer owns the ZED stream. The two have
to agree on which grabs belong to which time code, and the ZED network streamer
publishes continuously, so the agreement is a synchronous handshake: Isaac never
advances until the consumer has banked frames for the current time code.

This is a control channel only - no pixels cross it. Isaac writes ground truth
to disk and the consumer reads the imagery from the ZED stream, so the wire
carries nothing but time codes and counts. That is why this is a new module
rather than a reuse of view_ac_stream_protocol_v1.py, which exists to move
RGB-D arrays and stamps every packet with the View A/C experiment's version
string.

Message sequence:

    producer                         consumer
       |                                | listen, print LISTENING
       |------------- hello ----------->|
       |<------------ ready ------------| stream open, BODY_38 on, recording on
       |                                |
       |--- stepped (time_code=k) ----->| discard D grabs, keep M grabs
       |<--- banked (time_code=k) ------| rows written for k
       |            ... repeat ...      |
       |------------- finish ---------->|
       |<------------ closed -----------| recording off, state written

Every message carries the protocol version and is rejected on mismatch, so a
stale Isaac script cannot silently drive a new consumer.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any


PROTOCOL_VERSION = "body38_gt_capture_v1_20260808"
MAX_MESSAGE_BYTES = 1_048_576
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 58433

MESSAGE_TYPES = frozenset(
    {
        "hello",
        "ready",
        # stepped/banked belong to the pause-and-seek dwell design, which the
        # 2026-08-08 three-state stream probe ruled out: the ZED streamer stops
        # publishing the moment the timeline is not playing. They are kept so
        # the rejected design's producer still speaks a valid protocol.
        "stepped",
        "banked",
        # beacon drives the surviving design: the timeline plays continuously
        # and the producer announces each integer time code change. It is
        # advisory only - it narrows the NCC search window and cross-checks the
        # result, it never decides which time code a frame belongs to.
        "beacon",
        "finish",
        "closed",
        "abort",
    }
)


def recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        block = connection.recv(size - len(chunks))
        if not block:
            raise ConnectionError("socket closed while receiving a message")
        chunks.extend(block)
    return bytes(chunks)


def send_message(connection: socket.socket, message: dict[str, Any]) -> int:
    payload = dict(message)
    message_type = payload.get("type")
    if message_type not in MESSAGE_TYPES:
        raise ValueError(f"unknown message type: {message_type!r}")
    payload["protocol_version"] = PROTOCOL_VERSION
    encoded = json.dumps(
        payload, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ValueError("control message is too large")
    connection.sendall(struct.pack("!I", len(encoded)))
    connection.sendall(encoded)
    return 4 + len(encoded)


def recv_message(connection: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", recv_exact(connection, 4))[0]
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid control message size: {size}")
    message = json.loads(recv_exact(connection, size).decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("control message is not an object")
    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            "control protocol version mismatch: expected "
            f"{PROTOCOL_VERSION!r}, observed "
            f"{message.get('protocol_version')!r}"
        )
    if message.get("type") not in MESSAGE_TYPES:
        raise ValueError(f"unknown message type: {message.get('type')!r}")
    return message


class MessageReader:
    """Incremental reader for a non-blocking socket.

    The producer cannot use a blocking recv. Isaac only publishes ZED frames
    while the app ticks, so if the producer blocked on the reply the consumer
    would sit in grab() waiting for frames that Isaac is not producing - a
    genuine deadlock, not a slow path. Instead the producer pumps app updates
    and feeds whatever bytes have arrived into this reader until a message
    completes.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> None:
        self._buffer.extend(data)

    def next_message(self) -> dict[str, Any] | None:
        """Return one complete message, or None if more bytes are needed."""
        if len(self._buffer) < 4:
            return None
        size = struct.unpack("!I", bytes(self._buffer[:4]))[0]
        if not 0 < size <= MAX_MESSAGE_BYTES:
            raise ValueError(f"invalid control message size: {size}")
        if len(self._buffer) < 4 + size:
            return None
        encoded = bytes(self._buffer[4 : 4 + size])
        del self._buffer[: 4 + size]
        message = json.loads(encoded.decode("utf-8"))
        if not isinstance(message, dict):
            raise ValueError("control message is not an object")
        if message.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(
                "control protocol version mismatch: expected "
                f"{PROTOCOL_VERSION!r}, observed "
                f"{message.get('protocol_version')!r}"
            )
        if message.get("type") not in MESSAGE_TYPES:
            raise ValueError(f"unknown message type: {message.get('type')!r}")
        return message

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)


def require_message(message: dict[str, Any], expected_type: str) -> dict[str, Any]:
    if message.get("type") == "abort":
        raise RuntimeError(
            "peer aborted the capture: {}".format(message.get("reason"))
        )
    if message.get("type") != expected_type:
        raise RuntimeError(
            f"expected {expected_type!r}, observed {message.get('type')!r}"
        )
    return message
