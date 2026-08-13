"""Length-prefixed localhost protocol for conditional View A/C streaming."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

import numpy as np


PROTOCOL_VERSION = "view_ac_conditional_stream_v1_20260802"
MAX_HEADER_BYTES = 1_048_576
MAX_ARRAY_BYTES = 128 * 1024 * 1024
ALLOWED_DTYPES = {"uint8", "float32"}


def recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        block = connection.recv(size - len(chunks))
        if not block:
            raise ConnectionError("Socket closed while receiving a packet.")
        chunks.extend(block)
    return bytes(chunks)


def canonical_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array, dtype=np.uint8)
    if array.dtype.kind == "f":
        return np.ascontiguousarray(array, dtype=np.float32)
    raise ValueError(f"Unsupported streaming array dtype: {array.dtype}")


def send_packet(
    connection: socket.socket,
    header: dict[str, Any],
    arrays: dict[str, Any] | None = None,
) -> int:
    payload = dict(header)
    if payload.get("protocol_version", PROTOCOL_VERSION) != PROTOCOL_VERSION:
        raise ValueError("Protocol version mismatch in outgoing packet.")
    payload["protocol_version"] = PROTOCOL_VERSION
    encoded_arrays = []
    descriptors = []
    for name, value in (arrays or {}).items():
        if not name or not isinstance(name, str):
            raise ValueError("Array names must be non-empty strings.")
        array = canonical_array(value)
        raw = memoryview(array).cast("B")
        if len(raw) > MAX_ARRAY_BYTES:
            raise ValueError(f"Array payload too large: {name}")
        descriptors.append({
            "name": name,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "nbytes": len(raw),
        })
        encoded_arrays.append(raw)
    payload["arrays"] = descriptors
    header_bytes = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ValueError("Streaming packet header is too large.")
    connection.sendall(struct.pack("!I", len(header_bytes)))
    connection.sendall(header_bytes)
    total = 4 + len(header_bytes)
    for raw in encoded_arrays:
        connection.sendall(raw)
        total += len(raw)
    return total


def recv_packet(connection: socket.socket) -> tuple[dict[str, Any], dict[str, np.ndarray], int]:
    header_size = struct.unpack("!I", recv_exact(connection, 4))[0]
    if not 0 < header_size <= MAX_HEADER_BYTES:
        raise ValueError(f"Invalid streaming header size: {header_size}")
    header_bytes = recv_exact(connection, header_size)
    header = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(header, dict) or header.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Incoming protocol version mismatch.")
    arrays = {}
    total = 4 + header_size
    descriptors = header.pop("arrays", [])
    if not isinstance(descriptors, list):
        raise ValueError("Array descriptor list is invalid.")
    for descriptor in descriptors:
        name = descriptor["name"]
        dtype = descriptor["dtype"]
        shape = tuple(int(value) for value in descriptor["shape"])
        nbytes = int(descriptor["nbytes"])
        if dtype not in ALLOWED_DTYPES or nbytes < 0 or nbytes > MAX_ARRAY_BYTES:
            raise ValueError(f"Invalid array descriptor: {descriptor!r}")
        expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if expected != nbytes or name in arrays:
            raise ValueError(f"Array descriptor size/name mismatch: {descriptor!r}")
        raw = recv_exact(connection, nbytes)
        arrays[name] = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()
        total += nbytes
    return header, arrays, total


def require_message(header: dict[str, Any], expected_type: str) -> None:
    if header.get("type") != expected_type:
        raise RuntimeError(
            f"Expected streaming message {expected_type!r}, observed {header.get('type')!r}."
        )
