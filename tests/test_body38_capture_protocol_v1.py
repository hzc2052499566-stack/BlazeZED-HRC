from __future__ import annotations

import json
import socket
import struct
import sys
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE / "tools") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "tools"))

import body38_capture_protocol_v1 as protocol


def encode(message: dict) -> bytes:
    encoded = json.dumps(message, separators=(",", ":")).encode("utf-8")
    return struct.pack("!I", len(encoded)) + encoded


class RoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.left, self.right = socket.socketpair()
        self.addCleanup(self.left.close)
        self.addCleanup(self.right.close)

    def test_round_trip_preserves_payload(self) -> None:
        protocol.send_message(
            self.left, {"type": "stepped", "time_code": 42, "step_index": 7}
        )
        message = protocol.recv_message(self.right)
        self.assertEqual(message["type"], "stepped")
        self.assertEqual(message["time_code"], 42)
        self.assertEqual(message["protocol_version"], protocol.PROTOCOL_VERSION)

    def test_send_rejects_unknown_type(self) -> None:
        with self.assertRaises(ValueError):
            protocol.send_message(self.left, {"type": "not_a_message"})

    def test_recv_rejects_version_mismatch(self) -> None:
        self.left.sendall(
            encode({"type": "hello", "protocol_version": "something_else"})
        )
        with self.assertRaises(ValueError) as caught:
            protocol.recv_message(self.right)
        self.assertIn("version mismatch", str(caught.exception))

    def test_recv_rejects_unknown_type(self) -> None:
        self.left.sendall(
            encode(
                {
                    "type": "surprise",
                    "protocol_version": protocol.PROTOCOL_VERSION,
                }
            )
        )
        with self.assertRaises(ValueError):
            protocol.recv_message(self.right)


class MessageReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reader = protocol.MessageReader()

    def valid(self, **fields) -> bytes:
        payload = {"protocol_version": protocol.PROTOCOL_VERSION, **fields}
        return encode(payload)

    def test_returns_none_until_the_header_is_complete(self) -> None:
        data = self.valid(type="banked", time_code=1)
        self.reader.feed(data[:2])
        self.assertIsNone(self.reader.next_message())

    def test_returns_none_until_the_body_is_complete(self) -> None:
        data = self.valid(type="banked", time_code=1)
        self.reader.feed(data[:-1])
        self.assertIsNone(self.reader.next_message())
        self.reader.feed(data[-1:])
        self.assertEqual(self.reader.next_message()["time_code"], 1)

    def test_reads_several_messages_from_one_chunk(self) -> None:
        self.reader.feed(
            self.valid(type="banked", time_code=1)
            + self.valid(type="banked", time_code=2)
        )
        self.assertEqual(self.reader.next_message()["time_code"], 1)
        self.assertEqual(self.reader.next_message()["time_code"], 2)
        self.assertIsNone(self.reader.next_message())

    def test_buffer_is_drained_as_messages_are_consumed(self) -> None:
        self.reader.feed(self.valid(type="finish"))
        self.assertGreater(self.reader.buffered_bytes, 0)
        self.reader.next_message()
        self.assertEqual(self.reader.buffered_bytes, 0)

    def test_rejects_an_absurd_declared_size(self) -> None:
        self.reader.feed(struct.pack("!I", protocol.MAX_MESSAGE_BYTES + 1))
        with self.assertRaises(ValueError):
            self.reader.next_message()

    def test_rejects_version_mismatch(self) -> None:
        self.reader.feed(
            encode({"type": "hello", "protocol_version": "stale_producer"})
        )
        with self.assertRaises(ValueError):
            self.reader.next_message()


class RequireMessageTests(unittest.TestCase):
    def test_passes_the_expected_type_through(self) -> None:
        message = {"type": "ready", "stream": {}}
        self.assertIs(protocol.require_message(message, "ready"), message)

    def test_abort_is_reported_with_its_reason(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            protocol.require_message(
                {"type": "abort", "reason": "grab failed"}, "banked"
            )
        self.assertIn("grab failed", str(caught.exception))

    def test_wrong_type_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            protocol.require_message({"type": "ready"}, "banked")


if __name__ == "__main__":
    unittest.main()
