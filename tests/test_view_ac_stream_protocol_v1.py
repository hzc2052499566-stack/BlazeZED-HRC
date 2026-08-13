import socket
import sys
import threading
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import view_ac_stream_protocol_v1 as protocol


class StreamProtocolTest(unittest.TestCase):
    def test_round_trip_header_and_arrays(self):
        left, right = socket.socketpair()
        try:
            expected_rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
            expected_depth = np.arange(8, dtype=np.float32).reshape(2, 4)
            thread = threading.Thread(
                target=protocol.send_packet,
                args=(left, {"type": "frame", "sequence_index": 7}, {"rgb": expected_rgb, "depth": expected_depth}),
            )
            thread.start()
            header, arrays, received_bytes = protocol.recv_packet(right)
            thread.join()
            self.assertEqual(header["type"], "frame")
            self.assertEqual(header["sequence_index"], 7)
            np.testing.assert_array_equal(arrays["rgb"], expected_rgb)
            np.testing.assert_array_equal(arrays["depth"], expected_depth)
            self.assertGreater(received_bytes, expected_rgb.nbytes + expected_depth.nbytes)
        finally:
            left.close()
            right.close()

    def test_rejects_unsupported_dtype(self):
        left, right = socket.socketpair()
        try:
            with self.assertRaises(ValueError):
                protocol.send_packet(left, {"type": "bad"}, {"array": np.ones(2, dtype=np.int32)})
        finally:
            left.close()
            right.close()

    def test_require_message(self):
        with self.assertRaises(RuntimeError):
            protocol.require_message({"type": "wrong"}, "right")


if __name__ == "__main__":
    unittest.main()
