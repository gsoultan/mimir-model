"""The training section (spec/training.md): the corpus checkpoint decodes,
and damaged sections are rejected with bad-training, as the Go reference
rejects them."""

import os
import struct
import unittest

import _paths  # noqa: F401
from mlm.container import TRAINING, File
from mlm.errors import MIMError
from mlm.training import COUNTERS, PARAMS, STATES, STRINGS, decode_training

PATH = os.path.join(_paths.TESTDATA, "compat", "0.2-0.1", "training.mim")


class TestTraining(unittest.TestCase):
    def setUp(self):
        with File.open(PATH) as f:
            self.t = f.training()
            self.b = bytes(f.section_bytes(f.section(TRAINING)))

    def test_decode(self):
        t = self.t
        self.assertEqual(t["optimizer"], "adamw")
        self.assertEqual(t["counters"], [("step", 1200), ("epoch", 3)])
        self.assertIn(("weight_decay", 1, 0.0), t["hyperparameters"])
        self.assertEqual(t["parameters"][1], ("linear.bias", 1, [("m", "linear.bias@m"), ("v", "linear.bias@v")]))
        self.assertEqual(t["extras"], [("rng", "train.rng")])

    def test_hostile(self):
        def table(i):
            return struct.unpack_from("<Q", self.b, 16 + 16 * i)[0]

        def mut(off, fmt, v):
            b = bytearray(self.b)
            struct.pack_into(fmt, b, off, v)
            return bytes(b)

        cases = {
            "short": self.b[:127],
            "magic": mut(3, "<B", ord("X")),
            "major": mut(4, "<H", 1),
            "reserved": mut(12, "<I", 1),
            "misaligned table": mut(16 + 16 * COUNTERS, "<Q", table(COUNTERS) + 4),
            "table outside": mut(24 + 16 * STATES, "<Q", 1 << 20),
            "string 0": mut(table(STRINGS) + 4, "<I", 1),
            "string index": mut(8, "<I", 1000),
            "state run gap": mut(table(PARAMS) + 16 + 8, "<I", 3),
            "orphan state": mut(table(PARAMS) + 16 + 12, "<I", 1),
            "counter reserved": mut(table(COUNTERS) + 4, "<I", 1),
            "counter twice": mut(table(COUNTERS) + 16, "<I", struct.unpack_from("<I", self.b, table(COUNTERS))[0]),
        }
        for name, b in cases.items():
            with self.subTest(name):
                with self.assertRaises(MIMError) as cm:
                    decode_training(b)
                self.assertEqual(cm.exception.code, "bad-training")


if __name__ == "__main__":
    unittest.main()
