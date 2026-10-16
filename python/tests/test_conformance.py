"""Every case of the primitive conformance corpus, in IEEE mode for every
float dtype (spec/conformance.md)."""

import os
import unittest

import _paths  # noqa: F401
from mmf import conformance, prims


class TestConformance(unittest.TestCase):
    def test_corpus(self):
        cases = conformance.load_dir(os.path.join(_paths.TESTDATA, "conformance", "primitives"))
        self.assertGreaterEqual(len(cases), 1330)
        self.assertEqual({c.opcode for c in cases}, set(range(1, 34)), "the corpus covers every primitive")
        failures = conformance.run(cases, prims.evaluate)
        self.assertEqual(failures, [], "\n".join(f"{c}: {m}" for c, m in failures[:20]))


if __name__ == "__main__":
    unittest.main()
