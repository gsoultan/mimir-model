"""The compatibility corpus of spec/versioning.md section 4: every file of
every released version, read and run by this runtime."""

import os
import unittest

import _paths  # noqa: F401
from mlm import compat


class TestCompat(unittest.TestCase):
    def test_corpus(self):
        files = compat.load(os.path.join(_paths.TESTDATA, "compat"))
        self.assertTrue(files)
        for d, e in files:
            with self.subTest(version=os.path.basename(d), file=e["file"]):
                compat.check(d, e)


if __name__ == "__main__":
    unittest.main()
