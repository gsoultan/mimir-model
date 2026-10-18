"""The reader against hostile files (spec/container.md section 9.1): the
mutations of the Go reference's hostile corpus, applied to a corpus file,
must be rejected with the same codes."""

import os
import struct
import unittest

import _paths  # noqa: F401
from mlm.container import File
from mlm.errors import MIMError

BASE = os.path.join(_paths.TESTDATA, "compat", "0.1-0.1", "dtypes.mim")


def hostile_cases(base):
    ns = struct.unpack_from("<I", base, 64)[0]
    sec0 = 64 + 16

    def row(i):
        return sec0 + ns * 64 + i * 96

    def mut(f):
        b = bytearray(base)
        f(b)
        return bytes(b)

    def put(b, off, fmt, v):
        struct.pack_into(fmt, b, off, v)

    def swap(b):
        r0, r1 = row(0), row(1)
        b[r0 : r0 + 96], b[r1 : r1 + 96] = b[r1 : r1 + 96], b[r0 : r0 + 96]

    def overflow(b):
        b[row(0) + 12] = 2
        put(b, row(0) + 16, "<Q", 1 << 40)
        put(b, row(0) + 24, "<Q", 1 << 40)

    def overlap(b):
        put(b, row(0) + 80, "<Q", 0)
        put(b, row(1) + 80, "<Q", 0)

    def hole(b):
        p = struct.unpack_from("<Q", b, 40)[0] + 64
        put(b, 40, "<Q", p)
        put(b, sec0 + 16, "<Q", p)
        put(b, 48, "<Q", struct.unpack_from("<Q", b, 48)[0] + 64)
        b.extend(bytes(64))

    return [
        ("truncated-preamble", "truncated", base[:40]),
        ("bad-magic", "bad-magic", mut(lambda b: b.__setitem__(0, ord("X")))),
        ("big-endian", "big-endian", mut(lambda b: put(b, 8, ">I", 0x0A0B0C0D))),
        ("major-version", "unsupported-version", mut(lambda b: put(b, 4, "<H", 1))),
        ("reserved-preamble", "reserved-non-zero", mut(lambda b: b.__setitem__(60, 1))),
        ("alignment-not-pow2", "bad-alignment", mut(lambda b: put(b, 12, "<I", 5000))),
        ("truncated-file", "truncated", base[:-1]),
        ("trailing-bytes", "bad-length", base + b"\0"),
        ("index-too-large", "index-too-large", mut(lambda b: put(b, 24, "<Q", (256 << 20) + 1))),
        ("index-out-of-bounds", "out-of-bounds", mut(lambda b: put(b, 16, "<Q", (len(b) + 7) & ~7))),
        ("index-length-mismatch", "bad-length", mut(lambda b: put(b, 64 + 4, "<I", 7))),
        ("weights-disagree", "bad-length", mut(lambda b: put(b, sec0 + 16, "<Q", 0))),
        ("unknown-required-section", "unknown-required-section", mut(lambda b: put(b, sec0, "<I", 99))),
        ("bad-hash-alg", "bad-hash-algorithm", mut(lambda b: b.__setitem__(sec0 + 24, 9))),
        ("section-flags", "reserved-non-zero", mut(lambda b: put(b, sec0 + 4, "<I", 3))),
        ("name-out-of-pool", "out-of-bounds", mut(lambda b: put(b, row(0) + 4, "<I", 1 << 30))),
        ("unsorted-names", "unsorted", mut(swap)),
        ("bad-dtype", "bad-dtype", mut(lambda b: put(b, row(0) + 8, "<I", 999))),
        ("rank-too-big", "bad-shape", mut(lambda b: b.__setitem__(row(0) + 12, 9))),
        ("size-mismatch", "size-mismatch", mut(lambda b: put(b, row(0) + 16, "<Q", 5))),
        ("shape-overflow", "overflow", mut(overflow)),
        ("misaligned-tensor", "misaligned", mut(lambda b: put(b, row(0) + 80, "<Q", struct.unpack_from("<Q", b, row(0) + 80)[0] + 4))),
        ("tensor-out-of-bounds", "out-of-bounds", mut(lambda b: put(b, row(0) + 80, "<Q", 1 << 40))),
        ("overlapping-tensors", "overlap", mut(overlap)),
        ("hole-in-weights", "non-canonical-layout", mut(hole)),
    ]


class TestHostile(unittest.TestCase):
    def setUp(self):
        with open(BASE, "rb") as fh:
            self.base = fh.read()

    def test_base_is_valid(self):
        f = File(self.base)
        f.verify_hashes()
        self.assertEqual(f.tensor("large.u8").shape, (1 << 20,))
        self.assertIsNone(f.tensor("missing"))
        self.assertEqual(f.metadata()["test.uint"], ("uint", (1 << 64) - 1))

    def test_hostile(self):
        for name, code, data in hostile_cases(self.base):
            with self.subTest(name):
                with self.assertRaises(MIMError) as cm:
                    File(data)
                self.assertEqual(cm.exception.code, code, str(cm.exception))

    def test_hash_mismatch(self):
        b = bytearray(self.base)
        b[-1] ^= 1
        f = File(bytes(b))
        with self.assertRaises(MIMError) as cm:
            f.verify_hashes()
        self.assertEqual(cm.exception.code, "hash-mismatch")


if __name__ == "__main__":
    unittest.main()
