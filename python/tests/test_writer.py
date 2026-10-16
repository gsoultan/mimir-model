"""The writer: files it produces pass every check of the reader, and
metadata and graphs decode to what was written."""

import os
import struct
import tempfile
import unittest

import numpy as np

import _paths  # noqa: F401
from mmf import dtypes as D
from mmf import container as C
from mmf.container import File
from mmf.errors import MMFError
from mmf.executor import Program
from mmf.metadata import decode_metadata
from mmf.tensor import Tensor
from mmf.writer import CONST, INPUT, OUTPUT, WEIGHT, GraphBuilder, Writer, encode_metadata


def f32(*a):
    return np.asarray(a, dtype="<f4").tobytes()


class TestContainer(unittest.TestCase):
    def test_canonical_layout(self):
        w = Writer()
        big = np.arange(300_000, dtype="<f4").tobytes()  # over 1 MiB: 4 KiB aligned
        w.add_tensor("z.small", "f32", [3], f32(1, 2, 3))
        w.add_tensor("a.big", "f32", [300_000], big)
        w.add_tensor("m.empty", "f16", [0, 4], b"")
        w.add_tensor("q", "q8_0", [1, 32], bytes(34))
        f = File(w.to_bytes())  # the reader checks packing, order and alignment
        f.verify_hashes()
        self.assertEqual([t.name for t in f.tensors()], ["a.big", "m.empty", "q", "z.small"])
        self.assertEqual(f.tensor("a.big").offset % 4096, 0)
        self.assertEqual(bytes(f.tensor_data(f.tensor("a.big"))), big)
        self.assertEqual(bytes(f.tensor_data(f.tensor("z.small"))), f32(1, 2, 3))
        self.assertEqual(f.preamble.data_alignment, 16384)
        self.assertEqual(w.tensor_indices(), [3, 0, 1, 2])

    def test_no_hashes(self):
        w = Writer(data_alignment=4096, hashes=False)
        w.add_tensor("t", "i32", [2], np.array([1, 2], "<i4").tobytes())
        f = File(w.to_bytes())
        self.assertTrue(all(s.hash_alg == 0 for s in f.sections))
        self.assertEqual(f.preamble.data_alignment, 4096)

    def test_rejects(self):
        w = Writer()
        w.add_tensor("t", "f32", [1], f32(0))
        with self.assertRaises(MMFError) as e:
            w.add_tensor("t", "f32", [1], f32(0))
        self.assertEqual(e.exception.code, "duplicate")
        with self.assertRaises(MMFError) as e:
            w.add_tensor("u", "f32", [2], f32(0))
        self.assertEqual(e.exception.code, "size-mismatch")
        with self.assertRaises(MMFError) as e:
            w.add_tensor("v", "q4_0", [16], bytes(9))
        self.assertEqual(e.exception.code, "bad-shape")
        with self.assertRaises(MMFError) as e:
            w.add_tensor("", "f32", [1], f32(0))
        self.assertEqual(e.exception.code, "bad-name")
        with self.assertRaises(MMFError) as e:
            Writer(data_alignment=1000)
        self.assertEqual(e.exception.code, "bad-alignment")
        w.add_section(C.METADATA, encode_metadata({}))
        with self.assertRaises(MMFError):
            w.add_section(C.METADATA, encode_metadata({}))

    def test_write_is_atomic_and_readable(self):
        w = Writer()
        w.add_tensor("t", "f32", [2], f32(1, 2))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.mmf")
            n = w.write(p)
            self.assertEqual(os.path.getsize(p), n)
            self.assertEqual(os.listdir(d), ["m.mmf"])
            # Read rather than map: Windows cannot delete a mapped file.
            with open(p, "rb") as fh:
                File(fh.read()).verify_hashes()


class TestMetadata(unittest.TestCase):
    def test_round_trip(self):
        md = {
            "general.name": "tiny",
            "b": True,
            "i": -3,
            "u": ("uint", 1 << 63),
            "x": 1.5,
            "raw": b"\x00\x01",
            "ints": [1, -2, 3],
            "floats": [0.5, 2],
            "strings": ["a", "", "héllo"],
            "empty": ("strings", []),
        }
        got = decode_metadata(encode_metadata(md))
        self.assertEqual(list(got), sorted(got, key=lambda k: k.encode()))
        self.assertEqual(got["general.name"], ("string", "tiny"))
        self.assertEqual(got["b"], ("bool", True))
        self.assertEqual(got["i"], ("int", -3))
        self.assertEqual(got["u"], ("uint", 1 << 63))
        self.assertEqual(got["x"], ("float", 1.5))
        self.assertEqual(got["raw"], ("bytes", b"\x00\x01"))
        self.assertEqual(got["ints"], ("ints", [1, -2, 3]))
        self.assertEqual(got["floats"], ("floats", [0.5, 2.0]))
        self.assertEqual(got["strings"], ("strings", ["a", "", "héllo"]))
        self.assertEqual(got["empty"], ("strings", []))

    def test_decoded_values_re_encode(self):
        md = {"a": 1, "b": ["x"], "c": [1.0]}
        once = encode_metadata(md)
        self.assertEqual(encode_metadata(decode_metadata(once)), once)


class TestGraph(unittest.TestCase):
    def test_graph_runs(self):
        w = Writer()
        w.add_tensor("w", "f32", [2, 3], f32(1, 2, 3, 4, 5, 6))
        w.add_tensor("b", "f32", [2], f32(10, 20))
        g = GraphBuilder()
        g.symdim("N", 4)
        x = g.value("x", D.F32, ["N", 3], INPUT)
        pos = w.tensor_indices()
        wv = g.value("w", D.F32, [2, 3], WEIGHT, pos[0])
        bv = g.value("b", D.F32, [2], CONST, pos[1])
        h = g.value("h", D.F32, ["N", 2])
        g.op("call", [x, wv, bv], [h], [("name", "string", "linear"), ("version", "int", 1)])
        y = g.value("y", D.F32, ["N", 2], OUTPUT)
        zero = g.value("zero", D.F32, [1, 1])
        g.op("const", [], [zero], [("value", "float", 0.0)])
        g.op("max", [h, zero], [y])
        g.output(y)
        w.add_section(C.GRAPH, g.encode(), required=True)
        f = File(w.to_bytes())
        graph = f.graph()
        self.assertEqual(graph.symdims[0].name, "N")
        self.assertEqual(graph.symdims[0].bound, 4)
        self.assertEqual([graph.values[v].name for v in graph.main.inputs], ["x"])
        s = Program.load(f).session()
        out = s.run({"x": Tensor(D.F32, (2, 3), np.array([[1, 0, 0], [0, 0, -9]], np.float32))})["y"]
        np.testing.assert_array_equal(out.arr, [[11, 24], [0, 0]])

    def test_encoding_is_aligned(self):
        g = GraphBuilder()
        x = g.value("x", D.F32, [1], INPUT)
        y = g.value("y", D.F32, [1], OUTPUT)
        g.op("neg", [x], [y])
        g.output(y)
        b = g.encode()
        for i in range(11):
            off, _ = struct.unpack_from("<QQ", b, 16 + 16 * i)
            self.assertEqual(off % 8, 0)

    def test_unknown_symdim(self):
        g = GraphBuilder()
        with self.assertRaises(MMFError):
            g.value("x", D.F32, ["T"], INPUT)


if __name__ == "__main__":
    unittest.main()
