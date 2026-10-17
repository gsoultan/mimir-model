"""The executor's handling of sessions and errors, and the standard-library
composites that the corpora do not reach."""

import math
import os
import unittest

import numpy as np

import _paths  # noqa: F401
from mmf import compat, stdlib
from mmf import dtypes as D
from mmf.container import File
from mmf.errors import RunError
from mmf.executor import Program
from mmf.prims import Attr
from mmf.tensor import Tensor

GRAPH = os.path.join(_paths.TESTDATA, "compat", "0.1-0.1", "graph.mmf")


def f32(a):
    a = np.asarray(a, dtype=np.float32)
    return Tensor(D.F32, a.shape, a)


def call(name, attrs, ins, dts, shapes):
    return stdlib.COMPOSITES[name](attrs, ins, dts, shapes)


class TestSession(unittest.TestCase):
    def setUp(self):
        self.f = File.open(GRAPH)
        self.p = Program.load(self.f)
        (_, self.e), = [x for x in compat.load(os.path.join(_paths.TESTDATA, "compat")) if x[1]["file"] == "graph.mmf"]
        self.inputs = {k: compat.tensor_of(v) for k, v in self.e["runs"][0]["inputs"].items()}

    def tearDown(self):
        self.f.close()

    def test_failed_run_changes_no_variable(self):
        s = self.p.session()
        s.run(self.inputs)
        acc = s.variable("acc")
        bad = dict(self.inputs, k_t=f32(np.zeros((1, 2, 2))))
        with self.assertRaises(RunError):
            s.run(bad)
        self.assertIs(s.variable("acc"), acc)
        fresh = self.p.session().variable("acc")
        self.assertEqual(fresh.arr.tolist(), [[1, 2, 3, 4]])

    def test_input_errors(self):
        s = self.p.session()
        cases = {
            "missing": {k: v for k, v in self.inputs.items() if k != "n"},
            "extra": dict(self.inputs, extra=f32([1])),
            "above bound": dict(self.inputs, x=f32(np.zeros((9, 4)))),
            "dtype": dict(self.inputs, n=Tensor(D.I32, (), np.array(1, dtype=np.int32))),
        }
        for name, ins in cases.items():
            with self.subTest(name), self.assertRaises(RunError):
                s.run(ins)

    def test_iteration_limit(self):
        self.p.max_iterations = 3
        with self.assertRaisesRegex(RunError, "iterations"):
            self.p.session().run(self.inputs)


class TestComposites(unittest.TestCase):
    def test_top_k_and_argsort_order(self):
        x = f32([[1.0, np.nan, -0.0, 3.0, 0.0, -np.inf, 3.0]])
        vals, idx = call("top_k.v1", {"k": Attr("int", 4)}, [x], [D.F32, D.I64], [(1, 4), (1, 4)])
        self.assertEqual(idx.arr.tolist(), [[1, 3, 6, 0]])
        _, idx = call("top_k.v1", {"k": Attr("int", 3), "largest": Attr("int", 0)}, [x], [D.F32, D.I64], [(1, 3), (1, 3)])
        self.assertEqual(idx.arr.tolist(), [[5, 2, 4]])
        (o,) = call("argsort.v1", {"descending": Attr("int", 1)}, [x], [D.I64], [(1, 7)])
        self.assertEqual(o.arr.tolist(), [[1, 3, 6, 0, 2, 4, 5]])
        xi = Tensor(D.BY_NAME["u64"], (3,), np.array([2**64 - 1, 0, 2**64 - 1], dtype=np.uint64))
        (o,) = call("argsort.v1", {}, [xi], [D.I64], [(3,)])
        self.assertEqual(o.arr.tolist(), [1, 0, 2])

    def test_cumsum(self):
        x = Tensor(D.I64, (4,), np.array([1, 2, 3, 4]))
        for ex, rev, want in [(0, 0, [1, 3, 6, 10]), (1, 0, [0, 1, 3, 6]), (0, 1, [10, 9, 7, 4]), (1, 1, [9, 7, 4, 0])]:
            (y,) = call("cumsum.v1", {"axis": Attr("int", 0), "exclusive": Attr("int", ex), "reverse": Attr("int", rev)},
                        [x], [D.I64], [(4,)])
            self.assertEqual(y.arr.tolist(), want)

    def test_conv2d(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((2, 4, 7, 6)).astype(np.float32)
        W = rng.standard_normal((6, 2, 3, 2)).astype(np.float32)
        B = rng.standard_normal(6).astype(np.float32)
        st, dl, pads, g = (2, 1), (1, 2), (1, 0, 2, 1), 2
        attrs = {"strides": Attr("ints", list(st)), "dilations": Attr("ints", list(dl)),
                 "pads": Attr("ints", list(pads)), "groups": Attr("int", g)}
        Ho = (7 + 1 - 1 * 2 - 1) // 2 + 1
        Wo = (6 + 3 - 2 * 1 - 1) // 1 + 1
        (y,) = call("conv2d.v1", attrs, [f32(X), f32(W), f32(B)], [D.F32], [(2, 6, Ho, Wo)])
        xp = np.pad(X.astype(np.float64), ((0, 0), (0, 0), (1, 0), (2, 1)))
        want = np.zeros((2, 6, Ho, Wo))
        for n in range(2):
            for f in range(6):
                gi = f // 3
                for i in range(Ho):
                    for j in range(Wo):
                        s = B[f]
                        for c in range(2):
                            for a in range(3):
                                for b in range(2):
                                    s += xp[n, gi * 2 + c, i * 2 + a, j + 2 * b] * W[f, c, a, b]
                        want[n, f, i, j] = s
        np.testing.assert_allclose(y.arr, want, rtol=1e-6, atol=1e-6)

    def test_attention_causal_with_cache(self):
        rng = np.random.default_rng(2)
        q = rng.standard_normal((1, 4, 2, 8)).astype(np.float32)
        k = rng.standard_normal((1, 2, 5, 8)).astype(np.float32)
        v = rng.standard_normal((1, 2, 5, 3)).astype(np.float32)
        (o,) = call("attention.v1", {"causal": Attr("int", 1), "num_kv_heads": Attr("int", 2)},
                    [f32(q), f32(k), f32(v)], [D.F32], [(1, 4, 2, 3)])
        want = np.zeros((1, 4, 2, 3))
        for h in range(4):
            for i in range(2):
                n = i + 3 + 1  # query i sees keys 0 .. i + (Tk - Tq)
                s = q[0, h, i].astype(np.float64) @ k[0, h // 2, :n].T.astype(np.float64) / math.sqrt(8)
                p = np.exp(s - s.max())
                want[0, h, i] = (p / p.sum()) @ v[0, h // 2, :n]
        np.testing.assert_allclose(o.arr, want, rtol=1e-6, atol=1e-6)

    def test_attention_v2_f16_cache(self):
        rng = np.random.default_rng(3)
        q = rng.standard_normal((1, 4, 2, 8)).astype(np.float32)
        k = rng.standard_normal((1, 2, 5, 8)).astype(np.float16)
        v = rng.standard_normal((1, 2, 5, 3)).astype(np.float16)
        attrs = {"causal": Attr("int", 1), "num_kv_heads": Attr("int", 2)}
        (o,) = call("attention.v2", attrs, [f32(q), Tensor(D.F16, k.shape, k), Tensor(D.F16, v.shape, v)],
                    [D.F32], [(1, 4, 2, 3)])
        (want,) = call("attention.v1", attrs, [f32(q), f32(k.astype(np.float32)), f32(v.astype(np.float32))],
                       [D.F32], [(1, 4, 2, 3)])
        np.testing.assert_array_equal(o.arr, want.arr)
        with self.assertRaises(RunError):
            call("attention.v1", attrs, [f32(q), Tensor(D.F16, k.shape, k), f32(v.astype(np.float32))],
                 [D.F32], [(1, 4, 2, 3)])

    def test_norms_and_gelu(self):
        x = np.linspace(-4, 4, 24, dtype=np.float32).reshape(2, 3, 4)
        scale = np.linspace(0.5, 1.5, 12, dtype=np.float32).reshape(3, 4)
        (y,) = call("layer_norm.v1", {"axis": Attr("int", 1)}, [f32(x), f32(scale)], [D.F32], [x.shape])
        xd = x.astype(np.float64)
        mu = xd.mean(axis=(1, 2), keepdims=True)
        sd = np.sqrt(((xd - mu) ** 2).mean(axis=(1, 2), keepdims=True) + 1e-5)
        np.testing.assert_allclose(y.arr, (xd - mu) / sd * scale, rtol=1e-6)
        (g,) = call("gelu_erf.v1", {}, [f32(x)], [D.F32], [x.shape])
        exact = [0.5 * t * (1 + math.erf(t / math.sqrt(2))) for t in xd.ravel()]
        np.testing.assert_allclose(g.arr.ravel(), exact, atol=1e-6)

    def test_unknown_attribute(self):
        with self.assertRaises(RunError):
            call("softmax.v1", {"dim": Attr("int", 0)}, [f32([1.0])], [D.F32], [(1,)])

    def test_dequant_matmul_a8_v2(self):
        x = np.arange(6, dtype=np.float32).reshape(1, 2, 3) / 4
        w = Tensor(D.F16, (2, 3), np.array([[1, 2, 3], [-1, 0, 0.5]], dtype=np.float16))
        want = x @ w.arr.astype(np.float64).T
        for attrs in ({}, {"block": Attr("int", 32)}, {"block": Attr("int", 256)}):
            (y,) = call("dequant_matmul_a8.v2", attrs, [f32(x), w], [D.F32], [(1, 2, 2)])
            np.testing.assert_allclose(y.arr, want, rtol=1e-6)
        for attrs in ({"block": Attr("int", 64)}, {"other": Attr("int", 1)}):
            with self.assertRaises(RunError):
                call("dequant_matmul_a8.v2", attrs, [f32(x), w], [D.F32], [(1, 2, 2)])


if __name__ == "__main__":
    unittest.main()
