"""The tiny Llama of testdata/llama, run from its mim file: the prompt and
the greedy continuation, step by step through the key-value cache, against
the float64 NumPy oracle that the Go executor is also checked against."""

import base64
import json
import os
import unittest

import numpy as np

import _paths  # noqa: F401
from mim import dtypes as D
from mim.container import File
from mim.executor import Program
from mim.tensor import Tensor

MODEL = os.path.join(_paths.TESTDATA, "compat", "0.1-0.1", "tiny-llama.mim")


def logits(b64):
    return np.frombuffer(base64.b64decode(b64), dtype="<f4")


class TestLlama(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(_paths.TESTDATA, "llama", "expected.json"), encoding="utf-8") as fh:
            self.e = json.load(fh)
        self.f = File.open(MODEL)
        self.s = Program.load(self.f).session()

    def tearDown(self):
        self.f.close()

    def step(self, ids, pos):
        out = self.s.run({
            "tokens": Tensor(D.I32, (1, len(ids)), np.array([ids], dtype=np.int32)),
            "positions": Tensor(D.I64, (1, len(ids)), np.arange(pos, pos + len(ids), dtype=np.int64)[None]),
        })
        (y,) = out.values()
        self.assertEqual((y.dtype.name, y.shape), ("f32", (1, len(ids), self.e["vocab"])))
        return y.arr.reshape(-1)

    def near(self, what, got, want):
        big = np.abs(want).max()
        worst = np.abs(got.astype(np.float64) - want).max()
        self.assertLessEqual(worst, 2e-5 * big, f"{what}: max |logit difference| {worst}")

    def test_generate_matches_oracle(self):
        e = self.e
        v = e["vocab"]
        out = self.step(e["prompt_ids"], 0)
        self.near("prompt", out, logits(e["prompt_logits"]))
        last = out[-v:]
        pos = len(e["prompt_ids"])
        for i, want in enumerate(e["greedy"]):
            self.near(f"step {i}", last, logits(e["greedy_logits"][i]))
            self.assertEqual(int(np.argmax(last)), want, f"step {i}")
            last = self.step([want], pos)
            pos += 1

    def test_metadata(self):
        md = self.f.metadata()
        self.assertEqual(md["general.architecture"], ("string", "llama"))
        typ, tokens = md["tokenizer.ggml.tokens"]
        self.assertEqual((typ, len(tokens)), ("strings", self.e["vocab"]))


if __name__ == "__main__":
    unittest.main()
