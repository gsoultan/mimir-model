"""The models of testdata/pytorch, exported from PyTorch by mmf.pytorch,
run from their mmf files against PyTorch's outputs. Needs no PyTorch."""

import base64
import json
import os
import unittest

import numpy as np

import _paths  # noqa: F401
from mmf import dtypes as D
from mmf.container import File
from mmf.executor import Program
from mmf.tensor import Tensor

DIR = os.path.join(_paths.TESTDATA, "pytorch")


def tensor(j):
    dt = D.BY_NAME[j["dtype"]]
    return Tensor.from_bytes(dt, j["shape"], base64.b64decode(j["data"]))


class TestPyTorchFiles(unittest.TestCase):
    def test_matches_pytorch(self):
        with open(os.path.join(DIR, "expected.json"), encoding="utf-8") as fh:
            e = json.load(fh)
        self.assertEqual({m["name"] for m in e["models"]}, {"tiny-cnn", "audio", "tiny-gpt", "tiny-llama", "tiny-llama-kv"})
        for m in e["models"]:
            with self.subTest(model=m["name"]), File.open(os.path.join(DIR, m["mmf"])) as f:
                f.verify_hashes()
                self.assertEqual(f.metadata()["export.source_format"], ("string", "pytorch"))
                s = Program.load(f).session()
                for run in m["runs"]:
                    out = s.run({k: tensor(v) for k, v in run["inputs"].items()})
                    self.assertEqual(set(out), set(run["outputs"]))
                    for k, v in run["outputs"].items():
                        want = tensor(v).arr.astype(np.float64)
                        got = out[k]
                        self.assertIs(got.dtype, D.F32)
                        self.assertEqual(tuple(got.shape), want.shape)
                        d = np.abs(got.arr.astype(np.float64) - want).max()
                        self.assertLessEqual(d, e["tolerance"] * max(np.abs(want).max(), 1e-30), f"{k}: max |difference| {d:g}")


if __name__ == "__main__":
    unittest.main()
