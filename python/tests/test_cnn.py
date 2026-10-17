"""The CNN of testdata/onnx, converted from ONNX by the Go importer, run
from its mim file against onnxruntime's outputs for the original model."""

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

DIR = os.path.join(_paths.TESTDATA, "onnx")


def array(j):
    return np.frombuffer(base64.b64decode(j["data"]), dtype="<f4").reshape(j["shape"])


class TestCNN(unittest.TestCase):
    def test_matches_onnxruntime(self):
        with open(os.path.join(DIR, "expected.json"), encoding="utf-8") as fh:
            e = json.load(fh)
        with File.open(os.path.join(DIR, e["mim"])) as f:
            f.verify_hashes()
            md = f.metadata()
            self.assertEqual(md["import.source_format"][1], "onnx")
            s = Program.load(f).session()
            for run in e["runs"]:
                inputs = {k: Tensor(D.F32, tuple(v["shape"]), array(v).copy()) for k, v in run["inputs"].items()}
                out = s.run(inputs)
                self.assertEqual(set(out), set(run["outputs"]))
                for k, v in run["outputs"].items():
                    want = array(v)
                    got = out[k]
                    self.assertIs(got.dtype, D.F32)
                    self.assertEqual(tuple(got.shape), want.shape)
                    d = np.abs(got.arr.astype(np.float64) - want.astype(np.float64)).max()
                    self.assertLessEqual(d, e["tolerance"], f"{k}: max |difference| {d:g}")


if __name__ == "__main__":
    unittest.main()
