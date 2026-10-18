"""Checks this reader and executor against the compatibility corpus of
spec/versioning.md section 4 (testdata/compat)."""

import glob
import hashlib
import json
import os
import struct

import numpy as np

from .container import File
from .executor import Program
from .tensor import Tensor


def load(root):
    """Every expected.json under root, as (directory, description) pairs."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*.expected.json"))):
        with open(p, encoding="utf-8") as fh:
            e = json.load(fh)
        if e.get("format") != "mim-compat" or e.get("version") != 1:
            raise ValueError(f"{p}: not a version 1 compatibility file")
        out.append((os.path.dirname(p), e))
    return out


def _meta_json(typ, v):
    """The expected.json encoding of a metadata value."""
    f64 = lambda x: struct.pack("<d", x).hex()  # noqa: E731
    if typ == "float":
        return f64(v)
    if typ == "bytes":
        return v.hex()
    if typ == "floats":
        return [f64(x) for x in v]
    return v


def tensor_of(j):
    return Tensor.from_bytes(j["dtype"], j["shape"], bytes.fromhex(j["data"]))


def compare_output(got, want):
    w = tensor_of(want)
    if got is None:
        return "missing"
    if got.dtype is not w.dtype or got.shape != w.shape:
        return f"{got.dtype.name}{list(got.shape)}, want {w.dtype.name}{list(w.shape)}"
    if want["check"] == "bits":
        return None if got.tobytes() == w.tobytes() else "bytes differ"
    if want["check"] == "close":
        g = got.arr.astype(np.float64)
        x = w.arr.astype(np.float64)
        big = float(np.max(np.abs(x))) if x.size else 0.0
        d = np.abs(g - x)
        if np.isnan(d).any():
            return "NaN where a number is expected"
        worst = float(d.max()) if d.size else 0.0
        if worst > want["tol"] * big:
            return f"max |difference| {worst:g}, allowed {want['tol'] * big:g}"
        return None
    return f"unknown check {want['check']!r}"


def check(directory, e):
    """Opens the file e describes and compares what this reader sees, then
    reproduces its runs. Raises AssertionError on the first difference."""
    with File.open(os.path.join(directory, e["file"])) as f:
        f.verify_hashes()
        p = f.preamble
        got = {"major": p.major, "minor": p.minor, "data_alignment": p.data_alignment}
        assert got == e["container"], f"container {got}, want {e['container']}"
        secs = [{"kind": s.kind, "required": bool(s.flags & 1), "hash_algorithm": s.hash_alg, "length": s.length} for s in f.sections]
        assert secs == e["sections"], f"sections {secs}, want {e['sections']}"
        tensors = []
        for t in f.tensors():
            tensors.append({"name": t.name, "dtype": t.dtype.name, "shape": list(t.shape), "offset": t.offset,
                            "sha256": hashlib.sha256(f.tensor_data(t)).hexdigest()})
        assert len(tensors) == len(e["tensors"]), f"{len(tensors)} tensors, want {len(e['tensors'])}"
        for g, w in zip(tensors, e["tensors"]):
            assert g == w, f"tensor {g}, want {w}"
        md = [{"key": k, "type": typ, "value": _meta_json(typ, v)} for k, (typ, v) in f.metadata().items()]
        assert md == e["metadata"], f"metadata differs: {md} vs {e['metadata']}"
        tr = f.training()
        if tr is not None:
            tr = {"optimizer": tr["optimizer"],
                  "counters": [{"name": n, "value": v} for n, v in tr["counters"]],
                  "hyperparameters": [{"name": n, "group": g, "value": struct.pack("<d", v).hex()}
                                      for n, g, v in tr["hyperparameters"]],
                  "parameters": [{"tensor": t, "group": g, "states": [{"name": s, "tensor": st} for s, st in states]}
                                 for t, g, states in tr["parameters"]],
                  "extras": [{"name": n, "tensor": t} for n, t in tr["extras"]]}
        assert tr == e.get("training"), f"training {tr}, want {e.get('training')}"
        gr = f.graph()
        if gr is None:
            assert "graph" not in e, "the file has no graph"
            return
        info = {"inputs": [gr.values[i].name for i in gr.main.inputs], "outputs": [gr.values[i].name for i in gr.main.outputs]}
        assert info == e["graph"], f"graph {info}, want {e['graph']}"
        s = Program.load(f).session()
        for ri, run in enumerate(e.get("runs", [])):
            out = s.run({k: tensor_of(v) for k, v in run["inputs"].items()})
            assert len(out) == len(run["outputs"]), f"run {ri}: {len(out)} outputs"
            for name, want in run["outputs"].items():
                msg = compare_output(out.get(name), want)
                assert msg is None, f"run {ri} output {name!r}: {msg}"

