"""Loads the primitive conformance corpus (spec/conformance.md) and checks
results against it, in IEEE mode."""

import glob
import json
import math
import os
import struct

import numpy as np

from . import dtypes as D
from .prims import OP_BY_NAME, Attr
from .tensor import Tensor


def parse_tensor(j):
    return Tensor.from_bytes(D.BY_NAME[j["dtype"]], j["shape"], bytes.fromhex(j["data"]))


def parse_attrs(j):
    attrs = {}
    for k, v in (j or {}).items():
        (typ, val), = v.items()
        if typ == "float":
            val = struct.unpack("<d", bytes.fromhex(val))[0]
        elif typ == "dtype":
            val = D.BY_NAME[val].id
        attrs[k] = Attr(typ, val)
    return attrs


class Case:
    def __init__(self, op, j):
        self.op = op
        self.opcode = OP_BY_NAME[op]
        self.name = j["name"]
        self.inputs = [parse_tensor(t) for t in j["inputs"]]
        self.attrs = parse_attrs(j.get("attrs"))
        self.output = parse_tensor(j["output"])
        self.check = j["check"]
        self.dont_care = set(j.get("dont_care", []))

    @property
    def id(self):
        return f"{self.op}/{self.name}"


def load_dir(path):
    cases = []
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            j = json.load(fh)
        if j.get("format") != "mmf-conformance" or j.get("version") != 1:
            raise ValueError(f"{f}: not a version 1 conformance file")
        cases.extend(Case(j["op"], c) for c in j["cases"])
    return cases


def _ulp(e, dt):
    p, emin, _ = D._FORMATS[dt.name]
    if e == 0:
        return math.ldexp(1, emin - p + 1)
    return math.ldexp(1, max(math.frexp(abs(e))[1] - 1, emin) - p + 1)


def compare(case, got):
    """Returns None when got passes the case, else a description."""
    want = case.output
    if got.dtype is not want.dtype or got.shape != want.shape:
        return f"result {got.dtype.name}{list(got.shape)}, want {want.dtype.name}{list(want.shape)}"
    dt = want.dtype
    kind = case.check["kind"]
    n = want.size
    gb = np.frombuffer(got.tobytes(), dtype=np.uint8).reshape(n, -1) if n else None
    wb = np.frombuffer(want.tobytes(), dtype=np.uint8).reshape(n, -1) if n else None
    floaty = dt.kind in ("float", "storage")
    gv = D.to_f64(got.arr.reshape(-1), dt) if floaty else None
    wv = D.to_f64(want.arr.reshape(-1), dt) if floaty else None
    if kind == "tol":
        ref = np.frombuffer(bytes.fromhex(case.check["ref"]), dtype="<f8")
        bound = np.frombuffer(bytes.fromhex(case.check["bound"]), dtype="<f8")
        max_abs = case.check.get("max_abs")
    for i in range(n):
        if i in case.dont_care:
            continue
        if kind == "bits" or (kind == "exact" and not floaty):
            ok = bytes(gb[i]) == bytes(wb[i])
        elif kind == "exact":
            ok = (math.isnan(gv[i]) and math.isnan(wv[i])) or bytes(gb[i]) == bytes(wb[i])
        elif kind == "ulp":
            e, r = wv[i], gv[i]
            if math.isnan(e) or math.isinf(e):
                ok = (math.isnan(e) and math.isnan(r)) or r == e
            else:
                ok = not math.isnan(r) and abs(r - e) <= case.check["ulp"] * _ulp(e, dt)
        elif kind == "tol":
            e, r = ref[i], gv[i]
            if math.isnan(e) or math.isinf(e):
                ok = (math.isnan(e) and math.isnan(r)) or r == e
            else:
                ok = not math.isnan(r) and abs(r - e) <= bound[i]
                if ok and max_abs:
                    ok = abs(r) <= max_abs
        else:
            return f"unknown check {kind!r}"
        if not ok:
            g = gv[i] if floaty else bytes(gb[i]).hex()
            w = wv[i] if floaty else bytes(wb[i]).hex()
            return f"element {i}: got {g}, want {w} ({kind})"
    return None


def run(cases, evaluate):
    """Runs every case through evaluate(opcode, attrs, inputs, out_dtype,
    out_shape) and returns the failures as (case id, message) pairs. An
    exception counts as a failure of its case."""
    failures = []
    for c in cases:
        try:
            got = evaluate(c.opcode, c.attrs, c.inputs, c.output.dtype, c.output.shape)
            msg = compare(c, got)
        except Exception as e:  # noqa: BLE001 - every error is a failed case
            msg = f"{type(e).__name__}: {e}"
        if msg:
            failures.append((c.id, msg))
    return failures
