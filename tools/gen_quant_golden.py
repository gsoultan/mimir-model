#!/usr/bin/env python3
"""Generate testdata/quant_golden.json from the Python `gguf` package.

Run from the repository root with `python3 -I tools/gen_quant_golden.py`.

For each block dtype the file holds the raw block bytes (base64) and the
float32 values that gguf's reference dequantizer produces for them. The Go
test (quant_test.go) requires Dequantize to reproduce those values bit-exactly.

Q8_0, Q4_0 and MXFP4 are produced by gguf's quantize() from a deterministic
normal sample, which is stored too so the Go encoders can be checked against
gguf's byte for byte. gguf has no Python quantizer for Q4_K and Q6_K, so their blocks
are built from deterministic pseudo-random bytes with finite f16 scale fields;
only the dequantizer matters for the golden test.
"""

import base64
import json
import os
import sys

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize, quantize

N = 512  # elements per vector; a multiple of 256 so every block type fits
rng = np.random.default_rng(0)


def f16_bytes(rng: np.random.Generator, n: int) -> np.ndarray:
    """n random finite non-zero float16 values as little-endian bytes."""
    v = rng.uniform(0.01, 2.0, size=n).astype(np.float16)
    v *= rng.choice([-1.0, 1.0], size=n).astype(np.float16)
    return v.view(np.uint8).reshape(n, 2)


def from_quantizer(qtype: T, x: np.ndarray) -> bytes:
    q = quantize(x, qtype)
    return q.tobytes()


def raw_q4_k(rng: np.random.Generator, n_blocks: int) -> bytes:
    # f16 d; f16 dmin; u8 scales[12]; u8 qs[128]
    d = f16_bytes(rng, n_blocks)
    dmin = f16_bytes(rng, n_blocks)
    scales = rng.integers(0, 256, size=(n_blocks, 12), dtype=np.uint8)
    qs = rng.integers(0, 256, size=(n_blocks, 128), dtype=np.uint8)
    return np.concatenate([d, dmin, scales, qs], axis=1).astype(np.uint8).tobytes()


def raw_q6_k(rng: np.random.Generator, n_blocks: int) -> bytes:
    # u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d
    ql = rng.integers(0, 256, size=(n_blocks, 128), dtype=np.uint8)
    qh = rng.integers(0, 256, size=(n_blocks, 64), dtype=np.uint8)
    scales = rng.integers(0, 256, size=(n_blocks, 16), dtype=np.uint8)
    d = f16_bytes(rng, n_blocks)
    return np.concatenate([ql, qh, scales, d], axis=1).astype(np.uint8).tobytes()


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("testdata", "quant_golden.json")
    x = rng.standard_normal(N).astype(np.float32).reshape(1, N)

    inputs = {"q8_0": x, "q4_0": x, "mxfp4": x * np.float32(4.0)}
    vectors: dict[str, bytes] = {
        "q8_0": from_quantizer(T.Q8_0, x),
        "q4_0": from_quantizer(T.Q4_0, x),
        "mxfp4": from_quantizer(T.MXFP4, x * np.float32(4.0)),
        "q4_k": raw_q4_k(rng, N // 256),
        "q6_k": raw_q6_k(rng, N // 256),
    }
    types = {"q8_0": T.Q8_0, "q4_0": T.Q4_0, "mxfp4": T.MXFP4, "q4_k": T.Q4_K, "q6_k": T.Q6_K}

    golden = {}
    for name, raw in vectors.items():
        arr = np.frombuffer(raw, dtype=np.uint8)
        vals = dequantize(arr, types[name]).reshape(-1)
        assert vals.dtype == np.float32 and vals.shape == (N,), (name, vals.dtype, vals.shape)
        assert np.all(np.isfinite(vals)), name
        # Store exact bit patterns, not decimal strings, so the comparison is bit-exact.
        golden[name] = {
            "bytes": base64.b64encode(raw).decode("ascii"),
            # float32 bit patterns of the quantizer's input, when it had one
            "input_bits": [int(b) for b in inputs[name].reshape(-1).view(np.uint32)] if name in inputs else [],
            "bits": [int(b) for b in vals.view(np.uint32)],
            "values": [float(v) for v in vals],
        }

    with open(out_path, "w") as f:
        json.dump(golden, f, indent=1)
        f.write("\n")
    print(f"wrote {out_path}: " + ", ".join(f"{k}={len(v)}B" for k, v in vectors.items()))


if __name__ == "__main__":
    main()
