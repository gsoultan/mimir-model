#!/usr/bin/env python3
"""Generate the importer fixtures in testdata/import.

Run from the repository root with `python3 -I tools/gen_import_testdata.py`.
Needs NumPy and the `gguf` package (the GGUF writer maintained by llama.cpp).

tiny.gguf is written by gguf's own GGUFWriter, so the Go parser is checked
against the reference writer, not against itself. tiny.safetensors is
written by hand following the safetensors format. expected.json records,
for each tensor, the dtype, the shape in mim (outermost-first) order and the
SHA-256 of its bytes, plus the float32 values gguf's dequantizer produces for
the quantized tensors.
"""

import base64
import hashlib
import json
import struct

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf import GGUFWriter
from gguf.quants import dequantize, quantize

rng = np.random.default_rng(7)
expected = {"gguf": {}, "safetensors": {}}


def record(fmt, name, dtype, shape, raw, values=None):
    e = {"dtype": dtype, "shape": list(shape), "sha256": hashlib.sha256(raw).hexdigest()}
    if values is not None:
        e["f32"] = base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode()
    expected[fmt][name] = e


# ---- GGUF
w = GGUFWriter("testdata/import/tiny.gguf", "llama")
w.add_name("tiny")
w.add_uint32("tiny.block_count", 2)
w.add_float32("tiny.rope.freq_base", 10000.0)
w.add_bool("tiny.flag", True)
w.add_array("tokenizer.ggml.tokens", ["<s>", "</s>", "hello", "world"])
w.add_array("tokenizer.ggml.scores", [0.0, -1.0, -2.5, -3.25])
w.add_array("tokenizer.ggml.token_type", [3, 3, 1, 1])

f32 = rng.standard_normal((3, 5)).astype(np.float32)
w.add_tensor("f32.weight", f32)
record("gguf", "f32.weight", "f32", (3, 5), f32.tobytes())

f16 = rng.standard_normal((4, 8)).astype(np.float16)
w.add_tensor("f16.weight", f16)
record("gguf", "f16.weight", "f16", (4, 8), f16.tobytes())

i32 = np.arange(-6, 6, dtype=np.int32).reshape(2, 6)
w.add_tensor("i32.ids", i32)
record("gguf", "i32.ids", "i32", (2, 6), i32.tobytes())

for qt, name, dt in [(T.Q8_0, "q8.weight", "q8_0"), (T.Q4_0, "q4.weight", "q4_0")]:
    x = rng.standard_normal((2, 64)).astype(np.float32)
    q = quantize(x, qt)
    w.add_tensor(name, q, raw_dtype=qt)
    record("gguf", name, dt, (2, 64), q.tobytes(), dequantize(q, qt).reshape(-1))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()

# ---- safetensors
tensors = {
    "a.weight": ("F32", rng.standard_normal((2, 3)).astype("<f4")),
    "b.bias": ("F16", rng.standard_normal((4,)).astype("<f2")),
    "c.mask": ("BOOL", np.array([1, 0, 1], dtype=np.uint8)),
    "d.scalar": ("I64", np.array(42, dtype="<i8")),
    "e.bf16": ("BF16", (rng.standard_normal((2, 2)).astype("<f4").view("<u4") >> 16).astype("<u2")),
}
header = {"__metadata__": {"format": "pt", "note": "tiny"}}
blob = b""
for name in sorted(tensors):
    dt, arr = tensors[name]
    raw = arr.tobytes()
    header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [len(blob), len(blob) + len(raw)]}
    record("safetensors", name, {"F32": "f32", "F16": "f16", "BOOL": "bool", "I64": "i64", "BF16": "bf16"}[dt],
           arr.shape, raw)
    blob += raw
h = json.dumps(header, separators=(",", ":")).encode()
h += b" " * (-len(h) % 8)
with open("testdata/import/tiny.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(h)) + h + blob)

with open("testdata/import/expected.json", "w", newline="\n") as f:
    json.dump(expected, f, indent=1, sort_keys=True)
    f.write("\n")
