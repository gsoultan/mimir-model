#!/usr/bin/env python3
"""Generate the PyTorch fixture in testdata/pytorch.

Run from the repository root with `python3 -P tools/gen_pytorch_testdata.py`.
Needs NumPy and PyTorch.

Each model of python/tests/torch_models.py (a CNN, a 1-D convolutional
network, a GPT-style and a Llama-style decoder, with deterministic random
weights) is exported with mmf.pytorch to <name>.mmf, with a symbolic batch
or sequence length. expected.json holds two input sets per model, of
different sizes, and PyTorch's outputs for them, as base64 of
little-endian bytes. tiny-llama-kv.mmf is the Llama with a key-value cache;
its runs are a prompt and three decode steps of one session, in order. The Go test in interp and the Python test run the
files and compare with PyTorch; neither needs PyTorch installed.
"""

import base64
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "python", "tests"))

import torch_models as TM  # noqa: E402
from mmf.pytorch import export  # noqa: E402

OUT = os.path.join(ROOT, "testdata", "pytorch")

# Input sets per model: the traced size and a second one, so the symbolic
# dimensions are exercised.
RUNS = {
    "tiny-cnn": [lambda: (torch.randn(1, 3, 16, 16),), lambda: (torch.randn(3, 3, 16, 16),)],
    "audio": [lambda: (torch.randn(1, 4, 40),), lambda: (torch.randn(4, 4, 40),)],
    "tiny-gpt": [lambda: (torch.randint(0, 64, (1, 5)),), lambda: (torch.randint(0, 64, (2, 11)),)],
    "tiny-llama": [lambda: (torch.randint(0, 64, (1, 4)),), lambda: (torch.randint(0, 64, (1, 13)),)],
}
TYPES = {torch.float32: "f32", torch.int64: "i64"}


def arr(t):
    t = t.detach().contiguous()
    a = t.numpy()
    return {"dtype": TYPES[t.dtype], "shape": list(a.shape), "data": base64.b64encode(a.astype(a.dtype.newbyteorder("<")).tobytes()).decode()}


def kv_model():
    """The Llama with a key-value cache: one session runs a prompt of six
    tokens and then three single-token steps, each at its positions, so a
    runtime must keep the cache variables between runs."""
    name = "tiny-llama-kv"
    m = TM.build_kv()
    args, dyn = TM.TinyLlamaKV.example()
    path = os.path.join(OUT, f"{name}.mmf")
    data = export(m, args, path, dynamic_shapes=dyn, metadata={"general.license": "Apache-2.0"})
    m.reset()
    torch.manual_seed(2)
    ids = torch.randint(0, 64, (1, 9))
    runs, t = [], 0
    for n in (6, 1, 1, 1):
        x, pos = ids[:, t:t + n], torch.arange(t, t + n)
        with torch.no_grad():
            out = m(x, pos)
        runs.append({"inputs": {"ids": arr(x), "pos": arr(pos)}, "outputs": {"output": arr(out)}})
        t += n
    print(f"{name}: {len(data)} bytes")
    return {"name": name, "mmf": f"{name}.mmf", "runs": runs}


def main():
    os.makedirs(OUT, exist_ok=True)
    models = []
    for name, cls in TM.MODELS.items():
        m = TM.build(name)
        args, dyn = cls.example()
        path = os.path.join(OUT, f"{name}.mmf")
        data = export(m, args, path, dynamic_shapes=dyn, metadata={"general.license": "Apache-2.0"})
        input_names = list(dyn)
        runs = []
        torch.manual_seed(1)
        for make in RUNS[name]:
            ins = make()
            with torch.no_grad():
                out = m(*ins)
            out = out if isinstance(out, tuple) else (out,)
            onames = ["output"] if len(out) == 1 else [f"output_{i}" for i in range(len(out))]
            runs.append({
                "inputs": {k: arr(v) for k, v in zip(input_names, ins)},
                "outputs": {k: arr(v) for k, v in zip(onames, out)},
            })
        models.append({"name": name, "mmf": f"{name}.mmf", "runs": runs})
        print(f"{name}: {len(data)} bytes")
    models.append(kv_model())
    doc = {
        "torch": torch.__version__.split("+")[0],
        "tolerance": 2e-6,
        "note": "tolerance is relative to the largest absolute output of each run",
        "models": models,
    }
    with open(os.path.join(OUT, "expected.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main()
