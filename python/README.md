# mim for Python

A second runtime for mim, written from the specification in `../spec`
and sharing no code with the Go reference. It reads and validates files,
decodes every dtype, evaluates the 33 primitives and the standard library,
and runs graphs with variables, `while`, `if` and model-local composites.
Its only dependency is NumPy.

```python
from mim.container import File
from mim.executor import Program
from mim.tensor import Tensor

with File.open("model.mim") as f:          # mmap, validate, no copies
    print(f.metadata()["general.name"])     # ("string", "...")
    s = Program.load(f).session()           # holds variables, such as a KV cache
    out = s.run({"x": Tensor.from_bytes("f32", [1, 4], data)})
```

## Writing files and exporting from PyTorch

`mim.writer` writes files: `Writer` packs tensors canonically, sorts the
tensor table, hashes every section and writes atomically;
`encode_metadata` and `GraphBuilder` produce the metadata and graph
sections.

`mim.pytorch` exports a PyTorch model through `torch.export`:

```python
from torch.export import Dim
from mim.pytorch import export

model.eval()
export(model, (x,), "model.mim", dynamic_shapes={"x": {0: Dim("batch", max=64)}})
```

Each ATen operator becomes primitives or a standard-library composite, so
the output runs on any mim runtime. Parameters are stored byte for byte
under their state-dict names, buffers become constants, and operations
that depend only on buffers are computed at export time. Symbolic
dimensions keep the names of the `Dim` objects. An operator the exporter
cannot express raises `ExportError` naming it; `mim.pytorch.SUPPORTED`
lists the ones it handles. The exporter needs PyTorch 2.6 or later
(`pip install "mim[torch]"`); the rest of the package does not.

A buffer the model writes to in `forward`, such as a static key-value
cache, becomes a graph variable (ADR 0003), and its value at the end of
the call becomes the variable's update; a buffer of zeros is stored as
ZERO_INIT and takes no file space. Writes with `index_copy_` and with
index assignment of one index tensor (`cache[:, :, pos] = k`) become
`scatter`, which a runtime can apply in place. A session keeps the cache
between runs, so a prompt followed by one token per step gives the logits
of the full forward pass:

```python
s = Program.load(File.open("model.mim")).session()
s.run({"ids": prompt, "pos": positions})   # i64 tensors [1, n] and [n]: 0..n-1
s.run({"ids": next_id, "pos": position})   # [1, 1] and [1]: n, one decode step
```

Positions are tensors, not Python integers, so one export serves every
step; the model attends over the whole cache with a mask of the positions
written so far (see `TinyLlamaKV` in `tests/torch_models.py`). Mutating an
input is rejected.

## How it computes

- Primitives on f32, f16 and bf16 compute in float64 and round once to the
  output dtype. For `neg`, `add`, `mul`, `recip` and `sqrt` this is
  correctly rounded, because float64 has more than twice the precision of
  each of these formats plus two bits. Subnormals are kept: the runtime is
  in IEEE mode for every float dtype.
- Block dtypes decode with the ordered float32 procedures of
  spec/dtypes.md, so `dequantize` is bit-exact.
- Composites return their reference result (spec/stdlib.md section 1.3):
  the canonical body evaluated in float64 and rounded once. Tolerances are
  stated against that reference, so it passes every one.
- The reader applies the checks of spec/container.md in the spec's order
  and reports the spec's error codes.

## Tests

```
python3 -m pip install numpy
python3 -m unittest discover -s python/tests
```

With PyTorch installed, the exporter tests also run: they export the
models of `tests/torch_models.py` and compare with PyTorch.

The tests run the primitive conformance corpus, the hostile files of
spec/container.md section 9.1, the compatibility corpus of every released
version (including the tiny Llama, whose logits must match the Go
executor's), the tiny Llama's greedy continuation against the float64
oracle of `tools/gen_llama_testdata.py`, and the CNN in `testdata/onnx`,
converted from ONNX by the Go importer, against onnxruntime, and the
models in `testdata/pytorch`, exported by `mim.pytorch`, against PyTorch.
