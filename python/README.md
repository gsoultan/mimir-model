# mmf for Python

A second runtime for mmf, written from the specification in `../spec`
and sharing no code with the Go reference. It reads and validates files,
decodes every dtype, evaluates the 33 primitives and the standard library,
and runs graphs with variables, `while`, `if` and model-local composites.
Its only dependency is NumPy.

```python
from mmf.container import File
from mmf.executor import Program
from mmf.tensor import Tensor

with File.open("model.mmf") as f:          # mmap, validate, no copies
    print(f.metadata()["general.name"])     # ("string", "...")
    s = Program.load(f).session()           # holds variables, such as a KV cache
    out = s.run({"x": Tensor.from_bytes("f32", [1, 4], data)})
```

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

The tests run the primitive conformance corpus, the hostile files of
spec/container.md section 9.1, the compatibility corpus of every released
version (including the tiny Llama, whose logits must match the Go
executor's), the tiny Llama's greedy continuation against the float64
oracle of `tools/gen_llama_testdata.py`, and the CNN in `testdata/onnx`,
converted from ONNX by the Go importer, against onnxruntime.
