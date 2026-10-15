# mmf design notes

Notes from the first brainstorm. Nothing here is final; spec v0 comes next.

## Goal

An open model format that competes with ONNX and GGUF:

- **Lightweight:** a complete runtime is small to write.
- **Fast:** zero-copy loading and room for pre-optimized graphs.
- **Broad:** covers a large number of operations and any kind of model.

## Small core, broad coverage

"Handles many operations" is not the same as "has many operators".

- The spec defines about **30 primitive operations** (the same set mimir
  builds on: elementwise, reductions, movement, matmul, gather/scatter,
  and a few more).
- A **standard library** ships with the spec and defines hundreds of
  operations (GELU, softmax, layer norm, attention, every activation) as
  compositions of the primitives.
- A model file can use any standard-library operation, and any runtime that
  implements the 30 primitives runs them all.
- Each composite carries its name, so a runtime with a hand-tuned kernel
  (a fused GELU, flash attention) uses it as a **fast path**, and a runtime
  without one falls back to the primitives. Correctness never depends on the
  fast path.

The hard discipline: keep the primitive set at about 30 when people ask for
more. Operator creep is how ONNX reached about 190 operators.

## File layout

- Header (JSON, like safetensors), then raw tensor data, aligned so the file
  can be **memory-mapped and run** with no parsing or copying and no size
  limit.
- Sections in one file:
  - portable graph (primitives + standard-library calls)
  - weights
  - optional pre-optimized graph per target device (fused, memory-planned)
  - tokenizer (vocab, merges) and chat template
  - pre- and post-processing
  - optional training state: optimizer state, progress, LoRA adapters
  - metadata: lineage, metrics, quantization recipe, seeds and RNG scheme

## Dtypes

- float32, int32 (what mimir has today), bf16, f16, int8.
- Block-quantized formats in the GGUF family (Q4_K, Q5_K, Q8_0 and similar)
  as first-class dtypes, so a GGUF import keeps blocks as they are.

## Compared with ONNX

| | ONNX | mmf |
| --- | --- | --- |
| Operators a runtime must implement | about 190, versioned opsets | about 30 primitives |
| Encoding | protobuf, 2 GB limit, parse and copy | header + aligned raw data, mmap |
| Training state | no | yes |
| Quantization | limited | first-class block formats |
| Tokenizer, preprocessing | side files | in the file |

## Compared with GGUF

GGUF is a weights-plus-metadata container for llama.cpp. It has no graph:
the architecture is a metadata string and the model is hard-coded in C++.

Must match: block quantization, single memory-mapped file, tokenizer and chat
template inside the file.

Where mmf wins:

- A real graph, so a new architecture runs on an unchanged runtime.
- Not limited to LLMs.
- Training and fine-tuning state in the same file.
- The quantization recipe is recorded, so results are reproducible.

## How a format wins: adoption

ONNX has Microsoft, Meta and NVIDIA behind it, so features alone are not
enough.

1. **Importers from day one:** ONNX first, GGUF next. People convert, they do
   not abandon their models.
2. **PyTorch exporter:** a small Python package built on `torch.export`. The
   on-ramp for most users.
3. **Public spec with a conformance test suite:** what makes it a standard
   rather than a file layout.
4. **Fast reference runtime (mimir)** and a showcase (mimir-server): upload
   ONNX, get a faster mmf file, serve it, with benchmarks against ONNX
   Runtime and llama.cpp.
5. **A second runtime**, even a minimal C or Rust reader, to prove the spec
   stands on its own.

Niche to own first: deploying models without Python, as one static binary,
on servers and at the edge.

## Repository split

- **mmf** (this repo): spec, standard library, conformance corpus, reference
  Go reader and writer. Depends on nothing.
- **mimir**: imports mmf to load and save models; the fast reference runtime.
- **mimir-server**: imports both; holds the ONNX and GGUF importers and the
  PyTorch exporter.

## Open questions

- Final name. "mmf" ties it to mimir; a neutral name may help other runtimes
  adopt it.
- Graph encoding inside the header: JSON for readability, or a compact
  binary section for very large graphs.
- How control flow (loops, conditionals) and dynamic shapes are expressed.
- Versioning rules for the primitive set and the standard library.

## Next step

Write spec v0: the primitive set with exact semantics, the header schema, the
dtype list, and a first conformance corpus generated from mimir's golden
tests.
