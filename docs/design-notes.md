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

## Speed plan after v0.1

Every step of the adoption plan above has landed, and v0.1 is tagged. The
gap that is left is speed on a CPU against llama.cpp on the same Q4_K_M
model (docs/benchmarks.md): generation was 1.7 times slower and the prompt
4 to 5 times slower. A profile of mmf-run showed where the time goes: the
8-bit matrix product kernels took 65% of the CPU, the cores sat idle or
woke up threads for another 20%, and the kernels read the weights from
memory more slowly than the machine allows.

1. **Keep the cores busy and the weights streaming.** A pool of workers
   kept between the few hundred parallel calls of each token, instead of
   new goroutines for each; software prefetch of the weight rows in the
   block kernels; the activations quantized in parallel. Target: match
   llama.cpp at generation, which is bound by memory. Done: 29 tokens per
   second against llama.cpp's 25.
2. **Matrix by matrix for the prompt.** Kernels that multiply one weight
   row by several activation rows at once, decoding each block of weights
   and its scales once for all of them. Target: halve the prompt gap.
   Done, with the serial steps a profile showed next to them (attention
   over the whole cache, permutations): the prompt went from 4.2 to 2.2
   times slower than llama.cpp.
3. **AVX-512 and VNNI.** On CPUs that have them (Intel Xeon since Cascade
   Lake, AMD Zen 4), 512-bit kernels whose integer dot products take one
   instruction. Tried and dropped: 512-bit versions of the four-row
   kernels, bit for bit equal to the AVX2 ones, ran at the same speed for
   Q4_K and 20% slower for Q6_K on the 2.1 GHz Xeon of docs/benchmarks.md
   ('After the first speed work'). With activation blocks of 32, every 32
   products need their own conversion to float32 and multiply-add, and a
   512-bit instruction does twice the work on half the ports, so the count
   of port cycles per product does not change. VNNI saves one integer
   instruction in five.
4. **Activation blocks of 256.** dequant_matmul_a8.v1 fixes the
   activation blocks at 32 elements, where llama.cpp's K-quant kernels use
   blocks of 256 with one scale each. dequant_matmul_a8.v2 adds a `block`
   attribute, 32 by default or 256 (spec/stdlib.md section 3.4), and the
   GGUF importer writes 256 for Q4_K and Q6_K weights. The kernels then
   stay in integers across a whole weight block and convert once instead
   of eight times. Done: the prompt and generation each about 20% faster
   (docs/benchmarks.md, 'Activation blocks of 256'). The prompt is still
   2.1 times slower than llama.cpp, whose build on that machine has AMX
   int8 tiles and repacked weights.
5. **AMX tiles for the prompt.** On CPUs with AMX (Intel Xeon since
   Sapphire Rapids), the Q4_K and Q6_K products of 16 or more prompt rows
   run on int8 tile multiplies: each weight quant times its sub-block scale
   is split into a low and a high byte, two tile products per 64 elements
   give the block's integer sums for 16 weight rows by 32 activation rows,
   and the float32 work stays once per block. The block-256 AVX2 kernels
   now sum each block in integers the same way, so a row's result is bit
   for bit the same whichever kernel ran it and however many tokens were
   run together. Done: the prompt went from 66 to 113 tokens per second,
   1.4 times slower than llama.cpp instead of 2.1 (docs/benchmarks.md,
   'AMX tiles').
6. **ARM.** 8-bit Q4_K and Q6_K kernels for arm64 (NEON with the dot
   product instructions), for Apple Silicon and ARM servers, which ran
   the plain Go loops. Done for activation blocks of 256, one row and four
   rows: SDOT sums the products four at a time, and the kernels return
   what the AVX2 ones return, bit for bit, which a plain Go version of the
   numerics checks on both (internal/kernels/q8k_test.go). Linux reports
   the instructions in the auxiliary vector; every Apple Silicon CPU has
   them. Not yet measured on ARM hardware against llama.cpp.

Each step is measured end to end with tools/bench_generate.py against
llama.cpp on the same machine, and recorded in docs/benchmarks.md.
