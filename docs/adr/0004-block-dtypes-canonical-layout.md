# ADR 0004: Block dtypes keep ggml's canonical layouts; kernel interleavings are not dtypes

Status: accepted. Date: 2026-10.

## Context

Quantized models in the GGUF ecosystem are stored in block formats (Q8_0,
Q4_0, Q4_K, Q6_K and others) whose byte layouts are defined by ggml's
reference dequantizers. Fast kernels prefer different in-memory
arrangements: several blocks interleaved so that one SIMD load picks up the
same nibble position of four or eight blocks at once.

llama.cpp at one point exposed those arrangements as file-level types
(Q4_0_4_4, Q4_0_4_8, Q4_0_8_8). Users converted models into them for Arm
CPUs. In December 2024 llama.cpp removed the types and moved to repacking
Q4_0 at load time; files written in the removed types stopped loading.
The lesson: a layout tied to one kernel's register width is not a durable
dtype, and putting it in files breaks them when kernels change.

We also need `dequantize` (spec/primitives.md section 5.33) to be bit-exact
across runtimes, so a model quantized once behaves identically everywhere.

## Decision

- Every block dtype in spec/dtypes.md is defined by its canonical byte
  layout, which for the GGUF-derived types is exactly ggml's `block_*`
  struct, and by a decode procedure given as an ordered sequence of f32
  operations. Decoders MUST be bit-exact to that procedure. Encoders are
  not normative.
- A GGUF import copies blocks byte for byte.
- Kernel-specific interleavings, tilings and repackings are never dtypes
  and never appear in the weights section. A runtime repacks at load time,
  or stores the repacked bytes in the repack-cache section (kind 4), which
  is optional, is keyed by the target kernel, and can be dropped or
  regenerated at any time without changing the model.
- New block dtypes are admitted with a stable upstream definition, a
  bit-exact decode, and test vectors; a format whose layout exists only to
  suit one kernel is not admitted.

## Consequences

- Model files outlive kernels. A kernel can change its preferred layout
  without touching any file.
- Quantized models give identical logits on every conforming runtime for
  the dequantize path, which makes conformance tests meaningful.
- Load time may include a repack pass for runtimes that want an interleaved
  layout. The repack-cache section lets a runtime pay that once per machine.
- The repack-cache section's layout is not yet specified and is reserved
  until a second runtime needs it.
- Encoders may differ in the bits they choose, so two quantizers produce
  different files from the same source; that is accepted, because the
  quantization recipe is recorded in metadata.
