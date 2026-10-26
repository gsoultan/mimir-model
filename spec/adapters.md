# mim adapter files v0.1

Status: draft. This document defines adapter files: a mim file that holds
no model of its own but a change to another one, the base model. The only
kind of adapter defined so far is LoRA (low-rank adaptation), which
fine-tunes a large model by training a small pair of matrices next to
each of some of its weights while the weights themselves stay as they
are, quantized or not. The reference implementation is `adapter.go`.

The key words MUST, SHOULD and MAY are to be interpreted as described in
RFC 2119.

## 1. Design

- An adapter is an ordinary mim file: a preamble, an index, a weights
  section and a metadata section. It has no graph, so a runtime never
  mistakes it for a model, and it is a few megabytes where its base model
  is gigabytes, so many adapters can sit next to one base.
- The names are llama.cpp's. Its GGUF LoRA adapters use the same metadata
  keys and tensor names, so a converter can copy one into a mim adapter
  byte for byte, as it copies models.
- The adapter names its base by the SHA-256 of the base model's file, so
  a runtime can refuse one trained on another model. The name is
  optional: an adapter converted from GGUF carries none.

## 2. Metadata

| key | type | meaning |
| --- | --- | --- |
| `general.type` | string | `adapter`. Files without this key, or with another value, are models. |
| `adapter.type` | string | `lora`, the only kind defined |
| `adapter.lora.alpha` | float | the scale α of every pair (section 3) |
| `adapter.base.sha256` | string | optional: the lowercase hex SHA-256 of the base model's file |

Other keys, such as `general.name` and `general.architecture`, MAY be
present and mean what they mean for a model.

## 3. LoRA pairs

Each adapted weight W of the base model, a matrix of shape `[out, in]`
named `<target>` in the base's tensor table, has two tensors in the
adapter:

| tensor | shape | |
| --- | --- | --- |
| `<target>.lora_a` | `[r, in]` | A |
| `<target>.lora_b` | `[out, r]` | B |

r is the pair's rank. Both tensors are f32, f16 or bf16. With the
adapter applied, the base model computes as if W were

    W' = W + (α / r) · B · A

wherever the graph reads W. A runtime MAY compute that without forming
W', for example as `x·Wᵀ + (α/r)·((x·Aᵀ)·Bᵀ)` where W is the weight of a
`linear`, `dequant_matmul` or `dequant_matmul_a8` call, and it MAY refuse an
adapter of a weight its graph reads any other way. A quantized W stays
quantized: only the product changes.

## 4. Rules

A reader MUST reject an adapter with `bad-adapter` if

- `general.type` is not `adapter`, or `adapter.type` is not `lora`;
- `adapter.lora.alpha` is missing, not a number, not positive or not
  finite;
- `adapter.base.sha256` is present and not 64 lowercase hex digits;
- the file has a graph section;
- a tensor's name does not end in `.lora_a` or `.lora_b`, or ends in one
  with nothing before it;
- a `.lora_a` tensor has no `.lora_b` of the same target, or the other
  way round;
- a tensor of a pair is not f32, f16 or bf16, or is not a matrix with no
  zero dimension;
- the first dimension of A, r, is not the last dimension of B;
- there is no pair.

A runtime that applies an adapter to a base model MUST also reject it
with `bad-adapter` if `adapter.base.sha256` is present and is not the
SHA-256 of the base's file, if a target is not a tensor of the base, or
if a target is not a matrix of shape `[out, in]` of its pair.
