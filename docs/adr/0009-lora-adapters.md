# ADR 0009: LoRA adapters as files of their own

Status: accepted. Date: 2026-10.

## Context

Fine-tuning a language model of billions of parameters on someone's own
text is done with LoRA: the weights stay frozen, quantized or not, and a
pair of small matrices next to some of them is trained. The pairs are a
few megabytes. A server that keeps several fine-tunes of one base model
should store the base once and each fine-tune as its pairs. llama.cpp
keeps them in a GGUF file of their own, with `general.type = adapter`;
PEFT keeps them in safetensors with an `adapter_config.json`.

The alternatives were a model file whose graph adds the low-rank products
(which copies the base's weights into every fine-tune), and merging the
pairs into the weights (which means decoding quantized weights, adding,
and quantizing again, losing accuracy and the size advantage).

## Decision

An adapter is a mim file with no graph, `general.type = adapter` and
`adapter.type = lora` in its metadata, and two tensors per adapted weight,
`<weight>.lora_a` and `<weight>.lora_b`, with llama.cpp's names and
layouts (spec/adapters.md). Its meaning is the effective weight
`W + α/r · B·A`; how a runtime computes that is up to it. The base is
named, optionally, by the SHA-256 of its file. Readers check the file
with one error code, `bad-adapter`.

## Consequences

- A GGUF LoRA adapter can be copied into a mim adapter byte for byte.
- A runtime applies an adapter when it opens a model; nothing in the
  model's graph changes, so one base serves its own answers and those of
  every adapter.
- Only weights a graph reads as the weight of a matrix product can be
  adapted cheaply. Adapting an embedding table or a convolution is
  allowed by the format but a runtime may refuse it.
- Other kinds of adapter (IA³, prefix tuning) would be new values of
  `adapter.type` with their own tensor names.
