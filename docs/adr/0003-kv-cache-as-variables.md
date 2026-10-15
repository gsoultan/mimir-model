# ADR 0003: KV cache as declared variables with scatter writes

Status: accepted. Date: 2026-10.

## Context

An autoregressive model keeps a key-value cache across decode steps. Formats
and runtimes express this in two ways:

- Explicit past and present tensors: the graph takes `past_key_values` as
  inputs and returns `present_key_values` as outputs, and the host feeds
  one run's outputs into the next. ONNX Runtime GenAI does this, with
  extra configuration to tell the runtime which inputs and outputs form a
  pair so it can share buffers. The pairing is outside the graph, each
  framework invents its own naming, and the graph itself describes a copy
  of the whole cache per step.
- A runtime-owned cache: the architecture is hard-coded and the cache is an
  implementation detail (llama.cpp). This does not work for a format whose
  goal is to run a new architecture on an unchanged runtime.

The primitive set already has `scatter`, and spec/primitives.md section 7
defines graph variables that persist across runs within a session and are
replaced atomically at the end of a run.

## Decision

A key-value cache is a variable: a value of kind VARIABLE with a fixed
dtype and shape, declared in the graph with an initial value. Each decode
step reads the variable like an input, produces the updated cache with
`scatter` along the sequence axis at the current position, and lists the
pair (variable, new value) in the graph's updates. The runtime applies all
updates after the run, atomically; a failed run changes nothing.

The in-place rule is normative as an optimisation: when the new value of a
variable is a `scatter` whose data operand is that same variable, and no
other operation reads the variable after the scatter, a runtime SHOULD
perform the scatter in place on the variable's storage. The observable
result MUST equal the functional definition.

## Consequences

- The cache has one definition that every runtime understands; no
  per-framework naming of past and present tensors, no host-side plumbing.
- A runtime that implements the in-place rule gets the same cost as a
  hand-written cache: one row write per layer per step, no copy.
- A runtime that does not implement it still runs the model correctly, at
  the cost of a full cache copy per step.
- A cache starts empty, so its variable carries the ZERO_INIT flag
  (spec/graph.md) instead of a tensor of zeros, and costs no file space.
- The cache size is fixed at graph load (`T_max`); growing it means
  re-binding the variable. Paged or ring-buffer caches are expressed by
  changing the index tensor fed to `scatter`, not by new primitives.
- Session state is explicit, so the same mechanism serves other stateful
  models (RNN hidden states, streaming convolution buffers) and training
  state such as optimizer moments.
