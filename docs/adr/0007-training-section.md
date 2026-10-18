# ADR 0007: Training state as tensors named by a small section

Status: accepted. Date: 2026-10.

## Context

Section kind 5 was reserved for training state from the start, so that a
framework can save a checkpoint and resume from it with the same file it
deploys. The state has two kinds of content: arrays as large as the
weights or larger (optimizer moments, momentum buffers, factored
second-moment estimates), and a handful of scalars (the step, the
hyperparameters per parameter group). GGUF has no training state.
PyTorch's `state_dict` pickles both kinds together in a Python-specific
encoding. safetensors holds only arrays, so frameworks put the scalars in
free-form JSON metadata with no shared names.

## Decision

The arrays are ordinary tensors in the tensor table and weights section.
The training section (spec/training.md) is a directory of names: which
tensors are trainable, which state tensors belong to each and under what
slot name, the parameter group of each, the hyperparameters per group and
named progress counters. The optimizer is named; the update rule is not
encoded. A short table of well-known optimizer, slot, hyperparameter and
counter names fixes their meaning for writers that use them.

The section is never required, it is laid out like the graph section (a
directory of flat tables over a string pool), and its rules are checked by
both readers with one error code, `bad-training`. It arrives in container
minor 2.

## Consequences

- State tensors get everything weights have: memory mapping, alignment,
  per-section hashes, every dtype including the half-precision and
  quantized ones some frameworks store moments in.
- An inference runtime reads a checkpoint as a model and ignores the
  rest. Stripping training state is removing a section and the tensors it
  names.
- The format does not say how to apply an update, so two frameworks agree
  on a checkpoint only through the well-known names. A new optimizer is a
  new name and needs no format change.
- Scalars that are not hyperparameters or counters (a learning-rate
  schedule's internal state, say) go in an extra tensor or in `train.`
  metadata; neither has a fixed meaning across frameworks.
