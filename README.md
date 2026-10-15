# mmf

mmf (working name) is an open file format for machine learning models. It
aims to compete with ONNX and GGUF: smaller to implement, faster to load, and
able to hold any model, from a CNN to an LLM, for inference and for training.

> **Status: design.** There is no code yet. The ideas so far are in
> [docs/design-notes.md](docs/design-notes.md).

## In one paragraph

A model file holds a graph written in about 30 primitive operations, a
standard library that defines hundreds of common operations (GELU, softmax,
layer norm, attention) as compositions of those primitives, and the weights in
a memory-mappable, safetensors-style layout. A runtime that implements the 30
primitives runs every model; a runtime with faster kernels for named
operations uses them. Quantized dtypes, tokenizers, training state and
optional pre-optimized graphs per device live in the same file.

## Related projects

- [mimir](https://github.com/gsoultan/mimir): pure-Go ML framework, the
  reference runtime.
- [mimir-server](https://github.com/gsoultan/mimir-server): model server that
  serves mmf files and converts ONNX and GGUF into them.
