# mmf

mmf (working name) is an open file format for machine learning models. It
aims to compete with ONNX and GGUF: smaller to implement, faster to load, and
able to hold any model, from a CNN to an LLM, for inference and for training.

> **Status: spec v0.1 draft, with a Go reference reader and writer.** The
> design rationale is in [docs/design-notes.md](docs/design-notes.md) and
> [docs/adr](docs/adr); the normative text is in [spec](spec).

## In one paragraph

A model file is a binary container: a 64-byte preamble, a binary index, a
graph written in 33 primitive operations, typed metadata, and the weights in
a page-aligned, memory-mappable region. A standard library defines hundreds of
common operations (GELU, softmax, layer norm, attention) as compositions of
the primitives, each with a name and version so a runtime with a faster
kernel can use it. A runtime that implements the 33 primitives runs every
model. Block-quantized dtypes are specified byte for byte, and the file is
treated as untrusted input throughout.

## Layout of this repository

| path | contents |
| --- | --- |
| `spec/container.md` | the file container: preamble, index, sections, tensors, error codes |
| `spec/graph.md` | the graph section: flat tables, values, ops, subgraphs |
| `spec/primitives.md` | the 33 primitives, `call`, `while`, `if`, variables |
| `spec/stdlib.md` | the standard library of named composites |
| `spec/dtypes.md` | plain and block-quantized element types |
| `spec/metadata.md` | the typed key-value section |
| `spec/versioning.md`, `spec/security.md` | compatibility promise and threat model |
| `*.go` | reference reader and writer, zero dependencies |
| `cmd/mmf-inspect` | prints a file's index, metadata and graph |
| `docs/adr` | decision records |

## Using the Go package

```go
f, err := mmf.Open("model.mmf")   // mmap, validate, no copies
defer f.Close()
t, _ := f.Tensor("layers.0.attn.q.weight")
data := f.TensorData(t)           // a slice into the mapping
g, err := f.Graph()               // decoded, validated graph
```

Writing:

```go
w := mmf.NewWriter()
w.AddTensor("w", mmf.F32, []uint64{2, 3}, bytes)
w.AddSection(mmf.SectionGraph, true, graphBytes)
w.WriteFile("model.mmf")          // temp file, fsync, rename
```

Run `go test ./...` for the test suite, including a hostile-file corpus and
fuzz targets for the container, metadata and graph parsers.

## Related projects

- [mimir](https://github.com/gsoultan/mimir): pure-Go ML framework, the
  reference runtime.
- [mimir-server](https://github.com/gsoultan/mimir-server): model server that
  serves mmf files and converts ONNX and GGUF into them.
