# ADR 0006: Rename the format to Mimir Model (.mim)

Status: accepted. Date: 2026-10.

## Context

The format was called mmf, a working name. It is the model format of the
mimir framework and of mimir-server, which train, run and serve models
the way PyTorch and TensorFlow do, while this repository stays the format
with its reference reader, writer and runtime, as ONNX and GGUF are. The
name should say which family it belongs to.

## Decision

The format is Mimir Model, mim for short, and files end in `.mim`. The Go
module is `github.com/gsoultan/mim`, the commands are `mim-run`,
`mim-import`, `mim-inspect` and `mim-bench`, and the Python package is
`mim`.

The magic changes with the name: files start with `MIM\0`, and the
metadata and graph sections with `MIMM` and `MIMG`. Readers also accept
the old `MMF\0`, `MMFM` and `MMFG`, and writers never write them. Nothing
else in the layout changes: a file written before the rename differs from
the same file written after it only in those twelve bytes and the section
hashes that cover them.

Names the importer writes inside files, such as the tensor
`mmf.kv_positions` and the metadata key `mmf.context_length`, become
`mim.kv_positions` and `mim.context_length`. No reader looks them up, so
old files keep theirs.

## Consequences

- Every file written before the rename still opens, in Go and in Python;
  the 0.1 compatibility corpus, written before it, is the test.
- A reader released before the rename rejects new files with
  `bad-magic`. This breaks the backward window of spec/versioning.md
  section 3, which major 0 keeps on a best-effort basis (section 5); the
  break is recorded here. There were no releases outside this repository.
- Code that imports `github.com/gsoultan/mmf` changes its import path. The
  repository is renamed to match the module.
