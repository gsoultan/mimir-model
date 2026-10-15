# ADR 0001: Binary container with a 64-byte preamble and a binary index

Status: accepted. Date: 2026-10.

## Context

The container has to be memory-mapped and used in place, validated
completely before any tensor is touched, and safe against hostile input.
The formats we compared each fail one of these:

- safetensors stores its header as JSON. The header is parsed with a
  general JSON parser before any check can be made, its size is bounded only
  by convention, and a model with a hundred thousand tensors spends tens of
  milliseconds in the parser before the first byte of weight data is
  reachable. Offsets are JSON numbers, so overflow checks depend on the
  parser's number handling.
- ONNX uses protobuf. A protobuf message is limited to 2 GB and must be
  decoded and copied; weights cannot be used in place.
- TFLite uses FlatBuffers with 32-bit offsets, so a single file cannot
  address more than 4 GB of tensor data without external buffers.
- ExecuTorch's `.pte` format has a small fixed prelude that locates the
  program and a data segment with aligned, directly usable tensor storage.
  It is the shape we wanted, without the FlatBuffers dependency.

## Decision

The container is a fixed 64-byte preamble followed by a binary index of
fixed-width entries, then sections, then an aligned weights section
(spec/container.md). The preamble holds magic, version, byte order mark,
data alignment, the index location, the weights location and the file
length. The index is a 16-byte header, 64-byte section entries, 96-byte
tensor entries and a string pool, with a 256 MiB cap. All offsets are 64
bits. The index length is derived from the counts by one multiplication, so
validation is arithmetic, not parsing.

## Consequences

- A reader is a few hundred lines in any language with no dependencies. The
  reference Go reader is under 500 lines including validation.
- Open time is a 64-byte read, one index read, and a linear pass over the
  entries. Lookup by name is a binary search.
- Files have no size limit other than the 64-bit offsets, and the index
  alone is capped.
- The format is not human-readable; an inspection tool is required. A
  `mmf dump` command will ship with the reference implementation.
- Adding a field means a new minor version with a new reserved-bytes rule,
  rather than adding a JSON key. Extension goes through sections and
  metadata, not the preamble.
