# mmf versioning

Status: draft. This document defines how the parts of the format are
versioned, what a reader of one version accepts from another, and how long
that compatibility is kept.

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT,
RECOMMENDED, MAY and OPTIONAL are to be interpreted as described in RFC 2119.

## 1. Three independent versions

An mmf file carries three version numbers. They change independently.

| part | where stored | current |
|---|---|---|
| container | preamble `major.minor` (spec/container.md) | 0.1 |
| primitives | graph section header (spec/graph.md) | 0.1 |
| standard library | per composite, as the suffix of its name `name.vN` (spec/stdlib.md) | per composite |

### 1.1 Container

The container version describes the byte layout: preamble, index, section
table, tensor entries and the rules of spec/container.md.

- A reader MUST accept any file whose major version equals the major version
  it implements, whatever the minor version.
- A reader MUST reject a file with a different major version with
  `unsupported-version`.
- A minor version MAY add section kinds, flags that are not required for
  correct reading, hash algorithms, and metadata keys. It MUST NOT change the
  meaning or size of an existing field, and it MUST NOT make a file that was
  valid under an earlier minor version invalid.
- A reader that meets a section kind it does not know skips it unless the
  section carries the required flag. A writer sets the required flag only on
  a section a reader cannot do without; a new optional section therefore
  never breaks an older reader.
- A major version MAY change anything. It is a new format that shares the
  magic.

### 1.2 Primitives

The primitive version describes the operation set of spec/primitives.md.

- Adding a primitive takes a new opcode and a minor bump. A runtime that
  reads a graph with a newer minor version and meets an opcode it does not
  implement MUST reject the graph with `bad-graph`; it MUST NOT guess.
- Changing the semantics, shape rule, dtype rule or accuracy requirement of
  an existing primitive is a major bump. The opcode keeps its old meaning in
  the old major version and is not reused in the new one.
- A runtime MUST accept any graph whose primitive major version equals the
  one it implements.

### 1.3 Standard library

Each composite is named `name.vN`. The body of `name.vN` never changes once
published. A change in semantics, inputs, attributes or numerical behaviour
is a new composite `name.v(N+1)`. A runtime's fast path for `name.vN` MUST
implement exactly `name.vN`. Removing a composite from the standard library
does not remove its definition; it remains in spec/stdlib.md marked
withdrawn and files that call it stay valid.

## 2. Identifiers are never reused

- Dtype ids (spec/dtypes.md) are never reused. A withdrawn dtype keeps its
  id.
- Opcodes (spec/primitives.md) are never reused.
- Section kinds, hash algorithm numbers, section flag bits and error codes
  (spec/container.md) are never reused.
- Composite names with their version suffix are never reused.

A proposal that reuses an identifier MUST be rejected in review.

## 3. Compatibility promise

- Forward in time: a file written by any released version X of the format
  MUST remain readable, with identical results, by readers released during
  the 5 years after X. A reader released in that window that drops support
  for X is not conforming.
- Backward in time: a reader released at version X MUST accept files written
  by versions up to 2 years newer than X within the same major version,
  skipping optional sections and metadata it does not know. This is the
  window StableHLO uses, and it lets a deployed runtime load models produced
  by a newer exporter without a runtime upgrade.
- A major version bump of the container or the primitives ends neither
  window for the old major version; the old major continues to be read for
  its full 5 years.

Changes that would break either window are deferred to the next major
version.

## 4. Compatibility corpus

The repository holds a corpus of serialized files under `testdata/compat`,
one directory per released version:

```
testdata/compat/<container>-<primitives>/
    *.mmf            files written by the reference writer at that version
    *.expected.json  the tensors, sections and graph outputs a reader must see
```

Starting with v0.1, each release adds its files to the corpus and never
modifies or removes earlier directories. The test suite reads every file in
the corpus with the current reader and compares against the expected
output. A change that fails the corpus is a compatibility break and needs a
major version and an ADR.

### 4.1 Expected files (non-normative)

`<name>.expected.json` describes `<name>.mmf` as UTF-8 JSON with
`"format": "mmf-compat"` and `"version": 1`. It lists:

- `container`: the preamble's major and minor version and data alignment;
- `sections`: the section table in table order, each with its kind,
  required flag, hash algorithm and length;
- `tensors`: the tensor table in name order, each with its name, dtype
  name, shape, offset in the weights section and the SHA-256 of its bytes;
- `metadata`: every entry in key order with its type name and value (int
  and uint as JSON integers, float as the hex of its little-endian
  binary64, bytes as hex, floats as a list of such hex strings);
- `graph`: the names of the main graph's inputs and outputs, when the file
  has a graph;
- `runs`: for a file with a graph, a sequence of runs in one session, so
  variables carry over. Each run gives its inputs and expected outputs as
  tensors in the format of the conformance corpus (spec/conformance.md
  section 3). An output's `check` is `bits` (the same bytes) or `close`
  (every element within `tol` times the largest magnitude of the expected
  tensor, for outputs whose reductions the spec leaves unordered).

A reader passes a file when it opens it, verifies its hashes and reports
exactly the listed container, sections, tensors and metadata; a runtime
passes it when it also reproduces the runs. The reference implementation
of these checks is the Go package `compat`; `internal/compatgen` writes a
new version's directory and refuses to touch an existing one.

## 5. Pre-release status

Versions with major 0 are pre-release. During major 0 the compatibility
promise of section 3 applies from v0.1 onward on a best-effort basis; a
breaking change in a 0.x minor version MUST be recorded in an ADR and in
the corpus, and MUST be accompanied by a converter from the previous minor.
