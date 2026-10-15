# ADR 0002: Graph as flat fixed-width tables

Status: accepted. Date: 2026-10.

## Context

The graph section has to be loaded from a memory-mapped file with no
dependency, validated in one pass, and walked by a runtime in execution
order. The design notes left open whether to encode it as JSON inside the
header or as a binary section. We considered:

- JSON: readable, but requires a parser, loses the fixed-cost validation
  the container has, and large graphs (tens of thousands of nodes after
  unrolling a transformer) become slow to load.
- FlatBuffers (as in TFLite): zero-copy, but adds a schema compiler and a
  runtime library to every reader, and its offsets are 32-bit, which caps
  the size of a single buffer and of the whole table.
- protobuf (as in ONNX): decode and copy, 2 GB limit, dependency.

A graph in spec/primitives.md is small: a node has an opcode, a short list
of input value indices, output value indices and a few attributes. Nothing
in it needs variable-length nesting except subgraphs of `while` and `if`,
which are themselves graphs.

## Decision

The graph section (spec/graph.md) is a set of flat tables with fixed-width
rows: a value table (dtype, shape, kind), a node table (opcode, ranges into
an edge table, ranges into an attribute table), an edge table of u32 value
indices, an attribute table, a subgraph table and a string pool. All
references are indices into tables, never byte offsets or pointers.

Nodes are stored in topological order: every input of a node is a graph
input, a constant, a variable, or the output of an earlier node. A runtime
executes the table top to bottom; no scheduling pass is needed for
correctness.

Table sizes are u32 counts, row sizes are fixed, and the section length must
equal the sum, so the whole section is validated with the same arithmetic
check as the container index. Values and nodes are addressed by index, so a
reference is validated by one comparison against a count.

## Consequences

- A runtime walks the graph with no allocation beyond its own buffers; a
  reader in C is a struct cast after validation.
- Subgraphs are stored as nested ranges in the same tables, so control
  flow costs nothing extra in the reader.
- The encoding is not extensible by adding fields to a row; new node
  attributes go through the attribute table and new row fields need a minor
  version bump with reserved bytes, as in the container.
- Rewriting a graph (fusion, memory planning) means rebuilding the tables.
  Optimised graphs are written as separate sections, not patched in place.
- Tooling needs an inspection command; the format is not readable by eye.
