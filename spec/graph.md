# mim graph encoding v0.1

Status: draft. This document defines the binary layout of the graph section
(section kind 3). Operation semantics are in spec/primitives.md and
spec/stdlib.md; the container is in spec/container.md. The reference
implementation is `graph.go`.

The key words MUST, SHOULD and MAY are to be interpreted as described in
RFC 2119.

## 1. Design

The graph is a set of flat, fixed-width tables that reference each other by
index. There are no pointers to fix up and no variable-length records to
walk: a reader locates any table from a 192-byte directory and reads rows in
place. Nodes are stored in topological order, so the ops table of a subgraph
is already an execution order for an interpreter.

All integers are little-endian. All table offsets are relative to the start
of the section and are multiples of 8.

## 2. Header and directory

| offset | size | field |
| --- | --- | --- |
| 0 | 4 | magic `MIMG` (readers also accept the legacy `MMFG`) |
| 4 | 2 | format major (0) |
| 6 | 2 | format minor (1) |
| 8 | 4 | primitive version (1) |
| 12 | 4 | stdlib version (1) |
| 16 | 176 | directory: 11 entries of (u64 offset, u64 count) |

A reader MUST reject a graph whose format major differs from its own, or
whose primitive or stdlib version is newer than it supports. The directory
lists the tables in this fixed order:

| index | table | row size | count means |
| --- | --- | --- | --- |
| 0 | string pool | 1 | bytes |
| 1 | strings | 8 | strings |
| 2 | symbolic dims | 16 | symbolic dimensions |
| 3 | dims | 8 | i64 entries |
| 4 | values | 32 | values |
| 5 | ops | 32 | ops |
| 6 | runs | 4 | u32 entries |
| 7 | attrs | 24 | attributes |
| 8 | subgraphs | 32 | subgraphs |
| 9 | composites | 16 | model-local composites |
| 10 | updates | 8 | variable updates |

Every table MUST lie inside the section and no count may exceed 2^28.

## 3. Tables

### 3.1 strings

Row: u32 pool offset, u32 byte length. Each string MUST be valid UTF-8 and
at most 4096 bytes. String 0 MUST be the empty string, so a value with no
name refers to string 0.

### 3.2 symbolic dims

Row: u32 name (string index), u32 reserved (0), i64 bound. `bound` is the
largest size the dimension can take, or -1 when unbounded. Symbolic
dimensions are shape-polymorphic but never data-dependent (primitives §2.3).

### 3.3 dims

A flat array of i64. A shape or an `ints` attribute is a run (offset, count)
into this table. In a shape, a value >= 0 is a literal size and a value
d < 0 refers to symbolic dimension `-1 - d`.

### 3.4 values

| offset | size | field |
| --- | --- | --- |
| 0 | 4 | name (string index) |
| 4 | 4 | dtype (spec/dtypes.md) |
| 8 | 4 | shape offset into dims |
| 12 | 1 | rank (0..8) |
| 13 | 1 | kind |
| 14 | 1 | flags |
| 15 | 1 | reserved (0) |
| 16 | 4 | ref |
| 20 | 12 | reserved (0) |

Kinds: 0 intermediate (produced by an op), 1 input (supplied by the caller),
2 output (produced by an op and returned), 3 weight (`ref` is a tensor index
in the file's tensor table), 4 variable (persistent state; `ref` is the tensor
holding the initial value), 5 const (a tensor in the weights section that is
not a trained parameter, for example rotary tables).

A value of kind weight, variable or const MUST refer to a tensor with the
same dtype and rank whose literal dimensions match; a symbolic dimension in
the value matches any size.

Flags: bit 0 is ZERO_INIT; the other bits are reserved and MUST be 0. A
variable with ZERO_INIT set starts as all zero bytes (+0 for floats, 0 for
integers, false for bool) instead of a stored tensor: its `ref` MUST be 0
and is not a tensor index, its shape MUST be literal, and its dtype MUST
NOT be a block dtype. ZERO_INIT on a value of any other kind is invalid. A
key-value cache is the common case: it starts empty, and storing its zeros
would make the file larger by the size of the cache. A runtime allocates
the variable when it creates a session and SHOULD check the allocation's
size first (spec/security.md).

### 3.5 ops

| offset | size | field |
| --- | --- | --- |
| 0 | 2 | opcode (primitives §1) |
| 2 | 2 | reserved (0) |
| 4 | 4 | inputs run offset |
| 8 | 4 | input count |
| 12 | 4 | outputs run offset |
| 16 | 4 | output count |
| 20 | 4 | attrs offset |
| 24 | 4 | attr count |
| 28 | 4 | reserved (0) |

### 3.6 runs

A flat array of u32 value indices. Op inputs, op outputs, subgraph inputs and
subgraph outputs are runs into this table.

### 3.7 attrs

| offset | size | field |
| --- | --- | --- |
| 0 | 4 | key (string index) |
| 4 | 1 | type |
| 5 | 3 | reserved (0) |
| 8 | 4 | count (`ints` only, else 0) |
| 12 | 4 | reserved (0) |
| 16 | 8 | payload |

Types and payloads: 1 int (i64), 2 ints (offset into dims; `count` entries),
3 float (f64 bits), 4 string (string index), 5 dtype (dtype id), 6 subgraph
(subgraph index, never 0).

### 3.8 subgraphs

| offset | size | field |
| --- | --- | --- |
| 0 | 4 | first op index |
| 4 | 4 | op count |
| 8 | 4 | inputs run offset |
| 12 | 4 | input count |
| 16 | 4 | outputs run offset |
| 20 | 4 | output count |
| 24 | 8 | reserved (0) |

Subgraph 0 is the main graph. Every op MUST belong to exactly one subgraph
and the op ranges MUST NOT overlap. Every subgraph other than 0 MUST be
referenced exactly once, by one `subgraph` attribute or one composite.

### 3.9 composites

Row: u32 name (string index), u32 version, u32 subgraph, u32 reserved (0).
A model-local composite (stdlib §1) whose body is the given subgraph. The
subgraph's inputs and outputs are the composite's signature.

### 3.10 updates

Row: u32 variable (value index), u32 new value (value index). The variable
MUST be of kind variable, appear at most once, and the new value MUST have
the same dtype and shape (primitives §7).

## 4. Structural rules

A reader MUST reject a graph that violates any of these:

1. Every value of kind intermediate or output is produced by exactly one op
   output or one subgraph input; values of other kinds are produced by none.
2. Within a subgraph, every op input is defined before the op: it is a
   subgraph input, an output of an earlier op in the same subgraph, a value
   defined earlier in an enclosing subgraph, or a top-level input, weight,
   variable or const.
3. Subgraph outputs are defined within the subgraph by the time it ends.
4. The main graph's inputs are of kind input and its outputs of kind output.
5. Nested subgraph inputs are of kind intermediate.
6. Opcodes, dtypes, attribute types and kinds are known values; reserved
   bytes are zero.

Type and shape inference per operation is defined by spec/primitives.md and
is checked separately from the structural rules above.

## 5. Notes

- A composite body is referenced from `call` by `(name, version)`; the
  standard library's own bodies are not stored in model files.
- The text form printed by `mlm-inspect` is for humans and tools; the binary
  form is the only normative one.
