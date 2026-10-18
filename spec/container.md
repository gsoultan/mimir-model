# Mimir Model (mim) container v0.2

Status: draft. This document defines the byte layout of a mim file: the
preamble, the index, the section table, the tensor index and the weights
section. It is the normative description of what the reference reader
(`file.go`) accepts and what the reference writer (`writer.go`) produces.
Every constant in `format.go` appears here with the same value.

Element types are defined in spec/dtypes.md. The graph section is defined in
spec/graph.md. The metadata section is defined in spec/metadata.md. Version
rules are in spec/versioning.md and the threat model in spec/security.md.

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT,
RECOMMENDED, MAY and OPTIONAL are to be interpreted as described in RFC 2119.

## 1. Conventions

- All multi-byte integers are little-endian. `u8`, `u16`, `u32` and `u64`
  are unsigned integers of 1, 2, 4 and 8 bytes. There are no signed fields.
- Offsets are byte offsets. "Offset relative to X" means the byte offset
  from the first byte of X.
- A range `[off, off+len)` is "in bounds" of a region of `size` bytes when
  `off <= size` and `len <= size - off`, evaluated without overflow. Readers
  MUST evaluate bounds this way; `off + len <= size` overflows for hostile
  input.
- "Multiple of N" includes zero. "Power of two" excludes zero.
- A reader is "conforming" when it accepts every valid file defined here and
  rejects every invalid file with the error code given in section 9.

## 2. File structure

A file has four kinds of region, in this order of appearance in a file
written by the reference writer:

| region | offset | length |
|---|---|---|
| preamble | 0 | 64 |
| index | preamble.index_offset | preamble.index_length |
| sections other than weights | section_entry.offset | section_entry.length |
| weights section | preamble.data_offset | preamble.data_length |

The reader does not require this order. It requires that no two regions
overlap (section 5.4). Bytes between regions are padding and MUST be ignored
by readers; the reference writer writes zeros.

## 3. Preamble

The preamble is the first 64 bytes of the file (`PreambleSize = 64`). A file
shorter than 64 bytes is rejected with `truncated`.

| offset | size | field | value and constraint |
|---|---|---|---|
| 0 | 4 | magic | bytes `4D 49 4D 00` (`"MIM\0"`). Readers also accept `4D 4D 46 00` (`"MMF\0"`), the magic of files written before the format was renamed from mmf; writers MUST NOT write it. Otherwise `bad-magic`. |
| 4 | 2 | major | u16 container major version. v0 readers accept 0 only; otherwise `unsupported-version`. |
| 6 | 2 | minor | u16 container minor version. Any value is accepted (spec/versioning.md). The reference writer writes 2. Minor 2 defines the training section (kind 5); minor 1 files are the same layout without it. |
| 8 | 4 | byte order mark | u32 `0x0A0B0C0D`, stored as bytes `0D 0C 0B 0A`. A reader that decodes `0x0D0C0B0A` is looking at a file written big-endian and MUST report `big-endian`. Any other value is `bad-byte-order-mark`. v0 is little-endian only. |
| 12 | 4 | data alignment | u32. MUST be a power of two, at least 4096 (`MinDataAlignment = 4 KiB`) and at most 16777216 (`MaxDataAlignment = 16 MiB`). Otherwise `bad-alignment`. The reference writer defaults to 16384 (`DefaultDataAlignment = 16 KiB`). |
| 16 | 8 | index offset | u64. MUST be at least 64 and a multiple of 8 (`SectionAlignment`). Otherwise `misaligned`. The reference writer writes 64. |
| 24 | 8 | index length | u64. MUST be at most 268435456 (`MaxIndexLength = 256 MiB`); otherwise `index-too-large`. `[index offset, +index length)` MUST be in bounds of the file; otherwise `out-of-bounds`. |
| 32 | 8 | data offset | u64 offset of the weights section. MUST be a multiple of data alignment; otherwise `misaligned`. |
| 40 | 8 | data length | u64 length of the weights section. `[data offset, +data length)` MUST be in bounds of the file; otherwise `out-of-bounds`. |
| 48 | 8 | file length | u64. MUST equal the actual size of the file in bytes. If the file is shorter than this value the reader reports `truncated`; if it is longer, `bad-length`. |
| 56 | 8 | reserved | MUST be zero. Otherwise `reserved-non-zero`. |

The reader performs the preamble checks in the order of the table, except
that the byte order mark is checked before the major version, and the file
length is checked before the index and data fields. Every later bounds check
in this document is made against the file length, which by then equals the
actual size.

## 4. Index

The index occupies `[index offset, +index length)` and has three parts in
this order: a 16-byte header, the section table, the tensor table, then the
string pool. The index is at most 256 MiB, so a reader can hold it in memory
and validate it completely before touching any other byte of the file.

### 4.1 Index header

The header is 16 bytes (`IndexHeaderSize = 16`). An index shorter than 16
bytes is rejected with `truncated`.

| offset | size | field |
|---|---|---|
| 0 | 4 | u32 section count `ns` |
| 4 | 4 | u32 tensor count `nt` |
| 8 | 8 | u64 string pool length `pool` |

The index length MUST equal exactly

```
16 + 64 * ns + 96 * nt + pool
```

and `pool` MUST be at most 256 MiB. Otherwise `bad-length`. Readers MUST
compute this expression in 64-bit arithmetic; with the u32 counts and the
256 MiB cap on `pool` it cannot overflow.

### 4.2 Section entries

Immediately after the header come `ns` section entries of 64 bytes each
(`SectionEntrySize = 64`).

| offset | size | field | constraint |
|---|---|---|---|
| 0 | 4 | u32 kind | section 5.1 |
| 4 | 4 | u32 flags | only bit 0 is defined (section 5.2). Any other bit set is `reserved-non-zero`. |
| 8 | 8 | u64 offset | absolute offset of the section in the file |
| 16 | 8 | u64 length | length of the section in bytes |
| 24 | 1 | u8 hash algorithm | section 5.3 |
| 25 | 7 | reserved | MUST be zero; otherwise `reserved-non-zero` |
| 32 | 32 | hash | the section hash, or zero when the algorithm is 0 |

The order of section entries is not significant. The reference writer puts
the weights section first and then the others in the order they were added.

### 4.3 Tensor entries

Immediately after the section entries come `nt` tensor entries of 96 bytes
each (`TensorEntrySize = 96`).

| offset | size | field | constraint |
|---|---|---|---|
| 0 | 4 | u32 name offset | relative to the start of the string pool |
| 4 | 4 | u32 name length | `[name offset, +name length)` MUST be in bounds of the pool; otherwise `out-of-bounds` |
| 8 | 4 | u32 dtype | a dtype id defined in spec/dtypes.md; otherwise `bad-dtype` |
| 12 | 1 | u8 rank | at most 8 (`MaxRank`); otherwise `bad-shape` |
| 13 | 3 | reserved | MUST be zero; otherwise `reserved-non-zero` |
| 16 | 64 | u64 dims[8] | `dims[0..rank)` are the shape, row-major, last axis fastest. `dims[rank..8)` MUST be zero; otherwise `reserved-non-zero` |
| 80 | 8 | u64 offset | relative to the start of the weights section |
| 88 | 8 | u64 length | byte length of the tensor data |

Rank 0 is a scalar with one element. A dimension MAY be zero.

### 4.4 String pool

The string pool is the last `pool` bytes of the index. It holds the tensor
names. The reader addresses it only through the name offset and name length
of tensor entries; bytes of the pool not covered by any name are not
validated. The reference writer writes the names in tensor-table order with
no separators and no unused bytes.

## 5. Sections

### 5.1 Section kinds

| kind | name | status | contents |
|---|---|---|---|
| 1 | weights | defined | tensor data, laid out as in section 6 |
| 2 | metadata | defined elsewhere | key-value metadata, spec/metadata.md |
| 3 | graph | defined elsewhere | the model graph, spec/graph.md |
| 4 | repack-cache | reserved | kernel-specific weight layouts; layout not yet specified (ADR 0004) |
| 5 | training | defined elsewhere | optimizer state and training progress, spec/training.md (since container 0.2) |

Kinds 1 to 5 are "known". A file MUST contain at most one section of each
known kind; a second one is `duplicate`. A file MUST contain a weights
section; a file without one is `missing-section`. Reserved kinds, and kinds
defined in a document a reader does not implement, are known for the
purposes of the duplicate check; a reader that does not decode such a
section skips it, which it can because none of them carries the required
flag.

An entry with a kind outside 1 to 5 is an unknown section. A reader MUST
reject a file whose unknown section has the required flag set
(`unknown-required-section`). It MUST skip an unknown section without the
required flag, but it still applies the bounds, alignment and overlap checks
of section 5.4 to it.

### 5.2 Section flags

| bit | name | meaning |
|---|---|---|
| 0 | required (`FlagRequired = 1`) | a reader that does not understand the section's kind MUST reject the file |

All other bits are reserved and MUST be zero.

The reference writer sets the required flag on the weights section and on any
section the caller marks required.

### 5.3 Section hashes

| algorithm | name | hash bytes |
|---|---|---|
| 0 | none | MUST be all zero; otherwise `reserved-non-zero` |
| 1 | SHA-256 | the SHA-256 digest of the section's bytes, `[offset, +length)` |

Any other algorithm value is `bad-hash-algorithm`.

Verifying a hash reads every byte of the section, so it is not part of
opening a file. A reader MUST offer hash verification on request (the
reference reader's `VerifyHashes`) and MAY perform it at open. A verification
failure is `hash-mismatch`. For the weights section the hash covers the
whole section including the padding between tensors.

The reference writer writes SHA-256 hashes for every section unless hashing
is turned off, in which case it writes algorithm 0 and zero hash bytes.

### 5.4 Placement

- Every section's `[offset, +length)` MUST be in bounds of the file;
  otherwise `out-of-bounds`.
- The weights section entry MUST have exactly the preamble's data offset and
  data length; otherwise `bad-length`. Its alignment is therefore the data
  alignment.
- Every other section's offset, known or unknown, MUST be a multiple of 8
  (`SectionAlignment`); otherwise `misaligned`. The reference writer aligns
  these sections to 64 bytes.
- No two of the following regions may overlap: the preamble `[0, 64)`, the
  index, and every section. The reader sorts the regions by offset and
  rejects the file with `overlap` when a region starts before the previous
  one ends.

## 6. Tensors

### 6.1 Names

A tensor name is a sequence of 1 to 4096 bytes (`MaxNameLength`) of valid
UTF-8. An empty, over-long or malformed name is `bad-name`. Names are opaque
to the container; the graph gives them meaning.

Tensor entries MUST be sorted by name in ascending bytewise order (comparing
unsigned bytes, shorter prefix first). Two equal names are `duplicate`; any
other order violation is `unsorted`. Because the table is sorted, lookup by
name is a binary search over the entries and needs no auxiliary structure.

### 6.2 Dtype, shape and length

The dtype MUST be an id defined in spec/dtypes.md (`bad-dtype`). The rank
MUST be at most 8 (`bad-shape`).

The byte length of a tensor is derived from its dtype and shape:

```
elems = dims[0] * dims[1] * ... * dims[rank-1]        (1 when rank = 0)
bytes = (elems / block_elems) * block_bytes
```

where `block_elems` and `block_bytes` are the dtype's block size in elements
and bytes (1 and the item size for plain dtypes). Every multiplication MUST
be checked for 64-bit overflow; an overflow is `overflow`. For a block dtype
the rank MUST be at least 1 and the last dimension MUST be a multiple of
`block_elems`; otherwise `bad-shape`. The entry's length field MUST equal
`bytes`; otherwise `size-mismatch`. A tensor with zero elements has length 0
and is valid.

### 6.3 Alignment

A tensor's offset is relative to the start of the weights section and MUST
be a multiple of its required alignment:

| tensor length | alignment |
|---|---|
| less than 1048576 (`LargeTensorThreshold = 1 MiB`) | 64 (`TensorAlignment`) |
| 1 MiB or more | 4096 (`LargeTensorAlignment`) |

A violation is `misaligned`. Since the data offset is a multiple of the data
alignment, which is at least 4096, every tensor is also aligned to the same
boundary in the file.

`[offset, +length)` MUST be in bounds of the weights section (its length is
the preamble's data length); otherwise `out-of-bounds`.

### 6.4 Canonical packing

The weights section holds the tensors packed without choice. Let the tensors
be ordered by offset, and for equal offsets by length. Starting with
`end = 0`, each tensor in that order MUST start at

```
offset = align_up(end, alignment(length))
```

and then `end = offset + length`. After the last tensor, `end` MUST equal the
data length.

A tensor that starts before `end` is `overlap`; one that starts at any other
position than the required one, or a weights section that does not end
exactly at the last tensor, is `non-canonical-layout`. This single pass rules
out overlapping tensors, holes, and payloads hidden inside the weights
section (polyglot files). Several zero-length tensors MAY share an offset.

The reference writer packs tensors in the order they were added, which is
not necessarily name order. Readers MUST NOT assume that data order and
index order agree.

## 7. Loading guidance (non-normative)

The layout is designed so a runtime can map the whole file read-only and use
the bytes in place:

- The preamble is one read. The index is a second read of at most 256 MiB and
  is fully validated before any tensor is touched.
- The weights section is one contiguous, page-aligned byte range. A runtime
  can hand it to `madvise(MADV_WILLNEED)` or `MADV_HUGEPAGE`, register it
  with `cudaHostRegister`, wrap it with Metal's
  `newBufferWithBytesNoCopy`, or read it with `O_DIRECT` or cuFile straight
  into device memory, in one call.
- Default data alignment is 16 KiB because that is the page size on Apple
  Silicon; 4 KiB is the x86-64 and most Linux ARM page size, and the 16 MiB
  cap allows 2 MiB huge pages with room to spare.
- Tensors are 64-byte aligned so an AVX-512 or NEON load never crosses a
  cache line at the start of a tensor. Tensors of 1 MiB or more are 4 KiB
  aligned so they can be DMA'd to a GPU or read with `O_DIRECT` without a
  bounce buffer.
- The reference reader maps the file with `mmap(PROT_READ, MAP_SHARED)` and
  returns slices into the mapping; nothing is copied at open except the
  section table.

## 8. Split files

A model MAY be split across several files. Each part is named

```
<name>-<index>-of-<count>.mim
```

with `index` and `count` zero-padded to five decimal digits, `index` from
`00001` to `count`, for example `model-00001-of-00004.mim`. Each part carries
the metadata keys `split.index` and `split.count` with the same values. Each
part is a complete, valid file on its own: it has its own preamble, index and
weights section, and a reader validates it exactly as it validates a single
file. A tensor appears in exactly one part. The graph section, when present,
is in the first part.

## 9. Error codes

A reader MUST classify every rejected file with one of the codes below. The
codes are strings so that two readers in different languages can be compared.
`errors.go` defines the same set.

| code | raised when |
|---|---|
| `truncated` | the file is shorter than 64 bytes; the preamble's file length exceeds the actual size; the index is shorter than 16 bytes |
| `bad-magic` | the first four bytes are neither `"MIM\0"` nor the legacy `"MMF\0"` |
| `big-endian` | the byte order mark decodes as `0x0D0C0B0A` |
| `bad-byte-order-mark` | the byte order mark is neither `0x0A0B0C0D` nor `0x0D0C0B0A` |
| `unsupported-version` | the major version is not 0 |
| `reserved-non-zero` | preamble bytes 56..63, section entry bytes 25..31, undefined section flag bits, hash bytes with algorithm 0, tensor entry bytes 13..15, or unused dims are not zero |
| `bad-alignment` | the data alignment is not a power of two in 4 KiB..16 MiB |
| `bad-length` | the file is longer than the preamble's file length; the index length does not equal `16 + 64 ns + 96 nt + pool` or `pool` exceeds 256 MiB; the weights section entry disagrees with the preamble |
| `index-too-large` | the index length exceeds 256 MiB |
| `out-of-bounds` | the index, the weights region, a section, a tensor name within the pool, or tensor data within the weights section is out of bounds |
| `overlap` | two regions overlap; a tensor starts before the previous tensor in data order ends |
| `misaligned` | the index offset is below 64 or not a multiple of 8; the data offset is not a multiple of the data alignment; a non-weights section offset is not a multiple of 8; a tensor offset is not a multiple of its alignment |
| `non-canonical-layout` | a tensor does not start at the first aligned position after the previous one; the weights section does not end at the last tensor |
| `size-mismatch` | a tensor's length field differs from the length derived from dtype and shape |
| `overflow` | the element count or byte size overflows 64 bits; on `Open`, the file is too large to map |
| `bad-dtype` | the dtype id is not defined |
| `bad-shape` | the rank exceeds 8; a block dtype tensor has rank 0 or a last dimension that is not a multiple of the block size |
| `bad-name` | a tensor name is empty, longer than 4096 bytes or not valid UTF-8 |
| `unsorted` | tensor entries are not in ascending bytewise name order |
| `duplicate` | two tensors have the same name; two sections have the same known kind |
| `unknown-required-section` | a section of unknown kind has the required flag |
| `missing-section` | there is no weights section |
| `hash-mismatch` | a section's SHA-256 digest does not match (on verification) |
| `bad-hash-algorithm` | the hash algorithm is not 0 or 1 |
| `bad-metadata` | reserved for the metadata section parser (spec/metadata.md) |
| `bad-graph` | reserved for the graph section parser (spec/graph.md) |
| `bad-training` | reserved for the training section parser (spec/training.md) |
| `not-found` | reserved for lookups of a missing tensor or section by name by API layers; not a parse error |
| `unsupported-dtype` | reserved for runtimes that know a dtype id but cannot operate on it; not a parse error |

The reader applies the checks in the order they are described in this
document: preamble, index header, section entries in table order, tensor
entries in table order, then canonical packing. A file with several faults
is reported with the first one found in that order. Readers SHOULD follow
the same order so they report the same code.

### 9.1 Hostile corpus

The reference repository derives a corpus of invalid files from one valid
file (`hostile_test.go`). A conforming reader MUST reject each with the
listed code:

| file | code |
|---|---|
| truncated-preamble | `truncated` |
| bad-magic | `bad-magic` |
| big-endian | `big-endian` |
| major-version | `unsupported-version` |
| reserved-preamble | `reserved-non-zero` |
| alignment-not-pow2 | `bad-alignment` |
| truncated-file | `truncated` |
| trailing-bytes | `bad-length` |
| index-too-large | `index-too-large` |
| index-out-of-bounds | `out-of-bounds` |
| index-length-mismatch | `bad-length` |
| weights-disagree | `bad-length` |
| unknown-required-section | `unknown-required-section` |
| bad-hash-alg | `bad-hash-algorithm` |
| section-flags | `reserved-non-zero` |
| name-out-of-pool | `out-of-bounds` |
| unsorted-names | `unsorted` |
| bad-dtype | `bad-dtype` |
| rank-too-big | `bad-shape` |
| size-mismatch | `size-mismatch` |
| shape-overflow | `overflow` |
| misaligned-tensor | `misaligned` |
| tensor-out-of-bounds | `out-of-bounds` |
| overlapping-tensors | `overlap` |
| hole-in-weights | `non-canonical-layout` |

A valid file whose weights bytes are altered after writing MUST pass opening
and fail hash verification with `hash-mismatch`.

## 10. Writing

A writer MUST produce a file that satisfies every rule above. The reference
writer:

1. Packs tensors canonically in insertion order (section 6.4).
2. Sorts the tensor table by name and builds the string pool in that order.
3. Places the index at offset 64, each other section at the next 64-byte
   boundary after the index, and the weights section at the next multiple of
   the data alignment.
4. Writes SHA-256 hashes for every section by default.
5. Writes to a temporary file in the destination directory, flushes, calls
   `fsync`, closes, then renames over the destination path. A crash leaves
   either the old file or the new one, never a partial one. Writers SHOULD
   do the same.

A writer MUST refuse an index that would exceed 256 MiB (`index-too-large`)
and a data alignment outside the allowed range (`bad-alignment`).
