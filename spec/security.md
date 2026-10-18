# mim security

Status: draft. This document states the threat model for readers of mim
files and the requirements that follow from it. The requirements are
normative; the incidents in section 5 explain where each rule comes from.

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT,
RECOMMENDED, MAY and OPTIONAL are to be interpreted as described in RFC 2119.

## 1. Threat model

A model file is untrusted input. It arrives from a download, a model hub, a
user upload or a shared volume, and the party that produced it may be
hostile or compromised. A reader runs in the same process as the
application, often with access to accelerators, network and user data.

The attacker controls every byte of the file. The attacker does not control
the reader's binary, its configuration or its environment.

Goals, in order:

1. A reader MUST NOT read or write memory outside the file mapping and its
   own buffers for any input. Memory safety never depends on the file.
2. A reader MUST NOT execute code supplied by the file.
3. Opening a file MUST take time and memory proportional to the index, not
   to attacker-chosen numbers in the file.
4. A reader MUST detect corruption and tampering when asked to, with a
   cryptographic hash.
5. Two readers MUST agree on whether a file is valid, so an attacker cannot
   craft a file that one tool shows and another runs differently.

## 2. Validation requirements

Every length and offset in the file MUST be validated before it is used to
form a pointer, slice or read. The checks and the code a reader reports are
defined in spec/container.md section 9; this table groups them by what they
protect.

| protects against | checks (error code) |
|---|---|
| reading past the end of the file | file length equals actual size (`truncated`, `bad-length`); index in bounds, weights in bounds, each section in bounds, each tensor in bounds of the weights section (`out-of-bounds`) |
| reading past the end of the index | index at least 16 bytes (`truncated`); index length equals `16 + 64 ns + 96 nt + pool` (`bad-length`); tensor name in bounds of the pool (`out-of-bounds`) |
| unbounded allocation at open | index at most 256 MiB, pool at most 256 MiB (`index-too-large`, `bad-length`) |
| unbounded allocation at run time | a variable with ZERO_INIT (spec/graph.md) costs memory but no file space; a runtime computes its byte size with overflow checks and SHOULD refuse one larger than it is willing to allocate before creating a session |
| integer overflow in size arithmetic | bounds evaluated as `off <= size && len <= size - off`; element count and byte size multiplied with overflow detection (`overflow`); tensor length must equal the derived size (`size-mismatch`) |
| aliasing and hidden payloads | regions do not overlap (`overlap`); tensors packed canonically with no holes and the section ending at the last tensor (`non-canonical-layout`) |
| misaligned access faults | data alignment, index, section and tensor alignment (`bad-alignment`, `misaligned`) |
| type confusion | dtype must be defined (`bad-dtype`); rank at most 8 and block shape rules (`bad-shape`); names valid UTF-8 and bounded (`bad-name`); sorted unique names so lookup is deterministic (`unsorted`, `duplicate`) |
| silent misinterpretation | magic, byte order mark, major version, reserved bytes all zero (`bad-magic`, `big-endian`, `bad-byte-order-mark`, `unsupported-version`, `reserved-non-zero`); unknown required sections rejected (`unknown-required-section`); weights section present and consistent with the preamble (`missing-section`, `bad-length`) |
| tampering | per-section SHA-256 (`hash-mismatch`, `bad-hash-algorithm`) |

A reader MUST perform all of these checks before returning a handle to the
file. Lazy validation of tensor entries at first access is not permitted,
because a tool that lists tensors would then accept a file that a runtime
rejects.

## 3. Arithmetic

- All size computations MUST use unsigned 64-bit arithmetic with explicit
  overflow checks, or arbitrary precision. A reader written in a language
  with wrapping integers MUST use checked multiplication for the element
  count and the byte size.
- A reader MUST NOT trust the tensor length field. It derives the length
  from dtype and shape and rejects a mismatch.
- A reader MUST NOT compute `offset + length` and compare against a size.
  It compares `offset <= size` and `length <= size - offset`.

## 4. Content

- The file contains no executable content. The graph is data interpreted by
  the runtime (spec/primitives.md); every operation has bounded,
  shape-checked behaviour, and a runtime MUST reject a graph with an unknown
  opcode or a shape it cannot verify.
- Chat templates, tokenizer rules and pre- and post-processing recipes are
  data in a restricted template language that will be specified in
  spec/templates.md. That language has no code execution, no attribute
  access on host objects, no imports, no filesystem or network access, and
  bounded evaluation. Jinja and other general-purpose template engines MUST
  NOT be used to evaluate content from a model file.
- Metadata values are typed data (spec/metadata.md). A reader MUST NOT
  evaluate, deserialize with a general object deserializer (pickle, YAML
  with tags, Java serialization) or pass to a shell any string from the
  file.
- Tensor names and metadata strings are shown to users. A reader SHOULD
  escape or filter control characters before printing them.

## 5. Integrity

- Writers SHOULD write SHA-256 hashes for every section. The reference
  writer does so by default.
- A reader MUST offer hash verification and SHOULD verify hashes before the
  first run of a model obtained from an untrusted source. Verification reads
  the whole file, so it is optional at open.
- Hashes detect corruption and casual tampering. They do not authenticate
  the author: an attacker who can replace the file can replace the hashes.
  Signatures are out of scope for this document and belong in metadata
  signed over the whole file.

## 6. Implementation guidance

- Readers SHOULD be fuzzed. The reference repository ships a Go fuzz target,
  `FuzzParse` in `hostile_test.go`, seeded with the valid sample and the
  hostile corpus. Run `go test -fuzz FuzzParse -fuzztime 60s`. A reader in
  another language SHOULD run its own fuzzer against the same seed corpus.
- Readers SHOULD be written in, or isolate parsing in, a memory-safe
  language. A C or C++ reader SHOULD be run under AddressSanitizer in its
  test suite.
- Readers SHOULD map files read-only (`PROT_READ`). A file that changes
  under a mapping is a concern only for the hash check; every bounds check
  is made against the preamble's file length, which was validated against
  the size at open, so a shrinking file produces a fault, not an
  out-of-bounds read. Readers that cannot tolerate that fault SHOULD copy or
  lock the file.
- Error messages MAY include offsets and field values from the file. They
  MUST NOT include raw bytes of the file without escaping.

## 7. Incidents these rules come from

- CVE-2024-34359, llama-cpp-python: the chat template stored in a GGUF file
  was rendered with an unsandboxed Jinja2 environment, so a model file could
  execute arbitrary Python on load. Section 4: no general-purpose template
  engine, templates are data in a restricted language.
- CVE-2026-33298, ggml: `ggml_nbytes` computed a tensor's byte size from
  file-supplied dimensions without overflow checks, so a crafted GGUF
  produced a small allocation and a large copy. Section 3: overflow-checked
  size arithmetic and a derived, not trusted, tensor length.
- "Bleeding Llama", GGUF loader: tensor sizes from the header were used to
  read without checking them against the file size, allowing out-of-bounds
  reads of process memory. Section 2: every offset and length is bounds
  checked against a validated file length before use.
- Trail of Bits audit of safetensors (2023): the header was a JSON document
  of unbounded size parsed before validation; the audit recommended a hard
  header size limit, offset validation, overflow-safe arithmetic and
  fuzzing. Section 2: the 256 MiB index cap and a fixed-width binary index
  that is validated by arithmetic, not parsed; section 6: fuzzing.

Each future rule added to this document MUST cite the incident or analysis
that motivates it.
