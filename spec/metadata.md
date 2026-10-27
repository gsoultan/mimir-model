# mim metadata section v0.1

Status: draft. This document defines the metadata section (section kind 2):
a typed key-value store for everything that is neither a tensor nor part of
the graph, such as the model name, lineage, the quantization recipe, the
tokenizer and the chat template. The reference implementation is
`metadata.go`.

## 1. Layout

All integers are little-endian.

| offset | size | field |
| --- | --- | --- |
| 0 | 4 | magic `MIMM` (readers also accept the legacy `MMFM`) |
| 4 | 4 | entry count (at most 2^24) |
| 8 | 8 | string pool length |
| 16 | 32 × count | entries, sorted by key (bytewise ascending) |
| then | pool length | pool |

The section length MUST equal exactly 16 + 32 × count + pool length.

Entry:

| offset | size | field |
| --- | --- | --- |
| 0 | 4 | key offset in pool |
| 4 | 4 | key length (1..1024 bytes, UTF-8) |
| 8 | 1 | type |
| 9 | 3 | reserved (0) |
| 12 | 4 | count (arrays, strings and bytes), else 0 |
| 16 | 8 | payload |
| 24 | 8 | reserved (0) |

Types and payloads:

| type | name | payload | count |
| --- | --- | --- | --- |
| 1 | int | i64 | 0 |
| 2 | uint | u64 | 0 |
| 3 | float | f64 bits | 0 |
| 4 | bool | 0 or 1 | 0 |
| 5 | string | pool offset | byte length |
| 6 | bytes | pool offset | byte length |
| 7 | ints | pool offset (8-aligned) | elements of i64 |
| 8 | floats | pool offset (8-aligned) | elements of f64 |
| 9 | strings | pool offset of a table of (u32 offset, u32 length) pairs | elements |

Keys are sorted so a reader can find one key by binary search without
decoding the rest. Keys are namespaced with dots.

## 2. Well-known keys

| key | type | meaning |
| --- | --- | --- |
| `general.name` | string | human-readable model name |
| `general.architecture` | string | architecture family, informational; the graph is authoritative |
| `general.license` | string | SPDX identifier |
| `general.quantization_recipe` | string | how the weights were quantized, for reproducibility |
| `split.index`, `split.count` | int | position in a split set (container §split files) |
| `tokenizer.model` | string | tokenizer family (`bpe`, `unigram`, `wordpiece`) |
| `tokenizer.tokens` | strings | vocabulary, index = token id |
| `tokenizer.merges` | strings | BPE merges in priority order |
| `tokenizer.scores` | floats | unigram scores |
| `tokenizer.chat_template` | string | chat template, see §3 |
| `general.type` | string | `adapter` for an adapter file (spec/adapters.md); a model otherwise |
| `adapter.type`, `adapter.lora.alpha`, `adapter.base.sha256` | | an adapter's kind, scale and base (spec/adapters.md) |

Unknown keys MUST be preserved by tools that rewrite files.

## 3. Chat templates are data

`tokenizer.chat_template` is a string in a restricted template language to
be defined in a later revision. It MUST NOT be interpreted as Jinja or any
other general-purpose template language: rendering a model-supplied Jinja
template was the cause of CVE-2024-34359. Until the restricted language is
specified, readers MAY expose the string but MUST NOT execute it.
