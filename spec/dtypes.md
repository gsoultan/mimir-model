# mmf dtypes v0.1

Status: draft. This document defines every element type an mmf tensor may
have: its numeric id, its name, its storage size and, for quantized types,
the exact byte layout of a block and the exact procedure that turns a block
into f32 values. The reference implementation is `dtype.go`.

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT,
RECOMMENDED, MAY and OPTIONAL are to be interpreted as described in RFC 2119.

## 1. General rules

- A dtype id is a u32 stored in the tensor entry (spec/container.md,
  section 4.3). Ids 1 to 63 are plain dtypes, 64 and up are block dtypes.
  Id 0 is invalid.
- A dtype id is never reused. A dtype that is withdrawn keeps its id and is
  marked withdrawn here; files that use it are still well-formed.
- Every tensor is stored as a packed little-endian array with no padding
  between elements or between blocks. Elements and blocks run in row-major
  order, the last axis fastest.
- `f16` is IEEE 754 binary16. `bf16` is the upper 16 bits of an IEEE 754
  binary32. Wherever a block field is `f16` it is converted to f32 exactly;
  this conversion is lossless.
- For a block dtype, blocks run along the last axis. The last dimension of a
  block-dtype tensor MUST be a multiple of the block size in elements and the
  rank MUST be at least 1. A tensor of shape `[..., n]` with block size `B`
  and block bytes `S` occupies `prod(dims) / B * S` bytes.
- Kernel-specific interleavings, tilings or repackings of a block dtype are
  never dtypes. A runtime that wants a different in-memory layout repacks at
  load time, or stores the result in the repack-cache section (ADR 0004).
- The `dequantize` primitive (opcode 33, spec/primitives.md section 5.33)
  decodes every type in this document using the procedures below. For f32
  output the result MUST be bit-exact to these procedures.

### 1.1 Bit-exactness of decoders

For every block dtype the dequantization formula is given as a sequence of
f32 operations. A decoder MUST perform all arithmetic in IEEE 754 binary32
with round-to-nearest-even, in the order written, with no fused
multiply-add and no reassociation. Integer fields are converted to f32
exactly before use. This order matches the reference dequantizers in ggml
(`ggml-quants.c`), so a GGUF import decodes to the same bits.

Encoders (quantizers) are not normative. Any procedure that produces a valid
block is acceptable; two encoders MAY differ in the values they choose.

## 2. Plain dtypes

| id | name | bytes | semantics |
|---|---|---|---|
| 1 | `f32` | 4 | IEEE 754 binary32 |
| 2 | `f16` | 2 | IEEE 754 binary16 |
| 3 | `bf16` | 2 | bfloat16: sign, 8 exponent bits, 7 mantissa bits; the upper half of a binary32 |
| 4 | `f64` | 8 | IEEE 754 binary64 |
| 5 | `i8` | 1 | two's complement |
| 6 | `u8` | 1 | unsigned |
| 7 | `i16` | 2 | two's complement |
| 8 | `u16` | 2 | unsigned |
| 9 | `i32` | 4 | two's complement |
| 10 | `u32` | 4 | unsigned |
| 11 | `i64` | 8 | two's complement |
| 12 | `u64` | 8 | unsigned |
| 13 | `bool` | 1 | 0 is false, 1 is true. Writers MUST write 0 or 1; readers SHOULD treat any non-zero byte as true |
| 14 | `f8_e4m3` | 1 | OCP FP8 E4M3 (section 2.1) |
| 15 | `f8_e5m2` | 1 | OCP FP8 E5M2 (section 2.2) |

Ids 16 to 63 are reserved for future plain dtypes.

All floating-point types use the IEEE 754 encodings of signed zero,
subnormals, infinities and NaNs where the format has them. Readers MUST
preserve payload bits of NaNs when they copy or `bitcast`; they are not
required to preserve them through arithmetic.

### 2.1 f8_e4m3

The OCP 8-bit floating point E4M3 format (the "e4m3fn" variant): 1 sign bit,
4 exponent bits, 3 mantissa bits, exponent bias 7.

| bits | meaning |
|---|---|
| `S 0000 MMM` | subnormal: `(-1)^S * 2^-6 * (M / 8)` |
| `S EEEE MMM`, E in 1..15, not `1111 111` | normal: `(-1)^S * 2^(E-7) * (1 + M / 8)` |
| `0x7F`, `0xFF` | NaN |

There is no infinity. The largest finite value is `0x7E = 448`. The
encodings `0x7F` and `0xFF` are the only NaNs. `0x80` is negative zero.

### 2.2 f8_e5m2

The OCP 8-bit floating point E5M2 format: 1 sign bit, 5 exponent bits, 2
mantissa bits, exponent bias 15. It is the upper byte of an IEEE binary16 and
follows the IEEE conventions.

| bits | meaning |
|---|---|
| `S 00000 MM` | subnormal: `(-1)^S * 2^-14 * (M / 4)` |
| `S EEEEE MM`, E in 1..30 | normal: `(-1)^S * 2^(E-15) * (1 + M / 4)` |
| `S 11111 00` (`0x7C`, `0xFC`) | positive and negative infinity |
| `S 11111 MM`, M != 0 | NaN |

The largest finite value is `0x7B = 57344`.

## 3. Block dtypes

| id | name | block elements | block bytes | section |
|---|---|---|---|---|
| 64 | `q8_0` | 32 | 34 | 3.1 |
| 65 | `q4_0` | 32 | 18 | 3.2 |
| 66 | `q4_k` | 256 | 144 | 3.3 |
| 67 | `q6_k` | 256 | 210 | 3.4 |
| 68 | `mxfp4` | 32 | 17 | 3.5 |
| 69 | `affine4_g64` | 64 | 36 | 3.6 |

Ids 70 and up are reserved for future block dtypes; see the appendix for
candidates.

In the formulas below `y[i]` is the i-th decoded f32 element of the block,
`n` is an unsigned integer taken from a nibble or bit field, and all other
names are block fields. "Low nibble" is bits 0..3 of a byte, "high nibble"
bits 4..7.

### 3.1 q8_0 (64)

32 elements, 34 bytes. Identical to ggml `block_q8_0`.

| offset | bytes | field | type |
|---|---|---|---|
| 0 | 2 | `d` | f16 scale |
| 2 | 32 | `q[32]` | i8 quantized values |

```
y[j] = d * q[j]            j = 0..31
```

### 3.2 q4_0 (65)

32 elements, 18 bytes. Identical to ggml `block_q4_0`.

| offset | bytes | field | type |
|---|---|---|---|
| 0 | 2 | `d` | f16 scale |
| 2 | 16 | `qs[16]` | packed 4-bit values |

Element `j` for `j = 0..15` is the low nibble of `qs[j]`; element `j + 16`
is the high nibble of `qs[j]`.

```
n_j      = qs[j] & 0x0F
n_(j+16) = qs[j] >> 4
y[i]     = d * (n_i - 8)         n_i - 8 computed as an integer in -8..7
```

### 3.3 q4_k (66)

256 elements, 144 bytes. Identical to ggml `block_q4_K` with
`QK_K = 256`. The block is divided into 8 sub-blocks of 32 elements, each
with a 6-bit scale and a 6-bit minimum.

| offset | bytes | field | type |
|---|---|---|---|
| 0 | 2 | `d` | f16 super-block scale for scales |
| 2 | 2 | `dmin` | f16 super-block scale for minimums |
| 4 | 12 | `scales[12]` | packed 6-bit scales and minimums |
| 16 | 128 | `qs[128]` | packed 4-bit values |

Unpacking the scale `sc_j` and minimum `m_j` of sub-block `j`:

```
for j in 0..3:
    sc_j = scales[j]     & 63
    m_j  = scales[j + 4] & 63
for j in 4..7:
    sc_j = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
    m_j  = (scales[j + 4] >> 4)   | ((scales[j]     >> 6) << 4)
```

The quantized values are read in four chunks of 64 elements. Chunk `c`
(`c = 0..3`) uses `qs[32c .. 32c + 32)`: sub-block `2c` holds the low
nibbles and sub-block `2c + 1` the high nibbles.

```
for c in 0..3:
    for l in 0..31:
        n_lo = qs[32c + l] & 0x0F
        n_hi = qs[32c + l] >> 4
        y[64c + l]      = (d * sc_(2c))     * n_lo - (dmin * m_(2c))
        y[64c + 32 + l] = (d * sc_(2c + 1)) * n_hi - (dmin * m_(2c + 1))
```

The products `d * sc` and `dmin * m` are computed first, in f32, once per
sub-block; the result is then multiplied by `n` and the minimum subtracted,
in that order.

### 3.4 q6_k (67)

256 elements, 210 bytes. Identical to ggml `block_q6_K` with `QK_K = 256`.
Each element is 6 bits: 4 low bits from `ql` and 2 high bits from `qh`. The
block has 16 sub-blocks of 16 elements, each with an 8-bit signed scale.

| offset | bytes | field | type |
|---|---|---|---|
| 0 | 128 | `ql[128]` | low 4 bits of each value |
| 128 | 64 | `qh[64]` | high 2 bits of each value |
| 192 | 16 | `scales[16]` | i8 sub-block scales |
| 208 | 2 | `d` | f16 super-block scale |

The block is decoded in two halves `h = 0, 1` of 128 elements. For half `h`
let `ql_h = ql[64h ..]`, `qh_h = qh[32h ..]` and `sc_h = scales[8h ..]`.

```
for h in 0..1:
    for l in 0..31:
        is = l / 16                                   (0 or 1)
        q1 = ((ql_h[l]      & 0x0F) | (((qh_h[l] >> 0) & 3) << 4)) - 32
        q2 = ((ql_h[l + 32] & 0x0F) | (((qh_h[l] >> 2) & 3) << 4)) - 32
        q3 = ((ql_h[l]      >> 4)   | (((qh_h[l] >> 4) & 3) << 4)) - 32
        q4 = ((ql_h[l + 32] >> 4)   | (((qh_h[l] >> 6) & 3) << 4)) - 32
        y[128h + l]      = (d * sc_h[is])     * q1
        y[128h + l + 32] = (d * sc_h[is + 2]) * q2
        y[128h + l + 64] = (d * sc_h[is + 4]) * q3
        y[128h + l + 96] = (d * sc_h[is + 6]) * q4
```

`q1..q4` are integers in -32..31. `d * sc` is computed in f32, then
multiplied by the integer converted to f32.

### 3.5 mxfp4 (68)

32 elements, 17 bytes. The OCP Microscaling MXFP4 format with the element
order used by ggml `block_mxfp4`.

| offset | bytes | field | type |
|---|---|---|---|
| 0 | 1 | `e` | E8M0 shared scale |
| 1 | 16 | `qs[16]` | packed E2M1 values |

The scale is `2^(e - 127)` for `e = 0..254`. OCP MX defines `e = 0xFF` as
NaN; ggml and the Python `gguf` package instead decode it as the finite
scale `2^128`. For import compatibility mmf follows ggml: a decoder MUST
treat `e = 0xFF` as `2^128`, and an encoder MUST NOT emit it. A nibble of
`0x8` (negative zero) decodes to `+0`, as in ggml.

Element `j` for `j = 0..15` is the low nibble of `qs[j]`; element `j + 16` is
the high nibble. Each nibble is an E2M1 value: bit 3 is the sign, bits 1..2
the exponent, bit 0 the mantissa. The eight magnitudes are

| bits 0..2 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| value | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |

and bit 3 negates the value. E2M1 has no infinity and no NaN.

```
scale = 2^(e - 127)               (f32; 2^128 when e = 0xFF)
y[i]  = scale * e2m1(n_i)
```

The product of a power of two and an E2M1 value is exact in f32 whenever it
is within range, so the scale MAY be applied as any equivalent exact
factorization (ggml uses `2^(e - 128)` times doubled E2M1 values); the
result is the same bits.

### 3.6 affine4_g64 (69)

64 elements, 36 bytes. Asymmetric 4-bit quantization with a per-group scale
and bias, the layout produced by common weight-only quantizers (GPTQ, AWQ,
MLX) after regrouping to 64 elements.

| offset | bytes | field | type |
|---|---|---|---|
| 0 | 2 | `scale` | f16 |
| 2 | 2 | `bias` | f16 |
| 4 | 32 | `qs[32]` | packed unsigned 4-bit values |

Element `j` for `j = 0..31` is the low nibble of `qs[j]`; element `j + 32`
is the high nibble. `n` is unsigned, 0..15.

```
y[i] = scale * n_i + bias
```

The multiplication is performed first, then the addition, both in f32.

## Appendix A. Deferred block dtypes (non-normative)

These formats are candidates for future ids. Their layouts are recorded so
that an id can be assigned without redesign. None of them is valid in a v0.1
file.

| candidate | block elements | block bytes | layout |
|---|---|---|---|
| NVFP4 | 16 | 9 | f8_e4m3 per-block scale, then 8 bytes of E2M1 nibbles; one f32 per-tensor scale stored as a separate tensor or metadata key. `y = tensor_scale * block_scale * e2m1(n)` |
| q5_0 | 32 | 22 | f16 `d`; u8 `qh[4]` (fifth bits); u8 `qs[16]`. `y = d * (n - 16)` |
| q5_1 | 32 | 24 | f16 `d`; f16 `m`; u8 `qh[4]`; u8 `qs[16]`. `y = d * n + m` |
| q4_1 | 32 | 20 | f16 `d`; f16 `m`; u8 `qs[16]`. `y = d * n + m` |
| q8_1 | 32 | 36 | f16 `d`; f16 `s` (sum); i8 `qs[32]`. Activation format, not usually stored |
| q2_k | 256 | 84 | u8 `scales[16]`; u8 `qs[64]`; f16 `d`; f16 `dmin` |
| q3_k | 256 | 110 | u8 `hmask[32]`; u8 `qs[64]`; u8 `scales[12]`; f16 `d` |
| q5_k | 256 | 176 | f16 `d`; f16 `dmin`; u8 `scales[12]`; u8 `qh[32]`; u8 `qs[128]` |
| q8_k | 256 | 292 | f32 `d`; i8 `qs[256]`; i16 `bsums[16]`. Activation format |
| iq4_nl | 32 | 18 | f16 `d`; u8 `qs[16]`; non-linear 16-entry lookup table |
| iq4_xs | 256 | 136 | f16 `d`; u16 `scales_h`; u8 `scales_l[4]`; u8 `qs[128]`; same lookup table |

Admission of a candidate requires: a stable upstream definition, a
bit-exact decode procedure in the form used above, test vectors in
testdata/dtypes, and an ADR. A format whose layout is specific to one
kernel's register tiling is not admitted (ADR 0004).
