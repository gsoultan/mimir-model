# mmf conformance corpus v0.1

Status: draft. This document defines the primitive conformance corpus: what
it tests, how its expected results are produced, the file format, and how a
runtime's results are compared with it. The primitives are defined in
spec/primitives.md and the dtypes in spec/dtypes.md.

The key words MUST, SHOULD and MAY are to be interpreted as described in
RFC 2119.

## 1. Purpose

The corpus makes the spec, not any one implementation, the source of truth.
Every expected result is computed from the definitions in
spec/primitives.md and spec/dtypes.md by an oracle that shares no code with
the Go reference kernels. A runtime conforms for a primitive when it passes
every case of that primitive in the subnormal mode it declares.

The corpus lives in `testdata/conformance/primitives`, one file per
primitive, and covers all 33 primitives (opcodes 1 to 33). Composites
(spec/stdlib.md) and the structural operations are not covered by this
version.

## 2. The oracle

`tools/gen_conformance.py` generates the corpus. It needs Python 3, NumPy
and mpmath, and is deterministic: running it twice gives the same bytes.

- Operations listed as exact in spec/primitives.md section 3.5 are evaluated
  in rational arithmetic, so the reference is the exact result.
- `recip` is evaluated in rational arithmetic; `sqrt`, `exp2`, `log2`,
  `sin` and `pow` are evaluated with mpmath at 256 bits of precision.
- Every result is then rounded once, to nearest with ties to even, directly
  to the output dtype. There is no intermediate rounding to f64, so f32,
  f16 and bf16 results are correctly rounded and f64 results are within
  2^-256 relative of exact before the final rounding.
- Integer results are computed with unbounded integers and then wrapped,
  truncated or saturated as the primitive requires.
- Block-dtype decoders follow the ordered f32 procedures of spec/dtypes.md
  section 3 in NumPy float32 arithmetic.

`python3 -I tools/gen_conformance.py --check` regenerates the corpus in
memory and fails when the committed files differ; CI runs it.

## 3. File format

Each file is UTF-8 JSON:

```
{"format": "mmf-conformance", "version": 1, "op": "add", "opcode": 10,
 "cases": [ case, ... ]}
```

A case is an object with these members:

| member | meaning |
| --- | --- |
| `name` | unique within the file, for example `f32/broadcast` |
| `inputs` | the operation's inputs in order, as tensors |
| `attrs` | optional; attribute key to typed value |
| `output` | the expected result in IEEE mode, as a tensor; its dtype and shape are the declared output type |
| `check` | how to compare a result with `output` (section 4) |
| `output_ftz` | optional; `{"data", "check"}` for flush mode when it differs from IEEE mode |
| `dont_care` | optional; flat output indices whose value is unspecified |

A tensor is `{"dtype", "shape", "data"}`: the dtype name of spec/dtypes.md,
the shape as a list of non-negative integers (`[]` for a scalar), and the
packed little-endian element bytes as lowercase hex, laid out exactly as in
an mmf file. Block-dtype inputs use their block layout.

An attribute value is an object with one member naming its type: `int` (a
JSON integer), `ints` (a list), `float` (the 8 bytes of a little-endian
binary64, as hex, so that NaN and infinities are exact), `string`, or
`dtype` (a dtype name).

A runtime under test receives the opcode, the attributes, the inputs and the
declared output dtype and shape, and produces one output tensor.

## 4. Checks

Elements whose flat index is listed in `dont_care` are not compared. The
result's dtype and shape MUST equal the declared ones. Each element is then
compared by the case's check:

| kind | rule |
| --- | --- |
| `bits` | the element's bytes equal the expected bytes |
| `exact` | for float and storage-float dtypes: both are NaN, or the bit patterns are equal (so `-0` and `+0` differ); for other dtypes the bytes are equal |
| `ulp` | `\|r - e\| <= n * ulp(e)` with `n` from the member `ulp`; when `e` is infinite `r` MUST equal it, when `e` is NaN `r` MUST be NaN |
| `tol` | `\|r - ref_i\| <= bound_i`, where `ref` and `bound` are hex arrays of binary64 values, one per element; an infinite or NaN `ref_i` MUST be matched exactly; when `max_abs` is present `\|r\| <= max_abs` |

`ulp(e)` is the spacing of the output dtype at the magnitude of `e`:
`2^(max(floor(log2 |e|), emin) - p + 1)`, where `p` is the precision and
`emin` the exponent of the smallest normal number of the dtype; for `e = 0`
it is the smallest subnormal.

The checks implement spec/primitives.md section 3.5: `exact` for the exact
operations, `ulp` with the stated bound for `recip`, `sqrt`, `exp2`, `log2`
and `pow` (1 ULP for every f16 and bf16 result), `tol` for `sin` in f32 and
f64 (5 ULP or 2^-22, or 2^-12 and the range [-1, 1] beyond 2^16) and for
float `reduce add` and `reduce mul` (the bound of section 3.6, plus half an
f64 ULP for storing the exact sum as binary64 and one output ULP for f16 and
bf16), and `bits` for the movement operations and `bitcast`.

## 5. Unspecified results

A case lists in `dont_care` the elements the spec leaves unspecified:

- `idiv` and `imod` by zero (spec/primitives.md section 3.3);
- `gather` output elements selected by an out-of-range index (5.30);
- the whole output of a `scatter` with an out-of-range index, since the
  effect of such an update on `y` is unspecified (5.31).

These cases still check that the runtime completes the run without
crashing and without reading or writing outside the tensors.

## 6. Subnormal modes

The corpus gives expected results for both modes of spec/primitives.md
section 3.4. A runtime in IEEE mode compares against `output`. A runtime in
flush mode compares against `output_ftz` where present and `output`
elsewhere; the flush-mode results flush every subnormal float input and
output of the float-class dtypes to a zero of the same sign (`exp2` flushes
to `+0`). A conformance report MUST state the mode per float dtype.

## 7. Using the corpus from Go

The `conformance` package loads the corpus and checks results:

```go
cases, err := conformance.LoadDir("testdata/conformance/primitives")
report := conformance.Run(cases, myRuntime.EvalPrimitive, conformance.IEEE)
for _, f := range report.Failures {
    fmt.Println(f)
}
```

`Run` recovers from a panic in the runtime and reports it as a failure of
the case. The reference kernels in package `ref` pass every case in IEEE
mode.

## 8. Scope of version 1

The corpus covers each primitive across the dtypes it accepts, with special
values (signed zeros, subnormals, the largest finite values, infinities and
NaN), random values over the whole exponent range, broadcasting, rank-0 and
empty tensors, every cast between the 15 non-block dtypes, all 256 values of
each storage float, and every block dtype. It does not yet cover storage
floats as operands of `where` or `bitcast`, which spec/primitives.md
sections 2.2, 5.9 and 5.23 describe inconsistently; that question is open
for v0.2.
