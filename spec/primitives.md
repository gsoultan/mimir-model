# mmf primitives v0.1

Status: draft. This document defines the primitive operations of an mmf
graph: their inputs, outputs, attributes, shape and dtype rules, numerical
semantics and required accuracy. The binary encoding of graphs, values,
attributes and subgraphs is defined in spec/graph.md. Element types are
defined in spec/dtypes.md. Composite operations are defined in
spec/stdlib.md.

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT,
RECOMMENDED, MAY and OPTIONAL are to be interpreted as described in RFC 2119.

## 1. Overview

The operation set has three parts:

- 33 primitives (opcodes 1 to 33). Every conforming runtime MUST implement
  all of them.
- `call` (opcode 34), which invokes a composite from the standard library or
  from the model file's composites table.
- Two structural operations, `while` (opcode 64) and `if` (opcode 65), which
  carry subgraphs.

Opcode values not listed in this document are reserved. A runtime MUST
reject a graph that contains a reserved opcode.

| opcode | name | inputs -> outputs | attributes |
|---|---|---|---|
| 1 | neg | 1 -> 1 | |
| 2 | recip | 1 -> 1 | |
| 3 | sqrt | 1 -> 1 | |
| 4 | exp2 | 1 -> 1 | |
| 5 | log2 | 1 -> 1 | |
| 6 | sin | 1 -> 1 | |
| 7 | trunc | 1 -> 1 | |
| 8 | cast | 1 -> 1 | `to` (dtype) |
| 9 | bitcast | 1 -> 1 | `to` (dtype) |
| 10 | add | 2 -> 1 | |
| 11 | mul | 2 -> 1 | |
| 12 | max | 2 -> 1 | |
| 13 | pow | 2 -> 1 | |
| 14 | idiv | 2 -> 1 | |
| 15 | imod | 2 -> 1 | |
| 16 | shl | 2 -> 1 | |
| 17 | shr | 2 -> 1 | |
| 18 | and | 2 -> 1 | |
| 19 | or | 2 -> 1 | |
| 20 | xor | 2 -> 1 | |
| 21 | cmplt | 2 -> 1 | |
| 22 | cmpeq | 2 -> 1 | |
| 23 | where | 3 -> 1 | |
| 24 | reduce | 1 -> 1 | `mode` (string), `axes` (ints) |
| 25 | reshape | 1 -> 1 | `shape` (ints) |
| 26 | expand | 1 -> 1 | |
| 27 | permute | 1 -> 1 | `perm` (ints) |
| 28 | pad | 1 -> 1 | `pads` (ints), `value` (float, default 0) |
| 29 | shrink | 1 -> 1 | `begin` (ints), `end` (ints) |
| 30 | gather | 2 -> 1 | `axis` (int) |
| 31 | scatter | 3 -> 1 | `axis` (int) |
| 32 | const | 0 -> 1 | `value` (float or int) |
| 33 | dequantize | 1 -> 1 | |
| 34 | call | n -> m | `name` (string), `version` (int), composite attributes |
| 64 | while | n -> n | `cond` (subgraph), `body` (subgraph) |
| 65 | if | 1+n -> m | `then` (subgraph), `else` (subgraph) |

## 2. Values, shapes and dtypes

### 2.1 Tensors

Every value in a graph is a tensor with a dtype and a shape. The shape is a
list of dimensions; its length is the rank. A rank-0 tensor (a scalar) has
exactly one element. A dimension MAY be 0, in which case the tensor has no
elements. Elements are addressed in row-major order: the last axis varies
fastest. Axis numbers in attributes are non-negative and less than the rank of
the tensor they refer to; negative axis numbers are invalid in primitives.

Every value carries a declared dtype and a declared shape (spec/graph.md).
The output of every operation MUST have exactly the dtype and shape that the
rules below derive from its inputs and attributes; a graph whose declared
output types disagree with the derived ones is invalid and a runtime MUST
reject it when it can detect the mismatch at load time, and MUST report an
error when it detects it at run time.

### 2.2 Dtype classes

This document groups the dtypes of spec/dtypes.md as follows.

| class | dtypes |
|---|---|
| float | f64, f32, f16, bf16 |
| storage float | f8_e4m3, f8_e5m2 |
| signed integer | i8, i16, i32, i64 |
| unsigned integer | u8, u16, u32, u64 |
| integer | signed integer and unsigned integer |
| bool | bool |
| block | q8_0, q4_0, q4_k, q6_k, mxfp4, affine4_g64, and any block dtype added to spec/dtypes.md |

Storage floats MAY appear only as the input or output of `cast` and
`bitcast`, the input of `dequantize`, the data operand of the movement
operations, `gather` and `scatter`, and the selected operands of `where`.
These operations do no arithmetic on the values: they convert them by the
rules of spec/dtypes.md or copy them bit-exactly. Block dtypes MAY appear
only as the input of `dequantize`. Any other use is invalid.

"Arithmetic dtype" below means float or integer.

### 2.3 Symbolic dimensions

A dimension is either an integer or a symbolic dimension as defined in
spec/graph.md. Symbolic dimensions make a graph shape-polymorphic. They are
bound from the shapes of the graph inputs at the start of each run and stay
fixed for that run. Shapes are never data-dependent: no operation produces
an output whose shape depends on tensor element values. A runtime MUST check
every binding for consistency and MUST report a shape mismatch found at run
time as an error; it MUST NOT read or write outside any tensor because of a
mismatch.

## 3. General rules

### 3.1 Broadcasting

The elementwise operations with more than one input (opcodes 10 to 23)
require all inputs to have the same rank. For each axis, the input
dimensions MUST either be equal or be 1; a dimension of 1 is broadcast
against any size, including 0. The output dimension is the maximum of the
input dimensions on that axis, except that it is 0 if any input dimension is
0. Rank changes are never implicit; a graph uses `reshape` to add or remove
size-1 axes.

```
out_shape[i] = 0                 if any in_shape[i] == 0
             = max_k in_shape_k[i]  otherwise
in_k[idx]   := in_k[j] where j[i] = 0 if in_shape_k[i] == 1 else idx[i]
```

### 3.2 No implicit type promotion

All data operands of `add` through `cmpeq` (opcodes 10 to 22) MUST share one
dtype, and the `a` and `b` operands of `where` MUST share one dtype. Any
change of dtype is an explicit `cast`. The output dtype of opcodes 10 to 20
equals the operand dtype; the output dtype of `cmplt` and `cmpeq` is bool.

### 3.3 Integer arithmetic

Integer arithmetic is exact and wraps modulo 2^w on overflow, where w is the
bit width of the dtype (two's complement for signed dtypes). This applies to
`neg`, `add`, `mul`, `idiv`, `imod`, `shl`, `reduce` and `cast` between
integer dtypes. Overflow is never undefined behaviour.

Division or remainder by zero (`idiv`, `imod`) produces an unspecified value.
It MUST NOT trap, raise a signal or otherwise terminate the process.

### 3.4 Floating point

Float dtypes follow IEEE 754: f64 is binary64, f32 is binary32, f16 is
binary16 and bf16 is the bfloat16 format of spec/dtypes.md. The following
rules apply to every operation with a float output unless the operation
states otherwise:

- Rounding is round to nearest, ties to even.
- If any float data input is NaN, the output is NaN. The sign and payload of
  a NaN result are unspecified.
- Results that overflow the output dtype are an infinity with the sign of the
  exact result.
- Operations MUST NOT trap on any input, including NaN, infinities and
  division by zero.

Subnormals: for each float dtype, a runtime either flushes every subnormal
input and output of every float operation to a zero of the same sign
("flush mode"), or flushes none ("IEEE mode"). A runtime MUST declare its
mode per float dtype (for example in its conformance report) and MUST NOT
mix modes within one dtype. The conformance corpus contains expected results
for both modes.

f16 and bf16 operations: a runtime MAY compute in a wider float format
(f32 or f64) and round the result to the output dtype. The result MUST be
within 1 ULP of the f16 or bf16 rounding of the exact result. Operations that
section 3.5 lists as exact remain exact (correctly rounded) for f16 and bf16;
computing in f32 and rounding once satisfies this for `neg`, `add`, `mul`,
`max`, `trunc` and the comparisons.

### 3.5 Accuracy

The normative reference result of every primitive is the exact mathematical
result of the operation on the input values, rounded to the output dtype
with round to nearest, ties to even. For f32, f16 and bf16 outputs the
reference is computed in float64 and then rounded; for f64 outputs it is
computed with sufficient extra precision. The conformance corpus
(spec/conformance.md) is generated from this reference.

ULP below means the unit in the last place of the output dtype at the
reference result. A tolerance of n ULP means the runtime result r and the
reference result e satisfy |r - e| <= n * ulp(e), where infinities and NaN
must match exactly (any NaN matches any NaN).

| operations | bound for f32 and f64 |
|---|---|
| neg, add, mul, max, trunc, cmplt, cmpeq, where | exact (correctly rounded, 0.5 ULP) |
| cast between float dtypes, cast from integer to float | exact (correctly rounded) |
| cast to integer or bool, bitcast | exact |
| integer and bool operations (idiv, imod, shl, shr, and, or, xor) | exact |
| reshape, expand, permute, pad, shrink, gather, scatter, const | bit-exact |
| recip, sqrt | 1 ULP |
| exp2, log2 | 5 ULP |
| sin, for abs(x) <= 2^16 | 5 ULP, or absolute error <= 2^-22, whichever is larger |
| sin, for abs(x) > 2^16 | result in [-1, 1], absolute error <= 2^-12 |
| pow, for x > 0 | 5 ULP |
| pow, other x | special values in section 5.13 |
| reduce add, reduce mul | bound in section 3.6 |
| reduce max | exact |
| dequantize | bit-exact to spec/dtypes.md (section 5.33) |

For f16 and bf16 outputs every bound is replaced by the rule in section 3.4:
within 1 ULP of the f16 or bf16 rounding of the exact result, and exact where
the table says exact.

### 3.6 Reductions

The order in which `reduce` combines elements is unspecified. A runtime MAY
use any association order, tree shape or number of partial sums, and the
order MAY differ between runs. For an f32 or f64 reduction of N elements x_i
with exact result S, the result r MUST satisfy:

```
reduce add:  |r - S| <= N * u * sum_i |x_i|
reduce mul:  |r - S| <= N * u * |S|        (when no intermediate over- or underflows)
reduce max:  r == S
```

where u is the unit roundoff of the output dtype (2^-24 for f32, 2^-53 for
f64). For f16 and bf16 inputs a runtime MUST accumulate in at least f32: the
bound is computed with u = 2^-24 and then the result is rounded once to the
output dtype (an additional 0.5 ULP of the output dtype).

### 3.7 Movement operations

`reshape`, `expand`, `permute`, `pad` and `shrink` never change element
values: every output element is bit-identical to an input element or, for
`pad`, to the pad value. A runtime MAY implement them as views over the
input's storage without copying.

### 3.8 Run-time errors

A runtime MUST report an error and MUST NOT produce outputs for a run in which
a shape mismatch occurs (section 2.3). Out-of-range indices in `gather` and
`scatter` are not errors; their results are unspecified (sections 5.30 and
5.31), but a runtime MUST NOT read or write memory outside the tensors
involved. No input value of any operation may cause a runtime to crash.

## 4. Attributes

Attribute types are int, ints (a list of int), float, string, dtype and
subgraph. Their encoding is defined in spec/graph.md. An attribute shown
with a default MAY be omitted; every other attribute listed for an operation
is REQUIRED. A runtime MUST reject an operation with an unknown attribute or
a missing required attribute.

## 5. Primitive definitions

In this section x, a, b denote input tensors, y the output, and `[idx]` an
element index. "Float only" means the operand dtype MUST be in the float
class; "integer only" means it MUST be in the integer class.

### 5.1 neg (1)

`y = -x`. Shape: same as x. Dtype: arithmetic; output dtype equals input.
For floats, flips the sign bit, including for zeros, infinities and NaN.
For signed integers, wraps: `neg(MIN) = MIN`. For unsigned integers,
`y = (2^w - x) mod 2^w`.

### 5.2 recip (2)

`y = 1 / x`. Shape: same as x. Dtype: float only.
`recip(+0) = +inf`, `recip(-0) = -inf`, `recip(+-inf) = +-0`. Accuracy 1 ULP.

### 5.3 sqrt (3)

`y = sqrt(x)`. Shape: same as x. Dtype: float only.
`sqrt(-0) = -0`, `sqrt(+inf) = +inf`, `sqrt(x) = NaN` for x < 0. Accuracy
1 ULP.

### 5.4 exp2 (4)

`y = 2^x`. Shape: same as x. Dtype: float only.
`exp2(-inf) = +0`, `exp2(+inf) = +inf`. Results below the smallest subnormal
round to +0; in flush mode, subnormal results become +0. Accuracy 5 ULP.

### 5.5 log2 (5)

`y = log2(x)`. Shape: same as x. Dtype: float only.
`log2(+-0) = -inf`, `log2(+inf) = +inf`, `log2(x) = NaN` for x < 0,
`log2(1) = +0` exactly. Accuracy 5 ULP.

### 5.6 sin (6)

`y = sin(x)`, x in radians. Shape: same as x. Dtype: float only.
`sin(+-0) = +-0`, `sin(+-inf) = NaN`. Accuracy: see section 3.5. Cosine is
the composite `cos.v1` (spec/stdlib.md).

### 5.7 trunc (7)

Rounds toward zero to an integral value in the same dtype. Shape: same as x.
Dtype: arithmetic; for integer dtypes it is the identity.
`trunc(-0.5) = -0`; infinities and NaN are returned unchanged. Exact.

### 5.8 cast (8)

Converts each element to dtype `to`. Shape: same as x. Dtype: input and `to`
may be any float, storage float, integer or bool dtype.

| from -> to | rule |
|---|---|
| float -> float | round to nearest even; overflow -> signed infinity; NaN -> NaN |
| float -> integer | truncate toward zero, then saturate to the range of `to`; NaN -> 0 |
| integer -> float | round to nearest even; overflow -> signed infinity |
| integer -> integer | take the value modulo 2^w of `to` (two's complement wrap) |
| any -> bool | false if the value is zero (including -0), true otherwise (NaN -> true) |
| bool -> any | false -> 0, true -> 1 |
| storage float <-> any | through the reference conversion in spec/dtypes.md |

Conversions to and from storage floats, including saturation and NaN
encodings, follow spec/dtypes.md (decoding in sections 2.1 and 2.2,
encoding in section 2.3).

### 5.9 bitcast (9)

Reinterprets the bits of each element as dtype `to`. Shape: same as x. The
input dtype and `to` MUST have the same element bit width and MUST NOT be
bool or block dtypes; storage floats are allowed, so `bitcast` between `u8`
and `f8_e4m3` reads or builds FP8 bit patterns. The output is bit-identical to the input; it never
canonicalises NaN payloads.

### 5.10 add (10)

`y = a + b`. Shape: broadcast (3.1). Dtype: arithmetic, operands equal.
Floats: `(+inf) + (-inf) = NaN`; `(+0) + (-0) = +0`; `(-0) + (-0) = -0`.
Integers: wrap (3.3). Subtraction is `add(a, neg(b))`.

### 5.11 mul (11)

`y = a * b`. Shape: broadcast. Dtype: arithmetic, operands equal.
Floats: `0 * inf = NaN`; sign of a zero or infinite result is the XOR of the
operand signs. Integers: wrap, keeping the low w bits of the product.

### 5.12 max (12)

`y = max(a, b)`. Shape: broadcast. Dtype: arithmetic, operands equal.
If either operand is NaN the result is NaN. `max(-0, +0) = max(+0, -0) = +0`.
Minimum is `neg(max(neg(a), neg(b)))` for floats and signed integers.

### 5.13 pow (13)

`y = a ^ b`. Shape: broadcast. Dtype: float only, operands equal.
For a > 0 finite: accuracy 5 ULP. If either operand is NaN the result is
NaN (this departs from C, where `pow(1, NaN) = 1`). For the remaining
special values (a = +-0, a < 0, a = +-inf, b = +-inf) the result follows the
`pow` special cases of IEEE 754-2019 section 9.2.1; in particular
`pow(a, b) = NaN` for finite a < 0 and finite non-integral b,
`pow(-8, 3) = -512` (exactly rounded), and `pow(x, +-0) = 1` for non-NaN x.
For a < 0 and integral b, the accuracy bound is 5 ULP.

### 5.14 idiv (14)

`y = a / b`, truncated toward zero. Shape: broadcast. Dtype: integer only,
operands equal. `idiv(MIN, -1) = MIN` (wraps). Division by zero: the
result is unspecified and MUST NOT trap (3.3).

### 5.15 imod (15)

`y = a - idiv(a, b) * b`: the remainder has the sign of the dividend, as in
C. Shape: broadcast. Dtype: integer only, operands equal.
`imod(MIN, -1) = 0`. `imod(-7, 2) = -1`. Modulo by zero: unspecified, no
trap.

### 5.16 shl (16)

`y = a << b`. Shape: broadcast. Dtype: integer only, operands equal. The
shift amount b is interpreted as an unsigned integer of the same width (so a
negative signed amount is a large amount). Bits shifted out are discarded.
If the amount is >= w, the result is 0.

### 5.17 shr (17)

`y = a >> b`. Shape: broadcast. Dtype: integer only, operands equal. The
shift is arithmetic (sign-filling) for signed dtypes and logical
(zero-filling) for unsigned dtypes. b is interpreted as unsigned. If the
amount is >= w, the result is 0 for unsigned dtypes and for non-negative
signed values, and -1 for negative signed values.

### 5.18 and (18), 5.19 or (19), 5.20 xor (20)

Bitwise AND, OR and exclusive OR. Shape: broadcast. Dtype: integer or bool,
operands equal. For bool these are the logical operations. Logical NOT of a
bool b is `xor(b, const true)`; bitwise NOT of an integer is
`xor(x, const -1)` (all bits set; for unsigned dtypes the const value is
2^w - 1).

### 5.21 cmplt (21)

`y = a < b`. Shape: broadcast. Dtype: arithmetic operands, equal; output
bool. Any comparison with NaN is false. `-0 < +0` is false. Other
comparisons are derived: `a > b` is `cmplt(b, a)`; `a <= b` is
`xor(cmplt(b, a), true)` for non-NaN operands.

### 5.22 cmpeq (22)

`y = a == b`. Shape: broadcast. Dtype: arithmetic or bool operands, equal;
output bool. NaN compares unequal to everything, including itself.
`-0 == +0` is true. Comparison is by value, not by bits.

### 5.23 where (23)

`y = cond ? a : b`, elementwise. Inputs in order: cond, a, b. Shape:
broadcast over all three. Dtype: cond MUST be bool; a and b MUST share one
dtype (any non-block dtype, storage floats included); output dtype equals
that of a. The selected
element is copied bit-exactly; the unselected operand never affects the
result, even if it is NaN.

### 5.24 reduce (24)

Combines elements along `axes`. Attribute `mode` is one of "add", "mul",
"max". `axes` is a list of distinct axis numbers, in any order; an empty
list makes `reduce` the identity. Shape: same rank as x; each reduced axis
has size 1 in the output. Dtype: arithmetic; output dtype equals input.

```
y[idx] = fold(mode, { x[j] : j[i] == idx[i] for every i not in axes })
```

Reducing an axis of size 0 yields the identity: 0 for "add", 1 for "mul",
-inf for float "max", and the minimum value of the dtype for integer "max".
"max" propagates NaN. Integer "add" and "mul" wrap. Float accuracy: 3.6.

### 5.25 reshape (25)

Reinterprets the row-major element sequence of x with a new shape. The
target shape is the declared shape of the output value. Attribute `shape`
lists the target dimensions; an entry of -1 means "the corresponding
dimension of the declared output shape" and is how a symbolic dimension is
written. Shape: the element counts of input and output MUST be equal. Dtype:
any non-block dtype; unchanged. Bit-exact.

### 5.26 expand (26)

Broadcasts x to the declared output shape. Rank MUST be unchanged. For each
axis, the input dimension MUST equal the output dimension or be 1; only
size-1 dimensions expand. Dtype: any non-block dtype; unchanged.

```
y[idx] = x[j]  where j[i] = 0 if x_shape[i] == 1 else idx[i]
```

### 5.27 permute (27)

Reorders axes. `perm` MUST be a permutation of 0..r-1, where r is the rank
of x. Shape: `y_shape[i] = x_shape[perm[i]]`. Dtype: any non-block dtype;
unchanged.

```
y[idx] = x[j]  where j[perm[i]] = idx[i] for every i
```

### 5.28 pad (28)

Adds constant borders. `pads` has 2r non-negative entries, ordered
`[before_0, after_0, before_1, after_1, ...]`. Shape:
`y_shape[i] = x_shape[i] + before_i + after_i`. Dtype: any non-block dtype;
unchanged. `value` (default 0) is converted to the dtype as for `const`
(5.32); it MAY be +-inf or NaN for float dtypes.

```
y[idx] = x[idx - before]  if 0 <= idx[i] - before_i < x_shape[i] for every i
       = value            otherwise
```

Negative padding is not allowed; cropping is `shrink`.

### 5.29 shrink (29)

Extracts a contiguous box. `begin` and `end` each have r entries and select
the half-open range `[begin_i, end_i)` on axis i. They MUST satisfy
`0 <= begin_i <= end_i <= x_shape[i]`. An `end` entry of -1 means
`x_shape[i]` and is how the full extent of a symbolic axis is written.
Shape: `y_shape[i] = end_i - begin_i`. Dtype: any non-block dtype; unchanged.

```
y[idx] = x[idx + begin]
```

Strided slicing is not a primitive; it is expressed with `pad`, `reshape`
and `shrink` (see `take_strided` in spec/stdlib.md).

### 5.30 gather (30)

Selects slices of `data` along `axis` using `indices`. Inputs: data (rank
r >= 1), indices (rank q >= 0, dtype i32 or i64). Shape:
`data_shape[:axis] ++ indices_shape ++ data_shape[axis+1:]`, rank r + q - 1.
Dtype: data any non-block dtype; output dtype equals data.

```
y[a..., j..., b...] = data[a..., indices[j...], b...]
```

An index is in range if `0 <= index < data_shape[axis]`; negative indices
are out of range (they do not wrap). For an out-of-range index the
corresponding output elements are unspecified, but a runtime MUST NOT read
memory outside `data`. A runtime MAY report out-of-range indices as a
diagnostic, but MUST still complete the run unless configured otherwise by
the user.

### 5.31 scatter (31)

Functional inverse of `gather`: returns a copy of `data` with slices
replaced. Inputs: data (rank r >= 1), indices (rank q, i32 or i64),
updates. Shape: output equals data; updates MUST have shape
`data_shape[:axis] ++ indices_shape ++ data_shape[axis+1:]`. Dtype: data
and updates equal, any non-block dtype.

```
y = copy(data)
for j in row-major order over indices_shape:
    y[a..., indices[j], b...] = updates[a..., j, b...]   for all a, b
```

If two index positions hold the same index, the update with the later
position in row-major index order wins. Out-of-range indices: the
corresponding updates have unspecified effect on y, but a runtime MUST NOT
write memory outside y. `data` is never modified; see section 7 for
in-place execution.

### 5.32 const (32)

Produces a tensor filled with one value. No inputs. Shape and dtype: the
declared output shape and dtype (any non-block, non-storage dtype). The
attribute `value` is a float or an int:

- float output: `value` rounded to the output dtype (round to nearest even);
  it MAY be +-inf or NaN.
- integer output: `value` MUST be an integer exactly representable in the
  dtype; otherwise the graph is invalid.
- bool output: `value` MUST be 0 or 1.

Tensors with distinct elements (weights, tables) are not built with `const`;
they are constant values stored in the file (spec/graph.md).

### 5.33 dequantize (33)

Decodes a quantized or reduced-precision tensor to f32 or bf16. Input dtype:
any block dtype, f16, bf16, f8_e4m3 or f8_e5m2. Output dtype: f32 or bf16,
as declared on the output value. Shape: the logical element shape of the
input, unchanged. For block dtypes, blocks run along the last axis, whose
size is a multiple of the block size (spec/dtypes.md).

The reference decode of every dtype is defined in spec/dtypes.md. For f32
output the result MUST be bit-exact to that reference. For bf16 output the
result is the f32 reference value rounded to bf16 (nearest, ties to even).

## 6. Composite calls and structural operations

### 6.1 call (34)

Invokes the composite identified by attributes `name` (string) and `version`
(int), passing n inputs and producing m outputs. The remaining attributes are
the composite's own attributes. The composite is either a standard-library
composite (spec/stdlib.md) or a model-local composite from the file's
composites table (spec/graph.md). A runtime that recognises (name, version)
MAY execute its own implementation; otherwise it MUST inline the canonical
body. Inputs, outputs and attributes MUST match the composite's signature.
The call graph of composites MUST be acyclic.

### 6.2 while (64)

Runs a loop over n loop-carried values. Attributes: `cond`, a subgraph with
n inputs and one output (a bool scalar, rank 0); `body`, a subgraph with n
inputs and n outputs. Inputs and outputs of `while` are the n loop-carried
values. The k-th output of `body` MUST have the dtype and shape of the k-th
input of `while`; shapes are loop-invariant (symbolic dimensions keep their
binding).

```
v = inputs
loop:
    if not cond(v): break
    v = body(v)
outputs = v
```

If `cond` is false initially, the outputs equal the inputs. Termination is
the graph author's responsibility.

### 6.3 if (65)

Inputs: a bool scalar (rank 0), then n operands. Attributes: `then` and
`else`, each a subgraph with n inputs and m outputs; corresponding outputs of
the two branches MUST have the same dtype and shape. Exactly one branch is
evaluated.

```
outputs = then(operands) if pred else else(operands)
```

### 6.4 Subgraph scope

A subgraph MAY read, in addition to its own inputs, constant values and
variables of the top-level graph and any value of an enclosing graph that is
defined before the operation that owns the subgraph. A subgraph MUST NOT
declare variable updates; a value that changes across loop iterations is a
loop-carried value.

## 7. Variables and state

A graph MAY declare variables: values of kind VARIABLE with a fixed dtype and
shape and an initial value, stored as a tensor or given as all zeros by the
ZERO_INIT flag (spec/graph.md). A variable persists across runs
of the graph within one session. Within a run, every read of a variable sees
the value it had at the start of the run.

A graph also declares a list of updates, each a pair (variable, new value).
Each variable appears in at most one update; the new value MUST have the
variable's dtype and shape. After all outputs and all update values of a run
have been computed, the runtime replaces each updated variable with its new
value, as one atomic step. If a run fails, no variable changes.

This is how a key-value cache is expressed. The cache is a variable, read
like an input; the new cache is produced by `scatter` writing the new keys
and values at the current position. When the new value of a variable is the
output of a `scatter` whose `data` operand is that same variable, a runtime
SHOULD perform the scatter in place on the variable's storage. It MAY do so
only if no other operation reads the variable after the scatter executes;
the observable result MUST be the same as the functional definition.

Example: one decode step of one attention layer. Cache tensors hold up to
`T_max` positions; `pos` is the index of the current token; `kpos` is a
constant tensor holding 0..T_max-1.

```
inputs:     x: f16[1, 1, d_model], pos: i64[1]
variables:  k_cache: f16[1, H, T_max, D]   (ZERO_INIT, spec/graph.md)
            v_cache: f16[1, H, T_max, D]
constants:  kpos: i64[1, 1, 1, T_max]

q, k_t, v_t = projections(x)                     # each f16[1, H, 1, D]
k_new = scatter[axis=2](k_cache, pos, k_t)       # writes k_cache[:, :, pos, :]
v_new = scatter[axis=2](v_cache, pos, v_t)
p     = reshape[shape=[1,1,1,1]](pos)
valid = xor(cmplt(p, kpos), const true)          # kpos <= pos, bool[1,1,1,T_max]
o     = call attention.v1(q, k_new, v_new, valid)

outputs: o
updates: k_cache <- k_new
         v_cache <- v_new
```

Here `k_cache` has no reader other than the scatter, so a runtime writes the
new row into the existing cache buffer and the update step is free.

## 8. Admission rule for new primitives

The primitive set is deliberately small. A proposed operation is admitted as
a primitive only if no composite over the existing primitives can express
it. Speed is not a reason: an operation that is expressible but slow as a
composite becomes a named composite in the standard library, and runtimes
give it a fast path. A new primitive takes a new opcode and a new minor
version of this document; existing opcodes never change meaning.

`dequantize` is the one explicit exception. Every block dtype is expressible
with `bitcast`, `shr`, `and`, `cast`, `mul`, `add`, `reshape` and `gather`,
but the composite body would be dozens of nodes per block dtype, would have
to change whenever a block dtype is added, and every runtime that loads a
quantized model needs a direct decoder anyway. Making the decode a primitive
keeps model graphs small and puts the bit-level definition in one place,
spec/dtypes.md.
