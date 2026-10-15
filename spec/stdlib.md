# mmf standard library v0.1

Status: draft. This document defines the composite operations of the mmf
standard library: their signatures, canonical bodies over the primitives of
spec/primitives.md, and their accuracy tolerances. The encoding of `call`
nodes and of a model file's composites table is defined in spec/graph.md.

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT,
RECOMMENDED, MAY and OPTIONAL are to be interpreted as described in RFC 2119.

## How a runtime uses this

A graph invokes a composite with a `call` node that names it by
(name, version), for example `softmax.v1`. When a runtime loads the graph it
looks at each call. If it has its own kernel for that exact (name, version),
it MAY run the kernel (the fast path), provided the kernel meets the
composite's tolerance. Otherwise it MUST replace the call with the canonical
body given here, recursively, until only primitives remain. A runtime that
implements the 33 primitives and the inliner therefore runs every composite
in this document; fast paths change speed, never meaning.

| composite | purpose | tolerance (section 2.3 notation) |
|---|---|---|
| matmul.v1 | batched matrix product | abs K u sum_k abs(a b) |
| conv2d.v1 | 2-D convolution, NCHW | abs Q u sum abs(x w) |
| linear.v1 | x times transposed weight, plus bias | abs (K+1) u (sum abs(x w) + abs(bias)) |
| dequant_matmul.v1 | linear with a quantized weight | as linear.v1 |
| sigmoid.v1 | logistic function | rel (16 + 2 abs(x)) u |
| tanh.v1 | hyperbolic tangent | abs 16 u |
| erf.v1 | error function | abs 1.6e-7 + 16 u |
| exp.v1 | natural exponential | rel (16 + 2 abs(x)) u |
| log.v1 | natural logarithm | rel 16 u |
| cos.v1 | cosine | abs (16 + 2 abs(x)) u |
| rsqrt.v1 | reciprocal square root | rel 6 u |
| silu.v1 | x times sigmoid(x) | rel (20 + 2 abs(x)) u |
| gelu_erf.v1 | GELU, exact form | abs (1.6e-7 + 32 u) max(1, abs(x)) |
| gelu_tanh.v1 | GELU, tanh approximation | abs 32 u max(1, abs(x)) |
| swiglu.v1 | silu(a) times b | rel (20 + 2 abs(a)) u |
| geglu.v1 | gelu(a) times b | gelu bound times abs(b) |
| softmax.v1 | normalised exponential | rel (N + 32 + 2 abs(x - m)) u |
| log_softmax.v1 | log of softmax | abs (N + 40) u + 2 u abs(x - m) |
| layer_norm.v1 | layer normalisation | section 5.3 |
| rms_norm.v1 | RMS normalisation | rel (N + 24) u |
| mean.v1 | arithmetic mean over axes | abs (N + 4) u mean(abs(x)) |
| variance.v1 | variance over axes | abs (3N + 16) u mean(x^2) |
| cumsum.v1 | inclusive or exclusive prefix sum | abs L u sum_prefix abs(x) |
| rope.v1 | rotary position embedding, with YaRN | section 6.1 |
| attention.v1 | scaled dot-product attention, GQA | section 6.2 |
| embedding.v1 | table lookup | exact |
| top_k.v1 | k largest or smallest values and indices | exact |
| argsort.v1 | stable sort indices | exact |

## 1. Composite rules

### 1.1 Identity and versioning

A composite is identified by the pair (name, version), written
`name.vN`. A name never changes meaning: once `name.vN` is published, its
signature, attributes, defaults, canonical body and tolerance are fixed.
Any change of semantics, including a changed default, a new attribute or a
tighter or looser tolerance, is published as `name.v(N+1)`. Older versions
remain defined and runtimes SHOULD keep accepting them.

Standard-library names consist of lowercase letters, digits and `_`, and
never contain `/`.

### 1.2 Fast paths and inlining

For each call, a runtime either:

1. recognises (name, version) and runs its own implementation, which MUST
   produce results within the composite's tolerance for every input, or
2. inlines the canonical body, substituting inputs, outputs and attribute
   values, and recursively inlines calls inside the body.

A runtime MUST NOT run a fast path keyed on the name alone, and MUST NOT
substitute a kernel for a different version. A fast path MAY be more
accurate than the canonical body.

### 1.3 Reference evaluation and tolerances

The reference result of a composite is the float64 evaluation of its
canonical body: every float value in the body is computed in float64 with
correctly rounded primitives, every `cast` between float dtypes inside the
body is the identity, integer values are computed exactly, and each float
output is rounded once to its declared dtype. Tolerances are stated against
this reference and MUST be satisfied by any fast path.

The canonical body, evaluated in its computation dtype with primitives that
meet the bounds of spec/primitives.md, is designed to satisfy its own
tolerance. The tolerance numbers in this version are provisional and will be
calibrated against the conformance corpus in Phase 2.

### 1.4 Computation dtype

Unless a composite says otherwise:

- Float inputs of dtype f16 or bf16 are cast to f32 on entry to the body, and
  float outputs are cast back to the declared output dtype on exit. These
  casts are part of every canonical body and are not repeated below.
- f32 and f64 inputs are computed in their own dtype.
- All float inputs of one call MUST share one dtype.

The computation dtype is therefore f32 for f16, bf16 and f32 inputs, and f64
for f64 inputs.

### 1.5 Model-local composites

A model file MAY define composites in its composites table (spec/graph.md).
They follow every rule in this section. Their names MUST contain a `/`
(for example `acme/fused_moe`), so they never collide with standard-library
names; the part before the first `/` is a namespace whose owner is
responsible for keeping each (name, version) immutable. A runtime MAY run a
fast path for a model-local composite it recognises. If the definition
states no tolerance, a fast path MUST agree with the inlined body within the
primitive accuracy bounds.

Model-local composites are the only mechanism for custom operations. The
format has no opaque operator blobs: every operation in a file is either a
primitive or a composite whose body reduces to primitives.

## 2. Notation

### 2.1 Bodies

Bodies are pseudo-code over the primitives. `op[attr=value](inputs)` is a
primitive node; `name.vN[attrs](inputs)` is a call. `x_shape`, `r` and the
dimension names in a signature refer to the shapes of the inputs. Conditions
and loops over attribute values (`if causal:`, `for i in 0..KH-1:`) are
resolved and unrolled by the inliner; they are not `if` or `while` nodes.
Every reshape, expand and const declares its output shape from the shapes in
scope; a symbolic dimension is written with the -1 sentinel of
spec/primitives.md sections 5.25 and 5.29.

`tensor_const(expr)` is a constant tensor whose elements are computed by the
inliner from attribute values only, in float64, and rounded to the
computation dtype. The inliner materialises it as a constant value of the
graph (spec/graph.md).

### 2.2 Body macros

The following macros expand to primitives. They are notation, not
composites, and have no fast path.

```
k(v)                 = const[value=v]        # dtype: computation dtype,
                                             # shape: [1]*rank of context
sub(a, b)            = add(a, neg(b))
div(a, b)            = mul(a, recip(b))
abs(x)               = max(x, neg(x))
min(a, b)            = neg(max(neg(a), neg(b)))
sq(x)                = mul(x, x)
not(p)               = xor(p, const[value=1](bool))
slice(x, ax, b, e)   = shrink[begin, end](x)   # [b, e) on axis ax, full on others
unsqueeze(x, ax)     = reshape(x)              # insert a size-1 axis at ax
squeeze(x, ax)       = reshape(x)              # remove the size-1 axis ax
concat(a, b, ax)     = add(pad(a, after=len_b on ax), pad(b, before=len_a on ax))
                       # len_a and len_b MUST be static; a -0 element becomes +0
count(x, axes)       = reduce[add, axes](const[value=1])
                       # const shape: x_shape on axes, 1 elsewhere; exact for
                       # counts up to 2^24 in f32
take_strided(x, ax, start, n, step)
                     = squeeze(slice(reshape(slice(x, ax, start, start + n*step),
                                             [..., n, step, ...]),
                                     ax + 1, 0, 1), ax + 1)
                       # elements start, start+step, ..., start+(n-1)*step
iota(x, ax)          = cumsum.v1[axis=ax, exclusive=true](const[value=1](i64))
                       # const shape: x_shape on ax, 1 elsewhere; 0, 1, ...
riota(x, ax)         = cumsum.v1[axis=ax, exclusive=true, reverse=true](same const)
                       # L-1, L-2, ..., 0
```

Negative axis attributes of composites (for example `axis=-1`) are allowed
and mean `r + axis`; the inliner normalises them before emitting primitives.

### 2.3 Tolerance notation

`u` is the unit roundoff of the computation dtype: 2^-24 for f32 and 2^-53
for f64. `r` is the reference result and `y` the runtime result.

- `abs E` means `|y - r| <= E`.
- `rel E` means `|y - r| <= E * |r|`.

Every float tolerance additionally allows `n_min`, the smallest positive
normal number of the computation dtype (this absorbs underflow and flush
mode), and, when the output dtype is f16 or bf16, 1 ULP of the output dtype
for the final rounding. Infinities and NaN in the reference MUST be matched
exactly (any NaN matches any NaN). N is the number of elements reduced, K
the contracted length, L the scanned length.

## 3. Linear algebra

### 3.1 matmul.v1

Signature: inputs `A: [..., M, K]`, `B: [..., K, N]`; no attributes; output
`Y: [..., M, N]`. A and B have the same rank r >= 2; each batch dimension
is equal or 1 (broadcast). Dtype: float or integer, equal. Integer matmul
computes and wraps in the input dtype; a graph that wants i8 inputs with i32
accumulation casts to i32 first.

```
a = reshape(A, A_shape[:-1] ++ [K, 1])         # [..., M, K, 1]
b = unsqueeze(B, r - 2)                        # [..., 1, K, N]
P = Y_batch ++ [M, K, N]                       # broadcast batch dims
p = mul(expand(a, P), expand(b, P))            # [..., M, K, N]
s = reduce[add, axes=[r - 1]](p)               # [..., M, 1, N]
Y = squeeze(s, r - 1)                          # [..., M, N]
```

Tolerance: for each output element, `abs K u sum_k |A_ik B_kj|`. Integer:
exact. A fast path MUST accumulate f16 and bf16 inputs in at least f32.

### 3.2 linear.v1

Signature: inputs `x: [..., M, K]` (rank r >= 2), `w: [N, K]`, optional
`bias: [N]`; output `y: [..., M, N]`. The weight layout is [N, K] (output
features first), as in PyTorch.

```
wt = reshape(permute[perm=[1, 0]](w), [1]*(r-2) ++ [K, N])
y  = matmul.v1(x, expand(wt, x_shape[:-2] ++ [K, N]))
if bias: y = add(y, reshape(bias, [1]*(r-1) ++ [N]))
```

Tolerance: `abs (K+1) u (sum_k |x_ik w_jk| + |bias_j|)`.

### 3.3 dequant_matmul.v1

Signature: inputs `x: [..., M, K]` (f32, f16 or bf16), `w: [N, K]` of a
block dtype, f8_e4m3, f8_e5m2, f16 or bf16, optional `bias: [N]`; output
`y: [..., M, N]` with the dtype of x. For block dtypes the blocks run along
K.

```
wd = dequantize(w)                     # f32 [N, K], reference decode
y  = linear.v1(x, wd, bias)
```

This is the canonical fast-path target for quantized weights: a runtime
SHOULD decode blocks inside the matrix kernel rather than materialise `wd`.
Tolerance: as linear.v1, with w replaced by its reference decode. A kernel
that also quantizes the activations x (for example to 8-bit blocks) does not
meet this tolerance; see Open issues.

### 3.4 conv2d.v1

Signature: inputs `X: [Nb, C, H, W]`, `Wt: [F, C/g, KH, KW]`, optional
`B: [F]`; attributes `strides` (2 ints, default [1, 1]), `dilations`
(2 ints, default [1, 1]), `pads` (4 non-negative ints
`[top, bottom, left, right]`, default zeros), `groups` (int g, default 1);
output `Y: [Nb, F, Ho, Wo]`. C and F MUST be divisible by g. Dtype: float.

```
Ho = floor((H + top + bottom - dh*(KH-1) - 1) / sh) + 1
Wo = floor((W + left + right - dw*(KW-1) - 1) / sw) + 1
```

If H or W is symbolic, Ho and Wo are dimension expressions as permitted by
spec/graph.md.

```
Cg = C / g;  Fg = F / g
Xp = pad[pads=[0,0, 0,0, top, bottom + sh - 1, left, right + sw - 1]](X)
Xg = reshape(Xp, [Nb, g, Cg, 1, Hp, Wp])            # Hp, Wp: padded sizes
Wg = reshape(permute[perm=[0, 2, 1, 3, 4]](reshape(Wt, [g, Fg, Cg, KH, KW])),
             [1, g, Cg, Fg, KH, KW])
acc = none
for i in 0..KH-1:
  for j in 0..KW-1:
    xs = take_strided(Xg, 4, i*dh, Ho, sh)
    xs = take_strided(xs, 5, j*dw, Wo, sw)          # [Nb, g, Cg, 1, Ho, Wo]
    ws = slice(slice(Wg, 4, i, i+1), 5, j, j+1)     # [1, g, Cg, Fg, 1, 1]
    t  = reduce[add, axes=[2]](mul(xs, ws))         # [Nb, g, 1, Fg, Ho, Wo]
    acc = t if acc is none else add(acc, t)
Y = reshape(acc, [Nb, F, Ho, Wo])
if B: Y = add(Y, reshape(B, [1, F, 1, 1]))
```

The extra `s - 1` elements of padding at the end of each spatial axis keep
every `take_strided` within bounds and never contribute to the output.

Tolerance: with Q = Cg * KH * KW, `abs Q u sum |x w|` over the receptive
field of the output element, plus `u |B_f|`.

## 4. Elementwise functions

The constants below are written to float64 precision; `tensor_const` and
`k` round them to the computation dtype. `log2e = 1.4426950408889634`,
`ln2 = 0.6931471805599453`, `pi = 3.141592653589793`.

### 4.1 sigmoid.v1

Signature: `x` -> `y`, float, same shape.

```
y = recip(add(k(1), exp2(mul(neg(x), k(log2e)))))
```

For large negative x, `exp2` overflows to +inf and `y` is +0; for large
positive x, `y` is 1. Tolerance: `rel (16 + 2|x|) u`.

### 4.2 tanh.v1

```
y = sub(mul(k(2), sigmoid.v1(mul(k(2), x))), k(1))
```

Tolerance: `abs 16 u`. (Near zero this form loses relative accuracy; the
tolerance is absolute.)

### 4.3 erf.v1

Abramowitz and Stegun formula 7.1.26, whose absolute error against the true
error function is at most 1.5e-7.

```
p  = 0.3275911
a1 = 0.254829592;  a2 = -0.284496736;  a3 = 1.421413741
a4 = -1.453152027; a5 = 1.061405429
s  = where(cmplt(x, k(0)), k(-1), k(1))
ax = abs(x)
t  = recip(add(k(1), mul(k(p), ax)))
q  = mul(add(mul(add(mul(add(mul(add(mul(k(a5), t), k(a4)), t), k(a3)), t),
                     k(a2)), t), k(a1)), t)
y  = mul(s, sub(k(1), mul(q, exp.v1(neg(sq(ax))))))
```

Tolerance: `abs 1.6e-7 + 16 u`. The 1.6e-7 term lets a fast path that
computes the true error function pass against this approximate reference.

### 4.4 exp.v1

```
y = exp2(mul(x, k(log2e)))
```

Tolerance: `rel (16 + 2|x|) u`. The `|x|` term is the effect of rounding
`x * log2e` before `exp2`.

### 4.5 log.v1

```
y = mul(log2(x), k(ln2))
```

Tolerance: `rel 16 u`. Special values follow `log2`.

### 4.6 cos.v1

```
y = sin(sub(k(pi / 2), x))
```

Tolerance: for `|x| <= 2^16`, `abs (16 + 2|x|) u`; beyond, as `sin`
(spec/primitives.md section 3.5).

### 4.7 rsqrt.v1

```
y = recip(sqrt(x))
```

Tolerance: `rel 6 u`.

### 4.8 silu.v1

```
y = mul(x, sigmoid.v1(x))
```

Tolerance: `rel (20 + 2|x|) u`.

### 4.9 gelu_erf.v1

```
y = mul(mul(k(0.5), x), add(k(1), erf.v1(mul(x, k(0.7071067811865476)))))
```

Tolerance: `abs (1.6e-7 + 32 u) max(1, |x|)`.

### 4.10 gelu_tanh.v1

```
c = 0.7978845608028654                     # sqrt(2 / pi)
inner = mul(k(c), add(x, mul(k(0.044715), mul(x, sq(x)))))
y = mul(mul(k(0.5), x), add(k(1), tanh.v1(inner)))
```

Tolerance: `abs 32 u max(1, |x|)`.

### 4.11 swiglu.v1

Signature: inputs `a` (gate), `b` (up), equal dtype and broadcast-compatible
shapes; output `y`.

```
y = mul(silu.v1(a), b)
```

Tolerance: `rel (20 + 2|a|) u`.

### 4.12 geglu.v1

Signature: inputs `a` (gate), `b` (up); attribute `approximate` (string,
"none" or "tanh", default "none"); output `y`.

```
g = gelu_erf.v1(a) if approximate == "none" else gelu_tanh.v1(a)
y = mul(g, b)
```

Tolerance: the tolerance of the selected GELU times `|b|`, plus `u |r|`.

## 5. Normalisation and reductions

### 5.1 softmax.v1

Signature: input `x`; attribute `axis` (int, default -1); output `y`, same
shape.

```
m = reduce[max, axes=[axis]](x)
e = exp.v1(sub(x, m))
s = reduce[add, axes=[axis]](e)
y = div(e, s)
```

The body MUST subtract the maximum before exponentiating. Without it,
`exp` overflows for inputs above about 88 in f32 and the result is NaN. An
importer MUST NOT copy a softmax function body from another format (such as
an ONNX function definition) that lacks this step; it MUST map the operation
to softmax.v1. If every element along the axis is -inf, the result is NaN.

Tolerance: for each element, `rel (N + 32 + 2|x_i - m|) u`, where N is the
axis length and m the maximum.

### 5.2 log_softmax.v1

Signature: as softmax.v1.

```
m = reduce[max, axes=[axis]](x)
d = sub(x, m)
y = sub(d, log.v1(reduce[add, axes=[axis]](exp.v1(d))))
```

Tolerance: `abs (N + 40) u + 2 u |x_i - m|`.

### 5.3 layer_norm.v1

Signature: inputs `X`, `scale` with shape `X_shape[axis:]`, optional `bias`
of the same shape; attributes `axis` (int, default -1), `epsilon` (float,
default 1e-5); output `Y`, shape and dtype of X. The statistics are computed
in f32 (or f64 for f64 inputs) whatever the input dtype. This follows the
function body of ONNX LayerNormalization-17 with stash_type f32, including
its cast of the normalised value back to the input dtype before scale and
bias.

```
axes = [axis, ..., r-1]
Xf   = cast[f32](X)                                  # identity for f32
n    = count(Xf, axes)
mu   = div(reduce[add, axes](Xf), n)
D    = sub(Xf, mu)
var  = div(reduce[add, axes](sq(D)), n)
inv  = recip(sqrt(add(var, k(epsilon))))
Nm   = cast[dtype(X)](mul(D, inv))
Y    = mul(Nm, reshape(scale, [1]*axis ++ scale_shape))
if bias: Y = add(Y, reshape(bias, [1]*axis ++ bias_shape))
```

Here the f16 and bf16 entry cast of section 1.4 does not apply to X itself;
the body casts explicitly, as above. The optional mean and inverse standard
deviation outputs of the ONNX operator are not part of v1.

Tolerance: let N be the number of normalised elements, sigma =
sqrt(var + epsilon) and kappa = max(1, |mu| / sigma) from the reference,
and n_i the reference normalised value. Then
`abs (2N + 32) kappa u (1 + |n_i|) |scale_i| + 2 u |bias_i|`, plus 1 ULP of
the input dtype for the intermediate cast when it is f16 or bf16.

### 5.4 rms_norm.v1

Signature: inputs `x`, optional `scale` with shape `x_shape[axis:]`;
attributes `axis` (int, default -1), `epsilon` (float, default 1e-6);
output `y`, shape and dtype of x. Computed in f32 (f64 for f64 inputs).

```
axes = [axis, ..., r-1]
ms   = div(reduce[add, axes](sq(x)), count(x, axes))
y    = mul(x, rsqrt.v1(add(ms, k(epsilon))))
if scale: y = mul(y, reshape(scale, [1]*axis ++ scale_shape))
```

Tolerance: `rel (N + 24) u`.

### 5.5 mean.v1

Signature: input `x`; attribute `axes` (ints); output with the reduced axes
kept as size 1 (as `reduce`). Dtype: float.

```
y = div(reduce[add, axes](x), count(x, axes))
```

Tolerance: `abs (N + 4) u mean(|x|)`.

### 5.6 variance.v1

Signature: input `x`; attributes `axes` (ints), `correction` (int, default
0; 1 gives the unbiased estimator); output with reduced axes kept as size 1.
Dtype: float. Two-pass form.

```
n = count(x, axes)
d = sub(x, mean.v1[axes](x))
y = div(reduce[add, axes](sq(d)), sub(n, k(correction)))
```

If `N - correction <= 0` the result is +inf or NaN as the division gives.
Tolerance: `abs (3N + 16) u mean(x^2)`.

### 5.7 cumsum.v1

Signature: input `x`; attributes `axis` (int), `exclusive` (bool, default
false), `reverse` (bool, default false); output `y`, shape and dtype of x.
Dtype: float or integer (integers wrap). Element i along the axis is the sum
of elements 0..i (inclusive), 0..i-1 (exclusive), and with `reverse` the
same counted from the end.

The canonical body is a sequential `while` loop over the axis. It works for
a symbolic axis length L, because L is computed as data from a constant of
the same shape. It performs L iterations and is meant as a definition, not
an implementation; runtimes are expected to fast-path it.

```
a    = axis
ones = const[value=1](i64)                 # shape: x_shape on a, 1 elsewhere
L    = reshape(reduce[add, axes=[a]](ones), [])          # i64 scalar
acc0 = const[value=0]                      # shape: x_shape with axis a = 1
y0   = const[value=0]                      # shape: x_shape
i0   = const[value=0](i64)                 # shape: []

cond(i, acc, y) = cmplt(i, L)
body(i, acc, y):
    j    = reshape(sub(sub(L, k(1)), i) if reverse else i, [1])
    xj   = gather[axis=a](x, j)            # shape: x_shape with axis a = 1
    if exclusive:
        y'   = scatter[axis=a](y, j, acc)
        acc' = add(acc, xj)
    else:
        acc' = add(acc, xj)
        y'   = scatter[axis=a](y, j, acc')
    return (add(i, k(1)), acc', y')

_, _, y = while[cond, body](i0, acc0, y0)
```

Tolerance: float, `abs L u sum_{j in prefix} |x_j|`; integer, exact.

## 6. Transformer building blocks

### 6.1 rope.v1

Rotary position embedding. Signature: inputs `x: [B, H, T, D]` (float),
`positions: [B, T]` or `[1, T]` (i32 or i64); attributes:

| attribute | type | default | meaning |
|---|---|---|---|
| theta | float | 10000.0 | frequency base |
| rotary_dims | int | D | leading dims of the last axis that rotate; even, <= D |
| interleaved | bool | false | false: pairs (i, i + R/2) ("half" layout); true: pairs (2i, 2i+1) |
| yarn_factor | float | 1.0 | context extension factor; 1.0 disables YaRN |
| yarn_original_context | int | 0 | pre-extension context length; REQUIRED if yarn_factor != 1 |
| yarn_beta_fast | float | 32.0 | YaRN high-frequency boundary, in rotations |
| yarn_beta_slow | float | 1.0 | YaRN low-frequency boundary, in rotations |
| yarn_attn_factor | float | 0.1 ln(yarn_factor) + 1 | scale applied to cos and sin |

Output: shape and dtype of x. Dims at or beyond `rotary_dims` pass through
unchanged.

Frequencies, computed by the inliner in float64 from attributes only
(R = rotary_dims, h = R/2, i = 0..h-1):

```
f_i = theta^(-2i / R)
if yarn_factor == 1:
    inv_freq_i = f_i;  ms = 1
else:
    corr(n) = R * ln(yarn_original_context / (n * 2 * pi)) / (2 * ln(theta))
    lo = max(floor(corr(yarn_beta_fast)), 0)
    hi = min(ceil(corr(yarn_beta_slow)), R - 1)
    if lo == hi: hi = hi + 0.001
    ramp_i = clamp((i - lo) / (hi - lo), 0, 1)
    inv_freq_i = (f_i / yarn_factor) * ramp_i + f_i * (1 - ramp_i)
    ms = yarn_attn_factor
```

Body:

```
F   = tensor_const(inv_freq)                    # [1, 1, 1, h]
P   = reshape(cast[f32](positions), [-1, 1, T, 1])
ang = mul(P, F)                                 # [B, 1, T, h]
c   = mul(cos.v1(ang), k(ms))
s   = mul(sin(ang), k(ms))
xr  = slice(x, 3, 0, R)
if interleaved:
    xi = reshape(xr, [B, H, T, h, 2])
    x1 = squeeze(slice(xi, 4, 0, 1), 4);  x2 = squeeze(slice(xi, 4, 1, 2), 4)
else:
    x1 = slice(xr, 3, 0, h);  x2 = slice(xr, 3, h, R)
y1 = sub(mul(x1, c), mul(x2, s))
y2 = add(mul(x2, c), mul(x1, s))
if interleaved:
    yr = reshape(concat(unsqueeze(y1, 4), unsqueeze(y2, 4), 4), [B, H, T, R])
else:
    yr = concat(y1, y2, 3)
y = concat(yr, slice(x, 3, R, D), 3) if R < D else yr
```

The angle is computed in the computation dtype (f32 for f16, bf16 and f32
inputs), as most reference implementations do. Positions above 2^24 are not
exactly representable in f32.

Tolerance: let A be the largest `|positions| * inv_freq_0` in the call. For
each rotated pair, `abs (16 + 4A) u ms max(|x1|, |x2|)`; pass-through
elements are exact. A fast path MAY compute angles in higher precision.

### 6.2 attention.v1

Scaled dot-product attention with grouped-query heads. Signature:

| input | shape | notes |
|---|---|---|
| q | [B, Hq, Tq, D] | float |
| k | [B, Hkv, Tk, D] | dtype of q |
| v | [B, Hkv, Tk, Dv] | dtype of q |
| mask | rank 4, broadcastable to [B, Hq, Tq, Tk] | optional; bool (true = attend) or float (added to the scores) |
| sinks | [Hq] | optional; float; one extra logit per head that joins the softmax denominator only |

| attribute | type | default | meaning |
|---|---|---|---|
| scale | float | 1 / sqrt(D) | score scale |
| causal | bool | false | apply a causal mask aligned to the end (see below) |
| num_kv_heads | int | Hq | Hkv; Hq MUST be divisible by it |
| softcap | float | 0 | if > 0, scores become softcap * tanh(scores / softcap) |

Output: `o: [B, Hq, Tq, Dv]`, dtype of q. Query head h uses key and value
head `floor(h / G)`, where `G = Hq / Hkv`. With `causal`, query i may attend
key j if and only if `j <= i + (Tk - Tq)`, so a query block at the end of a
longer key sequence (decoding with a cache) is masked correctly. Scores and
softmax are computed in f32 (f64 for f64 inputs).

```
G  = Hq / Hkv
rep(t) = reshape(expand(unsqueeze(t, 2), [B, Hkv, G, Tk, -1]), [B, Hq, Tk, -1])
kf = rep(k);  vf = rep(v)
s  = mul(matmul.v1(q, permute[perm=[0, 1, 3, 2]](kf)), k(scale))   # [B, Hq, Tq, Tk]
if softcap > 0:
    s = mul(tanh.v1(div(s, k(softcap))), k(softcap))
if mask is bool:
    s = add(s, where(mask, k(0), k(-inf)))
if mask is float:
    s = add(s, cast[f32](mask))
if causal:
    rq = riota(s, 2)                       # Tq-1-i, shape [1, 1, Tq, 1]
    rk = riota(s, 3)                       # Tk-1-j, shape [1, 1, 1, Tk]
    s  = add(s, where(not(cmplt(rk, rq)), k(0), k(-inf)))
if sinks:
    z = reshape(cast[f32](sinks), [1, Hq, 1, 1])
    m = max(reduce[max, axes=[3]](s), z)
    e = exp.v1(sub(s, m))
    p = div(e, add(reduce[add, axes=[3]](e), exp.v1(sub(z, m))))
else:
    p = softmax.v1[axis=3](s)
o = matmul.v1(p, vf)
```

`rk >= rq` is the same condition as `j <= i + (Tk - Tq)`, written without
shape arithmetic so that it holds for symbolic Tq and Tk. A query row whose
keys are all masked yields NaN in the canonical body (without sinks); the
result for such rows is unspecified and a fast path MAY return NaN or 0.

Tolerance: let `S = scale * max_ij sum_d |q_id k_jd|` and
`V = max_j |v_j|` over the row. For each output element,
`abs (4 D S + 2 Tk + 64) u V`.

### 6.3 embedding.v1

Signature: inputs `table: [V, E]`, `ids` (any rank, i32 or i64); output
`[ids_shape..., E]`.

```
t = dequantize(table) if table is a block or storage-float dtype else table
y = gather[axis=0](t, ids)
```

Exact (bit-exact to the reference decode for quantized tables).
Out-of-range ids behave as in `gather`. A runtime SHOULD decode only the
selected rows of a quantized table.

## 7. Selection and sorting

The canonical bodies of top_k and argsort would be selection loops: for
each of k (or L) steps, find the maximum not yet taken with `reduce`, locate
its lowest index with `cmpeq`, `iota` and a minimum, and mark it taken with
`scatter`. That is expressible with the primitives, but costs O(k L) work
and O(k) loop iterations per row, and its tie and NaN handling would be
harder to read than the definition it implements. For these two composites
the reference definitions below are normative in place of a body. A runtime
without a fast path MUST implement them directly. See Open issues.

Ordering used by both: integers in their natural order; floats in the
order `-inf < ... < -0 = +0 < ... < +inf < NaN`, where -0 and +0 compare
equal and all NaN compare equal. Equal elements are ordered by lower index
first.

### 7.1 top_k.v1

Signature: input `x` (float or integer), attributes `k` (int, 0 <= k <=
L), `axis` (int, default -1), `largest` (bool, default true); outputs
`values` (dtype of x) and `indices` (i64), both with `x_shape` except that
the axis has length k. L is the axis length; `k > L` is a shape error.

```
for each position p of the other axes:
    row   = [(x[p, j], j) for j in 0..L-1]
    order = sort row by value, descending if largest else ascending,
            ties by ascending index j
    for t in 0..k-1:
        values[p, t]  = order[t].value          # bit-identical copy
        indices[p, t] = order[t].index
```

With `largest = true`, NaN elements come first; with `largest = false`,
they come last. Exact.

### 7.2 argsort.v1

Signature: input `x` (float or integer), attributes `axis` (int, default
-1), `descending` (bool, default false); output `indices` (i64), shape of x.

```
for each position p of the other axes:
    indices[p, :] = stable sort of 0..L-1 by x[p, j],
                    ascending, or descending if descending,
                    equal elements in ascending index order in both cases
```

NaN elements sort last when ascending and first when descending. Exact.

## 8. Open issues

1. Sort-like operations. top_k.v1 and argsort.v1 have reference definitions
   instead of canonical bodies. Under the admission rule of
   spec/primitives.md section 8 a sort is not a primitive, because it is
   expressible as a selection loop. But a runtime without a fast path then
   has no practical body to inline, which weakens the claim that the
   primitives alone run every model. Phase 2 decides between (a) admitting
   a `sort` primitive (returning values and stable indices) as a second
   explicit exception, alongside `dequantize`, or (b) publishing the
   selection-loop bodies as normative and keeping the reference definitions
   as informative.
2. Sequential scans. cumsum.v1, and `iota` and `riota` built on it, use an
   L-iteration `while` loop. They are correct for symbolic lengths but slow
   without a fast path, and `attention.v1` with `causal` depends on them.
   Speed is not grounds for a primitive; the alternative is a parallel-scan
   body for static lengths. To be decided with item 1.
3. Activation quantization. Kernels that quantize x in dequant_matmul (as
   llama.cpp does with 8-bit activation blocks) exceed the tolerance of
   dequant_matmul.v1. Either a separate composite (for example
   `dequant_matmul_q8act.v1`) with its own looser tolerance, or a declared
   per-runtime deviation, is needed.
4. Tolerances. All numbers in this document are provisional; Phase 2
   calibrates them by running the canonical bodies in f32 and f16 against
   the float64 reference on the conformance corpus.
5. Body macros. `sub`, `div`, `min`, `abs`, `concat` and `take_strided` are
   notation only. If importers need them as stable named operations, they
   can become composites without changing the primitive set.
6. Fully masked attention rows are unspecified. A later version may fix the
   result to 0, which is what most fused kernels return.
