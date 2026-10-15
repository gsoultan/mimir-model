#!/usr/bin/env python3
"""Generate the primitive conformance corpus under testdata/conformance.

Run from the repository root:

    python3 -I tools/gen_conformance.py            # write the corpus
    python3 -I tools/gen_conformance.py --check    # fail if it is out of date

The expected results are computed independently of the Go code, from the
definitions in spec/primitives.md and spec/dtypes.md. Exact operations are
evaluated in rational arithmetic (fractions.Fraction); transcendental
functions are evaluated with mpmath at 256 bits. Every result is then
rounded once, to nearest with ties to even, to the output dtype. NumPy is
used only for the deterministic random inputs and for the float32 order of
operations in the block-dtype decoders (spec/dtypes.md section 1.1).

The file format is described in spec/conformance.md.
"""

import json
import math
import os
import sys
from fractions import Fraction

import mpmath
import numpy as np

mpmath.mp.prec = 256

OUT_DIR = os.path.join("testdata", "conformance", "primitives")

# ---------------------------------------------------------------------------
# Number formats

# name: (precision including the hidden bit, emin, exponent bits, has inf)
FLOATS = {
    "f64": (53, -1022, 11, True),
    "f32": (24, -126, 8, True),
    "f16": (11, -14, 5, True),
    "bf16": (8, -126, 8, True),
    "f8_e4m3": (4, -6, 4, False),
    "f8_e5m2": (3, -14, 5, True),
}
ARITH_FLOATS = ["f64", "f32", "f16", "bf16"]
STORAGE_FLOATS = ["f8_e4m3", "f8_e5m2"]

# name: (bit width, signed)
INTS = {
    "i8": (8, True), "u8": (8, False), "i16": (16, True), "u16": (16, False),
    "i32": (32, True), "u32": (32, False), "i64": (64, True), "u64": (64, False),
}
INT_NAMES = list(INTS)

OPCODES = {
    "neg": 1, "recip": 2, "sqrt": 3, "exp2": 4, "log2": 5, "sin": 6, "trunc": 7,
    "cast": 8, "bitcast": 9, "add": 10, "mul": 11, "max": 12, "pow": 13,
    "idiv": 14, "imod": 15, "shl": 16, "shr": 17, "and": 18, "or": 19,
    "xor": 20, "cmplt": 21, "cmpeq": 22, "where": 23, "reduce": 24,
    "reshape": 25, "expand": 26, "permute": 27, "pad": 28, "shrink": 29,
    "gather": 30, "scatter": 31, "const": 32, "dequantize": 33,
}

BLOCKS = {  # name: (block elements, block bytes)
    "q8_0": (32, 34), "q4_0": (32, 18), "q4_k": (256, 144),
    "q6_k": (256, 210), "mxfp4": (32, 17), "affine4_g64": (64, 36),
}


def width(dt):
    if dt in FLOATS:
        p, _, eb, _ = FLOATS[dt]
        return p + eb
    if dt in INTS:
        return INTS[dt][0]
    if dt == "bool":
        return 8
    raise ValueError(dt)


def is_float(dt):
    return dt in ARITH_FLOATS


def fmt_params(dt):
    p, emin, eb, has_inf = FLOATS[dt]
    bias = 1 - emin
    if has_inf:
        emax = (1 << eb) - 2 - bias
        maxf = (2 - Fraction(1, 1 << (p - 1))) * Fraction(2) ** emax
    else:  # e4m3: top exponent is usable except for the all-ones mantissa
        emax = (1 << eb) - 1 - bias
        maxf = (2 - Fraction(2, 1 << (p - 1))) * Fraction(2) ** emax
    return p, emin, eb, has_inf, bias, maxf


def decode(dt, bits):
    """The exact value of a float bit pattern, as a Python float."""
    p, emin, eb, has_inf, bias, _ = fmt_params(dt)
    w = p + eb
    sign = bits >> (w - 1)
    e = (bits >> (p - 1)) & ((1 << eb) - 1)
    m = bits & ((1 << (p - 1)) - 1)
    allones = (1 << eb) - 1
    if has_inf and e == allones:
        v = math.inf if m == 0 else math.nan
    elif not has_inf and e == allones and m == (1 << (p - 1)) - 1:
        v = math.nan
    elif e == 0:
        v = math.ldexp(m, emin - (p - 1))
    else:
        v = math.ldexp((1 << (p - 1)) + m, e - bias - (p - 1))
    return -v if sign else v


def nan_bits(dt, neg=False):
    p, _, eb, has_inf, _, _ = fmt_params(dt)
    w = p + eb
    if has_inf:
        b = (((1 << eb) - 1) << (p - 1)) | (1 << (p - 2))
    else:
        b = (1 << (w - 1)) - 1
    return b | (1 << (w - 1)) if neg else b


def ilog2(a):
    """floor(log2(a)) for a positive Fraction."""
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    return e


def to_frac(v):
    if isinstance(v, Fraction):
        return v
    if isinstance(v, mpmath.mpf):
        s, man, exp, _ = v._mpf_
        f = Fraction(man) * Fraction(2) ** exp
        return -f if s else f
    return Fraction(v)


def round_to(dt, v, sat=False):
    """Round v to dt, nearest with ties to even, and return the bit pattern.

    v is a Python float (any value, including signed zeros, infinities and
    NaN) or an exact non-zero Fraction or mpf. Finite values beyond the
    largest finite value become infinity, or the signed largest finite value
    when sat is set or the format has no infinity.
    """
    p, emin, eb, has_inf, bias, maxf = fmt_params(dt)
    w = p + eb
    signbit = 1 << (w - 1)
    inf_bits = ((1 << eb) - 1) << (p - 1)
    if isinstance(v, float):
        if math.isnan(v):
            return nan_bits(dt)
        if math.isinf(v):
            s = signbit if v < 0 else 0
            if has_inf:
                return s | inf_bits
            return s | round_to(dt, maxf)
        if v == 0:
            return signbit if math.copysign(1, v) < 0 else 0
    v = to_frac(v)
    if v == 0:
        raise ValueError("an exact zero must be passed as a signed float")
    s = signbit if v < 0 else 0
    a = abs(v)
    e = max(ilog2(a), emin)
    q = Fraction(2) ** (e - (p - 1))
    n = a / q
    ni = n.numerator // n.denominator
    rem = n - ni
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and ni % 2 == 1):
        ni += 1
    if ni == 0:
        return s
    val = ni * q
    if val > maxf:
        if has_inf and not sat:
            return s | inf_bits
        val = maxf
        e = ilog2(maxf)
        ni = int(maxf / Fraction(2) ** (e - (p - 1)))
    if ni < (1 << (p - 1)):
        return s | ni
    if ni == (1 << p):
        ni >>= 1
        e += 1
    return s | ((e + bias) << (p - 1)) | (ni - (1 << (p - 1)))


def flush(dt, bits):
    """Flush-to-zero: a subnormal becomes a zero of the same sign."""
    p, _, eb, _, _, _ = fmt_params(dt)
    w = p + eb
    if (bits >> (p - 1)) & ((1 << eb) - 1) == 0:
        return bits & (1 << (w - 1))
    return bits


def is_subnormal(dt, bits):
    p, _, eb, _, _, _ = fmt_params(dt)
    return (bits >> (p - 1)) & ((1 << eb) - 1) == 0 and bits & ((1 << (p - 1)) - 1) != 0


def ulp(dt, bits):
    """The spacing of dt at the magnitude of the value bits (finite only)."""
    p, emin, _, _, _, _ = fmt_params(dt)
    v = abs(decode(dt, bits))
    e = emin if v == 0 else max(math.frexp(v)[1] - 1, emin)
    return math.ldexp(1.0, e - (p - 1))


# ---------------------------------------------------------------------------
# Integers

def ival(dt, bits):
    w, signed = INTS[dt]
    if signed and bits >> (w - 1):
        return bits - (1 << w)
    return bits


def ibits(dt, v):
    w, _ = INTS[dt]
    return v % (1 << w)


def irange(dt):
    w, signed = INTS[dt]
    if signed:
        return -(1 << (w - 1)), (1 << (w - 1)) - 1
    return 0, (1 << w) - 1


# ---------------------------------------------------------------------------
# Tensors and cases. A tensor is (dtype, shape, list of bit patterns).

def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def pack(dt, bits):
    nb = width(dt) // 8
    return b"".join(b.to_bytes(nb, "little") for b in bits).hex()


def tensor(dt, shape, bits):
    assert numel(shape) == len(bits), (dt, shape, len(bits))
    return {"dtype": dt, "shape": list(shape), "data": pack(dt, bits)}


def f64hex(values):
    return b"".join(np.array(values, dtype="<f8").tobytes() for _ in [0]).hex()


def attr_int(v):
    return {"int": int(v)}


def attr_ints(v):
    return {"ints": [int(x) for x in v]}


def attr_float(v):
    return {"float": np.array([v], dtype="<f8").tobytes().hex()}


def attr_str(v):
    return {"string": v}


def attr_dtype(v):
    return {"dtype": v}


class Corpus:
    def __init__(self):
        self.ops = {}

    def add(self, op, name, inputs, out, check, attrs=None, ftz=None, dont_care=None):
        case = {"name": name, "inputs": inputs, "output": out, "check": check}
        if attrs:
            case["attrs"] = attrs
        if ftz is not None:
            case["output_ftz"] = ftz
        if dont_care:
            case["dont_care"] = sorted(set(dont_care))
        self.ops.setdefault(op, []).append(case)

    def files(self):
        out = {}
        for op, cases in self.ops.items():
            names = [c["name"] for c in cases]
            assert len(names) == len(set(names)), op
            head = {"format": "mmf-conformance", "version": 1, "op": op, "opcode": OPCODES[op]}
            body = ",\n".join(json.dumps(c, sort_keys=True, separators=(",", ":")) for c in cases)
            text = json.dumps(head, sort_keys=True)[:-1] + ', "cases": [\n' + body + "\n]}\n"
            out[op + ".json"] = text
        return out


C = Corpus()


# ---------------------------------------------------------------------------
# Input values

def float_specials(dt):
    p, emin, eb, has_inf, bias, maxf = fmt_params(dt)
    vals = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 1.5, -2.5, 3.0, 0.1, -1 / 3, math.pi, 100.0,
            float(maxf), -float(maxf),
            math.ldexp(1.0, emin), -math.ldexp(1.0, emin),
            math.ldexp(1.0, emin - (p - 1)), -math.ldexp(1.0, emin - (p - 1)),
            math.ldexp(1.0, emin) - math.ldexp(1.0, emin - (p - 1)),
            math.inf, -math.inf, math.nan]
    return [round_to(dt, v) for v in vals]


def float_random(rng, dt, n, lo=-20, hi=20, neg=True):
    p, emin, eb, has_inf, bias, maxf = fmt_params(dt)
    e = rng.uniform(lo, hi, size=n)
    m = rng.uniform(1.0, 2.0, size=n)
    s = rng.choice([-1.0, 1.0], size=n) if neg else np.ones(n)
    return [round_to(dt, float(si * mi * 2.0 ** ei)) for si, mi, ei in zip(s, m, e)]


def int_specials(dt):
    lo, hi = irange(dt)
    vals = {0, 1, 2, 3, 7, hi, hi - 1, lo, lo + 1, (hi + 1) // 2, hi // 3}
    if lo < 0:
        vals |= {-1, -2, -3, -7, -8}
    return [ibits(dt, v) for v in sorted(vals)]


def int_random(rng, dt, n):
    w, _ = INTS[dt]
    return [int(x) for x in rng.integers(0, 1 << w, size=n, dtype=np.uint64)] if w == 64 \
        else [int(x) for x in rng.integers(0, 1 << w, size=n)]


# ---------------------------------------------------------------------------
# Elementwise reference functions on exact values.
#
# Each float function takes Python floats (exact inputs) and returns a float
# (for special or zero results) or an exact Fraction / mpf.

def mp(x):
    return mpmath.mpf(x)


def f_neg(x):
    return -x


def f_recip(x):
    if math.isnan(x):
        return math.nan
    if x == 0:
        return math.copysign(math.inf, x)
    if math.isinf(x):
        return math.copysign(0.0, x)
    return 1 / Fraction(x)


def f_sqrt(x):
    if math.isnan(x) or x < 0:
        return math.nan
    if x == 0 or math.isinf(x):
        return x
    return mpmath.sqrt(mp(x))


def f_exp2(x):
    if math.isnan(x):
        return math.nan
    if math.isinf(x):
        return 0.0 if x < 0 else math.inf
    if x == 0:
        return 1.0
    if x > 2000:
        return math.inf
    if x < -2000:
        return 0.0
    return mpmath.power(2, mp(x))


def f_log2(x):
    if math.isnan(x) or x < 0:
        return math.nan
    if x == 0:
        return -math.inf
    if math.isinf(x):
        return math.inf
    if x == 1:
        return 0.0
    m, e = math.frexp(x)
    if m == 0.5:
        return float(e - 1)
    return mpmath.log(mp(x), 2)


def f_sin(x):
    if math.isnan(x) or math.isinf(x):
        return math.nan
    if x == 0:
        return x
    return mpmath.sin(mp(x))


def f_trunc(x):
    if math.isnan(x) or math.isinf(x):
        return x
    t = float(math.trunc(x))
    return math.copysign(t, x) if t == 0 else t


def is_odd_int(y):
    return math.isfinite(y) and y == math.floor(y) and int(y) % 2 == 1


def f_pow(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.nan
    if b == 0:
        return 1.0
    if a == 0:
        if b < 0:
            return math.copysign(math.inf, a) if is_odd_int(b) else math.inf
        return math.copysign(0.0, a) if is_odd_int(b) else 0.0
    if math.isinf(b):
        if a == -1 or a == 1:
            return 1.0
        if (abs(a) < 1) == (b > 0):
            return 0.0
        return math.inf
    if math.isinf(a):
        if a > 0:
            return 0.0 if b < 0 else math.inf
        if b < 0:
            return -0.0 if is_odd_int(b) else 0.0
        return -math.inf if is_odd_int(b) else math.inf
    if a == 1:
        return 1.0
    if a < 0:
        if b != math.floor(b):
            return math.nan
        r = mpmath.power(mp(-a), mp(b))
        return -r if int(b) % 2 else r
    if b == math.floor(b) and abs(b) < 4096:
        return Fraction(a) ** int(b)
    return mpmath.power(mp(a), mp(b))


def f_add(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.nan
    if math.isinf(a) or math.isinf(b):
        if math.isinf(a) and math.isinf(b) and a != b:
            return math.nan
        return a if math.isinf(a) else b
    s = Fraction(a) + Fraction(b)
    if s == 0:
        if a == 0 and b == 0 and math.copysign(1, a) < 0 and math.copysign(1, b) < 0:
            return -0.0
        return 0.0
    return s


def f_mul(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.nan
    neg = (math.copysign(1, a) < 0) != (math.copysign(1, b) < 0)
    if math.isinf(a) or math.isinf(b):
        if a == 0 or b == 0:
            return math.nan
        return -math.inf if neg else math.inf
    if a == 0 or b == 0:
        return -0.0 if neg else 0.0
    return Fraction(a) * Fraction(b)


def f_max(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.nan
    if a == 0 and b == 0:
        return 0.0 if (math.copysign(1, a) > 0 or math.copysign(1, b) > 0) else -0.0
    return a if a > b else b


FLOAT_UNARY = {"neg": f_neg, "recip": f_recip, "sqrt": f_sqrt, "exp2": f_exp2,
               "log2": f_log2, "sin": f_sin, "trunc": f_trunc}
FLOAT_BINARY = {"add": f_add, "mul": f_mul, "max": f_max, "pow": f_pow}

EXACT_OPS = {"neg", "trunc", "add", "mul", "max"}
ULP_BOUND = {"recip": 1, "sqrt": 1, "exp2": 5, "log2": 5, "pow": 5}


def float_result_bits(op, dt, args):
    fn = FLOAT_UNARY.get(op) or FLOAT_BINARY[op]
    r = fn(*args)
    if op == "neg" and isinstance(r, float) and math.isnan(r):
        return nan_bits(dt)
    return round_to(dt, r)


def float_check(op, dt, xs, out_bits):
    """The check object for a float elementwise result."""
    if op in EXACT_OPS:
        return {"kind": "exact"}
    if dt in ("f16", "bf16"):
        return {"kind": "ulp", "ulp": 1}
    if op == "sin":
        refs, bounds = [], []
        for x, b in zip(xs, out_bits):
            e = decode(dt, b)
            refs.append(e)
            if not math.isfinite(e):
                bounds.append(0.0)
            elif abs(x) <= 65536:
                bounds.append(max(5 * ulp(dt, b), 2.0 ** -22))
            else:
                bounds.append(2.0 ** -12)
        return {"kind": "tol", "ref": f64hex(refs), "bound": f64hex(bounds), "max_abs": 1.0}
    return {"kind": "ulp", "ulp": ULP_BOUND[op]}


def gen_float_elementwise(op, dt, name, shapes, columns):
    """columns: list of bit lists, one per input, already broadcast-compatible."""
    out_shape = bcast_shape(shapes)
    idx_lists = bcast_indices(shapes, out_shape)
    vals = [[decode(dt, b) for b in col] for col in columns]
    out_bits, ftz_bits, xs = [], [], []
    any_ftz_diff = False
    for idx in idx_lists:
        args = [vals[k][idx[k]] for k in range(len(columns))]
        xs.append(args[0])
        r = float_result_bits(op, dt, args)
        out_bits.append(r)
        fargs = [decode(dt, flush(dt, columns[k][idx[k]])) for k in range(len(columns))]
        fr = flush(dt, float_result_bits(op, dt, fargs))
        if op == "exp2" and fr != r and fr & (1 << (width(dt) - 1)):
            fr = 0
        ftz_bits.append(fr)
        if fr != r:
            any_ftz_diff = True
    inputs = [tensor(dt, s, c) for s, c in zip(shapes, columns)]
    check = float_check(op, dt, xs, out_bits)
    ftz = None
    if any_ftz_diff:
        fxs = [decode(dt, flush(dt, b)) for b in [columns[0][i[0]] for i in idx_lists]]
        ftz = {"data": pack(dt, ftz_bits), "check": float_check(op, dt, fxs, ftz_bits)}
    C.add(op, name, inputs, tensor(dt, out_shape, out_bits), check, ftz=ftz)


def bcast_shape(shapes):
    r = len(shapes[0])
    out = []
    for i in range(r):
        ds = [s[i] for s in shapes]
        out.append(0 if 0 in ds else max(ds))
    return out


def strides(shape):
    st, acc = [], 1
    for s in reversed(shape):
        st.append(acc)
        acc *= s
    return list(reversed(st))


def unravel(i, shape):
    idx = []
    for s in reversed(shape):
        idx.append(i % s)
        i //= s
    return list(reversed(idx))


def ravel(idx, shape):
    i = 0
    for k, s in zip(idx, shape):
        i = i * s + k
    return i


def bcast_indices(shapes, out_shape):
    res = []
    for i in range(numel(out_shape)):
        idx = unravel(i, out_shape)
        res.append([ravel([0 if s[k] == 1 else idx[k] for k in range(len(idx))], s) for s in shapes])
    return res


# ---------------------------------------------------------------------------
# Float elementwise generators

def unary_inputs(rng, op, dt):
    sp = float_specials(dt)
    p, emin, eb, _, bias, maxf = fmt_params(dt)
    emax = ilog2(maxf)
    if op == "exp2":
        r = float_random(rng, dt, 48, -6, 3) + \
            [round_to(dt, float(x)) for x in rng.uniform(emin - p - 4, emax + 2, size=48)]
    elif op in ("log2", "sqrt"):
        r = float_random(rng, dt, 64, emin - p + 1, emax, neg=False) + float_random(rng, dt, 8, -4, 4)
    elif op == "sin":
        r = float_random(rng, dt, 64, -8, 4)
        if dt in ("f32", "f64", "bf16"):
            r += float_random(rng, dt, 24, 4, min(emax, 100))
    else:
        r = float_random(rng, dt, 64, emin - p + 1, emax)
    return sp + r


def gen_float_unary(rng, op):
    for dt in ARITH_FLOATS:
        col = unary_inputs(rng, op, dt)
        gen_float_elementwise(op, dt, "%s/values" % dt, [[len(col)]], [col])
        # rank 0 and empty tensors
        gen_float_elementwise(op, dt, "%s/rank0" % dt, [[]], [[round_to(dt, 0.75)]])
        gen_float_elementwise(op, dt, "%s/empty" % dt, [[2, 0, 3]], [[]])


def binary_inputs(rng, op, dt):
    sp = float_specials(dt)
    sp_small = [round_to(dt, v) for v in [0.0, -0.0, 1.0, -1.0, 2.0, -0.5, math.inf, -math.inf, math.nan, 3.0]]
    p, emin, eb, _, bias, maxf = fmt_params(dt)
    emax = ilog2(maxf)
    a, b = [], []
    for x in sp:
        for y in sp_small:
            a.append(x)
            b.append(y)
    if op == "pow":
        n = 64
        a += float_random(rng, dt, n, -6, 6, neg=False)
        b += float_random(rng, dt, n, -3, 3)
        ia = float_random(rng, dt, 32, -2, 2)
        a += ia
        b += [round_to(dt, float(v)) for v in rng.integers(-7, 8, size=32)]
        # results near overflow and underflow
        a += [round_to(dt, 2.0)] * 4 + [round_to(dt, 0.5)] * 4
        b += [round_to(dt, float(v)) for v in (emax, emax + 1, emax - 0.5, emin - p + 2, -emax, -(emax + 1), emin, emin - p)]
    else:
        n = 128
        a += float_random(rng, dt, n, emin - p + 1, emax)
        b += float_random(rng, dt, n, emin - p + 1, emax)
        # cancellation and nearby magnitudes
        x = float_random(rng, dt, 32, -4, 4)
        a += x
        b += [round_to(dt, -decode(dt, v) * (1 + 2.0 ** -(p - 1))) for v in x]
        y = float_random(rng, dt, 32, -4, 4)
        a += y
        b += [round_to(dt, -decode(dt, v)) for v in y]
    return a, b


def gen_float_binary(rng, op):
    for dt in ARITH_FLOATS:
        a, b = binary_inputs(rng, op, dt)
        gen_float_elementwise(op, dt, "%s/values" % dt, [[len(a)], [len(b)]], [a, b])
        ba = float_random(rng, dt, 12, -3, 3, neg=op != "pow")
        bb = float_random(rng, dt, 20, -3, 3)
        if op == "pow":
            ba = float_random(rng, dt, 12, -3, 3, neg=False)
        gen_float_elementwise(op, dt, "%s/broadcast" % dt, [[3, 1, 4], [1, 5, 4]], [ba, bb])
        gen_float_elementwise(op, dt, "%s/broadcast_empty" % dt, [[0, 3], [1, 3]],
                              [[], float_random(rng, dt, 3, -1, 1, neg=False)])
        gen_float_elementwise(op, dt, "%s/rank0" % dt, [[], []], [[round_to(dt, 1.25)], [round_to(dt, 2.0)]])


# ---------------------------------------------------------------------------
# Integer elementwise

def i_neg(dt, a):
    return -a


def i_trunc(dt, a):
    return a


def i_idiv(dt, a, b):
    if b == 0:
        return None
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def i_imod(dt, a, b):
    q = i_idiv(dt, a, b)
    return None if q is None else a - q * b


def i_shl(dt, a, b):
    w, _ = INTS[dt]
    amt = ibits(dt, b)
    return 0 if amt >= w else a << amt


def i_shr(dt, a, b):
    w, signed = INTS[dt]
    amt = ibits(dt, b)
    if amt >= w:
        return -1 if a < 0 else 0
    return a >> amt


def bitwise(fn):
    def g(dt, a, b):
        return fn(ibits(dt, a), ibits(dt, b))
    return g


INT_FUNCS = {
    "neg": i_neg, "trunc": i_trunc,
    "add": lambda dt, a, b: a + b, "mul": lambda dt, a, b: a * b,
    "max": lambda dt, a, b: max(a, b), "idiv": i_idiv, "imod": i_imod,
    "shl": i_shl, "shr": i_shr,
    "and": bitwise(lambda a, b: a & b), "or": bitwise(lambda a, b: a | b),
    "xor": bitwise(lambda a, b: a ^ b),
    "cmplt": lambda dt, a, b: a < b, "cmpeq": lambda dt, a, b: a == b,
}


def gen_int_elementwise(op, dt, name, shapes, columns):
    out_shape = bcast_shape(shapes)
    out_dt = "bool" if op in ("cmplt", "cmpeq") else dt
    res, dc = [], []
    for i, idx in enumerate(bcast_indices(shapes, out_shape)):
        args = [ival(dt, columns[k][idx[k]]) for k in range(len(columns))]
        r = INT_FUNCS[op](dt, *args)
        if r is None:
            dc.append(i)
            r = 0
        if out_dt == "bool":
            res.append(1 if r else 0)
        else:
            res.append(ibits(dt, r))
    inputs = [tensor(dt, s, c) for s, c in zip(shapes, columns)]
    C.add(op, name, inputs, tensor(out_dt, out_shape, res), {"kind": "exact"}, dont_care=dc)


def gen_int_ops(rng, op):
    unary = op in ("neg", "trunc")
    for dt in INT_NAMES:
        w, _ = INTS[dt]
        sp = int_specials(dt)
        if unary:
            col = sp + int_random(rng, dt, 32)
            gen_int_elementwise(op, dt, "%s/values" % dt, [[len(col)]], [col])
            continue
        a, b = [], []
        for x in sp:
            for y in sp:
                a.append(x)
                b.append(y)
        a += int_random(rng, dt, 64)
        if op in ("shl", "shr"):
            amts = list(range(0, w + 3)) + [ibits(dt, -1), ibits(dt, -w)] if INTS[dt][1] else list(range(0, w + 3)) + [(1 << w) - 1]
            b += [ibits(dt, int(rng.choice(amts)))] * 0 + [ibits(dt, amts[i % len(amts)]) for i in range(64)]
        else:
            b += int_random(rng, dt, 64)
        gen_int_elementwise(op, dt, "%s/values" % dt, [[len(a)], [len(b)]], [a, b])
        gen_int_elementwise(op, dt, "%s/broadcast" % dt, [[2, 1, 3], [1, 4, 3]],
                            [int_random(rng, dt, 6), [ibits(dt, int(v)) for v in rng.integers(1, 6, size=12)]])


def gen_bool_ops(op):
    a = [0, 0, 1, 1]
    b = [0, 1, 0, 1]
    fn = {"and": lambda x, y: x & y, "or": lambda x, y: x | y, "xor": lambda x, y: x ^ y,
          "cmpeq": lambda x, y: int(x == y)}[op]
    C.add(op, "bool/values", [tensor("bool", [4], a), tensor("bool", [4], b)],
          tensor("bool", [4], [fn(x, y) for x, y in zip(a, b)]), {"kind": "exact"})
    C.add(op, "bool/broadcast", [tensor("bool", [2, 1], [0, 1]), tensor("bool", [1, 2], [0, 1])],
          tensor("bool", [2, 2], [fn(x, y) for x in (0, 1) for y in (0, 1)]), {"kind": "exact"})


def gen_float_compare(rng, op):
    for dt in ARITH_FLOATS:
        a, b = binary_inputs(rng, "add", dt)
        # equal values and signed zeros
        x = float_random(rng, dt, 16, -4, 4)
        a += x
        b += x
        a += [round_to(dt, -0.0), round_to(dt, 0.0), nan_bits(dt)]
        b += [round_to(dt, 0.0), round_to(dt, -0.0), nan_bits(dt)]
        res = []
        for x, y in zip(a, b):
            u, v = decode(dt, x), decode(dt, y)
            res.append(int(u < v) if op == "cmplt" else int(u == v))
        ftz = []
        for x, y in zip(a, b):
            u, v = decode(dt, flush(dt, x)), decode(dt, flush(dt, y))
            ftz.append(int(u < v) if op == "cmplt" else int(u == v))
        f = {"data": pack("bool", ftz), "check": {"kind": "exact"}} if ftz != res else None
        C.add(op, "%s/values" % dt, [tensor(dt, [len(a)], a), tensor(dt, [len(b)], b)],
              tensor("bool", [len(a)], res), {"kind": "exact"}, ftz=f)


def gen_where(rng):
    for dt in ["f32", "f64", "f16", "bf16", "i8", "u16", "i32", "u64", "bool"]:
        n = 24
        if dt in FLOATS:
            nb = width(dt)
            a = [int(v) for v in rng.integers(0, 1 << nb, size=n, dtype=np.uint64)]
            b = [int(v) for v in rng.integers(0, 1 << nb, size=n, dtype=np.uint64)]
            a[0] = nan_bits(dt) | 1  # a NaN payload, copied bit-exactly
            b[1] = nan_bits(dt, True) | 1
        elif dt == "bool":
            a = [int(v) for v in rng.integers(0, 2, size=n)]
            b = [int(v) for v in rng.integers(0, 2, size=n)]
        else:
            a, b = int_random(rng, dt, n), int_random(rng, dt, n)
        c = [int(v) for v in rng.integers(0, 2, size=n)]
        c[0], c[1] = 1, 0
        out = [x if k else y for k, x, y in zip(c, a, b)]
        C.add("where", "%s/values" % dt, [tensor("bool", [n], c), tensor(dt, [n], a), tensor(dt, [n], b)],
              tensor(dt, [n], out), {"kind": "bits"})
        # broadcast across all three operands
        c2 = [1, 0, 1]
        a2 = a[:4]
        b2 = b[:12]
        shapes = [[3, 1], [1, 4], [3, 4]]
        out2 = []
        for idx in bcast_indices(shapes, [3, 4]):
            out2.append(a2[idx[1]] if c2[idx[0]] else b2[idx[2]])
        C.add("where", "%s/broadcast" % dt, [tensor("bool", [3, 1], c2), tensor(dt, [1, 4], a2), tensor(dt, [3, 4], b2)],
              tensor(dt, [3, 4], out2), {"kind": "bits"})


# ---------------------------------------------------------------------------
# cast and bitcast

CAST_DTYPES = ARITH_FLOATS + STORAGE_FLOATS + INT_NAMES + ["bool"]


def exact_value(dt, bits):
    """The exact value of an element as a float (floats, bool) or int."""
    if dt in FLOATS:
        return decode(dt, bits)
    if dt == "bool":
        return 1 if bits else 0
    return ival(dt, bits)


def cast_bits(src, dst, bits, ftz=False):
    if ftz and src in ARITH_FLOATS:
        bits = flush(src, bits)
    v = exact_value(src, bits)
    if dst == "bool":
        return 0 if v == 0 else 1  # NaN != 0
    if dst in FLOATS:
        sat = dst in STORAGE_FLOATS
        if isinstance(v, int):
            r = round_to(dst, float(v) if v == 0 else Fraction(v), sat)
        else:
            r = round_to(dst, v if not math.isfinite(v) or v == 0 else Fraction(v), sat)
        if ftz and dst in ARITH_FLOATS:
            r = flush(dst, r)
        return r
    lo, hi = irange(dst)
    if isinstance(v, float):
        if math.isnan(v):
            return 0
        if math.isinf(v):
            return ibits(dst, hi if v > 0 else lo)
        t = math.trunc(v)
        return ibits(dst, min(max(t, lo), hi))
    if src == "bool":
        return v
    return ibits(dst, v)


def cast_inputs(rng, dt):
    if dt in FLOATS:
        vals = float_specials(dt)
        p, emin, eb, _, bias, maxf = fmt_params(dt)
        extra = [-0.5, 0.49, -1.5, 2.5, 127.5, 128.0, -129.0, 255.9, 256.0, 32767.0, 32768.0, -32769.0,
                 65535.5, 2.0 ** 31, -(2.0 ** 31) - 1024, 2.0 ** 32, 2.0 ** 63, -(2.0 ** 63), 2.0 ** 64,
                 448.0, 464.0, 465.0, 57344.0, 61440.0, 65504.0, 65520.0, 1e-7, 3e-8]
        vals += [round_to(dt, v, sat=True) for v in extra]
        vals += float_random(rng, dt, 16, emin - p, min(ilog2(maxf), 70))
        return vals
    if dt == "bool":
        return [0, 1]
    return int_specials(dt) + int_random(rng, dt, 8) + [ibits(dt, v) for v in (100, 255, 256, 448, 65504, 65505, 16777217) if irange(dt)[0] <= v <= irange(dt)[1]]


def gen_cast(rng):
    for src in CAST_DTYPES:
        col = cast_inputs(rng, src)
        for dst in CAST_DTYPES:
            out = [cast_bits(src, dst, b) for b in col]
            fout = [cast_bits(src, dst, b, ftz=True) for b in col]
            ftz = None
            if fout != out and (src in ARITH_FLOATS or dst in ARITH_FLOATS):
                ftz = {"data": pack(dst, fout), "check": {"kind": "exact"}}
            C.add("cast", "%s_to_%s" % (src, dst), [tensor(src, [len(col)], col)],
                  tensor(dst, [len(col)], out), {"kind": "exact"},
                  attrs={"to": attr_dtype(dst)}, ftz=ftz)


BITCAST_PAIRS = [("f32", "i32"), ("f32", "u32"), ("i32", "f32"), ("u32", "f32"), ("i32", "u32"),
                 ("f64", "i64"), ("u64", "f64"), ("i64", "u64"), ("f16", "bf16"), ("bf16", "f16"),
                 ("f16", "i16"), ("u16", "bf16"), ("i8", "u8"), ("f32", "f32")]


def gen_bitcast(rng):
    for src, dst in BITCAST_PAIRS:
        w = width(src)
        col = [int(v) for v in rng.integers(0, 1 << w, size=32, dtype=np.uint64)]
        if src in FLOATS:
            col += [nan_bits(src) | 1, nan_bits(src, True) | 2, round_to(src, -0.0), round_to(src, -math.inf)]
        C.add("bitcast", "%s_to_%s" % (src, dst), [tensor(src, [2, len(col) // 2], col)],
              tensor(dst, [2, len(col) // 2], col), {"kind": "bits"}, attrs={"to": attr_dtype(dst)})


# ---------------------------------------------------------------------------
# reduce

def reduce_groups(shape, axes):
    out_shape = [1 if i in axes else s for i, s in enumerate(shape)]
    groups = [[] for _ in range(numel(out_shape))]
    for i in range(numel(shape)):
        idx = unravel(i, shape)
        o = ravel([0 if k in axes else idx[k] for k in range(len(shape))], out_shape)
        groups[o].append(i)
    return out_shape, groups


def gen_reduce_case(dt, mode, name, shape, axes, bits):
    out_shape, groups = reduce_groups(shape, axes)
    attrs = {"mode": attr_str(mode), "axes": attr_ints(axes)}
    inp = [tensor(dt, shape, bits)]
    if dt in INTS:
        out = []
        for g in groups:
            vs = [ival(dt, bits[i]) for i in g]
            if mode == "add":
                r = sum(vs)
            elif mode == "mul":
                r = 1
                for v in vs:
                    r *= v
            else:
                r = max(vs) if vs else irange(dt)[0]
            out.append(ibits(dt, r))
        C.add("reduce", name, inp, tensor(dt, out_shape, out), {"kind": "exact"}, attrs=attrs)
        return
    out, refs, bounds = [], [], []
    p_out = FLOATS[dt][0]
    u = 2.0 ** -53 if dt == "f64" else 2.0 ** -24
    for g in groups:
        vs = [decode(dt, bits[i]) for i in g]
        n = len(vs)
        if mode == "max":
            r = -math.inf
            for v in vs:
                r = f_max(r, v)
            out.append(round_to(dt, r))
            continue
        if any(math.isnan(v) for v in vs) or any(math.isinf(v) for v in vs):
            acc = 0.0 if mode == "add" else 1.0
            for v in vs:
                acc = f_add(acc, v) if mode == "add" else f_mul(acc, v)
                acc = float(acc) if isinstance(acc, Fraction) else acc
            b = round_to(dt, acc)
            out.append(b)
            refs.append(decode(dt, b))
            bounds.append(0.0)
            continue
        if mode == "add":
            s = sum((Fraction(v) for v in vs), Fraction(0))
            bound = n * u * float(sum((abs(Fraction(v)) for v in vs), Fraction(0)))
        else:
            s = Fraction(1)
            for v in vs:
                s *= Fraction(v)
            bound = n * u * abs(float(s))
        sf = float(s)  # exact S rounded to f64
        b = round_to(dt, s) if s != 0 else round_to(dt, 0.0)
        out.append(b)
        refs.append(sf)
        # rounding S to f64 for the file, and for f16 and bf16 the final rounding
        extra = abs(sf) * 2.0 ** -53
        if dt in ("f16", "bf16"):
            big = round_to(dt, abs(sf) + bound) if abs(sf) + bound > 0 else 0
            extra += ulp(dt, big) if not math.isinf(decode(dt, big)) else 0
        bounds.append(bound + extra)
    if mode == "max":
        C.add("reduce", name, inp, tensor(dt, out_shape, out), {"kind": "exact"}, attrs=attrs)
    else:
        C.add("reduce", name, inp, tensor(dt, out_shape, out),
              {"kind": "tol", "ref": f64hex(refs), "bound": f64hex(bounds)}, attrs=attrs)


def gen_reduce(rng):
    shape = [3, 4, 5]
    axes_list = [[], [0], [2], [1, 2], [2, 0], [0, 1, 2]]
    for dt in ARITH_FLOATS:
        for mode in ("add", "mul", "max"):
            n = numel(shape)
            if mode == "mul":
                bits = [round_to(dt, float(v)) for v in rng.uniform(0.6, 1.6, size=n) * rng.choice([-1, 1], size=n)]
            else:
                bits = float_random(rng, dt, n, -4, 4)
            for axes in axes_list:
                gen_reduce_case(dt, mode, "%s/%s/axes%s" % (dt, mode, "_".join(map(str, axes)) or "none"),
                                shape, axes, bits)
            # an empty reduced axis gives the identity
            gen_reduce_case(dt, mode, "%s/%s/empty_axis" % (dt, mode), [3, 0, 2], [1], [])
            # specials: NaN propagates, inf + -inf is NaN, signed zeros
            sp = [round_to(dt, v) for v in [1.0, math.nan, 2.0, math.inf, -math.inf, 3.0, -0.0, -0.0, math.inf, 1.0, -0.0, 0.0]]
            gen_reduce_case(dt, mode, "%s/%s/specials" % (dt, mode), [6, 2], [1], sp)
        # cancellation: large terms summing to a small result
        big = float_random(rng, dt, 8, 8, 10)
        canc = []
        for v in big:
            canc += [v, round_to(dt, -decode(dt, v))]
        canc += float_random(rng, dt, 4, -2, 0)
        gen_reduce_case(dt, "add", "%s/add/cancellation" % dt, [20], [0], canc)
    for dt in INT_NAMES:
        for mode in ("add", "mul", "max"):
            bits = int_random(rng, dt, numel(shape))
            for axes in ([1], [0, 2], []):
                gen_reduce_case(dt, mode, "%s/%s/axes%s" % (dt, mode, "_".join(map(str, axes)) or "none"),
                                shape, axes, bits)
            gen_reduce_case(dt, mode, "%s/%s/empty_axis" % (dt, mode), [2, 0], [1], [])


# ---------------------------------------------------------------------------
# Movement operations

MOVE_DTYPES = ["f32", "f16", "bf16", "f64", "i8", "u16", "i64", "bool", "f8_e4m3"]


def rand_bits(rng, dt, n):
    if dt == "bool":
        return [int(v) for v in rng.integers(0, 2, size=n)]
    w = width(dt)
    b = [int(v) for v in rng.integers(0, 1 << w, size=n, dtype=np.uint64)]
    if dt in FLOATS and n > 1:
        b[0] = nan_bits(dt) | 1
    return b


def gen_movement(rng):
    for dt in MOVE_DTYPES:
        x = rand_bits(rng, dt, 24)
        # reshape
        for name, s_in, s_out, attr in [("2x3x4_to_6x4", [2, 3, 4], [6, 4], [6, 4]),
                                        ("2x3x4_to_sym", [2, 3, 4], [4, 6], [-1, 6]),
                                        ("24_to_1x24x1", [24], [1, 24, 1], [1, 24, 1])]:
            C.add("reshape", "%s/%s" % (dt, name), [tensor(dt, s_in, x)], tensor(dt, s_out, x),
                  {"kind": "bits"}, attrs={"shape": attr_ints(attr)})
        C.add("reshape", "%s/rank0_to_1" % dt, [tensor(dt, [], x[:1])], tensor(dt, [1, 1], x[:1]),
              {"kind": "bits"}, attrs={"shape": attr_ints([1, 1])})
        C.add("reshape", "%s/empty" % dt, [tensor(dt, [0, 3], [])], tensor(dt, [3, 0], []),
              {"kind": "bits"}, attrs={"shape": attr_ints([3, 0])})
        # expand
        for name, s_in, s_out in [("3x1_to_3x4", [3, 1], [3, 4]), ("1x1x2_to_2x3x2", [1, 1, 2], [2, 3, 2]),
                                  ("1_to_0", [1], [0]), ("2x3_same", [2, 3], [2, 3])]:
            src = x[:numel(s_in)]
            out = [src[ravel([0 if s_in[k] == 1 else idx[k] for k in range(len(idx))], s_in)]
                   for idx in (unravel(i, s_out) for i in range(numel(s_out)))]
            C.add("expand", "%s/%s" % (dt, name), [tensor(dt, s_in, src)], tensor(dt, s_out, out), {"kind": "bits"})
        # permute
        for perm in ([2, 0, 1], [1, 0, 2], [0, 1, 2], [2, 1, 0]):
            s_in = [2, 3, 4]
            s_out = [s_in[p] for p in perm]
            out = []
            for i in range(24):
                idx = unravel(i, s_out)
                j = [0] * 3
                for k in range(3):
                    j[perm[k]] = idx[k]
                out.append(x[ravel(j, s_in)])
            C.add("permute", "%s/perm%s" % (dt, "".join(map(str, perm))), [tensor(dt, s_in, x)],
                  tensor(dt, s_out, out), {"kind": "bits"}, attrs={"perm": attr_ints(perm)})
        # pad
        pad_values = [0.0, 7.0] if dt not in ARITH_FLOATS else [0.0, 1.5, -math.inf, math.nan, -0.0]
        if dt == "bool":
            pad_values = [0.0, 1.0]
        for pv in pad_values:
            for pads, s_in in [([1, 0, 2, 1], [2, 3]), ([0, 0, 0, 0], [2, 3]), ([2, 1], [3]), ([1, 1, 0, 0], [0, 2])]:
                src = x[:numel(s_in)]
                r = len(s_in)
                s_out = [s_in[i] + pads[2 * i] + pads[2 * i + 1] for i in range(r)]
                if dt in FLOATS:
                    pbits = round_to(dt, pv, sat=dt in STORAGE_FLOATS)
                elif dt == "bool":
                    pbits = int(pv)
                else:
                    pbits = ibits(dt, int(pv))
                out = []
                for i in range(numel(s_out)):
                    idx = unravel(i, s_out)
                    j = [idx[k] - pads[2 * k] for k in range(r)]
                    out.append(src[ravel(j, s_in)] if all(0 <= j[k] < s_in[k] for k in range(r)) else pbits)
                attrs = {"pads": attr_ints(pads)}
                if pv != 0 or math.copysign(1, pv) < 0:
                    attrs["value"] = attr_float(pv)
                name = "%s/pads%s_value%s" % (dt, "_".join(map(str, pads)) + "_in" + "x".join(map(str, s_in)), repr(pv))
                C.add("pad", name, [tensor(dt, s_in, src)], tensor(dt, s_out, out), {"kind": "bits"}, attrs=attrs)
        # shrink
        for begin, end, name in [([0, 1, 1], [2, 3, 3], "box"), ([1, 0, 0], [-1, -1, 2], "full_sym"),
                                 ([0, 2, 0], [2, 2, 4], "empty")]:
            s_in = [2, 3, 4]
            e = [s_in[k] if end[k] == -1 else end[k] for k in range(3)]
            s_out = [e[k] - begin[k] for k in range(3)]
            out = [x[ravel([idx[k] + begin[k] for k in range(3)], s_in)]
                   for idx in (unravel(i, s_out) for i in range(numel(s_out)))]
            C.add("shrink", "%s/%s" % (dt, name), [tensor(dt, s_in, x)], tensor(dt, s_out, out), {"kind": "bits"},
                  attrs={"begin": attr_ints(begin), "end": attr_ints(end)})


def gather_ref(data, dshape, axis, ind, ishape):
    pre, post = dshape[:axis], dshape[axis + 1:]
    oshape = pre + ishape + post
    out, dc = [], []
    for i in range(numel(oshape)):
        idx = unravel(i, oshape)
        a = idx[:len(pre)]
        j = idx[len(pre):len(pre) + len(ishape)]
        b = idx[len(pre) + len(ishape):]
        k = ind[ravel(j, ishape)]
        if 0 <= k < dshape[axis]:
            out.append(data[ravel(a + [k] + b, dshape)])
        else:
            out.append(0)
            dc.append(i)
    return oshape, out, dc


def gen_gather_scatter(rng):
    for dt in ["f32", "f16", "i64", "bool", "f8_e5m2", "u8"]:
        for idt in ["i32", "i64"]:
            for name, dshape, axis, ishape, ind in [
                ("axis0", [5, 3], 0, [4], [4, 0, 2, 2]),
                ("axis1_idx2d", [5, 3], 1, [2, 2], [2, 0, 1, 1]),
                ("scalar_index", [2, 4, 3], 1, [], [3]),
                ("rank1_data", [6], 0, [3, 2], [5, 4, 3, 0, 1, 2]),
                ("empty_indices", [5, 3], 0, [0], []),
                ("out_of_range", [5, 3], 0, [5], [1, -1, 5, 1 << 30, 3]),
            ]:
                data = rand_bits(rng, dt, numel(dshape))
                oshape, out, dc = gather_ref(data, dshape, axis, ind, ishape)
                C.add("gather", "%s/%s/%s" % (dt, idt, name),
                      [tensor(dt, dshape, data), tensor(idt, ishape, [ibits(idt, v) for v in ind])],
                      tensor(dt, oshape, out), {"kind": "bits"}, attrs={"axis": attr_int(axis)}, dont_care=dc)
                # scatter with the same index pattern
                upd = rand_bits(rng, dt, numel(oshape))
                pre, post = dshape[:axis], dshape[axis + 1:]
                y = list(data)
                oor = False
                for jflat in range(numel(ishape)):
                    k = ind[jflat]
                    if not 0 <= k < dshape[axis]:
                        oor = True
                        continue
                    jidx = unravel(jflat, ishape)
                    for af in range(numel(pre)):
                        a = unravel(af, pre)
                        for bf in range(numel(post)):
                            b = unravel(bf, post)
                            y[ravel(a + [k] + b, dshape)] = upd[ravel(a + jidx + b, oshape)]
                dc = list(range(len(y))) if oor else None
                C.add("scatter", "%s/%s/%s" % (dt, idt, name),
                      [tensor(dt, dshape, data), tensor(idt, ishape, [ibits(idt, v) for v in ind]),
                       tensor(dt, oshape, upd)],
                      tensor(dt, dshape, y), {"kind": "bits"}, attrs={"axis": attr_int(axis)}, dont_care=dc)


def gen_const():
    cases = [("f32", 0.1, "float"), ("f32", math.nan, "float"), ("f32", -math.inf, "float"), ("f32", -0.0, "float"),
             ("f32", 1e39, "float"), ("f32", 16777217, "int"), ("f64", 0.1, "float"), ("f16", 0.1, "float"),
             ("f16", 65520.0, "float"), ("f16", 1e-8, "float"), ("bf16", 1 / 3, "float"), ("bf16", 3, "int"),
             ("i8", -128, "int"), ("u8", 255, "int"), ("i16", -2, "int"), ("u16", 65535, "int"),
             ("i32", 2147483647, "int"), ("u32", 4294967295, "int"), ("i64", -(1 << 63), "int"),
             ("u64", (1 << 63) - 1, "int"), ("i32", 5.0, "float"), ("bool", 1, "int"), ("bool", 0, "int")]
    for dt, v, kind in cases:
        for shape in ([], [2, 3], [0]):
            if dt in FLOATS:
                if kind == "float":
                    b = round_to(dt, v if not math.isfinite(v) or v == 0 else Fraction(v))
                else:
                    b = round_to(dt, Fraction(v))
            elif dt == "bool":
                b = int(v)
            else:
                b = ibits(dt, int(v))
            attr = attr_float(v) if kind == "float" else attr_int(v)
            name = "%s/%s_%s/shape%s" % (dt, kind, repr(v), "x".join(map(str, shape)) or "scalar")
            C.add("const", name, [], tensor(dt, shape, [b] * numel(shape)), {"kind": "exact"},
                  attrs={"value": attr})


# ---------------------------------------------------------------------------
# dequantize: the reference decoders of spec/dtypes.md, in float32.

f32 = np.float32


np.seterr(all="ignore")  # inf * 0 in a decoder is a NaN result, not an error


def h2f(lo, hi):
    return f32(np.frombuffer(bytes([lo, hi]), dtype="<f2")[0])


def dq_q8_0(b):
    d = h2f(b[0], b[1])
    return [f32(d * f32(np.int8(np.uint8(q)))) for q in b[2:34]]


def dq_q4_0(b):
    d = h2f(b[0], b[1])
    y = [None] * 32
    for j in range(16):
        y[j] = f32(d * f32((b[2 + j] & 15) - 8))
        y[j + 16] = f32(d * f32((b[2 + j] >> 4) - 8))
    return y


def dq_q4_k(b):
    d, dmin = h2f(b[0], b[1]), h2f(b[2], b[3])
    sc_b, qs = b[4:16], b[16:144]
    sc, mn = [0] * 8, [0] * 8
    for j in range(4):
        sc[j] = sc_b[j] & 63
        mn[j] = sc_b[j + 4] & 63
    for j in range(4, 8):
        sc[j] = (sc_b[j + 4] & 15) | ((sc_b[j - 4] >> 6) << 4)
        mn[j] = (sc_b[j + 4] >> 4) | ((sc_b[j] >> 6) << 4)
    y = [None] * 256
    for c in range(4):
        d1, m1 = f32(d * f32(sc[2 * c])), f32(dmin * f32(mn[2 * c]))
        d2, m2 = f32(d * f32(sc[2 * c + 1])), f32(dmin * f32(mn[2 * c + 1]))
        for l in range(32):
            q = qs[32 * c + l]
            y[64 * c + l] = f32(f32(d1 * f32(q & 15)) - m1)
            y[64 * c + 32 + l] = f32(f32(d2 * f32(q >> 4)) - m2)
    return y


def dq_q6_k(b):
    ql, qh, sc, d = b[0:128], b[128:192], b[192:208], h2f(b[208], b[209])
    sc = [int(np.int8(np.uint8(s))) for s in sc]
    y = [None] * 256
    for h in range(2):
        l0, h0, s0 = 64 * h, 32 * h, 8 * h
        for l in range(32):
            i = l // 16
            q1 = ((ql[l0 + l] & 15) | (((qh[h0 + l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[l0 + l + 32] & 15) | (((qh[h0 + l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[l0 + l] >> 4) | (((qh[h0 + l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[l0 + l + 32] >> 4) | (((qh[h0 + l] >> 6) & 3) << 4)) - 32
            y[128 * h + l] = f32(f32(d * f32(sc[s0 + i])) * f32(q1))
            y[128 * h + l + 32] = f32(f32(d * f32(sc[s0 + i + 2])) * f32(q2))
            y[128 * h + l + 64] = f32(f32(d * f32(sc[s0 + i + 4])) * f32(q3))
            y[128 * h + l + 96] = f32(f32(d * f32(sc[s0 + i + 6])) * f32(q4))
    return y


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def dq_mxfp4(b):
    scale = 2.0 ** (b[0] - 127)  # exact in f64, including 2^128 for 0xFF
    y = [None] * 32
    for j in range(16):
        for k, n in ((j, b[1 + j] & 15), (j + 16, b[1 + j] >> 4)):
            v = E2M1[n & 7] * (-1.0 if n & 8 else 1.0)
            r = scale * v
            y[k] = decode("f32", round_to("f32", r if r != 0 else 0.0))
    return y


def dq_affine4(b):
    s, bias = h2f(b[0], b[1]), h2f(b[2], b[3])
    y = [None] * 64
    for j in range(32):
        y[j] = f32(f32(s * f32(b[4 + j] & 15)) + bias)
        y[j + 32] = f32(f32(s * f32(b[4 + j] >> 4)) + bias)
    return y


DQ = {"q8_0": dq_q8_0, "q4_0": dq_q4_0, "q4_k": dq_q4_k, "q6_k": dq_q6_k,
      "mxfp4": dq_mxfp4, "affine4_g64": dq_affine4}


def f32bits(v):
    return int(np.array([v], dtype="<f4").view("<u4")[0])


def rand_f16_bytes(rng, n, lo=0.001, hi=2.0):
    v = (rng.uniform(lo, hi, size=n) * rng.choice([-1, 1], size=n)).astype("<f2")
    return v.view(np.uint8).reshape(n, 2)


def rand_blocks(rng, dt, nblocks):
    be, bb = BLOCKS[dt]
    raw = rng.integers(0, 256, size=(nblocks, bb), dtype=np.uint8)
    if dt in ("q8_0", "q4_0"):
        raw[:, 0:2] = rand_f16_bytes(rng, nblocks)
    elif dt == "q4_k":
        raw[:, 0:2] = rand_f16_bytes(rng, nblocks)
        raw[:, 2:4] = rand_f16_bytes(rng, nblocks)
    elif dt == "q6_k":
        raw[:, 208:210] = rand_f16_bytes(rng, nblocks)
    elif dt == "affine4_g64":
        raw[:, 0:2] = rand_f16_bytes(rng, nblocks)
        raw[:, 2:4] = rand_f16_bytes(rng, nblocks, 0.0, 8.0)
    elif dt == "mxfp4":
        raw[:, 0] = rng.integers(100, 150, size=nblocks)
    return raw


def gen_dequantize(rng):
    for dt, (be, bb) in BLOCKS.items():
        raw = rand_blocks(rng, dt, 4)
        if dt == "mxfp4":
            raw[1, 0] = 0xFF  # decodes as 2^128 (ggml), not NaN
            raw[2, 0] = 0x00
            raw[3, 1] = 0x88  # negative zero nibbles decode to +0
        if dt == "q8_0":
            raw[3, 0:2] = [0x00, 0x7C]  # an infinite scale: inf * 0 is NaN
        vals = []
        for blk in raw:
            vals += DQ[dt](bytes(blk))
        bits = [f32bits(v) for v in vals]
        inp = {"dtype": dt, "shape": [2, 2 * be], "data": raw.tobytes().hex()}
        C.add("dequantize", "%s/to_f32" % dt, [inp], tensor("f32", [2, 2 * be], bits), {"kind": "exact"})
        bf = [round_to("bf16", decode("f32", b) if not math.isfinite(decode("f32", b)) or decode("f32", b) == 0
                       else Fraction(decode("f32", b))) for b in bits]
        C.add("dequantize", "%s/to_bf16" % dt, [inp], tensor("bf16", [2, 2 * be], bf), {"kind": "exact"})
    for dt in ["f16", "bf16", "f8_e4m3", "f8_e5m2"]:
        w = width(dt)
        if w == 8:
            col = list(range(256))
        else:
            col = sorted(set(float_specials(dt) + [int(v) for v in rng.integers(0, 1 << 16, size=480)]))
        f = [round_to("f32", decode(dt, b) if not math.isfinite(decode(dt, b)) or decode(dt, b) == 0
                      else Fraction(decode(dt, b))) for b in col]
        C.add("dequantize", "%s/to_f32" % dt, [tensor(dt, [len(col)], col)], tensor("f32", [len(col)], f),
              {"kind": "exact"})
        bf = [round_to("bf16", decode(dt, b) if not math.isfinite(decode(dt, b)) or decode(dt, b) == 0
                       else Fraction(decode(dt, b))) for b in col]
        C.add("dequantize", "%s/to_bf16" % dt, [tensor(dt, [len(col)], col)], tensor("bf16", [len(col)], bf),
              {"kind": "exact"})


# ---------------------------------------------------------------------------

def build():
    seed = 20261015
    for i, op in enumerate(FLOAT_UNARY):
        gen_float_unary(np.random.default_rng([seed, 1, i]), op)
    for i, op in enumerate(FLOAT_BINARY):
        gen_float_binary(np.random.default_rng([seed, 2, i]), op)
    for i, op in enumerate(["neg", "trunc", "add", "mul", "max", "idiv", "imod", "shl", "shr",
                            "and", "or", "xor", "cmplt", "cmpeq"]):
        gen_int_ops(np.random.default_rng([seed, 3, i]), op)
    for op in ("and", "or", "xor", "cmpeq"):
        gen_bool_ops(op)
    for i, op in enumerate(["cmplt", "cmpeq"]):
        gen_float_compare(np.random.default_rng([seed, 4, i]), op)
    gen_where(np.random.default_rng([seed, 5]))
    gen_cast(np.random.default_rng([seed, 6]))
    gen_bitcast(np.random.default_rng([seed, 7]))
    gen_reduce(np.random.default_rng([seed, 8]))
    gen_movement(np.random.default_rng([seed, 9]))
    gen_gather_scatter(np.random.default_rng([seed, 10]))
    gen_const()
    gen_dequantize(np.random.default_rng([seed, 11]))
    missing = set(OPCODES) - set(C.ops)
    assert not missing, missing
    return C.files()


def main():
    check = "--check" in sys.argv[1:]
    files = build()
    stale = []
    if not check:
        os.makedirs(OUT_DIR, exist_ok=True)
    for name, text in sorted(files.items()):
        path = os.path.join(OUT_DIR, name)
        if check:
            try:
                with open(path, encoding="utf-8") as f:
                    if f.read() != text:
                        stale.append(path)
            except FileNotFoundError:
                stale.append(path)
        else:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
    if check:
        extra = set(os.listdir(OUT_DIR)) - set(files) if os.path.isdir(OUT_DIR) else set()
        stale += sorted(os.path.join(OUT_DIR, e) for e in extra)
        if stale:
            print("conformance corpus is out of date; run python3 -I tools/gen_conformance.py:")
            for s in stale:
                print("  " + s)
            sys.exit(1)
        print("conformance corpus is up to date (%d files)" % len(files))
    else:
        n = sum(len(c) for c in C.ops.values())
        print("wrote %d cases in %d files to %s" % (n, len(files), OUT_DIR))


if __name__ == "__main__":
    main()
