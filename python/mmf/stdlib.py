"""The standard library of spec/stdlib.md.

Every composite is evaluated as its reference result (spec/stdlib.md
section 1.3): the canonical body computed in float64, with float casts
inside the body as the identity, integers computed exactly, and each float
output rounded once to its declared dtype. A runtime may run any
implementation within a composite's tolerance, and the tolerances are
stated against this reference, so this is the most accurate choice there
is; it is also what NumPy computes fastest.
"""

import math

import numpy as np

from . import dtypes as D
from .errors import RunError
from .tensor import Tensor

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


def _need(cond, msg):
    if not cond:
        raise RunError(msg)


class _Attrs:
    """Typed access to a composite's attributes, rejecting unknown keys."""

    def __init__(self, attrs, name, *known):
        for k in attrs:
            _need(k in known, f"{name}: unknown attribute {k!r}")
        self.a = attrs
        self.name = name

    def _get(self, key, typ, default):
        a = self.a.get(key)
        if a is None:
            return default
        _need(a.type == typ, f"{self.name}: attribute {key!r} must be {typ}")
        return a.value

    def int(self, key, default):
        return self._get(key, "int", default)

    def float(self, key, default):
        return self._get(key, "float", default)

    def ints(self, key, default):
        v = self._get(key, "ints", default)
        return list(v) if v is not None else None

    def string(self, key, default):
        return self._get(key, "string", default)

    def bool(self, key, default):
        v = self._get(key, "int", None)
        if v is None:
            return default
        _need(v in (0, 1), f"{self.name}: attribute {key!r} must be an int 0 or 1")
        return v == 1


def values(t):
    """The elements of t as float64, decoding block and storage dtypes."""
    if t.dtype.is_block:
        return D.dequantize_block(t.raw, t.dtype, t.shape).astype(np.float64)
    return D.to_f64(t.arr, t.dtype)


def _out(dt, shape, v):
    v = np.asarray(v)
    _need(tuple(v.shape) == tuple(shape), f"result shape {list(v.shape)}, declared {list(shape)}")
    if dt.is_int:
        return Tensor(dt, shape, v.astype(dt.storage))
    _need(dt.is_float, f"result dtype {dt.name}")
    with np.errstate(all="ignore"):
        return Tensor(dt, shape, D.from_f64(v, dt))


def _float(t, name):
    _need(t.dtype.is_float, f"{name}: input of {t.dtype.name}")


def _axis(ax, r, name):
    if ax < 0:
        ax += r
    _need(0 <= ax < r, f"{name}: axis out of range")
    return ax


# Elementwise --------------------------------------------------------------


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp2(-x * LOG2E))


def _erf(x):
    p = 0.3275911
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    s = np.where(x < 0, -1.0, 1.0)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    q = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    return s * (1.0 - q * np.exp2(-(ax * ax) * LOG2E))


def _gelu_erf(x):
    return 0.5 * x * (1.0 + _erf(x * 0.7071067811865476))


def _tanh(x):
    return 2.0 * _sigmoid(2.0 * x) - 1.0


def _gelu_tanh(x):
    inner = 0.7978845608028654 * (x + 0.044715 * (x * (x * x)))
    return 0.5 * x * (1.0 + _tanh(inner))


_UNARY = {
    "sigmoid": _sigmoid,
    "tanh": _tanh,
    "erf": _erf,
    "exp": lambda x: np.exp2(x * LOG2E),
    "log": lambda x: np.log2(x) * LN2,
    "cos": lambda x: np.sin(math.pi / 2 - x),
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "silu": lambda x: x * _sigmoid(x),
    "gelu_erf": _gelu_erf,
    "gelu_tanh": _gelu_tanh,
}


def _unary(name):
    def fn(attrs, ins, dts, shapes):
        _Attrs(attrs, name)
        _need(len(ins) == 1, f"{name} takes one input")
        _float(ins[0], name)
        return [_out(dts[0], shapes[0], _UNARY[name](values(ins[0])))]

    return fn


def swiglu(attrs, ins, dts, shapes):
    _Attrs(attrs, "swiglu")
    _need(len(ins) == 2 and ins[0].dtype is ins[1].dtype, "swiglu takes a and b of one dtype")
    a, b = values(ins[0]), values(ins[1])
    return [_out(dts[0], shapes[0], a * _sigmoid(a) * b)]


def geglu(attrs, ins, dts, shapes):
    at = _Attrs(attrs, "geglu", "approximate")
    approx = at.string("approximate", "none")
    _need(approx in ("none", "tanh"), f"geglu: approximate {approx!r}")
    _need(len(ins) == 2 and ins[0].dtype is ins[1].dtype, "geglu takes a and b of one dtype")
    a, b = values(ins[0]), values(ins[1])
    g = _gelu_erf(a) if approx == "none" else _gelu_tanh(a)
    return [_out(dts[0], shapes[0], g * b)]


# Linear algebra -------------------------------------------------------------


def matmul(attrs, ins, dts, shapes):
    _Attrs(attrs, "matmul")
    _need(len(ins) == 2 and ins[0].dtype is ins[1].dtype, "matmul takes A and B of one dtype")
    A, B = ins
    _need(len(A.shape) >= 2 and len(A.shape) == len(B.shape) and A.shape[-1] == B.shape[-2], f"matmul {list(A.shape)} x {list(B.shape)}")
    if A.dtype.is_int:
        with np.errstate(all="ignore"):
            return [_out(dts[0], shapes[0], np.matmul(A.arr, B.arr))]
    _float(A, "matmul")
    return [_out(dts[0], shapes[0], np.matmul(values(A), values(B)))]


def _linear(x, w, bias):
    y = np.matmul(x, w.T)
    if bias is not None:
        y = y + bias
    return y


def linear(attrs, ins, dts, shapes):
    _Attrs(attrs, "linear")
    _need(2 <= len(ins) <= 3, "linear takes x, w and an optional bias")
    x, w = ins[0], ins[1]
    _need(len(x.shape) >= 2 and len(w.shape) == 2 and w.shape[1] == x.shape[-1], f"linear {list(x.shape)} by {list(w.shape)}")
    bias = values(ins[2]) if len(ins) == 3 else None
    _float(x, "linear")
    return [_out(dts[0], shapes[0], _linear(values(x), values(w), bias))]


def dequant_matmul(attrs, ins, dts, shapes):
    _Attrs(attrs, "dequant_matmul")
    _need(2 <= len(ins) <= 3, "dequant_matmul takes x, w and an optional bias")
    x, w = ins[0], ins[1]
    _float(x, "dequant_matmul")
    _need(w.dtype.is_block or w.dtype.name in ("f16", "bf16", "f8_e4m3", "f8_e5m2"), f"dequant_matmul weight of {w.dtype.name}")
    _need(len(w.shape) == 2 and w.shape[1] == x.shape[-1], f"dequant_matmul {list(x.shape)} by {list(w.shape)}")
    bias = values(ins[2]) if len(ins) == 3 else None
    return [_out(dts[0], shapes[0], _linear(values(x), values(w), bias))]


def conv2d(attrs, ins, dts, shapes):
    at = _Attrs(attrs, "conv2d", "strides", "dilations", "pads", "groups")
    sh, sw = at.ints("strides", [1, 1])
    dh, dw = at.ints("dilations", [1, 1])
    top, bottom, left, right = at.ints("pads", [0, 0, 0, 0])
    g = at.int("groups", 1)
    _need(2 <= len(ins) <= 3, "conv2d takes X, W and an optional bias")
    X, W = ins[0], ins[1]
    _float(X, "conv2d")
    _need(len(X.shape) == 4 and len(W.shape) == 4, "conv2d needs rank 4 X and W")
    Nb, C, H, Wd = X.shape
    F, Cg, KH, KW = W.shape
    _need(g > 0 and C % g == 0 and F % g == 0 and Cg == C // g, f"conv2d groups {g} for C {C}, F {F}")
    _need(min(sh, sw, dh, dw) > 0 and min(top, bottom, left, right) >= 0, "conv2d attributes")
    Ho = (H + top + bottom - dh * (KH - 1) - 1) // sh + 1
    Wo = (Wd + left + right - dw * (KW - 1) - 1) // sw + 1
    x = np.pad(values(X), ((0, 0), (0, 0), (top, bottom), (left, right)))
    w = values(W)
    Fg = F // g
    y = np.zeros((Nb, F, Ho, Wo))
    for gi in range(g):
        xg = x[:, gi * Cg : (gi + 1) * Cg]
        wg = w[gi * Fg : (gi + 1) * Fg]
        for i in range(KH):
            for j in range(KW):
                xs = xg[:, :, i * dh : i * dh + sh * (Ho - 1) + 1 : sh, j * dw : j * dw + sw * (Wo - 1) + 1 : sw]
                y[:, gi * Fg : (gi + 1) * Fg] += np.einsum("nchw,fc->nfhw", xs, wg[:, :, i, j])
    if len(ins) == 3:
        y = y + values(ins[2]).reshape(1, F, 1, 1)
    return [_out(dts[0], shapes[0], y)]


# Normalisation and reductions ------------------------------------------------


def softmax(attrs, ins, dts, shapes):
    ax = _Attrs(attrs, "softmax", "axis").int("axis", -1)
    _need(len(ins) == 1, "softmax takes one input")
    _float(ins[0], "softmax")
    x = values(ins[0])
    ax = _axis(ax, x.ndim, "softmax")
    with np.errstate(all="ignore"):
        e = np.exp2((x - np.max(x, axis=ax, keepdims=True)) * LOG2E)
        return [_out(dts[0], shapes[0], e / np.sum(e, axis=ax, keepdims=True))]


def log_softmax(attrs, ins, dts, shapes):
    ax = _Attrs(attrs, "log_softmax", "axis").int("axis", -1)
    _need(len(ins) == 1, "log_softmax takes one input")
    _float(ins[0], "log_softmax")
    x = values(ins[0])
    ax = _axis(ax, x.ndim, "log_softmax")
    with np.errstate(all="ignore"):
        d = x - np.max(x, axis=ax, keepdims=True)
        return [_out(dts[0], shapes[0], d - np.log(np.sum(np.exp(d), axis=ax, keepdims=True)))]


def layer_norm(attrs, ins, dts, shapes):
    at = _Attrs(attrs, "layer_norm", "axis", "epsilon")
    _need(2 <= len(ins) <= 3, "layer_norm takes X, scale and an optional bias")
    _float(ins[0], "layer_norm")
    x = values(ins[0])
    ax = _axis(at.int("axis", -1), x.ndim, "layer_norm")
    eps = at.float("epsilon", 1e-5)
    axes = tuple(range(ax, x.ndim))
    mu = np.mean(x, axis=axes, keepdims=True)
    d = x - mu
    var = np.mean(d * d, axis=axes, keepdims=True)
    y = d / np.sqrt(var + eps) * values(ins[1])
    if len(ins) == 3:
        y = y + values(ins[2])
    return [_out(dts[0], shapes[0], y)]


def rms_norm(attrs, ins, dts, shapes):
    at = _Attrs(attrs, "rms_norm", "axis", "epsilon")
    _need(1 <= len(ins) <= 2, "rms_norm takes x and an optional scale")
    _float(ins[0], "rms_norm")
    x = values(ins[0])
    ax = _axis(at.int("axis", -1), x.ndim, "rms_norm")
    eps = at.float("epsilon", 1e-6)
    ms = np.mean(x * x, axis=tuple(range(ax, x.ndim)), keepdims=True)
    y = x / np.sqrt(ms + eps)
    if len(ins) == 2:
        y = y * values(ins[1])
    return [_out(dts[0], shapes[0], y)]


def _norm_axes(axes, r, name):
    return tuple(sorted({_axis(a, r, name) for a in axes}))


def mean(attrs, ins, dts, shapes):
    at = _Attrs(attrs, "mean", "axes")
    _float(ins[0], "mean")
    x = values(ins[0])
    axes = _norm_axes(at.ints("axes", []), x.ndim, "mean")
    return [_out(dts[0], shapes[0], np.mean(x, axis=axes, keepdims=True))]


def variance(attrs, ins, dts, shapes):
    at = _Attrs(attrs, "variance", "axes", "correction")
    _float(ins[0], "variance")
    x = values(ins[0])
    axes = _norm_axes(at.ints("axes", []), x.ndim, "variance")
    c = at.int("correction", 0)
    n = int(np.prod([x.shape[a] for a in axes]))
    d = x - np.mean(x, axis=axes, keepdims=True)
    with np.errstate(all="ignore"):
        return [_out(dts[0], shapes[0], np.sum(d * d, axis=axes, keepdims=True) / np.float64(n - c))]


def cumsum(attrs, ins, dts, shapes):
    at = _Attrs(attrs, "cumsum", "axis", "exclusive", "reverse")
    x = ins[0]
    _need(x.dtype.is_float or x.dtype.is_int, f"cumsum of {x.dtype.name}")
    ax = at.int("axis", None)
    _need(ax is not None, "cumsum needs an axis")
    ax = _axis(ax, len(x.shape), "cumsum")
    exclusive, reverse = at.bool("exclusive", False), at.bool("reverse", False)
    v = x.arr if x.dtype.is_int else values(x)
    if reverse:
        v = np.flip(v, ax)
    with np.errstate(all="ignore"):
        y = np.cumsum(v, axis=ax, dtype=v.dtype)
        if exclusive:
            y = y - v
    if reverse:
        y = np.flip(y, ax)
    return [_out(dts[0], shapes[0], y)]


# Transformer blocks ------------------------------------------------------------


def rope(attrs, ins, dts, shapes):
    at = _Attrs(attrs, "rope", "theta", "rotary_dims", "interleaved", "yarn_factor", "yarn_original_context",
                "yarn_beta_fast", "yarn_beta_slow", "yarn_attn_factor")
    _need(len(ins) == 2, "rope takes x and positions")
    x, pos = ins
    _float(x, "rope")
    _need(len(x.shape) == 4, "rope: x must be [B, H, T, D]")
    B, H, T, Dd = x.shape
    _need(pos.dtype.name in ("i32", "i64") and len(pos.shape) == 2 and pos.shape[1] == T and pos.shape[0] in (1, B),
          f"rope: positions {pos.dtype.name}{list(pos.shape)}")
    theta = at.float("theta", 10000.0)
    R = at.int("rotary_dims", Dd)
    _need(0 < R <= Dd and R % 2 == 0, f"rope: rotary_dims {R} for D = {Dd}")
    inter = at.bool("interleaved", False)
    factor = at.float("yarn_factor", 1.0)
    orig = at.int("yarn_original_context", 0)
    fast, slow = at.float("yarn_beta_fast", 32.0), at.float("yarn_beta_slow", 1.0)
    attn = at.float("yarn_attn_factor", 0.1 * math.log(factor) + 1 if factor > 0 else 1.0)
    h = R // 2
    i = np.arange(h, dtype=np.float64)
    f = theta ** (-2 * i / R)
    if factor == 1:
        inv, ms = f, 1.0
    else:
        _need(orig > 0, "rope: yarn_original_context is required when yarn_factor != 1")

        def corr(n):
            return R * math.log(orig / (n * 2 * math.pi)) / (2 * math.log(theta))

        lo = max(math.floor(corr(fast)), 0)
        hi = min(math.ceil(corr(slow)), R - 1)
        if lo == hi:
            hi += 0.001
        ramp = np.clip((i - lo) / (hi - lo), 0, 1)
        inv, ms = (f / factor) * ramp + f * (1 - ramp), attn
    ang = pos.arr.astype(np.float64).reshape(pos.shape[0], 1, T, 1) * inv.reshape(1, 1, 1, h)
    c, s = np.cos(ang) * ms, np.sin(ang) * ms
    v = values(x)
    xr = v[..., :R]
    if inter:
        x1, x2 = xr[..., 0::2], xr[..., 1::2]
    else:
        x1, x2 = xr[..., :h], xr[..., h:]
    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s
    y = v.copy()
    if inter:
        y[..., 0:R:2], y[..., 1:R:2] = y1, y2
    else:
        y[..., :h], y[..., h:R] = y1, y2
    return [_out(dts[0], shapes[0], y)]


def attention(attrs, ins, dts, shapes):
    at = _Attrs(attrs, "attention", "scale", "causal", "num_kv_heads", "softcap")
    _need(3 <= len(ins) <= 5, "attention takes q, k, v, an optional mask and optional sinks")
    q, k, v = ins[:3]
    _float(q, "attention")
    _need(k.dtype is q.dtype and v.dtype is q.dtype, "attention: q, k and v must share a dtype")
    _need(len(q.shape) == len(k.shape) == len(v.shape) == 4, "attention: q, k and v must be rank 4")
    B, Hq, Tq, Dd = q.shape
    Hkv, Tk = k.shape[1], k.shape[2]
    scale = at.float("scale", 1 / math.sqrt(Dd))
    causal = at.bool("causal", False)
    nkv = at.int("num_kv_heads", Hq)
    softcap = at.float("softcap", 0.0)
    _need(nkv > 0 and Hq % nkv == 0 and Hkv == nkv and v.shape[1] == nkv and k.shape[0] == B and v.shape[0] == B
          and k.shape[3] == Dd and v.shape[2] == Tk, f"attention: q {list(q.shape)}, k {list(k.shape)}, v {list(v.shape)}")
    G = Hq // nkv
    kf = np.repeat(values(k), G, axis=1)
    vf = np.repeat(values(v), G, axis=1)
    with np.errstate(all="ignore"):
        s = np.matmul(values(q), np.swapaxes(kf, 2, 3)) * scale
        if softcap > 0:
            s = np.tanh(s / softcap) * softcap
        if len(ins) >= 4:
            m = ins[3]
            _need(len(m.shape) == 4, "attention: mask must be rank 4")
            if m.dtype.kind == "bool":
                s = s + np.where(m.arr, 0.0, -np.inf)
            else:
                _float(m, "attention mask")
                s = s + values(m)
        if causal:
            iq = np.arange(Tq).reshape(Tq, 1)
            jk = np.arange(Tk).reshape(1, Tk)
            s = s + np.where(jk <= iq + (Tk - Tq), 0.0, -np.inf)
        if len(ins) == 5:
            z = values(ins[4]).reshape(1, Hq, 1, 1)
            mx = np.maximum(np.max(s, axis=3, keepdims=True), z)
            e = np.exp(s - mx)
            p = e / (np.sum(e, axis=3, keepdims=True) + np.exp(z - mx))
        else:
            e = np.exp(s - np.max(s, axis=3, keepdims=True))
            p = e / np.sum(e, axis=3, keepdims=True)
        return [_out(dts[0], shapes[0], np.matmul(p, vf))]


def embedding(attrs, ins, dts, shapes):
    _Attrs(attrs, "embedding")
    _need(len(ins) == 2, "embedding takes table and ids")
    table, ids = ins
    _need(len(table.shape) == 2 and ids.dtype.name in ("i32", "i64"), "embedding: table [V, E] and integer ids")
    V = table.shape[0]
    idx = np.clip(ids.arr.astype(np.int64), 0, max(V - 1, 0))
    if table.dtype.is_block or table.dtype.kind == "storage":
        rows = np.unique(idx)
        if table.dtype.is_block:
            rb = table.dtype.block_bytes * table.shape[1] // table.dtype.block_elems
            raw = b"".join(bytes(table.raw[r * rb : (r + 1) * rb]) for r in rows.tolist())
            dec = D.dequantize_block(raw, table.dtype, (len(rows), table.shape[1]))
        else:
            dec = values(Tensor(table.dtype, (len(rows), table.shape[1]), table.arr[rows])).astype(np.float32)
        y = dec[np.searchsorted(rows, idx)]
        return [_out(dts[0], shapes[0], y.astype(np.float64))]
    if dts[0] is table.dtype:
        return [Tensor(dts[0], shapes[0], np.take(table.arr, idx, axis=0))]  # bit-exact
    # f16 and bf16 tables enter the body as f32 (spec/stdlib.md section 1.4).
    return [_out(dts[0], shapes[0], values(table)[idx])]


# Selection and sorting -------------------------------------------------------


def _ranks(x):
    """Ranks of the elements of x in the order of spec/stdlib.md section 7:
    -0 equals +0 and NaN is above everything."""
    if x.dtype.is_int:
        _, inv = np.unique(x.arr, return_inverse=True)
        return inv.reshape(x.shape)
    v = values(x)
    nan = np.isnan(v)
    v = np.where(nan, 0.0, np.where(v == 0, 0.0, v))
    _, inv = np.unique(v, return_inverse=True)
    inv = inv.reshape(x.shape)
    return np.where(nan, inv.max(initial=0) + 1, inv)


def _order(x, ax, descending):
    r = np.moveaxis(_ranks(x), ax, -1)
    L = r.shape[-1]
    idx = np.broadcast_to(np.arange(L), r.shape)
    key = -r if descending else r
    flat_k = key.reshape(-1, L)
    flat_i = idx.reshape(-1, L)
    out = np.empty_like(flat_i)
    for p in range(flat_k.shape[0]):
        out[p] = np.lexsort((flat_i[p], flat_k[p]))
    return out.reshape(r.shape)  # axis last


def top_k(attrs, ins, dts, shapes):
    at = _Attrs(attrs, "top_k", "k", "axis", "largest")
    x = ins[0]
    _need(x.dtype.is_float or x.dtype.is_int, f"top_k of {x.dtype.name}")
    ax = _axis(at.int("axis", -1), len(x.shape), "top_k")
    kk = at.int("k", None)
    _need(kk is not None and 0 <= kk <= x.shape[ax], "top_k: k out of range")
    order = _order(x, ax, at.bool("largest", True))[..., :kk]
    vals = np.take_along_axis(np.moveaxis(x.arr, ax, -1), order, axis=-1)
    return [Tensor(dts[0], shapes[0], np.moveaxis(vals, -1, ax)),
            Tensor(dts[1], shapes[1], np.moveaxis(order, -1, ax).astype(np.int64))]


def argsort(attrs, ins, dts, shapes):
    at = _Attrs(attrs, "argsort", "axis", "descending")
    x = ins[0]
    _need(x.dtype.is_float or x.dtype.is_int, f"argsort of {x.dtype.name}")
    ax = _axis(at.int("axis", -1), len(x.shape), "argsort")
    order = _order(x, ax, at.bool("descending", False))
    return [Tensor(dts[0], shapes[0], np.moveaxis(order, -1, ax).astype(np.int64))]


COMPOSITES = {
    "matmul.v1": matmul,
    "linear.v1": linear,
    "dequant_matmul.v1": dequant_matmul,
    "dequant_matmul_a8.v1": dequant_matmul,
    "conv2d.v1": conv2d,
    "swiglu.v1": swiglu,
    "geglu.v1": geglu,
    "softmax.v1": softmax,
    "log_softmax.v1": log_softmax,
    "layer_norm.v1": layer_norm,
    "rms_norm.v1": rms_norm,
    "mean.v1": mean,
    "variance.v1": variance,
    "cumsum.v1": cumsum,
    "rope.v1": rope,
    "attention.v1": attention,
    "embedding.v1": embedding,
    "top_k.v1": top_k,
    "argsort.v1": argsort,
}
for _n in _UNARY:
    COMPOSITES[_n + ".v1"] = _unary(_n)
