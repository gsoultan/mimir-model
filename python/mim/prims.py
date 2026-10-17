"""The 33 primitives of spec/primitives.md on NumPy arrays.

Float arithmetic on f32, f16 and bf16 is done in float64 and rounded once
to the output dtype. For neg, add, mul, recip and sqrt this is correctly
rounded, because float64 carries more than twice the precision of each of
these formats plus two bits, so rounding twice gives the same result as
rounding once. The transcendental functions are within their bounds the
same way. f64 arithmetic is done natively. Subnormals are kept (IEEE mode)
for every float dtype.
"""

import numpy as np

from . import dtypes as D
from .errors import RunError
from .tensor import Tensor, bits_view

OPCODES = {
    1: "neg", 2: "recip", 3: "sqrt", 4: "exp2", 5: "log2", 6: "sin", 7: "trunc", 8: "cast",
    9: "bitcast", 10: "add", 11: "mul", 12: "max", 13: "pow", 14: "idiv", 15: "imod", 16: "shl",
    17: "shr", 18: "and", 19: "or", 20: "xor", 21: "cmplt", 22: "cmpeq", 23: "where",
    24: "reduce", 25: "reshape", 26: "expand", 27: "permute", 28: "pad", 29: "shrink",
    30: "gather", 31: "scatter", 32: "const", 33: "dequantize", 34: "call", 64: "while", 65: "if",
}
OP_BY_NAME = {v: k for k, v in OPCODES.items()}

_ERR = dict(all="ignore")


class Attr:
    """An attribute: type is int, ints, float, string, dtype or subgraph."""

    __slots__ = ("type", "value")

    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"{self.type}:{self.value!r}"


def _attr(attrs, key, typ, default=None, required=True):
    a = attrs.get(key)
    if a is None:
        if required and default is None:
            raise RunError(f"missing attribute {key!r}")
        return default
    if a.type != typ:
        raise RunError(f"attribute {key!r} is {a.type}, want {typ}")
    return a.value


def _only(attrs, *keys):
    for k in attrs:
        if k not in keys:
            raise RunError(f"unknown attribute {k!r}")


def _f64(t):
    return D.to_f64(t.arr, t.dtype)


def _need(cond, msg):
    if not cond:
        raise RunError(msg)


def _broadcast(shapes):
    rank = len(shapes[0])
    _need(all(len(s) == rank for s in shapes), f"ranks differ: {shapes}")
    out = []
    for i in range(rank):
        dims = [s[i] for s in shapes]
        big = [d for d in dims if d != 1]
        _need(len(set(big)) <= 1, f"shapes {shapes} do not broadcast")
        out.append(0 if 0 in dims else max(dims))
    return tuple(out)


def _result(dt, shape, arr):
    arr = np.asarray(arr)
    if dt.kind == "bool":
        arr = arr.astype(np.bool_)
    if tuple(arr.shape) != tuple(shape):
        arr = np.broadcast_to(arr, shape)
    return Tensor(dt, shape, np.array(arr, dtype=dt.storage, copy=True))


# Elementwise unary ----------------------------------------------------------


def _neg(x, dt):
    _need(dt.is_float or dt.is_int, f"neg of {dt.name}")
    if dt.is_float:
        u = bits_view(x.arr)
        sign = np.array(1 << (dt.bits - 1), dtype=u.dtype)
        return (u ^ sign).view(x.arr.dtype)
    with np.errstate(**_ERR):
        return np.negative(x.arr)


def _float_unary(fn):
    def op(x, dt):
        _need(dt.is_float, f"float operation on {dt.name}")
        with np.errstate(**_ERR):
            return D.from_f64(fn(_f64(x)), dt)

    return op


def _recip(v):
    return 1.0 / v


def _trunc(x, dt):
    _need(dt.is_float or dt.is_int, f"trunc of {dt.name}")
    if dt.is_int:
        return x.arr.copy()
    with np.errstate(**_ERR):
        return D.from_f64(np.trunc(_f64(x)), dt)


# cast and bitcast -------------------------------------------------------------


def cast_array(x, to):
    """cast (spec/primitives.md section 5.8) from tensor x to dtype to."""
    src = x.dtype
    _need(not src.is_block and not to.is_block, "cast of a block dtype")
    a = x.arr
    if to.kind == "bool":
        if src.kind in ("float", "storage"):
            return _f64(x) != 0
        return a != 0
    if src.kind == "bool":
        a = a.astype(np.uint8)
        if to.is_int:
            return a.astype(to.storage)
        return D.from_f64(a.astype(np.float64), to)
    if src.is_int:
        if to.is_int:
            return a.astype(to.storage)
        return D.int_to_float(a, src, to)
    # src is a float or storage float
    v = _f64(x)
    if to.kind in ("float", "storage"):
        return D.from_f64(v, to)
    lo, hi = D.int_range(to)
    with np.errstate(**_ERR):
        t = np.trunc(v)
        nan = np.isnan(t)
        over = t >= float(hi) + 1  # 2^(w-1) or 2^w, exact in float64
        under = t < float(lo)
        safe = np.where(nan | over | under, 0.0, t)
        out = safe.astype(to.storage)
    out = np.where(over, to.storage(hi), out)
    out = np.where(under, to.storage(lo), out)
    return out.astype(to.storage)


def _bitcast(x, to):
    src = x.dtype
    _need(src.bits == to.bits and src.kind not in ("bool", "block") and to.kind not in ("bool", "block"),
          f"bitcast {src.name} to {to.name}")
    return bits_view(np.ascontiguousarray(x.arr)).view(to.storage)


# Elementwise binary -----------------------------------------------------------


def _binary_float(fn):
    def op(a, b, dt):
        with np.errstate(**_ERR):
            return D.from_f64(fn(_f64(a), _f64(b)), dt)

    return op


def _max_float(a, b):
    r = np.maximum(a, b)
    z = (a == 0) & (b == 0)
    if z.any():
        both_neg = np.signbit(a) & np.signbit(b)
        r = np.where(z, np.where(both_neg, -0.0, 0.0), r)
    return r


def _pow(a, b):
    r = np.power(a, b)
    return np.where(np.isnan(a) | np.isnan(b), np.nan, r)


def _idiv_imod(a, b, dt):
    """Truncated division and remainder; division by zero gives 0."""
    zero = b == 0
    if dt.kind == "unsigned":
        bs = np.where(zero, 1, b).astype(dt.storage)
        q = a // bs
        r = a - q * bs
    else:
        m1 = b == -1
        bs = np.where(zero | m1, 1, b).astype(dt.storage)
        q = a // bs  # floor
        r = a - q * bs
        fix = (r != 0) & ((a < 0) != (bs < 0))
        q = np.where(fix, q + 1, q).astype(dt.storage)
        r = a - q * bs
        q = np.where(m1, np.negative(a), q)
        r = np.where(m1, 0, r)
    q = np.where(zero, 0, q).astype(dt.storage)
    r = np.where(zero, 0, r).astype(dt.storage)
    return q, r


def _shift(a, b, dt, left):
    ub = b.view(np.dtype(f"u{b.dtype.itemsize}")) if dt.kind == "signed" else b
    big = ub >= dt.bits
    amt = np.where(big, 0, ub).astype(dt.storage)
    if left:
        r = np.left_shift(a, amt)
        return np.where(big, 0, r).astype(dt.storage)
    r = np.right_shift(a, amt)
    fill = np.where(a < 0, -1, 0) if dt.kind == "signed" else 0
    return np.where(big, fill, r).astype(dt.storage)


def _binary(name, a, b, out_dt):
    dt = a.dtype
    _need(a.dtype is b.dtype, f"{name}: operand dtypes {a.dtype.name} and {b.dtype.name} differ")
    shape = _broadcast([a.shape, b.shape])
    x, y = a.arr, b.arr
    with np.errstate(**_ERR):
        if name in ("add", "mul", "max"):
            _need(dt.is_float or dt.is_int, f"{name} of {dt.name}")
            if dt.is_float:
                fn = {"add": np.add, "mul": np.multiply, "max": _max_float}[name]
                return shape, D.from_f64(fn(*np.broadcast_arrays(_f64(a), _f64(b))), dt)
            fn = {"add": np.add, "mul": np.multiply, "max": np.maximum}[name]
            return shape, fn(x, y).astype(dt.storage)
        if name == "pow":
            _need(dt.is_float, f"pow of {dt.name}")
            return shape, D.from_f64(_pow(*np.broadcast_arrays(_f64(a), _f64(b))), dt)
        if name in ("idiv", "imod"):
            _need(dt.is_int, f"{name} of {dt.name}")
            x, y = np.broadcast_arrays(x, y)
            q, r = _idiv_imod(np.asarray(x), np.asarray(y), dt)
            return shape, q if name == "idiv" else r
        if name in ("shl", "shr"):
            _need(dt.is_int, f"{name} of {dt.name}")
            x, y = np.broadcast_arrays(x, y)
            return shape, _shift(np.asarray(x), np.asarray(y), dt, name == "shl")
        if name in ("and", "or", "xor"):
            _need(dt.is_int or dt.kind == "bool", f"{name} of {dt.name}")
            fn = {"and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor}[name]
            return shape, fn(x, y)
        if name in ("cmplt", "cmpeq"):
            ok = dt.is_float or dt.is_int or (name == "cmpeq" and dt.kind == "bool")
            _need(ok, f"{name} of {dt.name}")
            if dt.is_float:
                x, y = _f64(a), _f64(b)
            return shape, (x < y) if name == "cmplt" else (x == y)
    raise RunError(f"unknown binary op {name}")


# reduce -------------------------------------------------------------------------


def _reduce(x, mode, axes):
    dt = x.dtype
    _need(dt.is_float or dt.is_int, f"reduce of {dt.name}")
    r = len(x.shape)
    _need(len(set(axes)) == len(axes) and all(0 <= a < r for a in axes), f"reduce axes {axes}")
    axes = tuple(axes)
    if not axes:
        return x.arr.copy()
    with np.errstate(**_ERR):
        if dt.is_float:
            v = _f64(x)
            if mode == "add":
                return D.from_f64(np.sum(v, axis=axes, keepdims=True), dt)
            if mode == "mul":
                return D.from_f64(np.prod(v, axis=axes, keepdims=True), dt)
            if mode == "max":
                m = np.max(v, axis=axes, keepdims=True, initial=-np.inf)
                pos0 = np.any((v == 0) & ~np.signbit(v), axis=axes, keepdims=True)
                m = np.where((m == 0) & pos0, 0.0, m)
                return D.from_f64(m, dt)
        else:
            a = x.arr
            if mode == "add":
                return np.sum(a, axis=axes, keepdims=True, dtype=dt.storage)
            if mode == "mul":
                return np.prod(a, axis=axes, keepdims=True, dtype=dt.storage)
            if mode == "max":
                return np.max(a, axis=axes, keepdims=True, initial=D.int_range(dt)[0])
    raise RunError(f"reduce mode {mode!r}")


# const and pad values ---------------------------------------------------------


def scalar_value(attr, dt):
    """The storage value of a const or pad `value` attribute in dtype dt."""
    _need(not dt.is_block, f"const of {dt.name}")
    if attr.type == "float":
        v = float(attr.value)
        if dt.kind in ("float", "storage"):
            return D.from_f64(np.array(v), dt)
        _need(v == np.trunc(v), f"value {v} is not an integer")
        iv = int(v)
    elif attr.type == "int":
        iv = int(attr.value)
        if dt.kind in ("float", "storage"):
            return D.int_to_float(np.array(iv, dtype=np.int64), D.BY_NAME["i64"], dt)
    else:
        raise RunError(f"value attribute is {attr.type}")
    if dt.kind == "bool":
        _need(iv in (0, 1), f"bool value {iv}")
        return np.array(bool(iv))
    lo, hi = D.int_range(dt)
    _need(lo <= iv <= hi, f"value {iv} does not fit {dt.name}")
    return np.array(iv, dtype=dt.storage)


# gather and scatter ---------------------------------------------------------------


def _gather(data, idx, axis, out_shape):
    _need(len(data.shape) >= 1 and 0 <= axis < len(data.shape), f"gather axis {axis}")
    _need(idx.dtype.name in ("i32", "i64"), f"gather indices of {idx.dtype.name}")
    want = data.shape[:axis] + idx.shape + data.shape[axis + 1 :]
    _need(tuple(out_shape) == want, f"gather output {list(out_shape)}, derived {list(want)}")
    n = data.shape[axis]
    ind = idx.arr.astype(np.int64)
    if n == 0:
        return np.zeros(out_shape, dtype=data.dtype.storage)
    # Out-of-range indices give unspecified elements but never read outside
    # the data: they are clamped.
    safe = np.clip(ind, 0, n - 1)
    return np.take(data.arr, safe, axis=axis)


def _scatter(data, idx, upd, axis):
    _need(len(data.shape) >= 1 and 0 <= axis < len(data.shape), f"scatter axis {axis}")
    _need(idx.dtype.name in ("i32", "i64"), f"scatter indices of {idx.dtype.name}")
    _need(upd.dtype is data.dtype, "scatter updates dtype")
    want = data.shape[:axis] + idx.shape + data.shape[axis + 1 :]
    _need(upd.shape == want, f"scatter updates {list(upd.shape)}, want {list(want)}")
    y = bits_view(data.arr).copy()
    u = bits_view(upd.arr).reshape(data.shape[:axis] + (-1,) + data.shape[axis + 1 :]) if idx.size else None
    n = data.shape[axis]
    pre = (slice(None),) * axis
    for j, i in enumerate(idx.arr.reshape(-1).tolist()):
        if 0 <= i < n:  # an out-of-range update is dropped, never written outside y
            y[pre + (i,)] = u[pre + (j,)]
    return y.view(data.arr.dtype)


# dispatch -------------------------------------------------------------------------


def evaluate(op, attrs, inputs, out_dt, out_shape):
    """Evaluates primitive `op` (a name or an opcode) and returns a Tensor of
    the declared dtype and shape. attrs maps keys to Attr."""
    if isinstance(op, int):
        op = OPCODES.get(op)
        _need(op is not None, "unknown opcode")
    out_shape = tuple(int(s) for s in out_shape)
    n_in = {"where": 3, "scatter": 3, "const": 0}.get(op, 2 if op in _BINARY or op == "gather" else 1)
    _need(len(inputs) == n_in, f"{op} takes {n_in} inputs, got {len(inputs)}")
    for t in inputs:
        _need(not t.dtype.is_block or op == "dequantize", f"{op} of block dtype {t.dtype.name}")
        _need(t.dtype.kind != "storage" or op in _STORAGE_OK, f"{op} of storage float {t.dtype.name}")

    if op in _UNARY:
        _only(attrs)
        x = inputs[0]
        _need(out_dt is x.dtype and out_shape == x.shape, f"{op}: declared {out_dt.name}{list(out_shape)}")
        return _result(out_dt, out_shape, _UNARY[op](x, x.dtype))
    if op in _BINARY:
        _only(attrs)
        a, b = inputs
        want_dt = D.BOOL if op in ("cmplt", "cmpeq") else a.dtype
        shape, r = _binary(op, a, b, want_dt)
        _need(out_dt is want_dt and out_shape == shape, f"{op}: declared {out_dt.name}{list(out_shape)}, derived {want_dt.name}{list(shape)}")
        return _result(out_dt, out_shape, r)
    if op == "where":
        _only(attrs)
        c, a, b = inputs
        _need(c.dtype.kind == "bool" and a.dtype is b.dtype, "where operand dtypes")
        shape = _broadcast([c.shape, a.shape, b.shape])
        _need(out_dt is a.dtype and out_shape == shape, "where: declared type")
        r = np.where(c.arr, bits_view(a.arr), bits_view(b.arr))
        return _result(out_dt, out_shape, np.asarray(r).view(a.arr.dtype))
    if op == "cast":
        _only(attrs, "to")
        to = D.BY_ID[_attr(attrs, "to", "dtype")]
        _need(out_dt is to and out_shape == inputs[0].shape, "cast: declared type")
        return _result(out_dt, out_shape, cast_array(inputs[0], to))
    if op == "bitcast":
        _only(attrs, "to")
        to = D.BY_ID[_attr(attrs, "to", "dtype")]
        _need(out_dt is to and out_shape == inputs[0].shape, "bitcast: declared type")
        return _result(out_dt, out_shape, _bitcast(inputs[0], to))
    if op == "reduce":
        _only(attrs, "mode", "axes")
        x = inputs[0]
        axes = list(_attr(attrs, "axes", "ints"))
        mode = _attr(attrs, "mode", "string")
        want = tuple(1 if i in axes else d for i, d in enumerate(x.shape))
        _need(out_dt is x.dtype and out_shape == want, f"reduce: declared {list(out_shape)}, derived {list(want)}")
        return _result(out_dt, out_shape, _reduce(x, mode, axes))
    if op == "reshape":
        _only(attrs, "shape")
        x = inputs[0]
        spec = list(_attr(attrs, "shape", "ints"))
        _need(len(spec) == len(out_shape) and all(s == -1 or s == d for s, d in zip(spec, out_shape)),
              f"reshape: shape {spec}, declared {list(out_shape)}")
        _need(int(np.prod(out_shape, dtype=np.int64)) == x.size, "reshape changes the element count")
        _need(out_dt is x.dtype, "reshape: declared dtype")
        return Tensor(out_dt, out_shape, x.arr.reshape(out_shape))
    if op == "expand":
        _only(attrs)
        x = inputs[0]
        _need(len(out_shape) == len(x.shape) and all(i == o or i == 1 for i, o in zip(x.shape, out_shape)),
              f"expand {list(x.shape)} to {list(out_shape)}")
        _need(out_dt is x.dtype, "expand: declared dtype")
        return Tensor(out_dt, out_shape, np.broadcast_to(x.arr, out_shape))
    if op == "permute":
        _only(attrs, "perm")
        x = inputs[0]
        perm = list(_attr(attrs, "perm", "ints"))
        _need(sorted(perm) == list(range(len(x.shape))), f"perm {perm}")
        _need(out_shape == tuple(x.shape[p] for p in perm) and out_dt is x.dtype, "permute: declared type")
        return Tensor(out_dt, out_shape, np.transpose(x.arr, perm))
    if op == "pad":
        _only(attrs, "pads", "value")
        x = inputs[0]
        pads = list(_attr(attrs, "pads", "ints"))
        r = len(x.shape)
        _need(len(pads) == 2 * r and all(p >= 0 for p in pads), f"pads {pads}")
        want = tuple(x.shape[i] + pads[2 * i] + pads[2 * i + 1] for i in range(r))
        _need(out_shape == want and out_dt is x.dtype, "pad: declared type")
        value = attrs.get("value", Attr("float", 0.0))
        fill = bits_view(np.asarray(scalar_value(value, x.dtype), dtype=x.dtype.storage).reshape(()))
        y = np.full(out_shape, fill, dtype=fill.dtype)
        y[tuple(slice(pads[2 * i], pads[2 * i] + x.shape[i]) for i in range(r))] = bits_view(x.arr)
        return Tensor(out_dt, out_shape, y.view(x.arr.dtype))
    if op == "shrink":
        _only(attrs, "begin", "end")
        x = inputs[0]
        begin = list(_attr(attrs, "begin", "ints"))
        end = list(_attr(attrs, "end", "ints"))
        r = len(x.shape)
        _need(len(begin) == r and len(end) == r, "shrink begin and end")
        end = [x.shape[i] if e == -1 else e for i, e in enumerate(end)]
        _need(all(0 <= b <= e <= d for b, e, d in zip(begin, end, x.shape)), f"shrink [{begin}, {end})")
        want = tuple(e - b for b, e in zip(begin, end))
        _need(out_shape == want and out_dt is x.dtype, "shrink: declared type")
        return Tensor(out_dt, out_shape, x.arr[tuple(slice(b, e) for b, e in zip(begin, end))])
    if op == "gather":
        _only(attrs, "axis")
        data, idx = inputs
        _need(out_dt is data.dtype, "gather: declared dtype")
        r = _gather(data, idx, _attr(attrs, "axis", "int"), out_shape)
        return Tensor(out_dt, out_shape, r)
    if op == "scatter":
        _only(attrs, "axis")
        data, idx, upd = inputs
        _need(out_dt is data.dtype and out_shape == data.shape, "scatter: declared type")
        return Tensor(out_dt, out_shape, _scatter(data, idx, upd, _attr(attrs, "axis", "int")))
    if op == "const":
        _only(attrs, "value")
        _need(out_dt.kind != "storage", "const of a storage float")
        v = attrs.get("value")
        _need(v is not None, "const needs a value")
        s = np.asarray(scalar_value(v, out_dt), dtype=out_dt.storage)
        return Tensor(out_dt, out_shape, np.broadcast_to(s, out_shape))
    if op == "dequantize":
        _only(attrs)
        x = inputs[0]
        _need(x.dtype.is_block or x.dtype.name in ("f16", "bf16", "f8_e4m3", "f8_e5m2"), f"dequantize of {x.dtype.name}")
        _need(out_dt.name in ("f32", "bf16") and out_shape == x.shape, "dequantize: declared type")
        if x.dtype.is_block:
            y = D.dequantize_block(x.raw, x.dtype, x.shape)
        else:
            with np.errstate(**_ERR):
                y = _f64(x).astype(np.float32)
        if out_dt.name == "bf16":
            y = D.from_f64(y.astype(np.float64), out_dt)
        return Tensor(out_dt, out_shape, y)
    raise RunError(f"{op} is not a primitive")


_UNARY = {
    "neg": _neg,
    "recip": _float_unary(_recip),
    "sqrt": _float_unary(np.sqrt),
    "exp2": _float_unary(np.exp2),
    "log2": _float_unary(np.log2),
    "sin": _float_unary(np.sin),
    "trunc": _trunc,
}
_BINARY = ("add", "mul", "max", "pow", "idiv", "imod", "shl", "shr", "and", "or", "xor", "cmplt", "cmpeq")
_STORAGE_OK = ("cast", "bitcast", "dequantize", "reshape", "expand", "permute", "pad", "shrink", "gather", "scatter", "where")
