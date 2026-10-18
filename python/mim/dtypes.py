"""Element types of spec/dtypes.md: ids, sizes, conversions between stored
bits and values, and the block decoders.

A tensor's elements are held in a NumPy array of the dtype's storage type:
the IEEE type for f32, f16 and f64, the integer type for integers, NumPy
bool for bool, and the unsigned integer of the same width for bf16, f8_e4m3
and f8_e5m2, whose bits NumPy cannot interpret. Block-dtype tensors are held
as their raw bytes.
"""

import numpy as np

from .errors import MIMError


class DType:
    __slots__ = ("id", "name", "block_elems", "block_bytes", "kind", "storage")

    def __init__(self, id, name, block_elems, block_bytes, kind, storage):
        self.id = id
        self.name = name
        self.block_elems = block_elems
        self.block_bytes = block_bytes
        # float, storage (f8), signed, unsigned, bool or block
        self.kind = kind
        self.storage = storage

    @property
    def is_block(self):
        return self.kind == "block"

    @property
    def is_float(self):
        return self.kind == "float"

    @property
    def is_int(self):
        return self.kind in ("signed", "unsigned")

    @property
    def bits(self):
        return 8 * self.block_bytes

    def __repr__(self):
        return self.name


_TABLE = [
    (1, "f32", 1, 4, "float", np.float32),
    (2, "f16", 1, 2, "float", np.float16),
    (3, "bf16", 1, 2, "float", np.uint16),
    (4, "f64", 1, 8, "float", np.float64),
    (5, "i8", 1, 1, "signed", np.int8),
    (6, "u8", 1, 1, "unsigned", np.uint8),
    (7, "i16", 1, 2, "signed", np.int16),
    (8, "u16", 1, 2, "unsigned", np.uint16),
    (9, "i32", 1, 4, "signed", np.int32),
    (10, "u32", 1, 4, "unsigned", np.uint32),
    (11, "i64", 1, 8, "signed", np.int64),
    (12, "u64", 1, 8, "unsigned", np.uint64),
    (13, "bool", 1, 1, "bool", np.bool_),
    (14, "f8_e4m3", 1, 1, "storage", np.uint8),
    (15, "f8_e5m2", 1, 1, "storage", np.uint8),
    (64, "q8_0", 32, 34, "block", np.uint8),
    (65, "q4_0", 32, 18, "block", np.uint8),
    (66, "q4_k", 256, 144, "block", np.uint8),
    (67, "q6_k", 256, 210, "block", np.uint8),
    (68, "mxfp4", 32, 17, "block", np.uint8),
    (69, "affine4_g64", 64, 36, "block", np.uint8),
]

BY_ID = {}
BY_NAME = {}
for _row in _TABLE:
    _d = DType(*_row)
    BY_ID[_d.id] = _d
    BY_NAME[_d.name] = _d

F32, F16, BF16, F64 = (BY_NAME[n] for n in ("f32", "f16", "bf16", "f64"))
I32, I64, BOOL = BY_NAME["i32"], BY_NAME["i64"], BY_NAME["bool"]
F8E4M3, F8E5M2 = BY_NAME["f8_e4m3"], BY_NAME["f8_e5m2"]


def byte_size(dt, shape):
    """Bytes of a tensor (spec/container.md section 6.2), raising the
    container's error codes."""
    n = 1
    for s in shape:
        n *= s
        if n >= 1 << 64:
            raise MIMError("overflow", "element count overflows u64")
    if dt.is_block:
        if not shape:
            raise MIMError("bad-shape", f"{dt.name} tensor must have rank >= 1")
        if shape[-1] % dt.block_elems:
            raise MIMError("bad-shape", f"{dt.name}: last dimension {shape[-1]} is not a multiple of {dt.block_elems}")
    size = n // dt.block_elems * dt.block_bytes
    if size >= 1 << 64:
        raise MIMError("overflow", "byte size overflows u64")
    return size


def int_range(dt):
    if dt.kind == "signed":
        return -(1 << (dt.bits - 1)), (1 << (dt.bits - 1)) - 1
    return 0, (1 << dt.bits) - 1


# Float formats: precision in bits including the implicit one, the exponent
# of the smallest normal number, and the largest finite value.
_FORMATS = {
    "f32": (24, -126, float(np.finfo(np.float32).max)),
    "f16": (11, -14, 65504.0),
    "bf16": (8, -126, float.fromhex("0x1.fep127")),
    "f64": (53, -1022, float(np.finfo(np.float64).max)),
    "f8_e4m3": (4, -6, 448.0),
    "f8_e5m2": (3, -14, 57344.0),
}


def _round_format(x, p, emin):
    """Rounds float64 values to p bits of precision with the subnormal
    floor of emin, to nearest with ties to even. The result is a float64
    holding a value of the target format, before any overflow handling."""
    x = np.array(x, dtype=np.float64, copy=True)
    fin = np.isfinite(x) & (x != 0)
    if fin.any():
        v = x[fin]
        _, e = np.frexp(v)  # |v| in [2^(e-1), 2^e)
        ue = np.maximum(e - 1, emin) - (p - 1)
        x[fin] = np.ldexp(np.rint(np.ldexp(v, -ue)), ue)
    return x


def _f8_codes(dt):
    """Value bits of every finite f8 code, for encoding."""
    vals = decode_f8(np.arange(256, dtype=np.uint8), dt)
    table = {}
    for code in range(256):
        v = vals[code]
        if np.isfinite(v):
            table[np.float64(v).tobytes()] = code
    return table


_F8_TABLES = {}


def to_f64(arr, dt):
    """The values of a plain-dtype storage array as float64. Integers above
    2^53 lose precision; callers that convert integers exactly use
    int_values."""
    a = np.asarray(arr)
    with np.errstate(invalid="ignore"):
        return _to_f64(a, dt)


def _to_f64(a, dt):
    if dt.name == "bf16":
        return (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if dt.kind == "storage":
        return decode_f8(a, dt)
    return a.astype(np.float64)


def decode_f8(a, dt):
    a = np.asarray(a, dtype=np.uint8)
    s = np.where(a & 0x80, -1.0, 1.0)
    if dt.name == "f8_e4m3":
        e = (a >> 3) & 0xF
        m = (a & 7).astype(np.float64)
        v = np.where(e == 0, np.ldexp(m / 8, -6), np.ldexp(1 + m / 8, e.astype(np.int32) - 7))
        v = np.where((a & 0x7F) == 0x7F, np.nan, v)
    else:
        e = (a >> 2) & 0x1F
        m = (a & 3).astype(np.float64)
        v = np.where(e == 0, np.ldexp(m / 4, -14), np.ldexp(1 + m / 4, e.astype(np.int32) - 15))
        v = np.where(e == 31, np.where(m == 0, np.inf, np.nan), v)
    return s * v


def from_f64(x, dt):
    """Rounds float64 values once to the float or storage-float dtype dt
    and returns the storage array. f64 values are taken as exact."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if dt.name == "f64":
            return x.copy()
        if dt.name == "f32":
            return x.astype(np.float32)
        if dt.name == "f16":
            return x.astype(np.float16)
        p, emin, big = _FORMATS[dt.name]
        r = _round_format(x, p, emin)
        if dt.name == "bf16":
            r = np.where(np.abs(r) > big, np.copysign(np.inf, r), r)
            b = (r.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            return np.where(np.isnan(r), np.uint16(0x7FC0), b).astype(np.uint16)
        # Storage floats (spec/dtypes.md section 2.3).
        nan = np.isnan(r)
        r = np.where(np.isfinite(x) & (np.abs(r) > big), np.copysign(big, r), r)
        if dt.name == "f8_e4m3":
            r = np.where(np.isinf(r), np.copysign(big, r), r)
        if dt.name not in _F8_TABLES:
            _F8_TABLES[dt.name] = _f8_codes(dt)
        table = _F8_TABLES[dt.name]
        out = np.zeros(r.shape, dtype=np.uint8)
        flat, of = r.reshape(-1), out.reshape(-1)
        isnan = nan.reshape(-1)
        for i in range(flat.size):
            if isnan[i]:
                of[i] = 0x7F if dt.name == "f8_e4m3" else 0x7E
            elif np.isinf(flat[i]):
                of[i] = 0x7C if flat[i] > 0 else 0xFC
            else:
                of[i] = table[np.float64(flat[i]).tobytes()]
        return out


def int_to_float(a, src, dt):
    """Converts integers exactly rounded to a float or storage-float dtype."""
    a = np.asarray(a)
    lo, hi = int_range(src)
    if max(abs(lo), hi) <= 1 << 53:
        return from_f64(a.astype(np.float64), dt)
    # 64-bit integers: round from the exact value, element by element.
    p, emin, big = _FORMATS[dt.name]
    out = np.empty(a.shape, dtype=np.float64)
    flat, of = a.reshape(-1), out.reshape(-1)
    for i in range(flat.size):
        of[i] = _round_int(int(flat[i]), p)
    return from_f64(out, dt)


def _round_int(v, p):
    """The integer v rounded to p significant bits, ties to even, as a
    float64 (or inf when it exceeds float64)."""
    if v == 0:
        return 0.0
    s = -1 if v < 0 else 1
    m = abs(v)
    shift = m.bit_length() - p
    if shift > 0:
        q, r = divmod(m, 1 << shift)
        half = 1 << (shift - 1)
        if r > half or (r == half and q & 1):
            q += 1
        m = q << shift
    try:
        return s * float(m)
    except OverflowError:
        return s * float("inf")


# Block decoders, spec/dtypes.md section 3. Every step is a float32
# operation in the written order.


def _f16(b):
    return np.ascontiguousarray(b).view(np.float16).astype(np.float32).reshape(b.shape[:-1])


def dequantize_block(raw, dt, shape):
    """Decodes a block-dtype tensor of logical shape `shape` to float32."""
    nb = int(np.prod(shape, dtype=np.int64)) // dt.block_elems
    b = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(nb, dt.block_bytes)
    with np.errstate(all="ignore"):
        y = _DECODERS[dt.name](b)
    return y.reshape(shape)


def _q8_0(b):
    d = _f16(b[:, 0:2])[:, None]
    q = b[:, 2:34].view(np.int8).astype(np.float32)
    return d * q


def _q4_0(b):
    d = _f16(b[:, 0:2])[:, None]
    qs = b[:, 2:18]
    n = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.int32) - 8
    return d * n.astype(np.float32)


def _q4_k(b):
    nb = b.shape[0]
    d = _f16(b[:, 0:2])
    dmin = _f16(b[:, 2:4])
    sc_raw = b[:, 4:16].astype(np.int32)
    qs = b[:, 16:144]
    sc = np.empty((nb, 8), dtype=np.int32)
    m = np.empty((nb, 8), dtype=np.int32)
    for j in range(4):
        sc[:, j] = sc_raw[:, j] & 63
        m[:, j] = sc_raw[:, j + 4] & 63
    for j in range(4, 8):
        sc[:, j] = (sc_raw[:, j + 4] & 0x0F) | ((sc_raw[:, j - 4] >> 6) << 4)
        m[:, j] = (sc_raw[:, j + 4] >> 4) | ((sc_raw[:, j] >> 6) << 4)
    ds = d[:, None] * sc.astype(np.float32)  # d * sc, once per sub-block
    dm = dmin[:, None] * m.astype(np.float32)
    y = np.empty((nb, 256), dtype=np.float32)
    for c in range(4):
        q = qs[:, 32 * c : 32 * c + 32]
        lo = (q & 0x0F).astype(np.float32)
        hi = (q >> 4).astype(np.float32)
        y[:, 64 * c : 64 * c + 32] = ds[:, 2 * c, None] * lo - dm[:, 2 * c, None]
        y[:, 64 * c + 32 : 64 * c + 64] = ds[:, 2 * c + 1, None] * hi - dm[:, 2 * c + 1, None]
    return y


def _q6_k(b):
    nb = b.shape[0]
    ql = b[:, 0:128].astype(np.int32)
    qh = b[:, 128:192].astype(np.int32)
    scales = b[:, 192:208].view(np.int8).astype(np.float32)
    d = _f16(b[:, 208:210])
    y = np.empty((nb, 256), dtype=np.float32)
    for h in range(2):
        qlh = ql[:, 64 * h : 64 * h + 64]
        qhh = qh[:, 32 * h : 32 * h + 32]
        sch = scales[:, 8 * h : 8 * h + 8]
        for l in range(32):
            i = l // 16
            q1 = ((qlh[:, l] & 0x0F) | (((qhh[:, l] >> 0) & 3) << 4)) - 32
            q2 = ((qlh[:, l + 32] & 0x0F) | (((qhh[:, l] >> 2) & 3) << 4)) - 32
            q3 = ((qlh[:, l] >> 4) | (((qhh[:, l] >> 4) & 3) << 4)) - 32
            q4 = ((qlh[:, l + 32] >> 4) | (((qhh[:, l] >> 6) & 3) << 4)) - 32
            base = 128 * h + l
            y[:, base] = (d * sch[:, i]) * q1.astype(np.float32)
            y[:, base + 32] = (d * sch[:, i + 2]) * q2.astype(np.float32)
            y[:, base + 64] = (d * sch[:, i + 4]) * q3.astype(np.float32)
            y[:, base + 96] = (d * sch[:, i + 6]) * q4.astype(np.float32)
    return y


_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def _mxfp4(b):
    e = b[:, 0].astype(np.int32)
    qs = b[:, 1:17]
    n = np.concatenate([qs & 0x0F, qs >> 4], axis=1)
    # 2^(e-127) times an E2M1 value is exact in f32 when in range, so it is
    # computed exactly in f64 and converted once.
    return np.ldexp(_E2M1[n], (e - 127)[:, None]).astype(np.float32)


def _affine4_g64(b):
    scale = _f16(b[:, 0:2])[:, None]
    bias = _f16(b[:, 2:4])[:, None]
    qs = b[:, 4:36]
    n = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.float32)
    return scale * n + bias


_DECODERS = {
    "q8_0": _q8_0,
    "q4_0": _q4_0,
    "q4_k": _q4_k,
    "q6_k": _q6_k,
    "mxfp4": _mxfp4,
    "affine4_g64": _affine4_g64,
}
