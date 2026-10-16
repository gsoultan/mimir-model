"""Writing mmf files: the container of spec/container.md, the metadata
section of spec/metadata.md and the graph section of spec/graph.md.

The writer follows spec/container.md section 10: tensors are packed
canonically in the order they are added, the tensor table is sorted by name,
every section carries a SHA-256 hash, and a file is written to a temporary
file that is flushed, synced and renamed over the destination.
"""

import hashlib
import os
import struct
import tempfile

from . import container as C
from . import dtypes as D
from .errors import MMFError
from .graph import CONST, INPUT, INTERMEDIATE, OUTPUT, PRIMITIVE_VERSION, STDLIB_VERSION, VARIABLE, WEIGHT
from .prims import OP_BY_NAME

__all__ = [
    "Writer", "write_file", "encode_metadata", "GraphBuilder",
    "INTERMEDIATE", "INPUT", "OUTPUT", "WEIGHT", "VARIABLE", "CONST",
]


def _align_up(n, a):
    return (n + a - 1) // a * a


def _check_name(name):
    b = name.encode("utf-8")
    if not 0 < len(b) <= C.MAX_NAME_LENGTH:
        raise MMFError("bad-name", f"tensor name {name!r}")
    return b


class Writer:
    """Collects tensors and sections and writes a complete file."""

    def __init__(self, data_alignment=16384, hashes=True):
        if not C._pow2(data_alignment) or not C.MIN_DATA_ALIGNMENT <= data_alignment <= C.MAX_DATA_ALIGNMENT:
            raise MMFError("bad-alignment", f"data alignment {data_alignment}")
        self.data_alignment = data_alignment
        self.hashes = hashes
        self._tensors = []  # (name bytes, dtype, shape, data)
        self._names = set()
        self._sections = []  # (kind, required, data)

    def add_tensor(self, name, dtype, shape, data):
        """Adds a tensor. dtype is a DType or its name; data holds the
        packed little-endian element bytes, as Tensor.tobytes returns."""
        if isinstance(dtype, str):
            dtype = D.BY_NAME[dtype]
        shape = tuple(int(d) for d in shape)
        b = _check_name(name)
        if b in self._names:
            raise MMFError("duplicate", f"tensor {name!r}")
        if len(shape) > C.MAX_RANK or any(d < 0 for d in shape):
            raise MMFError("bad-shape", f"tensor {name!r} shape {list(shape)}")
        if dtype.is_block and (not shape or shape[-1] % dtype.block_elems):
            raise MMFError("bad-shape", f"tensor {name!r}: {dtype.name} needs a last dimension that is a multiple of {dtype.block_elems}")
        size = D.byte_size(dtype, shape)
        data = bytes(data)
        if len(data) != size:
            raise MMFError("size-mismatch", f"tensor {name!r}: {dtype.name}{list(shape)} needs {size} bytes, got {len(data)}")
        self._names.add(b)
        self._tensors.append((b, dtype, shape, data))
        return len(self._tensors) - 1

    def tensor_index(self, i):
        """The index in the written file's tensor table of the i-th tensor
        added. The table is sorted by name, so this is what a graph value's
        `ref` must hold."""
        order = sorted(range(len(self._tensors)), key=lambda j: self._tensors[j][0])
        return order.index(i)

    def tensor_indices(self):
        """tensor_index for every added tensor, in the order added."""
        order = sorted(range(len(self._tensors)), key=lambda j: self._tensors[j][0])
        pos = [0] * len(order)
        for k, j in enumerate(order):
            pos[j] = k
        return pos

    def add_section(self, kind, data, required=False):
        if kind == C.WEIGHTS:
            raise MMFError("duplicate", "the weights section is written by the writer")
        if 1 <= kind <= 5 and any(k == kind for k, _, _ in self._sections):
            raise MMFError("duplicate", f"two sections of kind {kind}")
        self._sections.append((kind, required, bytes(data)))

    def to_bytes(self):
        # Canonical packing in insertion order (section 6.4).
        offsets, end = [], 0
        for _, _, _, data in self._tensors:
            off = _align_up(end, C._tensor_alignment(len(data)))
            offsets.append(off)
            end = off + len(data)
        weights = bytearray(end)
        for (_, _, _, data), off in zip(self._tensors, offsets):
            weights[off : off + len(data)] = data

        order = sorted(range(len(self._tensors)), key=lambda j: self._tensors[j][0])
        pool = b"".join(self._tensors[j][0] for j in order)
        ns, nt = 1 + len(self._sections), len(self._tensors)
        index_len = C.INDEX_HEADER_SIZE + C.SECTION_ENTRY_SIZE * ns + C.TENSOR_ENTRY_SIZE * nt + len(pool)
        if index_len > C.MAX_INDEX_LENGTH:
            raise MMFError("index-too-large", f"index would be {index_len} bytes")

        pos = _align_up(C.PREAMBLE_SIZE + index_len, 64)
        placed = []
        for kind, required, data in self._sections:
            placed.append((kind, C.FLAG_REQUIRED if required else 0, pos, data))
            pos = _align_up(pos + len(data), 64)
        data_off = _align_up(pos, self.data_alignment)
        file_len = data_off + len(weights)

        def entry(kind, flags, off, data):
            if self.hashes:
                return struct.pack("<IIQQB7x", kind, flags, off, len(data), 1) + hashlib.sha256(data).digest()
            return struct.pack("<IIQQB7x32x", kind, flags, off, len(data), 0)

        idx = bytearray(struct.pack("<IIQ", ns, nt, len(pool)))
        idx += entry(C.WEIGHTS, C.FLAG_REQUIRED, data_off, weights)
        for kind, flags, off, data in placed:
            idx += entry(kind, flags, off, data)
        name_off = 0
        for j in order:
            name, dtype, shape, data = self._tensors[j]
            dims = list(shape) + [0] * (C.MAX_RANK - len(shape))
            idx += struct.pack("<IIIB3x8QQQ", name_off, len(name), dtype.id, len(shape), *dims, offsets[j], len(data))
            name_off += len(name)
        idx += pool
        assert len(idx) == index_len

        out = bytearray(file_len)
        out[0 : C.PREAMBLE_SIZE] = struct.pack(
            "<4sHHIIQQQQQQ", C.MAGIC, 0, 1, C.BYTE_ORDER_MARK, self.data_alignment,
            C.PREAMBLE_SIZE, index_len, data_off, len(weights), file_len, 0,
        )
        out[C.PREAMBLE_SIZE : C.PREAMBLE_SIZE + index_len] = idx
        for _, _, off, data in placed:
            out[off : off + len(data)] = data
        out[data_off:] = weights
        return bytes(out)

    def write(self, path):
        """Writes the file atomically and returns its size."""
        return write_file(path, self.to_bytes())


def write_file(path, data):
    """Writes data to path atomically: a temporary file in the destination
    directory, flushed and synced, then renamed over path."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".mmf-", suffix=".tmp", dir=d)
    try:
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return len(data)


# Metadata --------------------------------------------------------------------

_META_TYPES = {"int": 1, "uint": 2, "float": 3, "bool": 4, "string": 5, "bytes": 6, "ints": 7, "floats": 8, "strings": 9}


def _meta_type(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "string"
    if isinstance(v, (bytes, bytearray)):
        return "bytes"
    if isinstance(v, (list, tuple)):
        if all(isinstance(x, str) for x in v):
            return "strings"
        if all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            return "ints"
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            return "floats"
    raise MMFError("bad-metadata", f"no metadata type for {type(v).__name__}")


def encode_metadata(entries):
    """Encodes a metadata section. entries maps a key to a value, or to a
    (type name, value) pair as decode_metadata returns; a bare value's type
    is inferred (bool, int, float, str, bytes, or a list of str, int or
    float)."""
    items = []
    for key, v in entries.items():
        kb = key.encode("utf-8")
        if not 0 < len(kb) <= 1024:
            raise MMFError("bad-metadata", f"key {key!r}")
        if isinstance(v, tuple) and len(v) == 2 and v[0] in _META_TYPES:
            typ, v = v
        else:
            typ = _meta_type(v)
        items.append((kb, typ, v))
    items.sort(key=lambda e: e[0])
    for a, b in zip(items, items[1:]):
        if a[0] == b[0]:
            raise MMFError("bad-metadata", f"key {a[0].decode()!r} given twice")
    pool = bytearray()

    def put(b, align=1):
        off = _align_up(len(pool), align)
        pool.extend(bytes(off - len(pool)))
        pool.extend(b)
        return off

    rows = []
    for kb, typ, v in items:
        koff = put(kb)
        count, payload = 0, 0
        if typ == "int":
            payload = struct.unpack("<Q", struct.pack("<q", v))[0]
        elif typ == "uint":
            payload = v
        elif typ == "float":
            payload = struct.unpack("<Q", struct.pack("<d", v))[0]
        elif typ == "bool":
            payload = 1 if v else 0
        elif typ in ("string", "bytes"):
            b = v.encode("utf-8") if typ == "string" else bytes(v)
            count, payload = len(b), put(b)
        elif typ in ("ints", "floats"):
            count = len(v)
            payload = put(struct.pack(f"<{count}{'q' if typ == 'ints' else 'd'}", *v), 8)
        else:
            bs = [s.encode("utf-8") for s in v]
            count = len(bs)
            base = _align_up(len(pool), 8)
            tab, body, o = bytearray(), bytearray(), base + 8 * count
            for b in bs:
                tab += struct.pack("<II", o + len(body), len(b))
                body += b
            payload = put(bytes(tab) + bytes(body), 8)
        rows.append(struct.pack("<IIB3xIQ8x", koff, len(kb), _META_TYPES[typ], count, payload))
    return b"MMFM" + struct.pack("<IQ", len(rows), len(pool)) + b"".join(rows) + bytes(pool)


# Graph -----------------------------------------------------------------------

_ATTR_TYPES = {"int": 1, "ints": 2, "float": 3, "string": 4, "dtype": 5, "subgraph": 6}


class GraphBuilder:
    """Builds the graph section of spec/graph.md for a graph with one
    subgraph, the main graph, and optional variable updates.

    A dimension in a shape is an int (a literal size) or a str naming a
    symbolic dimension declared with symdim. Attributes are given as
    (key, type, value) with the type names of spec/graph.md section 3.7:
    int, ints, float, string, dtype (a DType) and subgraph."""

    def __init__(self):
        self.symdims = []  # (name, bound)
        self._sym = {}
        self.values = []  # (name, dtype, shape, kind, ref, zero)
        self.ops = []  # (opcode, inputs, outputs, attrs)
        self.inputs = []
        self.outputs = []
        self.updates = []

    def symdim(self, name, bound=-1):
        if name in self._sym:
            raise MMFError("bad-graph", f"symbolic dimension {name!r} declared twice")
        self._sym[name] = len(self.symdims)
        self.symdims.append((name, bound))
        return name

    def _dim(self, d):
        if isinstance(d, str):
            if d not in self._sym:
                raise MMFError("bad-graph", f"unknown symbolic dimension {d!r}")
            return -1 - self._sym[d]
        if d < 0:
            raise MMFError("bad-graph", f"dimension {d}")
        return int(d)

    def value(self, name, dtype, shape, kind=INTERMEDIATE, ref=0, zero=False):
        if len(shape) > C.MAX_RANK:
            raise MMFError("bad-shape", f"value {name!r} rank {len(shape)}")
        for d in shape:
            self._dim(d)
        self.values.append((name, dtype, tuple(shape), kind, ref, zero))
        vid = len(self.values) - 1
        if kind == INPUT:
            self.inputs.append(vid)
        return vid

    def shape(self, vid):
        return self.values[vid][2]

    def dtype(self, vid):
        return self.values[vid][1]

    def op(self, name, inputs, outputs, attrs=()):
        self.ops.append((OP_BY_NAME[name], list(inputs), list(outputs), list(attrs)))

    def output(self, vid):
        if self.values[vid][3] != OUTPUT:
            raise MMFError("bad-graph", f"value {vid} is not of kind output")
        self.outputs.append(vid)

    def update(self, variable, value):
        self.updates.append((variable, value))

    def encode(self):
        strings, pool, str_idx = [], bytearray(), {}

        def s(x):
            i = str_idx.get(x)
            if i is None:
                b = x.encode("utf-8")
                if len(b) > 4096:
                    raise MMFError("bad-name", f"string {x[:40]!r}... is longer than 4096 bytes")
                i = str_idx[x] = len(strings)
                strings.append(struct.pack("<II", len(pool), len(b)))
                pool.extend(b)
            return i

        dims, runs = [], []

        def dim_run(ds):
            off = len(dims)
            dims.extend(ds)
            return off

        def run(ids):
            off = len(runs)
            runs.extend(ids)
            return off

        s("")
        sym = b"".join(struct.pack("<IIq", s(n), 0, b) for n, b in self.symdims)
        vals = bytearray()
        for name, dt, shape, kind, ref, zero in self.values:
            off = dim_run([self._dim(d) for d in shape])
            vals += struct.pack("<IIIBBBBI12x", s(name), dt.id, off, len(shape), kind, 1 if zero else 0, 0, ref)
        ops, attrs = bytearray(), bytearray()
        for opc, ins, outs, ats in self.ops:
            aoff = len(attrs) // 24
            for key, typ, v in ats:
                count = 0
                if typ == "int":
                    payload = struct.unpack("<Q", struct.pack("<q", int(v)))[0]
                elif typ == "ints":
                    count, payload = len(v), dim_run([int(x) for x in v])
                elif typ == "float":
                    payload = struct.unpack("<Q", struct.pack("<d", float(v)))[0]
                elif typ == "string":
                    payload = s(v)
                elif typ == "dtype":
                    payload = v.id
                else:
                    payload = int(v)
                attrs += struct.pack("<IB3xI4xQ", s(key), _ATTR_TYPES[typ], count, payload)
            ops += struct.pack("<HHIIIIII4x", opc, 0, run(ins), len(ins), run(outs), len(outs), aoff, len(ats))
        sub = struct.pack("<6I8x", 0, len(self.ops), run(self.inputs), len(self.inputs), run(self.outputs), len(self.outputs))
        upd = b"".join(struct.pack("<II", a, b) for a, b in self.updates)
        tables = [
            bytes(pool), b"".join(strings), sym, struct.pack(f"<{len(dims)}q", *dims), bytes(vals), bytes(ops),
            struct.pack(f"<{len(runs)}I", *runs), bytes(attrs), sub, b"", upd,
        ]
        counts = [len(pool), len(strings), len(self.symdims), len(dims), len(self.values), len(self.ops),
                  len(runs), len(attrs) // 24, 1, 0, len(self.updates)]
        out = bytearray(b"MMFG" + struct.pack("<HHII", 0, 1, PRIMITIVE_VERSION, STDLIB_VERSION))
        out += bytes(176)
        for i, t in enumerate(tables):
            off = _align_up(len(out), 8)
            out += bytes(off - len(out))
            struct.pack_into("<QQ", out, 16 + 16 * i, off, counts[i])
            out += t
        return bytes(out)

