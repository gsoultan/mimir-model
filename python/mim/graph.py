"""The graph section of spec/graph.md: decoding and the structural rules of
section 4."""

import struct
from collections import namedtuple

from . import dtypes as D
from .errors import MIMError
from .prims import OPCODES, Attr

INTERMEDIATE, INPUT, OUTPUT, WEIGHT, VARIABLE, CONST = range(6)
ROW_SIZES = (1, 8, 16, 8, 32, 32, 4, 24, 32, 16, 8)
MAX_ROWS = 1 << 28
PRIMITIVE_VERSION = 1
STDLIB_VERSION = 1

SymDim = namedtuple("SymDim", "name bound")
Composite = namedtuple("Composite", "name version subgraph")
Update = namedtuple("Update", "variable value")


class Value:
    __slots__ = ("name", "dtype", "shape", "kind", "ref", "zero")

    def __init__(self, name, dtype, shape, kind, ref, zero):
        self.name, self.dtype, self.shape, self.kind, self.ref, self.zero = name, dtype, shape, kind, ref, zero

    def __repr__(self):
        return f"Value({self.name!r}, {self.dtype.name}{self.shape}, kind={self.kind})"


class Op:
    __slots__ = ("opcode", "inputs", "outputs", "attrs")

    def __init__(self, opcode, inputs, outputs, attrs):
        self.opcode, self.inputs, self.outputs, self.attrs = opcode, inputs, outputs, attrs

    @property
    def name(self):
        return OPCODES[self.opcode]


class Subgraph:
    __slots__ = ("ops", "inputs", "outputs")

    def __init__(self, ops, inputs, outputs):
        self.ops, self.inputs, self.outputs = ops, inputs, outputs


class Graph:
    def __init__(self):
        self.primitive_version = 0
        self.stdlib_version = 0
        self.symdims = []
        self.values = []
        self.subgraphs = []
        self.composites = []
        self.updates = []

    @property
    def main(self):
        return self.subgraphs[0]


def _bad(msg):
    return MIMError("bad-graph", msg)


def decode_graph(b):
    b = memoryview(b)
    if len(b) < 192 or bytes(b[:4]) not in (b"MIMG", b"MMFG"):
        raise _bad("missing header")
    major, _minor, prim, std = struct.unpack_from("<HHII", b, 4)
    if major != 0:
        raise MIMError("unsupported-version", f"graph format {major}")
    if prim > PRIMITIVE_VERSION or std > STDLIB_VERSION:
        raise MIMError("unsupported-version", f"primitives v{prim}, stdlib v{std}")
    g = Graph()
    g.primitive_version, g.stdlib_version = prim, std
    tabs, counts = [], []
    for i in range(11):
        off, n = struct.unpack_from("<QQ", b, 16 + 16 * i)
        size = n * ROW_SIZES[i]
        if n > MAX_ROWS or off % 8 or off > len(b) or size > len(b) - off:
            raise _bad(f"table {i} out of bounds")
        tabs.append(b[off : off + size])
        counts.append(n)
    pool, strtab, symtab, dims, vals, ops, runs, attrs, subs, comps, upds = tabs
    if counts[1] == 0:
        raise _bad("string table is empty")

    strings = []
    for i in range(counts[1]):
        off, ln = struct.unpack_from("<II", strtab, 8 * i)
        if off > len(pool) or ln > len(pool) - off:
            raise _bad(f"string {i} outside pool")
        raw = bytes(pool[off : off + ln])
        try:
            s = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise MIMError("bad-name", f"string {i}") from None
        if len(raw) > 4096:
            raise MIMError("bad-name", f"string {i}")
        strings.append(s)

    def string(i, what):
        if i >= len(strings):
            raise _bad(f"{what}: string {i} out of range")
        return strings[i]

    def dims_at(off, n, what):
        if off > counts[3] or n > counts[3] - off:
            raise _bad(f"{what}: dims out of range")
        return list(struct.unpack_from(f"<{n}q", dims, 8 * off))

    def run_at(off, n, what):
        if off > counts[6] or n > counts[6] - off:
            raise _bad(f"{what}: run out of range")
        r = list(struct.unpack_from(f"<{n}I", runs, 4 * off))
        for v in r:
            if v >= counts[4]:
                raise _bad(f"{what}: value {v} out of range")
        return r

    for i in range(counts[2]):
        name_i, reserved, bound = struct.unpack_from("<IIq", symtab, 16 * i)
        name = string(name_i, "symdim")
        if reserved:
            raise MIMError("reserved-non-zero", f"symdim {i}")
        if bound < -1:
            raise _bad(f"symdim {name!r} bound {bound}")
        g.symdims.append(SymDim(name, bound))

    for i in range(counts[4]):
        e = 32 * i
        name_i, dt_id, shape_off, rank, kind, flags, res = struct.unpack_from("<IIIBBBB", vals, e)
        ref = struct.unpack_from("<I", vals, e + 16)[0]
        name = string(name_i, "value")
        if rank > 8 or flags & ~1 or res or any(vals[e + 20 : e + 32]):
            raise _bad(f"value {i} header")
        dt = D.BY_ID.get(dt_id)
        if dt is None:
            raise MIMError("bad-dtype", f"value {i} dtype {dt_id}")
        if kind > CONST:
            raise _bad(f"value {i} kind {kind}")
        shape = dims_at(shape_off, rank, "value")
        for d in shape:
            if d < 0 and -1 - d >= counts[2]:
                raise _bad(f"value {i} symbolic dim out of range")
        g.values.append(Value(name, dt, shape, kind, ref, flags == 1))

    if counts[8] == 0:
        raise _bad("no main subgraph")
    owner = [-1] * counts[5]
    for si in range(counts[8]):
        e = 32 * si
        begin, count, ioff, inn, ooff, on = struct.unpack_from("<6I", subs, e)
        if begin > counts[5] or count > counts[5] - begin or any(subs[e + 24 : e + 32]):
            raise _bad(f"subgraph {si} op range")
        sg = Subgraph([], run_at(ioff, inn, "subgraph inputs"), run_at(ooff, on, "subgraph outputs"))
        for k in range(count):
            oi = begin + k
            if owner[oi] != -1:
                raise _bad(f"op {oi} in two subgraphs")
            owner[oi] = si
            oe = 32 * oi
            opc, res2, inoff, inn2, outoff, outn, aoff, an, res3 = struct.unpack_from("<HH7I", ops, oe)
            if res2 or res3:
                raise MIMError("reserved-non-zero", f"op {oi}")
            if opc not in OPCODES:
                raise _bad(f"op {oi} opcode {opc}")
            op = Op(opc, run_at(inoff, inn2, "op inputs"), run_at(outoff, outn, "op outputs"), {})
            if aoff > counts[7] or an > counts[7] - aoff:
                raise _bad(f"op {oi} attrs out of range")
            for j in range(an):
                ae = 24 * (aoff + j)
                key_i, typ = struct.unpack_from("<IB", attrs, ae)
                key = string(key_i, "attr")
                if any(attrs[ae + 5 : ae + 8]) or struct.unpack_from("<I", attrs, ae + 12)[0]:
                    raise MIMError("reserved-non-zero", f"attr {key!r}")
                cnt = struct.unpack_from("<I", attrs, ae + 8)[0]
                payload = struct.unpack_from("<Q", attrs, ae + 16)[0]
                if typ != 2 and cnt:
                    raise _bad(f"attr {key!r}: count on a scalar")
                if typ == 1:
                    a = Attr("int", struct.unpack("<q", struct.pack("<Q", payload))[0])
                elif typ == 2:
                    a = Attr("ints", dims_at(payload, cnt, "attr " + key))
                elif typ == 3:
                    a = Attr("float", struct.unpack("<d", struct.pack("<Q", payload))[0])
                elif typ == 4:
                    if payload >= 1 << 32:
                        raise _bad(f"attr {key!r} string")
                    a = Attr("string", string(payload, "attr " + key))
                elif typ == 5:
                    if payload not in D.BY_ID:
                        raise MIMError("bad-dtype", f"attr {key!r}")
                    a = Attr("dtype", payload)
                elif typ == 6:
                    if payload >= counts[8] or payload == 0:
                        raise _bad(f"attr {key!r} subgraph {payload}")
                    a = Attr("subgraph", payload)
                else:
                    raise _bad(f"attr {key!r} type {typ}")
                if key in op.attrs:
                    raise _bad(f"attr {key!r} given twice")
                op.attrs[key] = a
            sg.ops.append(op)
        g.subgraphs.append(sg)
    if -1 in owner:
        raise _bad(f"op {owner.index(-1)} belongs to no subgraph")
    for i in range(counts[9]):
        name_i, ver, sub, res = struct.unpack_from("<4I", comps, 16 * i)
        name = string(name_i, "composite")
        if res:
            raise MIMError("reserved-non-zero", f"composite {name!r}")
        if sub == 0 or sub >= counts[8]:
            raise _bad(f"composite {name!r} subgraph {sub}")
        g.composites.append(Composite(name, ver, sub))
    for i in range(counts[10]):
        var, val = struct.unpack_from("<II", upds, 8 * i)
        if var >= counts[4] or val >= counts[4]:
            raise _bad(f"update {i} value out of range")
        g.updates.append(Update(var, val))
    validate(g)
    return g


def validate(g):
    """The structural rules of spec/graph.md section 4."""
    nv = len(g.values)
    for i, v in enumerate(g.values):
        if v.zero and (v.kind != VARIABLE or v.ref != 0 or v.dtype.is_block or any(d < 0 for d in v.shape)):
            raise _bad(f"value {i} ({v.name!r}): only a variable with a literal shape and a plain dtype can start as zeros")
    produced = [False] * nv
    refs = [0] * len(g.subgraphs)
    refs[0] = 1
    for c in g.composites:
        refs[c.subgraph] += 1
    for si, sg in enumerate(g.subgraphs):
        if si:
            for vid in sg.inputs:
                if produced[vid] or g.values[vid].kind != INTERMEDIATE:
                    raise _bad(f"subgraph {si} input {vid} is defined twice or is not an intermediate value")
                produced[vid] = True
        for oi, op in enumerate(sg.ops):
            for vid in op.outputs:
                if produced[vid] or g.values[vid].kind not in (INTERMEDIATE, OUTPUT):
                    raise _bad(f"subgraph {si} op {oi} output {vid} is defined twice or is not an op-produced value")
                produced[vid] = True
            for a in op.attrs.values():
                if a.type == "subgraph":
                    refs[a.value] += 1
    for r, n in enumerate(refs):
        if n != 1:
            raise _bad(f"subgraph {r} is referenced {n} times")
    for vid, v in enumerate(g.values):
        if v.kind in (INTERMEDIATE, OUTPUT):
            if not produced[vid]:
                raise _bad(f"value {vid} ({v.name!r}) has no definition")
        elif produced[vid]:
            raise _bad(f"value {vid} ({v.name!r}) of kind {v.kind} is also produced by an op")
    top = frozenset(vid for vid, v in enumerate(g.values) if v.kind in (INPUT, WEIGHT, VARIABLE, CONST))

    def walk(si, defined):
        sg = g.subgraphs[si]
        local = set(defined)
        local.update(sg.inputs)
        for oi, op in enumerate(sg.ops):
            for vid in op.inputs:
                if vid not in local:
                    raise _bad(f"subgraph {si} op {oi} uses value {vid} before its definition")
            for a in op.attrs.values():
                if a.type == "subgraph":
                    walk(a.value, local)
            local.update(op.outputs)
        for vid in sg.outputs:
            if vid not in local:
                raise _bad(f"subgraph {si} output {vid} is never defined")

    walk(0, top)
    for c in g.composites:
        walk(c.subgraph, top)
    for vid in g.main.inputs:
        if g.values[vid].kind != INPUT:
            raise _bad(f"main input {vid} is not of kind input")
    for vid in g.main.outputs:
        if g.values[vid].kind != OUTPUT:
            raise _bad(f"main output {vid} is not of kind output")
    seen = set()
    for u in g.updates:
        v, n = g.values[u.variable], g.values[u.value]
        if v.kind != VARIABLE or u.variable in seen:
            raise _bad(f"update target {u.variable} is not a variable or is updated twice")
        seen.add(u.variable)
        if v.dtype is not n.dtype or v.shape != n.shape:
            raise _bad(f"update of {v.name!r} changes dtype or shape")
