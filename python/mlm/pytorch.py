"""Exporting PyTorch models to mim, built on torch.export.

    import torch
    from torch.export import Dim
    from mlm.pytorch import export

    model.eval()
    export(model, (x,), "model.mim", dynamic_shapes={"x": {0: Dim("batch", max=64)}})

`torch.export.export` traces the model into a graph of ATen operators with
the parameters lifted out. Each operator becomes primitives of
spec/primitives.md or a call of a standard-library composite of
spec/stdlib.md (linear.v1, conv2d.v1, layer_norm.v1, attention.v1 and
others), so a runtime with a fast kernel for a composite uses it, and every
other runtime inlines the canonical body. Parameters become weights, stored
byte for byte under their names in the state dict; buffers and tensor
constants become constant values. A dimension marked dynamic in
`dynamic_shapes` becomes a symbolic dimension, bounded by its `max`.

Operations that depend only on buffers and constants, such as a rotary
table computed from a buffer of frequencies, are evaluated once at export
time and stored as constants. Parameters are never folded, so they keep
their names.

A buffer the model mutates, such as a static key-value cache, becomes a
graph variable and its final value the variable's update (ADR 0003);
index_copy_ and index assignment become scatter. A buffer that starts as
zeros is ZERO_INIT and takes no file space.

Anything the exporter cannot express is rejected with the operator's name;
nothing is approximated. See SUPPORTED for the operators it handles.
"""

import math
import operator
import sys

import torch

from . import container as C
from . import dtypes as D
from .writer import CONST, INPUT, OUTPUT, VARIABLE, WEIGHT, GraphBuilder, Writer, encode_metadata, write_file

__all__ = ["export", "convert", "ExportError", "SUPPORTED"]

EXPORTER_VERSION = "0.1"


class ExportError(Exception):
    """The model uses something the exporter cannot express."""


_DTYPES = {
    torch.float32: D.F32, torch.float16: D.F16, torch.bfloat16: D.BF16, torch.float64: D.F64,
    torch.int8: D.BY_NAME["i8"], torch.uint8: D.BY_NAME["u8"], torch.int16: D.BY_NAME["i16"],
    torch.int32: D.I32, torch.int64: D.I64, torch.bool: D.BOOL,
    torch.uint16: D.BY_NAME["u16"], torch.uint32: D.BY_NAME["u32"], torch.uint64: D.BY_NAME["u64"],
    torch.float8_e4m3fn: D.BY_NAME["f8_e4m3"], torch.float8_e5m2: D.BY_NAME["f8_e5m2"],
}
_TORCH = {v: k for k, v in _DTYPES.items()}


def _mim_dtype(td):
    dt = _DTYPES.get(td)
    if dt is None:
        raise ExportError(f"dtype {td} has no mim equivalent")
    return dt


def _tensor_bytes(t):
    if sys.byteorder != "little":
        raise ExportError("the exporter runs on little-endian hosts only")
    t = t.detach().cpu().contiguous()
    if t.numel() == 0:
        return b""
    return t.reshape(-1).view(torch.uint8).numpy().tobytes()


class _Const:
    """A tensor known at export time: a parameter, a buffer, a lifted
    constant or a folded result. It enters the graph only when an operation
    that depends on an input uses it."""

    __slots__ = ("tensor", "name", "param", "vid")

    def __init__(self, tensor, name, param):
        self.tensor, self.name, self.param, self.vid = tensor, name, param, None


def export(model, args, path=None, *, kwargs=None, dynamic_shapes=None, metadata=None, name=None,
           output_names=None, strict=False):
    """Exports model, called with args (and kwargs), with torch.export and
    converts the result. Writes the file to path when given and returns its
    bytes. The model must be in eval mode."""
    if isinstance(model, torch.nn.Module) and model.training:
        raise ExportError("the model is in training mode; call model.eval() before exporting")
    ep = torch.export.export(model, tuple(args), kwargs=kwargs, dynamic_shapes=dynamic_shapes, strict=strict)
    if name is None:
        name = type(model).__name__
    w = convert(ep, metadata=metadata, name=name, output_names=output_names,
                dim_names=_dim_names(ep, dynamic_shapes))
    data = w.to_bytes()
    if path is not None:
        write_file(path, data)
    return data


def convert(ep, *, metadata=None, name=None, output_names=None, dim_names=None):
    """Converts a torch.export.ExportedProgram into a Writer holding the
    weights, the graph and the metadata."""
    if _has_mutation(ep):
        # torch.export keeps in-place operators such as index_copy_ in its
        # graph; functionalizing turns them into functional ones and the
        # mutated buffers into outputs. The empty table decomposes nothing.
        ep = ep.run_decompositions({})
    return _Converter(ep, dim_names or {}).run(metadata or {}, name, output_names)


def _has_mutation(ep):
    for node in ep.graph.nodes:
        schema = getattr(node.target, "_schema", None) if node.op == "call_function" else None
        if schema is not None and schema.is_mutable:
            return True
    return False


def _dim_names(ep, dynamic_shapes):
    """Maps (input name, axis) to the name of the torch.export.Dim the user
    gave it, so symbolic dimensions keep their names."""
    out = {}
    if not dynamic_shapes:
        return out
    users = [s.arg.name for s in ep.graph_signature.input_specs if s.kind.name == "USER_INPUT"]
    items = dynamic_shapes.items() if isinstance(dynamic_shapes, dict) else zip(users, dynamic_shapes)
    for inp, spec in items:
        if isinstance(spec, dict):
            axes = spec.items()
        elif isinstance(spec, (list, tuple)):
            axes = enumerate(spec)
        else:
            continue
        for ax, d in axes:
            n = getattr(d, "__name__", None)
            if n:
                out[(inp, ax)] = n
    return out


def _bind(node):
    """The node's arguments by schema name, with defaults filled in."""
    schema = node.target._schema
    a = {}
    for i, arg in enumerate(schema.arguments):
        if i < len(node.args):
            a[arg.name] = node.args[i]
        elif arg.name in node.kwargs:
            a[arg.name] = node.kwargs[arg.name]
        elif arg.has_default_value():
            a[arg.name] = arg.default_value
        else:
            a[arg.name] = None
    return a


class _Converter:
    def __init__(self, ep, dim_names):
        self.ep = ep
        self.dim_names = dim_names
        self.w = Writer()
        self.g = GraphBuilder()
        self.env = {}  # node -> _Const, a list of results, or a size
        self.tv = {}  # node -> graph value of a tensor result
        self.sym = {}  # sympy symbol -> symbolic dimension name
        self.node = None

    # graph helpers -------------------------------------------------------

    def fail(self, msg):
        where = f" ({self.node.target})" if self.node is not None and self.node.op == "call_function" else ""
        raise ExportError(f"{self.node.name if self.node is not None else 'graph'}{where}: {msg}")

    def dim(self, d):
        if isinstance(d, int):
            return d
        if isinstance(d, torch.SymInt):
            expr = d.node.expr
            if expr.is_Integer:
                return int(expr)
            if expr.is_Symbol and expr in self.sym:
                return self.sym[expr]
            self.fail(f"dimension {expr} is an expression of symbolic sizes; mim dimensions are literal or a single symbol")
        self.fail(f"dimension {d!r}")

    def meta(self, node):
        return node.meta.get("val")

    def shape_of_meta(self, t):
        return [self.dim(d) for d in t.shape]

    def new(self, dt, shape, name=""):
        return self.g.value(name, dt, shape)

    def dt(self, v):
        return self.g.dtype(v)

    def shp(self, v):
        return list(self.g.shape(v))

    def prim(self, op, ins, dt, shape, attrs=()):
        y = self.new(dt, shape)
        self.g.op(op, ins, [y], attrs)
        return y

    def call(self, name, ins, dt, shape, attrs=()):
        y = self.new(dt, shape)
        self.g.op("call", ins, [y], [("name", "string", name), ("version", "int", 1), *attrs])
        return y

    def k(self, value, dt, rank_or_shape):
        """A const filled with value, of dtype dt, with every dimension 1
        (given a rank) or the given shape."""
        shape = [1] * rank_or_shape if isinstance(rank_or_shape, int) else list(rank_or_shape)
        if dt.is_float:
            attr = ("value", "float", float(value))
        else:
            if isinstance(value, float):
                if not value.is_integer():
                    self.fail(f"value {value} for an integer or bool tensor")
                value = int(value)
            if dt.kind == "bool":
                value = 1 if value else 0
            attr = ("value", "int", int(value))
        return self.prim("const", [], dt, shape, [attr])

    def cast(self, v, dt):
        if self.dt(v) is dt:
            return v
        return self.prim("cast", [v], dt, self.shp(v), [("to", "dtype", dt)])

    def reshape(self, v, shape):
        shape = list(shape)
        if self.shp(v) == shape:
            return v
        return self.prim("reshape", [v], self.dt(v), shape, [("shape", "ints", [d if isinstance(d, int) else -1 for d in shape])])

    def rank_to(self, v, r):
        s = self.shp(v)
        if len(s) > r:
            self.fail(f"rank {len(s)} cannot broadcast to rank {r}")
        return self.reshape(v, [1] * (r - len(s)) + s)

    def expand(self, v, shape):
        v = self.rank_to(v, len(shape))
        if self.shp(v) == list(shape):
            return v
        return self.prim("expand", [v], self.dt(v), shape)

    def permute(self, v, perm):
        s = self.shp(v)
        if list(perm) == list(range(len(s))):
            return v
        return self.prim("permute", [v], self.dt(v), [s[p] for p in perm], [("perm", "ints", list(perm))])

    def shrink(self, v, axis, begin, end):
        s = self.shp(v)
        if begin == 0 and (end == s[axis] or (end == -1 and isinstance(s[axis], str))):
            return v
        b, e = [0] * len(s), [-1] * len(s)
        b[axis], e[axis] = begin, end
        out = list(s)
        out[axis] = end - begin if end >= 0 else s[axis]
        return self.prim("shrink", [v], self.dt(v), out, [("begin", "ints", b), ("end", "ints", e)])

    def pad(self, v, pads, value=0.0):
        s = self.shp(v)
        if not any(pads):
            return v
        out = []
        for i, d in enumerate(s):
            if isinstance(d, str) and (pads[2 * i] or pads[2 * i + 1]):
                self.fail("padding a symbolic dimension")
            out.append(d + pads[2 * i] + pads[2 * i + 1] if isinstance(d, int) else d)
        return self.prim("pad", [v], self.dt(v), out, [("pads", "ints", list(pads)), ("value", "float", value)])

    def broadcast_shape(self, shapes):
        r = max(len(s) for s in shapes)
        out = []
        for i in range(r):
            ds = [s[i - r + len(s)] for s in shapes if i - r + len(s) >= 0]
            if 0 in ds:
                out.append(0)
                continue
            big = [d for d in ds if d != 1]
            if len(set(big)) > 1:
                self.fail(f"shapes {shapes} do not broadcast")
            out.append(big[0] if big else 1)
        return out

    def bin(self, op, a, b, dt=None):
        """An elementwise primitive over a and b with broadcasting, after
        casting both to dt (default: a's dtype)."""
        dt = dt or self.dt(a)
        a, b = self.cast(a, dt), self.cast(b, dt)
        shape = self.broadcast_shape([self.shp(a), self.shp(b)])
        a, b = self.rank_to(a, len(shape)), self.rank_to(b, len(shape))
        out_dt = D.BOOL if op in ("cmplt", "cmpeq") else dt
        return self.prim(op, [a, b], out_dt, shape)

    def where(self, c, a, b):
        shape = self.broadcast_shape([self.shp(c), self.shp(a), self.shp(b)])
        r = len(shape)
        return self.prim("where", [self.rank_to(c, r), self.rank_to(a, r), self.rank_to(b, r)], self.dt(a), shape)

    def minimum(self, a, b):
        dt = self.dt(a)
        if dt.kind == "unsigned":
            # min(a, b) = where(a < b, a, b) is exact; neg would wrap.
            return self.where(self.bin("cmplt", a, b), a, self.cast(b, dt))
        return self.prim("neg", [self.bin("max", self.un("neg", a), self.un("neg", self.cast(b, dt)))], dt,
                         self.broadcast_shape([self.shp(a), self.shp(b)]))

    def un(self, op, x):
        return self.prim(op, [x], self.dt(x), self.shp(x))

    def not_(self, b):
        return self.bin("xor", b, self.k(1, D.BOOL, len(self.shp(b))))

    def scalar_or(self, x, dt, rank):
        """A value for x, which is a tensor argument or a Python number."""
        if isinstance(x, torch.fx.Node):
            return self.cast(self.v(x), dt)
        return self.k(self.number(x), dt, rank)

    def number(self, x):
        if isinstance(x, (bool, int, float)):
            return x
        if isinstance(x, torch.SymInt):
            d = self.dim(x)
            if isinstance(d, int):
                return d
            self.fail("a value computed from a symbolic size; mim graphs cannot use sizes as data")
        if isinstance(x, torch.fx.Node):
            v = self.env.get(x)
            if isinstance(v, (int, float, bool)):
                return v
            if isinstance(v, torch.SymInt):
                return self.number(v)
        self.fail(f"expected a number, got {x!r}")

    # values ----------------------------------------------------------------

    def v(self, x):
        """The graph value of a tensor argument, materialising constants."""
        if not isinstance(x, torch.fx.Node):
            self.fail(f"expected a tensor, got {x!r}")
        vid = self.tv.get(x)
        if vid is not None:
            return vid
        e = self.env.get(x)
        if isinstance(e, _Const):
            return self.materialize(e)
        self.fail(f"argument {x.name} is not a tensor")

    def materialize(self, c):
        if c.vid is None:
            t = c.tensor
            dt = _mim_dtype(t.dtype)
            name = c.name
            if name in self.tensor_names:
                n = 1
                while f"{name}.{n}" in self.tensor_names:
                    n += 1
                name = f"{name}.{n}"
            self.tensor_names.add(name)
            idx = self.w.add_tensor(name, dt, list(t.shape), _tensor_bytes(t))
            c.vid = self.g.value(name, dt, list(t.shape), WEIGHT if c.param else CONST, idx)
            self.tensor_refs.append(c.vid)
        return c.vid

    def variable(self, name, t):
        """A graph variable for the mutated buffer t. A buffer of zeros, such
        as an empty key-value cache, is ZERO_INIT and takes no file space;
        any other starts from its stored value."""
        dt = _mim_dtype(t.dtype)
        shape = list(t.shape)
        self.tensor_names.add(name)
        if t.numel() and not torch.count_nonzero(t) and not (t.is_floating_point() and torch.signbit(t).any()):
            return self.g.value(name, dt, shape, VARIABLE, 0, zero=True)
        idx = self.w.add_tensor(name, dt, shape, _tensor_bytes(t))
        vid = self.g.value(name, dt, shape, VARIABLE, idx)
        self.tensor_refs.append(vid)
        return vid

    def const_of(self, x):
        """The tensor of an argument known at export time, or None."""
        if isinstance(x, torch.fx.Node):
            e = self.env.get(x)
            return e.tensor if isinstance(e, _Const) else None
        return None

    # driver ----------------------------------------------------------------

    def run(self, metadata, name, output_names):
        ep, g = self.ep, self.g
        self.tensor_names = set()
        self.tensor_refs = []
        specs = {s.arg.name: s for s in ep.graph_signature.input_specs}
        state = ep.state_dict
        consts = getattr(ep, "constants", {}) or {}
        nodes = list(ep.graph.nodes)
        out_specs = list(ep.graph_signature.output_specs)
        for s in out_specs:
            if s.kind.name not in ("USER_OUTPUT", "BUFFER_MUTATION"):
                what = "an input" if s.kind.name == "USER_INPUT_MUTATION" else s.kind.name.lower().replace("_", " ")
                raise ExportError(f"the program mutates {what} ({s.target}); only buffers may be mutated, "
                                  "and they become graph variables")
        mutated = {s.target for s in out_specs if s.kind.name == "BUFFER_MUTATION"}
        variables = {}  # buffer name -> value of kind variable

        # Symbolic dimensions, named after the user's Dim objects, in the
        # order they first appear in the inputs.
        for node in nodes:
            if node.op != "placeholder" or specs[node.name].kind.name != "USER_INPUT":
                continue
            t = self.meta(node)
            if not isinstance(t, torch.Tensor):
                raise ExportError(f"input {node.name} is not a tensor")
            for ax, d in enumerate(t.shape):
                if isinstance(d, torch.SymInt) and d.node.expr.is_Symbol and d.node.expr not in self.sym:
                    expr = d.node.expr
                    nm = self.dim_names.get((node.name, ax)) or f"{node.name}_dim{ax}"
                    while nm in self.sym.values():
                        nm += "_"
                    bound = -1
                    rc = ep.range_constraints.get(expr)
                    if rc is not None:
                        try:
                            bound = int(rc.upper)
                        except (TypeError, ValueError, OverflowError):
                            bound = -1
                        if bound >= 1 << 62:
                            bound = -1
                    self.sym[expr] = g.symdim(nm, bound)

        outputs = []
        for node in nodes:
            self.node = node
            if node.op == "placeholder":
                spec = specs[node.name]
                kind = spec.kind.name
                if kind == "USER_INPUT":
                    t = self.meta(node)
                    self.tv[node] = g.value(node.name, _mim_dtype(t.dtype), self.shape_of_meta(t), INPUT)
                elif kind in ("PARAMETER", "BUFFER", "CONSTANT_TENSOR"):
                    src = state if kind != "CONSTANT_TENSOR" else consts
                    t = src.get(spec.target)
                    if t is None:
                        t = consts.get(spec.target, state.get(spec.target))
                    if not isinstance(t, torch.Tensor):
                        self.fail(f"{kind.lower()} {spec.target} is not a tensor")
                    if kind == "BUFFER" and spec.target in mutated:
                        self.tv[node] = variables[spec.target] = self.variable(spec.target, t.detach())
                    else:
                        self.env[node] = _Const(t.detach(), spec.target, kind == "PARAMETER")
                else:
                    self.fail(f"input kind {kind} is not supported")
            elif node.op == "get_attr":
                t = getattr(ep.graph_module, node.target)
                if not isinstance(t, torch.Tensor):
                    self.fail("only tensor attributes are supported")
                self.env[node] = _Const(t.detach(), node.target, False)
            elif node.op == "call_function":
                self.call_function(node)
            elif node.op == "output":
                outputs = list(node.args[0]) if isinstance(node.args[0], (list, tuple)) else [node.args[0]]
            else:
                self.fail(f"node kind {node.op} is not supported")
        self.node = None

        if len(outputs) != len(out_specs):
            raise ExportError(f"{len(outputs)} graph outputs for {len(out_specs)} output specs")
        # A mutated buffer is a variable, and its value at the end of the
        # run is the variable's update (spec/primitives.md section 7).
        for o, spec in zip(outputs, out_specs):
            if spec.kind.name != "BUFFER_MUTATION":
                continue
            var = variables.get(spec.target)
            if var is None:
                raise ExportError(f"buffer {spec.target} is mutated but not read")
            new = self.v(o) if isinstance(o, torch.fx.Node) else None
            if new is None:
                raise ExportError(f"the new value of buffer {spec.target} is not a tensor")
            new = self.cast(new, self.dt(var))
            if self.shp(new) != self.shp(var):
                raise ExportError(f"buffer {spec.target} changes shape from {self.shp(var)} to {self.shp(new)}")
            if new != var:
                g.update(var, new)
        outputs = [o for o, spec in zip(outputs, out_specs) if spec.kind.name == "USER_OUTPUT"]
        if output_names is None:
            output_names = ["output"] if len(outputs) == 1 else [f"output_{i}" for i in range(len(outputs))]
        if len(output_names) != len(outputs):
            raise ExportError(f"{len(output_names)} output names for {len(outputs)} outputs")
        taken = {v[0] for v in g.values}
        for o, nm in zip(outputs, output_names):
            if nm in taken:
                raise ExportError(f"output name {nm!r} is already used by an input or a weight")
            taken.add(nm)
            src = self.v(o) if isinstance(o, torch.fx.Node) else None
            if src is None:
                raise ExportError(f"output {nm} is not a tensor")
            s = self.shp(src)
            y = g.value(nm, self.dt(src), s, OUTPUT)
            g.op("reshape", [src], [y], [("shape", "ints", [d if isinstance(d, int) else -1 for d in s])])
            g.output(y)

        # Tensor refs index the sorted tensor table.
        pos = self.w.tensor_indices()
        for vid in self.tensor_refs:
            row = list(g.values[vid])
            row[4] = pos[row[4]]
            g.values[vid] = tuple(row)

        md = {"export.source_format": "pytorch", "export.torch_version": str(torch.__version__),
              "export.exporter": f"mim.pytorch {EXPORTER_VERSION}"}
        if name:
            md["general.name"] = name
        md.update(metadata)
        self.w.add_section(C.GRAPH, g.encode(), required=True)
        self.w.add_section(C.METADATA, encode_metadata(md))
        return self.w

    def call_function(self, node):
        t = node.target
        if t is operator.getitem:
            seq, i = node.args
            e = self.env.get(seq)
            if not isinstance(e, (list, tuple)) or e[i] is None:
                self.fail(f"output {i} of {seq.name} is not supported")
            if isinstance(e[i], _Const):
                self.env[node] = e[i]
            else:
                self.tv[node] = e[i]
            return
        name = str(t)
        if not name.startswith("aten.") and t not in _SYM_FUNCS:
            self.fail(f"operator {name} is not supported")
        val = self.meta(node)
        # Shape arithmetic: sizes and products of sizes.
        if t in _SYM_FUNCS or isinstance(val, (int, float, bool, torch.SymInt, torch.SymFloat)):
            if name.startswith("aten._assert") or name.startswith("aten.sym_constrain"):
                return
            self.env[node] = val
            return
        if val is None:
            if name.startswith("aten._assert") or name.startswith("aten.sym_constrain"):
                return
            self.fail("operator without a result")
        key = name[len("aten."):]
        # Fold operations on buffers and constants.
        if self.foldable(node):
            self.fold(node)
            return
        fn = _LOWER.get(key)
        if fn is None:
            self.fail(f"operator {name} is not supported")
        before = len(self.g.values)
        r = fn(self, node, _bind(node))
        if isinstance(r, (list, tuple)):
            self.env[node] = list(r)
            return
        # Check the result against what torch.export recorded, and name it.
        if isinstance(val, torch.Tensor):
            want_dt, want_shape = _mim_dtype(val.dtype), self.shape_of_meta(val)
            if self.dt(r) is not want_dt or self.shp(r) != want_shape:
                self.fail(f"lowered to {self.dt(r).name}{self.shp(r)}, torch.export recorded {want_dt.name}{want_shape}")
        if r >= before and not self.g.values[r][0]:
            row = list(self.g.values[r])
            row[0] = node.name
            self.g.values[r] = tuple(row)
        self.tv[node] = r

    def foldable(self, node):
        found = False
        for a in node.all_input_nodes:
            e = self.env.get(a)
            if isinstance(e, _Const):
                if e.param:
                    return False
                found = True
            elif isinstance(e, (int, float, bool)) and not isinstance(e, torch.SymInt):
                continue
            else:
                return False
        if not found and not node.all_input_nodes:
            # A creation op (arange, ones, ...) with static arguments.
            val = self.meta(node)
            return isinstance(val, torch.Tensor) and all(isinstance(d, int) for d in val.shape)
        return found

    def fold(self, node):
        def sub(x):
            if isinstance(x, torch.fx.Node):
                e = self.env[x]
                return e.tensor if isinstance(e, _Const) else e
            if isinstance(x, (list, tuple)):
                return type(x)(sub(y) for y in x)
            return x

        with torch.no_grad():
            r = node.target(*sub(node.args), **{k: sub(v) for k, v in node.kwargs.items()})
        if isinstance(r, torch.Tensor):
            self.env[node] = _Const(r, f"const.{node.name}", False)
        elif isinstance(r, (list, tuple)):
            self.env[node] = [
                _Const(x, f"const.{node.name}.{i}", False) if isinstance(x, torch.Tensor) else None for i, x in enumerate(r)
            ]
        else:
            self.fail("folding gave no tensor")


_SYM_FUNCS = {operator.mul, operator.add, operator.sub, operator.floordiv, operator.mod, operator.truediv, operator.neg}

# Lowerings -------------------------------------------------------------------

_LOWER = {}


def _lower(*names):
    def reg(fn):
        for n in names:
            _LOWER[n] = fn
        return fn

    return reg


def _out(cv, node):
    val = cv.meta(node)
    return _mim_dtype(val.dtype), cv.shape_of_meta(val)


def _axis(ax, r):
    return ax + r if ax < 0 else ax


# identity and casts


@_lower("contiguous.default", "clone.default", "alias.default", "detach.default", "lift_fresh_copy.default",
        "_to_copy.default", "to.dtype", "to.dtype_layout", "to.device", "to.other", "type_as.default")
def _identity_or_cast(cv, node, a):
    dt, _ = _out(cv, node)
    return cv.cast(cv.v(node.args[0]), dt)


@_lower("dropout.default", "feature_dropout.default", "alpha_dropout.default")
def _dropout(cv, node, a):
    if a.get("train"):
        cv.fail("dropout in training mode; call model.eval() before exporting")
    return cv.v(a["input"])


# elementwise


def _unary(op, call=None):
    def fn(cv, node, a):
        dt, _ = _out(cv, node)
        x = cv.cast(cv.v(node.args[0]), dt)
        if call:
            return cv.call(call, [x], dt, cv.shp(x))
        return cv.un(op, x)

    return fn


for _n, _op, _call in [
    ("neg.default", "neg", None), ("reciprocal.default", "recip", None), ("sqrt.default", "sqrt", None),
    ("exp2.default", "exp2", None), ("log2.default", "log2", None), ("sin.default", "sin", None),
    ("trunc.default", "trunc", None), ("fix.default", "trunc", None),
    ("rsqrt.default", None, "rsqrt"), ("exp.default", None, "exp"), ("log.default", None, "log"),
    ("cos.default", None, "cos"), ("tanh.default", None, "tanh"), ("sigmoid.default", None, "sigmoid"),
    ("silu.default", None, "silu"), ("erf.default", None, "erf"),
]:
    _LOWER[_n] = _unary(_op, _call)


@_lower("gelu.default")
def _gelu(cv, node, a):
    x = cv.v(a["self"])
    approx = a.get("approximate") or "none"
    if approx not in ("none", "tanh"):
        cv.fail(f"gelu approximate={approx!r}")
    return cv.call("gelu_erf" if approx == "none" else "gelu_tanh", [x], cv.dt(x), cv.shp(x))


@_lower("relu.default")
def _relu(cv, node, a):
    x = cv.v(a["self"])
    return cv.bin("max", x, cv.k(0, cv.dt(x), len(cv.shp(x))))


@_lower("abs.default")
def _abs(cv, node, a):
    x = cv.v(a["self"])
    if cv.dt(x).kind in ("unsigned", "bool"):
        return x
    return cv.bin("max", x, cv.un("neg", x))


def _clamp(cv, x, lo, hi):
    dt = cv.dt(x)
    r = len(cv.shp(x))
    y = x
    if lo is not None:
        y = cv.bin("max", y, cv.scalar_or(lo, dt, r))
    if hi is not None:
        y = cv.minimum(y, cv.scalar_or(hi, dt, r))
    return y


@_lower("clamp.default", "clamp.Tensor", "clip.default")
def _clamp_l(cv, node, a):
    dt, _ = _out(cv, node)
    return _clamp(cv, cv.cast(cv.v(a["self"]), dt), a.get("min"), a.get("max"))


@_lower("clamp_min.default")
def _clamp_min(cv, node, a):
    return _clamp(cv, cv.v(a["self"]), a["min"], None)


@_lower("clamp_max.default")
def _clamp_max(cv, node, a):
    return _clamp(cv, cv.v(a["self"]), None, a["max"])


@_lower("hardtanh.default")
def _hardtanh(cv, node, a):
    return _clamp(cv, cv.v(a["self"]), a["min_val"], a["max_val"])


@_lower("relu6.default")
def _relu6(cv, node, a):
    return _clamp(cv, cv.v(a["self"]), 0, 6)


@_lower("leaky_relu.default")
def _leaky_relu(cv, node, a):
    x = cv.v(a["self"])
    dt, r = cv.dt(x), len(cv.shp(x))
    slope = cv.number(a["negative_slope"])
    neg = cv.bin("cmplt", x, cv.k(0, dt, r))
    return cv.where(neg, cv.bin("mul", x, cv.k(slope, dt, r)), x)


@_lower("hardsigmoid.default")
def _hardsigmoid(cv, node, a):
    x = cv.v(a["self"])
    dt, r = cv.dt(x), len(cv.shp(x))
    y = _clamp(cv, cv.bin("add", x, cv.k(3, dt, r)), 0, 6)
    return cv.bin("mul", y, cv.k(1 / 6, dt, r))


@_lower("hardswish.default")
def _hardswish(cv, node, a):
    x = cv.v(a["self"])
    return cv.bin("mul", x, _hardsigmoid(cv, node, a))


def _operands(cv, node, a, b, dt=None):
    """Both operands of a binary operation as values of the dtype torch
    computes in: the result dtype for arithmetic, the promoted operand
    dtype for comparisons."""
    if dt is None:
        dt, _ = _out(cv, node)

    def as_val(x, other):
        if isinstance(x, torch.fx.Node) and isinstance(cv.meta(x), torch.Tensor):
            return cv.cast(cv.v(x), dt)
        r = len(cv.shp(cv.v(other))) if isinstance(other, torch.fx.Node) else 0
        return cv.k(cv.number(x), dt, r)

    return as_val(a, b), as_val(b, a), dt


def _promoted(cv, a, b):
    def proto(x):
        if isinstance(x, torch.fx.Node):
            m = cv.meta(x)
            if isinstance(m, torch.Tensor):
                return torch.empty([1] * m.dim(), dtype=m.dtype)
            x = cv.number(x)
        return x

    return _mim_dtype(torch.result_type(proto(a), proto(b)))


def _arith(op, reverse=False, negate=False):
    def fn(cv, node, a):
        x, y = a["self"], a["other"]
        if reverse:
            x, y = y, x
        alpha = a.get("alpha")
        xv, yv, dt = _operands(cv, node, x, y)
        if alpha is not None and alpha != 1:
            yv = cv.bin("mul", yv, cv.k(cv.number(alpha), dt, len(cv.shp(yv))))
        if negate:
            if dt.kind == "bool":
                cv.fail("subtraction of bool tensors")
            yv = cv.un("neg", yv)
        return cv.bin(op, xv, yv)

    return fn


for _n in ("add.Tensor", "add.Scalar"):
    _LOWER[_n] = _arith("add")
for _n in ("sub.Tensor", "sub.Scalar"):
    _LOWER[_n] = _arith("add", negate=True)
for _n in ("rsub.Tensor", "rsub.Scalar"):
    _LOWER[_n] = _arith("add", reverse=True, negate=True)
for _n in ("mul.Tensor", "mul.Scalar"):
    _LOWER[_n] = _arith("mul")
for _n in ("maximum.default",):
    _LOWER[_n] = _arith("max")
for _n in ("pow.Tensor_Tensor",):
    _LOWER[_n] = lambda cv, node, a: cv.bin("pow", *_operands(cv, node, a["self"], a["exponent"])[:2])
for _n in ("bitwise_and.Tensor", "logical_and.default", "__and__.Tensor"):
    _LOWER[_n] = lambda cv, node, a: _logic(cv, node, a, "and")
for _n in ("bitwise_or.Tensor", "logical_or.default", "__or__.Tensor"):
    _LOWER[_n] = lambda cv, node, a: _logic(cv, node, a, "or")
for _n in ("bitwise_xor.Tensor", "logical_xor.default", "__xor__.Tensor"):
    _LOWER[_n] = lambda cv, node, a: _logic(cv, node, a, "xor")


def _logic(cv, node, a, op):
    dt, _ = _out(cv, node)
    if dt.kind == "bool":
        x = cv.v(a["self"])
        y = a["other"]
        xb = x if cv.dt(x) is D.BOOL else cv.not_(cv.bin("cmpeq", x, cv.k(0, cv.dt(x), len(cv.shp(x)))))
        yv = cv.v(y)
        yb = yv if cv.dt(yv) is D.BOOL else cv.not_(cv.bin("cmpeq", yv, cv.k(0, cv.dt(yv), len(cv.shp(yv)))))
        return cv.bin(op, xb, yb)
    xv, yv, _ = _operands(cv, node, a["self"], a["other"])
    return cv.bin(op, xv, yv)


@_lower("minimum.default")
def _minimum(cv, node, a):
    xv, yv, _ = _operands(cv, node, a["self"], a["other"])
    return cv.minimum(xv, yv)


@_lower("div.Tensor", "div.Scalar", "div.Tensor_mode", "div.Scalar_mode")
def _div(cv, node, a):
    mode = a.get("rounding_mode")
    dt, _ = _out(cv, node)
    xv, yv, _ = _operands(cv, node, a["self"], a["other"])
    if dt.is_float:
        # Division is mul(a, recip(b)), the div macro of spec/stdlib.md.
        q = cv.bin("mul", xv, cv.un("recip", yv))
        if mode is None:
            return q
        t = cv.un("trunc", q)
        if mode == "trunc":
            return t
        return cv.where(cv.bin("cmplt", q, t), cv.bin("add", t, cv.k(-1, dt, len(cv.shp(t)))), t)
    if mode is None:
        cv.fail("true division of integers")
    q = cv.bin("idiv", xv, yv)
    if mode == "trunc":
        return q
    r = len(cv.shp(q))
    rem = cv.bin("imod", xv, yv)
    z = cv.k(0, dt, r)
    adjust = cv.bin("and", cv.not_(cv.bin("cmpeq", rem, z)),
                    cv.bin("xor", cv.bin("cmplt", rem, z), cv.bin("cmplt", yv, z)))
    return cv.where(adjust, cv.bin("add", q, cv.k(-1, dt, r)), q)


@_lower("pow.Tensor_Scalar")
def _pow(cv, node, a):
    dt, _ = _out(cv, node)
    x = cv.cast(cv.v(a["self"]), dt)
    e = cv.number(a["exponent"])
    if e == 1:
        return x
    if e == 2:
        return cv.bin("mul", x, x)
    if e == 0.5 and dt.is_float:
        return cv.un("sqrt", x)
    if e == -1 and dt.is_float:
        return cv.un("recip", x)
    if e == -0.5 and dt.is_float:
        return cv.call("rsqrt", [x], dt, cv.shp(x))
    if not dt.is_float:
        cv.fail("integer pow")
    return cv.bin("pow", x, cv.k(e, dt, len(cv.shp(x))))


def _compare(kind):
    def fn(cv, node, a):
        x, y = a["self"], a["other"]
        dt = _promoted(cv, x, y)
        xv, yv, _ = _operands(cv, node, x, y, dt)
        if kind == "eq":
            return cv.bin("cmpeq", xv, yv)
        if kind == "ne":
            return cv.not_(cv.bin("cmpeq", xv, yv))
        if kind == "lt":
            return cv.bin("cmplt", xv, yv)
        if kind == "gt":
            return cv.bin("cmplt", yv, xv)
        if kind == "le":
            return cv.bin("or", cv.bin("cmplt", xv, yv), cv.bin("cmpeq", xv, yv))
        return cv.bin("or", cv.bin("cmplt", yv, xv), cv.bin("cmpeq", xv, yv))

    return fn


for _k in ("eq", "ne", "lt", "gt", "le", "ge"):
    _LOWER[f"{_k}.Tensor"] = _LOWER[f"{_k}.Scalar"] = _compare(_k)


@_lower("logical_not.default", "bitwise_not.default", "__invert__.default")
def _not(cv, node, a):
    x = cv.v(a["self"])
    dt = cv.dt(x)
    if dt.kind == "bool":
        return cv.not_(x)
    if str(node.target).startswith("aten.logical"):
        return cv.bin("cmpeq", x, cv.k(0, dt, len(cv.shp(x))))
    allbits = (1 << dt.bits) - 1 if dt.kind == "unsigned" else -1
    return cv.bin("xor", x, cv.k(allbits, dt, len(cv.shp(x))))


@_lower("where.self", "where.ScalarSelf", "where.ScalarOther", "where.Scalar")
def _where(cv, node, a):
    dt, _ = _out(cv, node)
    c = cv.v(a["condition"])
    xv, yv, _ = _operands(cv, node, a["self"], a["other"], dt)
    return cv.where(c, xv, yv)


@_lower("masked_fill.Scalar", "masked_fill.Tensor")
def _masked_fill(cv, node, a):
    x = cv.v(a["self"])
    dt, r = cv.dt(x), len(cv.shp(x))
    m = cv.v(a["mask"])
    val = a["value"]
    fill = cv.reshape(cv.cast(cv.v(val), dt), [1] * r) if isinstance(val, torch.fx.Node) else cv.k(cv.number(val), dt, r)
    return cv.where(m, fill, x)


# creation with symbolic sizes (static ones are folded)


@_lower("full.default", "zeros.default", "ones.default", "full_like.default", "zeros_like.default",
        "ones_like.default", "new_zeros.default", "new_ones.default", "new_full.default", "scalar_tensor.default")
def _full(cv, node, a):
    dt, shape = _out(cv, node)
    name = str(node.target)
    if "zeros" in name:
        value = 0
    elif "ones" in name:
        value = 1
    else:
        value = cv.number(a.get("fill_value", a.get("s")))
    return cv.k(value, dt, shape)


# movement


@_lower("view.default", "reshape.default", "_unsafe_view.default", "view_copy.default", "flatten.using_ints",
        "unflatten.int", "squeeze.dim", "squeeze.dims", "squeeze.default", "unsqueeze.default")
def _reshape(cv, node, a):
    _, shape = _out(cv, node)
    return cv.reshape(cv.v(node.args[0]), shape)


@_lower("permute.default")
def _permute(cv, node, a):
    x = cv.v(a["self"])
    r = len(cv.shp(x))
    return cv.permute(x, [_axis(d, r) for d in a["dims"]])


@_lower("transpose.int")
def _transpose(cv, node, a):
    x = cv.v(a["self"])
    r = len(cv.shp(x))
    p = list(range(r))
    i, j = _axis(a["dim0"], r), _axis(a["dim1"], r)
    p[i], p[j] = p[j], p[i]
    return cv.permute(x, p)


@_lower("t.default", "numpy_T.default")
def _t(cv, node, a):
    x = cv.v(a["self"])
    r = len(cv.shp(x))
    return cv.permute(x, list(range(r))[::-1])


@_lower("expand.default", "expand_as.default", "expand_copy.default")
def _expand(cv, node, a):
    _, shape = _out(cv, node)
    return cv.expand(cv.v(a["self"]), shape)


def _static_index(cv, i, size, what):
    i = cv.number(i)
    if i < 0:
        if not isinstance(size, int):
            cv.fail(f"negative {what} on a symbolic dimension")
        i += size
    return i


def _take_strided(cv, x, ax, start, n, step):
    s = cv.shp(x)
    need = start + n * step
    if s[ax] < need:
        p = [0] * (2 * len(s))
        p[2 * ax + 1] = need - s[ax]
        x = cv.pad(x, p)
    x = cv.shrink(x, ax, start, need)
    s = cv.shp(x)
    x = cv.reshape(x, s[:ax] + [n, step] + s[ax + 1:])
    x = cv.shrink(x, ax + 1, 0, 1)
    return cv.reshape(x, s[:ax] + [n] + s[ax + 1:])


@_lower("slice.Tensor", "slice_copy.Tensor")
def _slice(cv, node, a):
    x = cv.v(a["self"])
    s = cv.shp(x)
    ax = _axis(a["dim"], len(s))
    size = s[ax]
    _, out_shape = _out(cv, node)
    step = cv.number(a["step"] or 1)
    start = a["start"]
    end = a["end"]
    if isinstance(size, str):
        # Only a slice that keeps the whole symbolic axis is expressible.
        if out_shape[ax] == size and (start in (None, 0)) and step == 1:
            return x
        cv.fail("slicing a symbolic dimension")
    if isinstance(out_shape[ax], str):
        # x[start:start + T] of a literal axis with a symbolic length T,
        # such as a rotary table or a causal mask cut to the sequence
        # length: a gather of start + iota(T).
        if step != 1:
            cv.fail("a strided slice of symbolic length")
        start = 0 if start is None else _static_index(cv, start, size, "start")
        idx = _iota(cv, out_shape[ax], start)
        out = list(s)
        out[ax] = out_shape[ax]
        return cv.prim("gather", [x, idx], cv.dt(x), out, [("axis", "int", ax)])
    start = 0 if start is None else _static_index(cv, start, size, "start")
    end = size if end is None else cv.number(end)
    if end < 0:
        end += size
    start, end = max(0, min(start, size)), max(0, min(end, size))
    end = max(start, end)
    if step == 1:
        return cv.shrink(x, ax, start, end)
    n = out_shape[ax]
    return _take_strided(cv, x, ax, start, n, step)


def _iota(cv, n, start=0, dt=D.I64):
    """start, start + 1, ..., start + n - 1 as a 1-D tensor, for a literal
    or symbolic n: the iota macro of spec/stdlib.md section 2.2."""
    ones = cv.k(1, D.I64, [n])
    y = cv.call("cumsum", [ones], D.I64, [n], [("axis", "int", 0), ("exclusive", "int", 1)])
    if start:
        y = cv.bin("add", y, cv.k(start, D.I64, 1))
    return cv.cast(y, dt)


@_lower("arange.default", "arange.start", "arange.start_step")
def _arange(cv, node, a):
    # Static ranges are folded; this is a range with a symbolic length.
    dt, shape = _out(cv, node)
    start = cv.number(a.get("start") or 0) if "start" in a else 0
    step = cv.number(a.get("step") or 1) if "step" in a else 1
    if step != 1 or not float(start).is_integer():
        cv.fail("a range of symbolic length with a step or fractional start")
    return _iota(cv, shape[0], int(start), dt)


@_lower("narrow.default")
def _narrow(cv, node, a):
    x = cv.v(a["self"])
    s = cv.shp(x)
    ax = _axis(a["dim"], len(s))
    start = _static_index(cv, a["start"], s[ax], "start")
    return cv.shrink(x, ax, start, start + cv.number(a["length"]))


@_lower("select.int")
def _select(cv, node, a):
    x = cv.v(a["self"])
    s = cv.shp(x)
    ax = _axis(a["dim"], len(s))
    i = _static_index(cv, a["index"], s[ax], "index")
    y = cv.shrink(x, ax, i, i + 1)
    return cv.reshape(y, s[:ax] + s[ax + 1:])


@_lower("split.Tensor", "split_with_sizes.default", "chunk.default", "unbind.int", "split.sizes",
        "tensor_split.sections")
def _split(cv, node, a):
    x = cv.v(a["self"])
    s = cv.shp(x)
    ax = _axis(a.get("dim", 0) or 0, len(s))
    if not isinstance(s[ax], int):
        cv.fail("splitting a symbolic dimension")
    outs, off = [], 0
    unbind = str(node.target) == "aten.unbind.int"
    for m in cv.meta(node):
        n = 1 if unbind else cv.dim(m.shape[ax])
        y = cv.shrink(x, ax, off, off + n)
        if unbind:
            y = cv.reshape(y, cv.shape_of_meta(m))
        outs.append(y)
        off += n
    return outs


def _concat(cv, vals, ax, out_shape):
    vals = [v for v in vals if cv.shp(v)[ax] != 0]
    if not vals:
        cv.fail("concatenation of empty tensors")
    if len(vals) == 1:
        return vals[0]
    for v in vals:
        if not isinstance(cv.shp(v)[ax], int):
            cv.fail("concatenation along a symbolic dimension")
    total = sum(cv.shp(v)[ax] for v in vals)
    dt = cv.dt(vals[0])
    if dt.is_block or dt.kind == "storage":
        cv.fail(f"concatenation of {dt.name}")
    op = "or" if dt.kind == "bool" else "add"
    acc, off = None, 0
    for v in vals:
        n = cv.shp(v)[ax]
        p = [0] * (2 * len(cv.shp(v)))
        p[2 * ax], p[2 * ax + 1] = off, total - off - n
        pv = cv.pad(v, p)
        acc = pv if acc is None else cv.bin(op, acc, pv)
        off += n
    return acc


@_lower("cat.default", "concat.default")
def _cat(cv, node, a):
    dt, shape = _out(cv, node)
    vals = [cv.cast(cv.v(t), dt) for t in a["tensors"]]
    vals = [v for v in vals if cv.shp(v) != [0]]  # torch accepts 1-D empty tensors of any rank
    ax = _axis(a["dim"], len(shape))
    return _concat(cv, vals, ax, shape)


@_lower("stack.default")
def _stack(cv, node, a):
    dt, shape = _out(cv, node)
    ax = _axis(a["dim"], len(shape))
    vals = []
    for t in a["tensors"]:
        v = cv.cast(cv.v(t), dt)
        s = cv.shp(v)
        vals.append(cv.reshape(v, s[:ax] + [1] + s[ax:]))
    return _concat(cv, vals, ax, shape)


@_lower("index_select.default")
def _index_select(cv, node, a):
    dt, shape = _out(cv, node)
    x = cv.v(a["self"])
    ax = _axis(a["dim"], len(cv.shp(x)))
    return cv.prim("gather", [x, cv.v(a["index"])], dt, shape, [("axis", "int", ax)])


def _scatter(cv, node, x, ax, idx, src):
    """A copy of x with the slices at idx along axis ax replaced by src,
    broadcast to their shape: the scatter primitive. Indices must be in
    range; unlike PyTorch, a negative index does not count from the end."""
    dt, shape = _out(cv, node)
    if cv.dt(idx).name not in ("i32", "i64"):
        cv.fail(f"indices of {cv.dt(idx).name}; only integer indices are supported")
    s = cv.shp(x)
    want = s[:ax] + cv.shp(idx) + s[ax + 1:]
    src = cv.expand(cv.cast(src, dt), want)
    return cv.prim("scatter", [cv.cast(x, dt), idx, src], dt, shape, [("axis", "int", ax)])


@_lower("index_copy.default")
def _index_copy(cv, node, a):
    x = cv.v(a["self"])
    ax = _axis(a["dim"], len(cv.shp(x)))
    idx = cv.v(a["index"])
    if len(cv.shp(idx)) != 1:
        cv.fail("index_copy with an index that is not 1-D")
    return _scatter(cv, node, x, ax, idx, cv.v(a["source"]))


@_lower("index_put.default")
def _index_put(cv, node, a):
    # x[..., idx, ...] = values with one index tensor, as in writing new
    # keys at their positions in a cache.
    if a.get("accumulate"):
        cv.fail("index_put with accumulate=True")
    x = cv.v(a["self"])
    found = [(i, t) for i, t in enumerate(a["indices"]) if t is not None]
    if len(found) != 1:
        cv.fail(f"index_put with {len(found)} index tensors; only one is supported")
    ax, t = found[0]
    idx = cv.v(t)
    if cv.dt(idx) is D.BOOL:
        cv.fail("index_put with a boolean mask")
    val = a["values"]
    src = cv.v(val) if isinstance(val, torch.fx.Node) else None
    if src is None:
        cv.fail("index_put of a value that is not a tensor")
    return _scatter(cv, node, x, ax, idx, src)


@_lower("embedding.default")
def _embedding(cv, node, a):
    dt, shape = _out(cv, node)
    return cv.call("embedding", [cv.v(a["weight"]), cv.v(a["indices"])], dt, shape)


# linear algebra


@_lower("linear.default")
def _linear(cv, node, a):
    dt, shape = _out(cv, node)
    x, w = cv.v(a["input"]), cv.v(a["weight"])
    ins = [x, w]
    if a.get("bias") is not None:
        ins.append(cv.v(a["bias"]))
    xs = cv.shp(x)
    if len(xs) == 1:
        ins[0] = cv.reshape(x, [1] + xs)
        y = cv.call("linear", ins, dt, [1] + shape)
        return cv.reshape(y, shape)
    return cv.call("linear", ins, dt, shape)


def _matmul(cv, x, y, dt, shape):
    xs, ys = cv.shp(x), cv.shp(y)
    if len(xs) == 1:
        x, xs = cv.reshape(x, [1] + xs), [1] + xs
    if len(ys) == 1:
        y, ys = cv.reshape(y, ys + [1]), ys + [1]
    r = max(len(xs), len(ys))
    x, y = cv.rank_to(x, r), cv.rank_to(y, r)
    xs, ys = cv.shp(x), cv.shp(y)
    batch = cv.broadcast_shape([xs[:-2], ys[:-2]])
    z = cv.call("matmul", [x, y], dt, batch + [xs[-2], ys[-1]])
    return cv.reshape(z, shape)


@_lower("matmul.default", "mm.default", "bmm.default")
def _matmul_l(cv, node, a):
    dt, shape = _out(cv, node)
    return _matmul(cv, cv.v(a["self"]), cv.v(a["mat2"] if "mat2" in a else a["other"]), dt, shape)


@_lower("addmm.default")
def _addmm(cv, node, a):
    dt, shape = _out(cv, node)
    z = _matmul(cv, cv.v(a["mat1"]), cv.v(a["mat2"]), dt, shape)
    alpha, beta = cv.number(a.get("alpha", 1)), cv.number(a.get("beta", 1))
    if alpha != 1:
        z = cv.bin("mul", z, cv.k(alpha, dt, 2))
    if beta == 0:
        return z
    b = cv.v(a["self"])
    if beta != 1:
        b = cv.bin("mul", b, cv.k(beta, dt, len(cv.shp(b))))
    return cv.bin("add", z, b)


def _conv_padding(cv, padding, ks, dil, nd):
    if isinstance(padding, str):
        if padding == "valid":
            return [0] * (2 * nd)
        if padding == "same":
            out = []
            for k, d in zip(ks, dil):
                total = d * (k - 1)
                out += [total // 2, total - total // 2]
            return out
        cv.fail(f"padding {padding!r}")
    padding = list(padding) if isinstance(padding, (list, tuple)) else [padding] * nd
    if len(padding) == 1:
        padding = padding * nd
    out = []
    for p in padding:
        out += [cv.number(p), cv.number(p)]
    return out


def _pair(v, n):
    v = list(v) if isinstance(v, (list, tuple)) else [v]
    return v * n if len(v) == 1 else v


@_lower("conv2d.default", "conv2d.padding", "conv1d.default", "conv1d.padding", "convolution.default")
def _conv(cv, node, a):
    dt, shape = _out(cv, node)
    x, w = cv.v(a["input"]), cv.v(a["weight"])
    if a.get("transposed"):
        cv.fail("transposed convolution")
    ws = cv.shp(w)
    nd = len(ws) - 2
    if nd not in (1, 2):
        cv.fail(f"{nd}-D convolution")
    stride, dil = _pair(a.get("stride", 1), nd), _pair(a.get("dilation", 1), nd)
    pads = _conv_padding(cv, a.get("padding", 0), ws[2:], dil, nd)
    groups = cv.number(a.get("groups", 1))
    if nd == 1:
        # A 1-D convolution is a 2-D one over a height of 1.
        xs = cv.shp(x)
        x = cv.reshape(x, xs[:2] + [1] + xs[2:])
        w = cv.reshape(w, ws[:2] + [1] + ws[2:])
        stride, dil, pads = [1] + stride, [1] + dil, [0, 0] + pads
    ins = [x, w]
    if a.get("bias") is not None:
        ins.append(cv.v(a["bias"]))
    out4 = shape if nd == 2 else shape[:2] + [1] + shape[2:]
    y = cv.call("conv2d", ins, dt, out4, [
        ("strides", "ints", stride), ("dilations", "ints", dil),
        ("pads", "ints", [pads[0], pads[1], pads[2], pads[3]]), ("groups", "int", groups)])
    return cv.reshape(y, shape)


# normalisation


@_lower("batch_norm.default", "_native_batch_norm_legit_no_training.default", "native_batch_norm.default",
        "_native_batch_norm_legit.default", "cudnn_batch_norm.default")
def _batch_norm(cv, node, a):
    if a.get("training"):
        cv.fail("batch norm in training mode; call model.eval() before exporting")
    x = cv.v(a["input"])
    xs = cv.shp(x)
    dt = cv.dt(x)
    ch = xs[1]
    if not isinstance(ch, int):
        cv.fail("batch norm over a symbolic channel axis")
    p = []
    for k, default in (("weight", 1.0), ("bias", 0.0), ("running_mean", None), ("running_var", None)):
        t = cv.const_of(a.get(k)) if a.get(k) is not None else None
        if t is None:
            if a.get(k) is not None or default is None:
                cv.fail(f"batch norm {k} must be a parameter or buffer")
            t = torch.full([ch], default, dtype=torch.float64)
        p.append(t.detach().double())
    eps = cv.number(a["eps"])
    scale = p[0] / torch.sqrt(p[3] + eps)
    shift = p[1] - p[2] * scale
    dims = [1] * len(xs)
    dims[1] = ch
    td = _TORCH[dt]
    base = node.name
    rm = a.get("running_mean")
    if isinstance(rm, torch.fx.Node) and isinstance(cv.env.get(rm), _Const):
        base = cv.env[rm].name.rpartition(".")[0] or base
    sv = cv.materialize(_Const(scale.to(td).reshape(dims), f"{base}.folded_scale", False))
    bv = cv.materialize(_Const(shift.to(td).reshape(dims), f"{base}.folded_shift", False))
    y = cv.bin("add", cv.bin("mul", x, sv), bv)
    val = cv.meta(node)
    if isinstance(val, (list, tuple)):
        return [y] + [None] * (len(val) - 1)
    return y


@_lower("layer_norm.default", "native_layer_norm.default")
def _layer_norm(cv, node, a):
    x = cv.v(a["input"])
    xs, dt = cv.shp(x), cv.dt(x)
    ns = list(a["normalized_shape"])
    axis = len(xs) - len(ns)
    w = a.get("weight")
    scale = cv.v(w) if w is not None else cv.k(1, dt, [cv.number(n) for n in ns])
    ins = [x, scale]
    if a.get("bias") is not None:
        ins.append(cv.v(a["bias"]))
    eps = a.get("eps")
    y = cv.call("layer_norm", ins, dt, xs, [("axis", "int", axis), ("epsilon", "float", cv.number(eps))])
    if str(node.target) == "aten.native_layer_norm.default":
        return [y, None, None]
    return y


@_lower("rms_norm.default")
def _rms_norm(cv, node, a):
    x = cv.v(a["input"])
    xs, dt = cv.shp(x), cv.dt(x)
    axis = len(xs) - len(a["normalized_shape"])
    eps = a.get("eps")
    if eps is None:
        eps = torch.finfo(_TORCH[dt]).eps
    ins = [x]
    if a.get("weight") is not None:
        ins.append(cv.v(a["weight"]))
    return cv.call("rms_norm", ins, dt, xs, [("axis", "int", axis), ("epsilon", "float", cv.number(eps))])


def _with_dtype(cv, x, a):
    d = a.get("dtype")
    return cv.cast(x, _mim_dtype(d)) if d is not None else x


@_lower("softmax.int", "_softmax.default", "log_softmax.int", "_log_softmax.default")
def _softmax(cv, node, a):
    dt, shape = _out(cv, node)
    x = cv.cast(_with_dtype(cv, cv.v(a["self"]), a), dt)
    name = "log_softmax" if "log_softmax" in str(node.target) else "softmax"
    return cv.call(name, [x], dt, shape, [("axis", "int", _axis(a["dim"], len(shape)))])


def _axes(cv, dims, r):
    if dims is None or (isinstance(dims, (list, tuple)) and len(dims) == 0):
        return list(range(r))
    if isinstance(dims, int):
        dims = [dims]
    return sorted({_axis(d, r) for d in dims})


def _reduced(cv, y, keepdim, shape):
    if keepdim:
        return y
    return cv.reshape(y, shape)


@_lower("mean.dim", "mean.default")
def _mean(cv, node, a):
    dt, shape = _out(cv, node)
    x = cv.cast(cv.v(a["self"]), dt)
    xs = cv.shp(x)
    axes = _axes(cv, a.get("dim"), len(xs))
    kept = [1 if i in axes else d for i, d in enumerate(xs)]
    y = cv.call("mean", [x], dt, kept, [("axes", "ints", axes)])
    return _reduced(cv, y, a.get("keepdim", False), shape)


def _reduce(mode):
    def fn(cv, node, a):
        dt, shape = _out(cv, node)
        x = cv.cast(cv.v(a["self"]), dt)
        xs = cv.shp(x)
        axes = _axes(cv, a.get("dim"), len(xs))
        kept = [1 if i in axes else d for i, d in enumerate(xs)]
        y = cv.prim("reduce", [x], dt, kept, [("mode", "string", mode), ("axes", "ints", axes)])
        return _reduced(cv, y, a.get("keepdim", False), shape)

    return fn


_LOWER["sum.dim_IntList"] = _LOWER["sum.default"] = _reduce("add")
_LOWER["prod.dim_int"] = _LOWER["prod.default"] = _reduce("mul")
_LOWER["amax.default"] = _reduce("max")


@_lower("amin.default")
def _amin(cv, node, a):
    dt, shape = _out(cv, node)
    x = cv.v(a["self"])
    if dt.kind == "unsigned" or dt.kind == "bool":
        cv.fail(f"amin of {dt.name}")
    xs = cv.shp(x)
    axes = _axes(cv, a.get("dim"), len(xs))
    kept = [1 if i in axes else d for i, d in enumerate(xs)]
    y = cv.prim("reduce", [cv.un("neg", x)], dt, kept, [("mode", "string", "max"), ("axes", "ints", axes)])
    return _reduced(cv, cv.un("neg", y), a.get("keepdim", False), shape)


@_lower("var.correction", "var.dim", "std.correction", "std.dim")
def _var(cv, node, a):
    dt, shape = _out(cv, node)
    x = cv.v(a["self"])
    xs = cv.shp(x)
    axes = _axes(cv, a.get("dim"), len(xs))
    corr = a.get("correction")
    if corr is None:
        corr = 1 if a.get("unbiased", True) else 0
    kept = [1 if i in axes else d for i, d in enumerate(xs)]
    y = cv.call("variance", [x], dt, kept, [("axes", "ints", axes), ("correction", "int", int(cv.number(corr)))])
    if str(node.target).startswith("aten.std"):
        y = cv.un("sqrt", y)
    return _reduced(cv, y, a.get("keepdim", False), shape)


@_lower("cumsum.default")
def _cumsum(cv, node, a):
    dt, shape = _out(cv, node)
    x = cv.cast(cv.v(a["self"]), dt)
    return cv.call("cumsum", [x], dt, shape, [("axis", "int", _axis(a["dim"], len(shape)))])


# attention


@_lower("scaled_dot_product_attention.default")
def _sdpa(cv, node, a):
    dt, shape = _out(cv, node)
    q, k, v = cv.v(a["query"]), cv.v(a["key"]), cv.v(a["value"])
    qs, ks = cv.shp(q), cv.shp(k)
    if len(qs) != 4:
        cv.fail(f"attention on rank-{len(qs)} tensors; mim's attention.v1 takes [B, H, T, D]")
    if a.get("dropout_p"):
        cv.fail("attention dropout")
    attrs = []
    if a.get("scale") is not None:
        attrs.append(("scale", "float", cv.number(a["scale"])))
    if qs[1] != ks[1]:
        if not a.get("enable_gqa"):
            cv.fail("query and key head counts differ without enable_gqa")
        attrs.append(("num_kv_heads", "int", ks[1]))
    if a.get("is_causal"):
        # torch aligns the causal mask to the top left, mim to the end of
        # the keys; they agree when the lengths are equal.
        if qs[2] != ks[2]:
            cv.fail("causal attention with different query and key lengths")
        attrs.append(("causal", "int", 1))
    ins = [q, k, v]
    if a.get("attn_mask") is not None:
        m = cv.v(a["attn_mask"])
        if cv.dt(m) is not D.BOOL:
            m = cv.cast(m, dt)
        ins.append(cv.rank_to(m, 4))
    return cv.call("attention", ins, dt, shape, attrs)


# pooling


def _pool(cv, x, kernel, stride, pads, dil, is_max, include_pad, shape):
    xs = cv.shp(x)
    if len(xs) != 4:
        cv.fail(f"pooling over rank {len(xs)}; only 2-D pooling is supported")
    h, w = xs[2], xs[3]
    if not isinstance(h, int) or not isinstance(w, int):
        cv.fail("pooling needs known spatial sizes")
    (kh, kw), (sh, sw), (dh, dw) = kernel, stride, dil
    top, bottom, left, right = pads
    ho, wo = shape[2], shape[3]
    dt = cv.dt(x)
    # As in conv2d.v1: pad with s - 1 extra elements at the end so every
    # strided take stays in bounds, then combine one view per kernel tap.
    need_h, need_w = (ho - 1) * sh + dh * (kh - 1) + 1, (wo - 1) * sw + dw * (kw - 1) + 1
    extra_b = max(0, need_h - (h + top + bottom))
    extra_r = max(0, need_w - (w + left + right))
    if extra_b or extra_r:
        cv.fail("ceil_mode with a partial last window is not supported")
    padv = -math.inf if is_max else 0.0
    xp = cv.pad(x, [0, 0, 0, 0, top, bottom + sh - 1, left, right + sw - 1], padv)
    acc = None
    for i in range(kh):
        for j in range(kw):
            t = _take_strided(cv, xp, 2, i * dh, ho, sh)
            t = _take_strided(cv, t, 3, j * dw, wo, sw)
            acc = t if acc is None else cv.bin("max" if is_max else "add", acc, t)
    if is_max:
        return acc
    if include_pad or not any(pads):
        return cv.bin("mul", acc, cv.k(1.0 / (kh * kw), dt, 4))
    inv = torch.zeros([1, 1, ho, wo], dtype=torch.float64)
    for oy in range(ho):
        for ox in range(wo):
            cnt = 0
            for i in range(kh):
                for j in range(kw):
                    yy, xx = oy * sh + i - top, ox * sw + j - left
                    if 0 <= yy < h and 0 <= xx < w:
                        cnt += 1
            inv[0, 0, oy, ox] = 1.0 / cnt
    iv = cv.materialize(_Const(inv.to(_TORCH[dt]), f"{cv.node.name}.inv_count", False))
    return cv.bin("mul", acc, iv)


@_lower("max_pool2d.default", "max_pool2d_with_indices.default", "avg_pool2d.default")
def _pool_l(cv, node, a):
    is_max = "max_pool" in str(node.target)
    val = cv.meta(node)
    out_meta = val[0] if isinstance(val, (list, tuple)) else val
    shape = cv.shape_of_meta(out_meta)
    k = _pair(a["kernel_size"], 2)
    s = _pair(a.get("stride") or k, 2)
    p = _pair(a.get("padding", 0), 2)
    d = _pair(a.get("dilation", 1), 2) if is_max else [1, 1]
    if not is_max and a.get("divisor_override") is not None:
        cv.fail("divisor_override")
    y = _pool(cv, cv.v(a["self"]), k, s, [p[0], p[0], p[1], p[1]], d, is_max,
              a.get("count_include_pad", True), shape)
    y = cv.reshape(y, shape)
    if "with_indices" in str(node.target):
        return [y, None]
    return y


@_lower("adaptive_avg_pool2d.default", "_adaptive_avg_pool2d.default", "adaptive_max_pool2d.default")
def _adaptive_pool(cv, node, a):
    is_max = "max" in str(node.target)
    val = cv.meta(node)
    out_meta = val[0] if isinstance(val, (list, tuple)) else val
    dt, shape = _mim_dtype(out_meta.dtype), cv.shape_of_meta(out_meta)
    x = cv.v(a["self"])
    xs = cv.shp(x)
    if len(xs) != 4:
        cv.fail("only 2-D adaptive pooling is supported")
    h, w = xs[2], xs[3]
    oh, ow = shape[2], shape[3]
    if not all(isinstance(d, int) for d in (h, w)) or h % oh or w % ow:
        cv.fail(f"adaptive pooling from {h}x{w} to {oh}x{ow}; the input must be a multiple of the output")
    if oh == 1 and ow == 1:
        y = cv.prim("reduce", [x], dt, xs[:2] + [1, 1], [("mode", "string", "max" if is_max else "add"), ("axes", "ints", [2, 3])])
        if not is_max:
            y = cv.bin("mul", y, cv.k(1.0 / (h * w), dt, 4))
    else:
        kh, kw = h // oh, w // ow
        y = _pool(cv, x, (kh, kw), (kh, kw), [0, 0, 0, 0], (1, 1), is_max, True, shape)
    return [y, None] if is_max else y


SUPPORTED = sorted(_LOWER)
