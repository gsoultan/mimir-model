"""Runs a mim graph: binds symbolic dimensions from the inputs, evaluates
primitives, `call`, `while` and `if`, and applies variable updates
atomically after each run (spec/primitives.md sections 6 and 7)."""

import numpy as np

from . import graph as G
from . import prims
from .errors import RunError
from .stdlib import COMPOSITES
from .tensor import Tensor

DEFAULT_MAX_ITERATIONS = 1 << 24
MAX_ZERO_BYTES = 1 << 40


class Program:
    """A loaded graph with its weights, constants and initial variable
    values resolved to tensors."""

    def __init__(self, g, tensor, max_iterations=DEFAULT_MAX_ITERATIONS):
        """tensor(i) returns the file tensor with index i."""
        G.validate(g)
        self.graph = g
        self.max_iterations = max_iterations
        self.fixed = {}
        self.init_vars = {}
        self.zero_vars = {}
        for vid, v in enumerate(g.values):
            if v.kind not in (G.WEIGHT, G.CONST, G.VARIABLE):
                continue
            if v.zero:
                n = v.dtype.block_bytes
                for d in v.shape:
                    n *= d
                if n > MAX_ZERO_BYTES:
                    raise RunError(f"variable {v.name!r} is larger than {MAX_ZERO_BYTES} bytes")
                self.zero_vars[vid] = v
                continue
            t = tensor(v.ref)
            if t.dtype is not v.dtype or len(t.shape) != len(v.shape) or any(
                d >= 0 and d != s for d, s in zip(v.shape, t.shape)
            ):
                raise RunError(f"value {v.name!r} is {v.dtype.name}{v.shape}, tensor is {t.dtype.name}{list(t.shape)}")
            (self.init_vars if v.kind == G.VARIABLE else self.fixed)[vid] = t
        self.composites = {}
        for c in g.composites:
            key = f"{c.name}.v{c.version}"
            if key in self.composites:
                raise RunError(f"composite {key} is defined twice")
            self.composites[key] = c.subgraph

    @classmethod
    def load(cls, f, **kw):
        """Loads the graph of a container.File. Tensor data is not copied."""
        g = f.graph()
        if g is None:
            raise RunError("the file has no graph")

        def tensor(i):
            if i >= len(f):
                raise RunError(f"tensor {i} does not exist")
            ti = f.tensor_at(i)
            return Tensor.from_bytes(ti.dtype, ti.shape, f.tensor_data(ti))

        return cls(g, tensor, **kw)

    def session(self):
        return Session(self)


class Session:
    """The variable state of one sequence of runs, such as one conversation
    with its key-value cache."""

    def __init__(self, p):
        self.p = p
        self.vars = dict(p.init_vars)
        for vid, v in p.zero_vars.items():
            self.vars[vid] = Tensor(v.dtype, v.shape, np.zeros(v.shape, dtype=v.dtype.storage))

    def variable(self, name):
        for vid, t in self.vars.items():
            if self.p.graph.values[vid].name == name:
                return t
        return None

    def run(self, inputs):
        """Runs the main graph once. inputs maps input names to Tensors; the
        result maps output names to Tensors. If the run fails, no variable
        changes."""
        g = self.p.graph
        r = _Run(self.p)
        top = dict(self.p.fixed)
        for vid, t in self.vars.items():
            r.bind(g.values[vid], t)
            top[vid] = t
        used = 0
        for vid in g.main.inputs:
            v = g.values[vid]
            t = inputs.get(v.name)
            if t is None:
                raise RunError(f"missing input {v.name!r}")
            used += 1
            r.bind(v, t)
            top[vid] = t
        if used != len(inputs):
            raise RunError(f"{len(inputs)} inputs given, the graph takes {used}")
        scope = _Scope(top, None)
        r.exec(0, scope)
        out = {g.values[vid].name: scope.get(vid) for vid in g.main.outputs}
        nxt = dict(self.vars)
        for u in g.updates:
            t = scope.get(u.value)
            r.check(g.values[u.variable], t)
            nxt[u.variable] = t
        self.vars = nxt  # the one atomic step
        return out


class _Scope:
    __slots__ = ("vals", "parent")

    def __init__(self, vals, parent):
        self.vals, self.parent = vals, parent

    def get(self, vid):
        s = self
        while s is not None:
            t = s.vals.get(vid)
            if t is not None:
                return t
            s = s.parent
        raise RunError(f"value {vid} is not available")

    def top(self):
        s = self
        while s.parent is not None:
            s = s.parent
        return s


class _Run:
    def __init__(self, p):
        self.p = p
        self.g = p.graph
        self.sym = [-1] * len(self.g.symdims)

    def bind(self, v, t):
        if t.dtype is not v.dtype or len(t.shape) != len(v.shape):
            raise RunError(f"{v.name!r} is {t.dtype.name}{list(t.shape)}, declared {v.dtype.name} rank {len(v.shape)}")
        for d, n in zip(v.shape, t.shape):
            if d >= 0:
                if d != n:
                    raise RunError(f"{v.name!r} is {list(t.shape)}, declared {v.shape}")
                continue
            si = -1 - d
            sd = self.g.symdims[si]
            if self.sym[si] == -1:
                if 0 <= sd.bound < n:
                    raise RunError(f"{v.name!r} dimension {sd.name} = {n} exceeds its bound {sd.bound}")
                self.sym[si] = n
            elif self.sym[si] != n:
                raise RunError(f"{v.name!r} dimension {sd.name} is {n}, bound to {self.sym[si]}")

    def shape(self, v):
        out = []
        for d in v.shape:
            if d >= 0:
                out.append(d)
                continue
            n = self.sym[-1 - d]
            if n < 0:
                raise RunError(f"{v.name!r}: symbolic dimension {self.g.symdims[-1 - d].name} is not bound by any input")
            out.append(n)
        return tuple(out)

    def check(self, v, t):
        want = self.shape(v)
        if t.dtype is not v.dtype or tuple(t.shape) != want:
            raise RunError(f"{v.name!r} is {t.dtype.name}{list(t.shape)}, declared {v.dtype.name}{list(want)}")

    def exec(self, si, scope):
        sg = self.g.subgraphs[si]
        for oi, op in enumerate(sg.ops):
            ins = [scope.get(vid) for vid in op.inputs]
            try:
                outs = self.op(op, ins, scope)
                if len(outs) != len(op.outputs):
                    raise RunError(f"{len(outs)} results for {len(op.outputs)} outputs")
                for vid, t in zip(op.outputs, outs):
                    self.check(self.g.values[vid], t)
                    scope.vals[vid] = t
            except RunError as e:
                raise RunError(f"subgraph {si} op {oi} ({op.name}): {e}") from None

    def sub(self, si, ins, parent):
        sg = self.g.subgraphs[si]
        if len(ins) != len(sg.inputs):
            raise RunError(f"subgraph {si} takes {len(sg.inputs)} inputs, got {len(ins)}")
        scope = _Scope({}, parent)
        for vid, t in zip(sg.inputs, ins):
            self.check(self.g.values[vid], t)
            scope.vals[vid] = t
        self.exec(si, scope)
        return [scope.get(vid) for vid in sg.outputs]

    def op(self, op, ins, scope):
        name = op.name
        if name == "while":
            return self.while_(op, ins, scope)
        if name == "if":
            return self.if_(op, ins, scope)
        if name == "call":
            return self.call(op, ins, scope)
        if len(op.outputs) != 1:
            raise RunError("a primitive has exactly one output")
        v = self.g.values[op.outputs[0]]
        return [prims.evaluate(name, op.attrs, ins, v.dtype, self.shape(v))]

    def _subgraph_attrs(self, op, *keys):
        if set(op.attrs) != set(keys) or any(op.attrs[k].type != "subgraph" for k in keys):
            raise RunError(f"{op.name} takes exactly the subgraph attributes {', '.join(keys)}")
        return [op.attrs[k].value for k in keys]

    @staticmethod
    def _pred(t):
        if t.dtype.kind != "bool" or t.shape != ():
            raise RunError(f"condition is {t.dtype.name}{list(t.shape)}, want a bool scalar")
        return bool(t.arr)

    def while_(self, op, ins, scope):
        cond, body = self._subgraph_attrs(op, "cond", "body")
        v = ins
        i = 0
        while self._pred(self.sub(cond, v, scope)[0]):
            if i >= self.p.max_iterations:
                raise RunError(f"while loop exceeded {self.p.max_iterations} iterations")
            v = self.sub(body, v, scope)
            i += 1
        return v

    def if_(self, op, ins, scope):
        then, els = self._subgraph_attrs(op, "then", "else")
        if not ins:
            raise RunError("if takes a predicate")
        return self.sub(then if self._pred(ins[0]) else els, ins[1:], scope)

    def call(self, op, ins, scope):
        name, ver = op.attrs.get("name"), op.attrs.get("version")
        if name is None or name.type != "string" or ver is None or ver.type != "int":
            raise RunError("call needs a string name and an int version")
        key = f"{name.value}.v{ver.value}"
        si = self.p.composites.get(key)
        if si is not None:
            # A model-local composite runs in the top-level scope, not the
            # caller's (spec/primitives.md section 6.4).
            return self.sub(si, ins, scope.top())
        fn = COMPOSITES.get(key)
        if fn is None:
            raise RunError(f"composite {key} is not supported")
        dts, shapes = [], []
        for vid in op.outputs:
            v = self.g.values[vid]
            dts.append(v.dtype)
            shapes.append(self.shape(v))
        attrs = {k: a for k, a in op.attrs.items() if k not in ("name", "version")}
        return fn(attrs, ins, dts, shapes)
