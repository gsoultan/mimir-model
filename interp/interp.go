// Package interp is a minimal reference executor for mmf graphs. It runs a
// graph end to end on the reference kernels of package ref: symbolic
// dimensions are bound from the inputs, `while` and `if` evaluate their
// subgraphs, `call` runs model-local composites and a subset of the standard
// library, and variable updates are applied atomically after each run
// (spec/primitives.md sections 6 and 7).
//
// The executor is written for clarity and for checking other runtimes, not
// for speed: every value is materialised and every element is computed by
// the reference kernels.
package interp

import (
	"fmt"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/ref"
)

// DefaultMaxIterations bounds each `while` loop so that a hostile or broken
// graph cannot hang the caller.
const DefaultMaxIterations = 1 << 24

// Program is a loaded graph with its weights and constants resolved.
type Program struct {
	// MaxIterations bounds the iterations of each `while` op in a run;
	// 0 means DefaultMaxIterations.
	MaxIterations int

	// FastPaths enables the executor's own float32 kernels for the
	// standard-library composites that have one (fast.go). When false,
	// every composite runs its canonical body on the reference kernels.
	FastPaths bool

	g          *mmf.Graph
	fixed      map[uint32]*ref.Tensor // weights and consts, by value id
	initVars   map[uint32]*ref.Tensor // initial variable values, by value id
	composites map[string]uint32      // "name.vN" to subgraph index
}

// Load decodes f's graph and resolves its weight, const and variable values
// to the tensors they refer to. Tensor data is not copied.
func Load(f *mmf.File) (*Program, error) {
	g, err := f.Graph()
	if err != nil {
		return nil, err
	}
	return New(g, func(i uint32) (*ref.Tensor, error) {
		if int(i) >= f.NumTensors() {
			return nil, fmt.Errorf("tensor %d does not exist", i)
		}
		t := f.TensorAt(int(i))
		shape := make([]int, len(t.Shape))
		for k, d := range t.Shape {
			if d > 1<<40 {
				return nil, fmt.Errorf("tensor %q dimension %d is too large", t.Name, d)
			}
			shape[k] = int(d)
		}
		return &ref.Tensor{DType: t.DType, Shape: shape, Data: f.TensorData(t)}, nil
	})
}

// New builds a program from a decoded graph. tensor returns the file tensor
// with the given index, for weight, const and variable values.
func New(g *mmf.Graph, tensor func(uint32) (*ref.Tensor, error)) (*Program, error) {
	if err := g.Validate(); err != nil {
		return nil, err
	}
	p := &Program{
		g:          g,
		fixed:      map[uint32]*ref.Tensor{},
		initVars:   map[uint32]*ref.Tensor{},
		composites: map[string]uint32{},
	}
	for id, v := range g.Values {
		if v.Kind != mmf.KindWeight && v.Kind != mmf.KindConst && v.Kind != mmf.KindVariable {
			continue
		}
		t, err := tensor(v.Ref)
		if err != nil {
			return nil, fmt.Errorf("value %q: %w", v.Name, err)
		}
		if t.DType != v.DType || len(t.Shape) != len(v.Shape) {
			return nil, fmt.Errorf("value %q is %s rank %d, tensor is %s%v", v.Name, v.DType, len(v.Shape), t.DType, t.Shape)
		}
		for k, d := range v.Shape {
			if !d.Symbolic() && int64(d) != int64(t.Shape[k]) {
				return nil, fmt.Errorf("value %q dimension %d is %d, tensor has %d", v.Name, k, d, t.Shape[k])
			}
		}
		if err := t.Validate(); err != nil {
			return nil, fmt.Errorf("value %q: %w", v.Name, err)
		}
		if v.Kind == mmf.KindVariable {
			p.initVars[uint32(id)] = t
		} else {
			p.fixed[uint32(id)] = t
		}
	}
	for _, c := range g.Composites {
		key := fmt.Sprintf("%s.v%d", c.Name, c.Version)
		if _, dup := p.composites[key]; dup {
			return nil, fmt.Errorf("composite %s is defined twice", key)
		}
		p.composites[key] = c.Subgraph
	}
	return p, nil
}

// Graph returns the program's graph.
func (p *Program) Graph() *mmf.Graph { return p.g }

// Session holds the variable state of one sequence of runs, such as one
// conversation with its key-value cache.
type Session struct {
	p    *Program
	vars map[uint32]*ref.Tensor
}

// NewSession starts a session with every variable at its initial value.
func (p *Program) NewSession() *Session {
	s := &Session{p: p, vars: map[uint32]*ref.Tensor{}}
	for id, t := range p.initVars {
		s.vars[id] = t
	}
	return s
}

// Variable returns the current value of the variable with the given name.
func (s *Session) Variable(name string) (*ref.Tensor, bool) {
	for id, t := range s.vars {
		if s.p.g.Values[id].Name == name {
			return t, true
		}
	}
	return nil, false
}

// run is the state of one Run call.
type run struct {
	p   *Program
	sym []int // bound size of each symbolic dimension, -1 when unbound
}

// scope maps value ids to tensors; a subgraph's scope sees its parent's
// values (spec/primitives.md section 6.4).
type scope struct {
	vals   map[uint32]*ref.Tensor
	parent *scope
}

func (s *scope) get(id uint32) (*ref.Tensor, bool) {
	for ; s != nil; s = s.parent {
		if t, ok := s.vals[id]; ok {
			return t, true
		}
	}
	return nil, false
}

// Run executes the main graph once. inputs maps the names of the main
// graph's inputs to tensors; the result maps output names to tensors. If
// the run fails, no variable changes (spec/primitives.md section 7).
func (s *Session) Run(inputs map[string]*ref.Tensor) (map[string]*ref.Tensor, error) {
	g := s.p.g
	r := &run{p: s.p, sym: make([]int, len(g.SymDims))}
	for i := range r.sym {
		r.sym[i] = -1
	}
	top := &scope{vals: map[uint32]*ref.Tensor{}}
	for id, t := range s.p.fixed {
		top.vals[id] = t
	}
	for id, t := range s.vars {
		top.vals[id] = t
		// A variable with a symbolic dimension binds it from its value.
		if err := r.bind(g.Values[id], t); err != nil {
			return nil, err
		}
	}
	main := g.Main()
	used := 0
	for _, id := range main.Inputs {
		v := g.Values[id]
		t, ok := inputs[v.Name]
		if !ok || t == nil {
			return nil, fmt.Errorf("missing input %q", v.Name)
		}
		used++
		if err := t.Validate(); err != nil {
			return nil, fmt.Errorf("input %q: %w", v.Name, err)
		}
		if err := r.bind(v, t); err != nil {
			return nil, err
		}
		top.vals[id] = t
	}
	if used != len(inputs) {
		return nil, fmt.Errorf("%d inputs given, the graph takes %d", len(inputs), used)
	}
	if err := r.exec(0, top); err != nil {
		return nil, err
	}
	out := map[string]*ref.Tensor{}
	for _, id := range main.Outputs {
		t, _ := top.get(id)
		out[g.Values[id].Name] = t
	}
	next := map[uint32]*ref.Tensor{}
	for id, t := range s.vars {
		next[id] = t
	}
	for _, u := range g.Updates {
		t, ok := top.get(u.Value)
		if !ok {
			return nil, fmt.Errorf("update value %d was not computed", u.Value)
		}
		if err := r.check(g.Values[u.Variable], t); err != nil {
			return nil, fmt.Errorf("update of %q: %w", g.Values[u.Variable].Name, err)
		}
		next[u.Variable] = t
	}
	s.vars = next // the one atomic step
	return out, nil
}

// bind checks t against the declared type of v, binding v's symbolic
// dimensions on first use.
func (r *run) bind(v mmf.Value, t *ref.Tensor) error {
	if t.DType != v.DType || len(t.Shape) != len(v.Shape) {
		return fmt.Errorf("%q is %s%v, declared %s rank %d", v.Name, t.DType, t.Shape, v.DType, len(v.Shape))
	}
	for k, d := range v.Shape {
		n := t.Shape[k]
		if !d.Symbolic() {
			if int64(d) != int64(n) {
				return fmt.Errorf("%q dimension %d is %d, declared %d", v.Name, k, n, d)
			}
			continue
		}
		si := d.SymIndex()
		sd := r.p.g.SymDims[si]
		switch {
		case r.sym[si] == -1:
			if sd.Bound >= 0 && int64(n) > sd.Bound {
				return fmt.Errorf("%q dimension %s = %d exceeds its bound %d", v.Name, sd.Name, n, sd.Bound)
			}
			r.sym[si] = n
		case r.sym[si] != n:
			return fmt.Errorf("%q dimension %s is %d, bound to %d", v.Name, sd.Name, n, r.sym[si])
		}
	}
	return nil
}

// shape resolves the declared shape of v with the current bindings.
func (r *run) shape(v mmf.Value) ([]int, error) {
	s := make([]int, len(v.Shape))
	for k, d := range v.Shape {
		if !d.Symbolic() {
			s[k] = int(d)
			continue
		}
		n := r.sym[d.SymIndex()]
		if n < 0 {
			return nil, fmt.Errorf("%q: symbolic dimension %s is not bound by any input", v.Name, r.p.g.SymDims[d.SymIndex()].Name)
		}
		s[k] = n
	}
	return s, nil
}

// check verifies that t has the declared dtype and resolved shape of v.
func (r *run) check(v mmf.Value, t *ref.Tensor) error {
	want, err := r.shape(v)
	if err != nil {
		return err
	}
	if t.DType != v.DType || len(t.Shape) != len(want) {
		return fmt.Errorf("%q is %s%v, declared %s%v", v.Name, t.DType, t.Shape, v.DType, want)
	}
	for k := range want {
		if t.Shape[k] != want[k] {
			return fmt.Errorf("%q is %s%v, declared %s%v", v.Name, t.DType, t.Shape, v.DType, want)
		}
	}
	return nil
}

func attr(op *mmf.Op, key string) *mmf.Attr {
	for i := range op.Attrs {
		if op.Attrs[i].Key == key {
			return &op.Attrs[i]
		}
	}
	return nil
}

// exec runs the ops of subgraph si in sc. The subgraph's inputs must
// already be in sc.
func (r *run) exec(si int, sc *scope) error {
	g := r.p.g
	sg := &g.Subgraphs[si]
	for oi := range sg.Ops {
		op := &sg.Ops[oi]
		in := make([]*ref.Tensor, len(op.Inputs))
		for k, id := range op.Inputs {
			t, ok := sc.get(id)
			if !ok {
				return fmt.Errorf("subgraph %d op %d (%s): input %d is not available", si, oi, op.Opcode, id)
			}
			in[k] = t
		}
		outs, err := r.op(op, in, sc)
		if err != nil {
			return fmt.Errorf("subgraph %d op %d (%s): %w", si, oi, op.Opcode, err)
		}
		if len(outs) != len(op.Outputs) {
			return fmt.Errorf("subgraph %d op %d (%s): %d results for %d outputs", si, oi, op.Opcode, len(outs), len(op.Outputs))
		}
		for k, id := range op.Outputs {
			if err := r.check(g.Values[id], outs[k]); err != nil {
				return fmt.Errorf("subgraph %d op %d (%s): %w", si, oi, op.Opcode, err)
			}
			sc.vals[id] = outs[k]
		}
	}
	return nil
}

// sub runs subgraph si with the given inputs in a child scope of parent
// and returns its outputs.
func (r *run) sub(si int, in []*ref.Tensor, parent *scope) ([]*ref.Tensor, error) {
	g := r.p.g
	sg := &g.Subgraphs[si]
	if len(in) != len(sg.Inputs) {
		return nil, fmt.Errorf("subgraph %d takes %d inputs, got %d", si, len(sg.Inputs), len(in))
	}
	sc := &scope{vals: map[uint32]*ref.Tensor{}, parent: parent}
	for k, id := range sg.Inputs {
		if err := r.check(g.Values[id], in[k]); err != nil {
			return nil, fmt.Errorf("subgraph %d input %d: %w", si, k, err)
		}
		sc.vals[id] = in[k]
	}
	if err := r.exec(si, sc); err != nil {
		return nil, err
	}
	out := make([]*ref.Tensor, len(sg.Outputs))
	for k, id := range sg.Outputs {
		out[k], _ = sc.get(id)
	}
	return out, nil
}

func (r *run) op(op *mmf.Op, in []*ref.Tensor, sc *scope) ([]*ref.Tensor, error) {
	g := r.p.g
	switch op.Opcode {
	case mmf.OpWhile:
		return r.while(op, in, sc)
	case mmf.OpIf:
		return r.ifOp(op, in, sc)
	case mmf.OpCall:
		return r.call(op, in, sc)
	}
	if len(op.Outputs) != 1 {
		return nil, fmt.Errorf("a primitive has exactly one output")
	}
	v := g.Values[op.Outputs[0]]
	shape, err := r.shape(v)
	if err != nil {
		return nil, err
	}
	t, err := ref.Eval(op.Opcode, op.Attrs, in, v.DType, shape)
	if err != nil {
		return nil, err
	}
	return []*ref.Tensor{t}, nil
}

func subgraphAttr(op *mmf.Op, key string) (int, error) {
	a := attr(op, key)
	if a == nil || a.Type != mmf.AttrSubgraph {
		return 0, fmt.Errorf("missing subgraph attribute %q", key)
	}
	return int(a.Subgraph), nil
}

func scalarBool(t *ref.Tensor) (bool, error) {
	if t.DType != mmf.Bool || len(t.Shape) != 0 {
		return false, fmt.Errorf("condition is %s%v, want a bool scalar", t.DType, t.Shape)
	}
	return t.Data[0] != 0, nil
}

func (r *run) while(op *mmf.Op, in []*ref.Tensor, sc *scope) ([]*ref.Tensor, error) {
	cond, err := subgraphAttr(op, "cond")
	if err != nil {
		return nil, err
	}
	body, err := subgraphAttr(op, "body")
	if err != nil {
		return nil, err
	}
	if len(op.Attrs) != 2 {
		return nil, fmt.Errorf("while takes exactly the attributes cond and body")
	}
	limit := r.p.MaxIterations
	if limit <= 0 {
		limit = DefaultMaxIterations
	}
	v := in
	for i := 0; ; i++ {
		c, err := r.sub(cond, v, sc)
		if err != nil {
			return nil, err
		}
		ok, err := scalarBool(c[0])
		if err != nil {
			return nil, err
		}
		if !ok {
			return v, nil
		}
		if i >= limit {
			return nil, fmt.Errorf("while loop exceeded %d iterations", limit)
		}
		if v, err = r.sub(body, v, sc); err != nil {
			return nil, err
		}
	}
}

func (r *run) ifOp(op *mmf.Op, in []*ref.Tensor, sc *scope) ([]*ref.Tensor, error) {
	thenG, err := subgraphAttr(op, "then")
	if err != nil {
		return nil, err
	}
	elseG, err := subgraphAttr(op, "else")
	if err != nil {
		return nil, err
	}
	if len(op.Attrs) != 2 || len(in) == 0 {
		return nil, fmt.Errorf("if takes a predicate and the attributes then and else")
	}
	pred, err := scalarBool(in[0])
	if err != nil {
		return nil, err
	}
	if pred {
		return r.sub(thenG, in[1:], sc)
	}
	return r.sub(elseG, in[1:], sc)
}

func (r *run) call(op *mmf.Op, in []*ref.Tensor, sc *scope) ([]*ref.Tensor, error) {
	name, ver := attr(op, "name"), attr(op, "version")
	if name == nil || name.Type != mmf.AttrString || ver == nil || ver.Type != mmf.AttrInt {
		return nil, fmt.Errorf("call needs a string name and an int version")
	}
	key := fmt.Sprintf("%s.v%d", name.String, ver.Int)
	if si, ok := r.p.composites[key]; ok {
		// Model-local composite: run its body. Its scope is the top level,
		// not the caller's (spec/primitives.md section 6.4).
		top := sc
		for top.parent != nil {
			top = top.parent
		}
		return r.sub(int(si), in, top)
	}
	fn, ok := stdlib[key]
	if !ok {
		return nil, fmt.Errorf("composite %s is not supported by this executor", key)
	}
	fast := r.p.FastPaths && fastPaths[key] != nil
	shapes := make([][]int, len(op.Outputs))
	dts := make([]mmf.DType, len(op.Outputs))
	for k, id := range op.Outputs {
		v := r.p.g.Values[id]
		s, err := r.shape(v)
		if err != nil {
			return nil, err
		}
		shapes[k], dts[k] = s, v.DType
	}
	var cattrs []mmf.Attr
	for _, a := range op.Attrs {
		if a.Key != "name" && a.Key != "version" {
			cattrs = append(cattrs, a)
		}
	}
	if fast {
		out, handled, err := fastPaths[key](cattrs, in, dts, shapes)
		if err != nil || handled {
			return out, err
		}
	}
	return fn(cattrs, in, dts, shapes)
}
