// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mim, created and owned by Gembit Soultan Shirazi.
// It is licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License in the LICENSE file at the root
// of this repository or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

// Package interp is a minimal reference executor for mim graphs. It runs a
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
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/ref"
)

// DefaultMaxIterations bounds each `while` loop so that a hostile or broken
// graph cannot hang the caller.
const DefaultMaxIterations = 1 << 24

// DefaultMaxBytes bounds the memory a session allocates, so that a hostile
// or broken graph that declares huge variables or intermediate values
// fails with an error instead of exhausting memory.
const DefaultMaxBytes = 4 << 30

// Program is a loaded graph with its weights and constants resolved.
type Program struct {
	// MaxIterations bounds the iterations of each `while` op in a run;
	// 0 means DefaultMaxIterations.
	MaxIterations int

	// MaxBytes bounds the bytes a session allocates: the variables that
	// start as zeros, together, and in each run the values computed by the
	// main graph that are still held plus the value being computed. 0 means
	// DefaultMaxBytes; a negative value means no bound. Weights, constants,
	// variables stored in the file and run inputs are not counted.
	MaxBytes int64

	// FastPaths enables the executor's own float32 kernels for the
	// standard-library composites that have one (fast.go). When false,
	// every composite runs its canonical body on the reference kernels.
	FastPaths bool

	g           *mim.Graph
	fixed       map[uint32]*ref.Tensor // weights and consts, by value id
	initVars    map[uint32]*ref.Tensor // initial variable values, by value id
	zeroVars    map[uint32]zeroVar     // variables that start as zeros
	composites  map[string]uint32      // "name.vN" to subgraph index
	inplace     map[*mim.Op]uint32     // scatters that may run in place, to their variable
	frees       [][]uint32             // values each op of the main graph is the last to need
	noView      map[uint32]bool        // values a reshape must copy from or into (reshapeView)
	isFixed     map[*ref.Tensor]bool   // the tensors of fixed, which no run changes
	packed      sync.Map               // a weight of isFixed to its layout for a kernel (convGemm)
	chains      map[int]*chain         // chains of movement ops of the main graph, by first op
	convFusions map[int]*convFusion    // convolutions with an epilogue, by op of the main graph
	gathers     sync.Map               // a chain and input shape to its gather indices
	gatherElems atomic.Int64           // the indices in gathers, and those not kept
	arenas      sync.Pool              // *arena, one for each run going on at once
	planable    bool                   // the main graph can be compiled (plan.go)
	plans       sync.Map               // a binding of the symbolic dimensions to its *planSlot
	planKeys    atomic.Int32           // the bindings in plans
	static      [][]int                // each value's shape when it has no symbolic dimension, by value id
	calls       map[*mim.Op]*callSite  // the call ops of every subgraph, resolved
}

// callSite is a call op with its composite's key and attributes worked
// out once, rather than on every run.
type callSite struct {
	key   string     // "name.vN"
	attrs []mim.Attr // the attributes other than name and version
	err   error      // why the op is not a well-formed call
}

// Load decodes f's graph and resolves its weight, const and variable values
// to the tensors they refer to. Tensor data is not copied.
func Load(f *mim.File) (*Program, error) {
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
func New(g *mim.Graph, tensor func(uint32) (*ref.Tensor, error)) (*Program, error) {
	if err := g.Validate(); err != nil {
		return nil, err
	}
	p := &Program{
		g:          g,
		fixed:      map[uint32]*ref.Tensor{},
		initVars:   map[uint32]*ref.Tensor{},
		zeroVars:   map[uint32]zeroVar{},
		composites: map[string]uint32{},
	}
	for id, v := range g.Values {
		if v.Kind != mim.KindWeight && v.Kind != mim.KindConst && v.Kind != mim.KindVariable {
			continue
		}
		if v.Zero {
			z, err := newZeroVar(v)
			if err != nil {
				return nil, err
			}
			p.zeroVars[uint32(id)] = z
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
		if v.Kind == mim.KindVariable {
			p.initVars[uint32(id)] = t
		} else {
			p.fixed[uint32(id)] = t
		}
	}
	p.inplace = inplaceScatters(g)
	p.frees = lastUses(g)
	p.noView = noViews(g)
	p.chains = moveChains(g)
	p.convFusions = convFusions(g)
	p.planable = planable(g)
	p.static = make([][]int, len(g.Values))
	for id, v := range g.Values {
		s := make([]int, len(v.Shape))
		for k, d := range v.Shape {
			if d.Symbolic() {
				s = nil
				break
			}
			s[k] = int(d)
		}
		p.static[id] = s
	}
	p.calls = map[*mim.Op]*callSite{}
	for si := range g.Subgraphs {
		for oi := range g.Subgraphs[si].Ops {
			if op := &g.Subgraphs[si].Ops[oi]; op.Opcode == mim.OpCall {
				p.calls[op] = newCallSite(op)
			}
		}
	}
	p.isFixed = map[*ref.Tensor]bool{}
	for _, t := range p.fixed {
		p.isFixed[t] = true
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

// lastUses returns, for each op of the main graph, the values computed in
// a run that nothing needs after that op: no later op of the main graph
// reads them and they are not outputs or variable updates. A run drops
// them once the op has run, so that it holds only the values still to be
// read rather than every value of every layer. Values read by an op of
// another subgraph, which may see its parent's values, are kept.
func lastUses(g *mim.Graph) [][]uint32 {
	if len(g.Subgraphs) == 0 {
		return nil
	}
	main := g.Main()
	keep := map[uint32]bool{}
	for _, id := range main.Outputs {
		keep[id] = true
	}
	for _, u := range g.Updates {
		keep[u.Value] = true
	}
	for si := 1; si < len(g.Subgraphs); si++ {
		sg := &g.Subgraphs[si]
		for _, op := range sg.Ops {
			for _, id := range op.Inputs {
				keep[id] = true
			}
		}
		for _, id := range sg.Outputs {
			keep[id] = true
		}
	}
	last := map[uint32]int{}
	for oi, op := range main.Ops {
		for _, id := range op.Outputs {
			last[id] = oi // a value nothing reads is dropped at once
		}
		for _, id := range op.Inputs {
			last[id] = oi
		}
	}
	frees := make([][]uint32, len(main.Ops))
	for id, oi := range last {
		switch g.Values[id].Kind {
		case mim.KindWeight, mim.KindConst, mim.KindVariable:
			continue
		}
		if !keep[id] {
			frees[oi] = append(frees[oi], id)
		}
	}
	return frees
}

// Graph returns the program's graph.
func (p *Program) Graph() *mim.Graph { return p.g }

// Session holds the variable state of one sequence of runs, such as one
// conversation with its key-value cache.
type Session struct {
	p    *Program
	vars map[uint32]*ref.Tensor
	// owned marks the variables whose current tensor this session made
	// and nothing else refers to, which an in-place scatter may modify.
	owned map[uint32]bool
	err   error // why the session could not start; every Run returns it
}

// maxBytes returns the byte budget of p's sessions, or -1 for none.
func (p *Program) maxBytes() int64 {
	switch {
	case p.MaxBytes == 0:
		return DefaultMaxBytes
	case p.MaxBytes < 0:
		return -1
	}
	return p.MaxBytes
}

// NewSession starts a session with every variable at its initial value.
// If the variables that start as zeros do not fit in MaxBytes, nothing is
// allocated and every Run of the session returns that error; StartSession
// reports it directly.
func (p *Program) NewSession() *Session {
	s, err := p.StartSession()
	if err != nil {
		return &Session{p: p, err: err}
	}
	return s
}

// StartSession is NewSession, returning an error when the variables that
// start as zeros do not fit in MaxBytes.
func (p *Program) StartSession() (*Session, error) {
	if budget := p.maxBytes(); budget >= 0 {
		var total int64
		for id, z := range p.zeroVars {
			if total += int64(z.bytes); total > budget {
				return nil, fmt.Errorf("variable %q takes the zero variables past the %d-byte budget", p.g.Values[id].Name, budget)
			}
		}
	}
	s := &Session{p: p, vars: map[uint32]*ref.Tensor{}, owned: map[uint32]bool{}}
	for id, t := range p.initVars {
		s.vars[id] = t
	}
	// A zero variable's buffer is the session's own from the start, so its
	// first update can already be made in place.
	for id, z := range p.zeroVars {
		s.vars[id] = &ref.Tensor{DType: z.dt, Shape: z.shape, Data: make([]byte, z.bytes)}
		s.owned[id] = true
	}
	return s, nil
}

// zeroVar is a variable whose initial value is all zero bytes.
type zeroVar struct {
	dt    mim.DType
	shape []int
	bytes int
}

// maxZeroBytes bounds the size of a variable that starts as zeros, which
// costs memory without costing file space.
const maxZeroBytes = 1 << 40

func newZeroVar(v mim.Value) (zeroVar, error) {
	z := zeroVar{dt: v.DType, shape: make([]int, len(v.Shape))}
	n := int64(v.DType.BlockBytes())
	for k, d := range v.Shape {
		if d.Symbolic() {
			return zeroVar{}, fmt.Errorf("variable %q dimension %d is symbolic", v.Name, k)
		}
		if d > maxZeroBytes || d != 0 && n > maxZeroBytes/int64(d) {
			return zeroVar{}, fmt.Errorf("variable %q is larger than %d bytes", v.Name, int64(maxZeroBytes))
		}
		n *= int64(d)
		z.shape[k] = int(d)
	}
	if int64(int(n)) != n {
		return zeroVar{}, fmt.Errorf("variable %q does not fit in memory", v.Name)
	}
	z.bytes = int(n)
	return z, nil
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
	ctx   context.Context
	p     *Program
	s     *Session
	sym   []int  // bound size of each symbolic dimension, -1 when unbound
	undo  []undo // bytes of owned variables overwritten in place
	live  int64  // bytes of the main graph's computed values still held
	alloc *arena // buffers of the main graph's values on the fast paths; nil elsewhere
}

// reserve checks that a value of dtype dt and the given shape fits in the
// byte budget beside the values the run still holds, before it is
// allocated.
func (r *run) reserve(dt mim.DType, shape []int) error {
	budget := r.p.maxBytes()
	if budget < 0 {
		return nil
	}
	n, err := ref.ByteSize(dt, shape)
	if err != nil {
		return err
	}
	if int64(n) > budget-r.live {
		return fmt.Errorf("a %s%v value of %d bytes, with %d bytes held, exceeds the %d-byte budget", dt, shape, n, r.live, budget)
	}
	return nil
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
	return s.RunContext(context.Background(), inputs)
}

// RunContext is Run, stopping with ctx's error once ctx is done. It looks
// before each op and each while iteration, so a run stops within one op of
// being cancelled; as for any failed run, no variable changes.
func (s *Session) RunContext(ctx context.Context, inputs map[string]*ref.Tensor) (out map[string]*ref.Tensor, err error) {
	if s.err != nil {
		return nil, s.err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	g := s.p.g
	r := &run{ctx: ctx, p: s.p, s: s, sym: make([]int, len(g.SymDims))}
	defer func() {
		if err != nil {
			for i := len(r.undo) - 1; i >= 0; i-- {
				copy(r.undo[i].dst, r.undo[i].old)
			}
		}
	}()
	for i := range r.sym {
		r.sym[i] = -1
	}
	if s.p.FastPaths {
		a, _ := s.p.arenas.Get().(*arena)
		if a == nil {
			a = newArena()
		}
		a.begin()
		r.alloc = a
		defer func() {
			a.end()
			s.p.arenas.Put(a)
		}()
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
	if pl := s.p.planFor(r.sym); pl != nil {
		defer s.p.release(r.sym, pl)
		in := make([]*ref.Tensor, len(main.Inputs))
		for i, id := range main.Inputs {
			in[i] = top.vals[id]
		}
		return pl.run(ctx, in)
	}
	if err := r.exec(0, top); err != nil {
		return nil, err
	}
	out = map[string]*ref.Tensor{}
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
	// The scatters that may run in place made fresh tensors for their
	// variables, or wrote into tensors the session already owned.
	for _, v := range s.p.inplace {
		s.owned[v] = true
	}
	return out, nil
}

// bind checks t against the declared type of v, binding v's symbolic
// dimensions on first use.
func (r *run) bind(v mim.Value, t *ref.Tensor) error {
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
func (r *run) shape(v mim.Value) ([]int, error) {
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

// shapeOf is shape for the value with the given id. The result of a value
// with no symbolic dimension is the program's own: the caller must not
// change it.
func (r *run) shapeOf(id uint32) ([]int, error) {
	if s := r.p.static[id]; s != nil {
		return s, nil
	}
	return r.shape(r.p.g.Values[id])
}

// check verifies that t has the declared dtype and resolved shape of v.
func (r *run) check(v mim.Value, t *ref.Tensor) error {
	ok := t.DType == v.DType && len(t.Shape) == len(v.Shape)
	for k := 0; ok && k < len(v.Shape); k++ {
		want := int64(v.Shape[k])
		if d := v.Shape[k]; d.Symbolic() {
			want = int64(r.sym[d.SymIndex()])
		}
		ok = want >= 0 && int64(t.Shape[k]) == want
	}
	if ok {
		return nil
	}
	want, err := r.shape(v)
	if err != nil {
		return err
	}
	return fmt.Errorf("%q is %s%v, declared %s%v", v.Name, t.DType, t.Shape, v.DType, want)
}

func attr(op *mim.Op, key string) *mim.Attr {
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
	if si != 0 && r.alloc != nil {
		a := r.alloc
		r.alloc = nil
		defer func() { r.alloc = a }()
	}
	var buf []*ref.Tensor // the inputs of each op in turn
	for oi := 0; oi < len(sg.Ops); oi++ {
		if err := r.ctx.Err(); err != nil {
			return err
		}
		op := &sg.Ops[oi]
		if cap(buf) < len(op.Inputs) {
			buf = make([]*ref.Tensor, len(op.Inputs))
		}
		in := buf[:len(op.Inputs)]
		for k, id := range op.Inputs {
			t, ok := sc.get(id)
			if !ok {
				return fmt.Errorf("subgraph %d op %d (%s): input %d is not available", si, oi, op.Opcode, id)
			}
			in[k] = t
		}
		// A chain of movement ops runs as one gather (moveChain); its
		// intermediate values are never made.
		first := oi
		var outs []*ref.Tensor
		var err error
		var fused bool
		if fz := r.p.convFusions[oi]; si == 0 && r.p.FastPaths && fz != nil {
			var t *ref.Tensor
			if t, fused, err = r.fusedConv(oi, fz, in, sc); err == nil && fused {
				outs = []*ref.Tensor{t}
				oi = fz.last
				op = &sg.Ops[oi]
			}
		}
		if fused || err != nil {
		} else if ch := r.p.chains[oi]; si == 0 && r.p.FastPaths && ch != nil {
			var t *ref.Tensor
			if t, err = r.moveChain(oi, ch, in[0]); err == nil {
				outs = []*ref.Tensor{t}
				oi = ch.last
				op = &sg.Ops[oi]
			}
		} else {
			outs, err = r.op(op, in, sc)
		}
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
			if si == 0 {
				r.live += int64(len(outs[k].Data))
				r.alloc.retain(outs[k])
			}
		}
		if si == 0 {
			for fi := first; fi <= oi; fi++ {
				for _, id := range r.p.frees[fi] {
					// Run inputs are dropped too, but were never counted.
					if t, ok := sc.vals[id]; ok && g.Values[id].Kind != mim.KindInput {
						r.live -= int64(len(t.Data))
						r.alloc.release(t)
					}
					delete(sc.vals, id)
				}
			}
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

func (r *run) op(op *mim.Op, in []*ref.Tensor, sc *scope) ([]*ref.Tensor, error) {
	g := r.p.g
	switch op.Opcode {
	case mim.OpWhile:
		return r.while(op, in, sc)
	case mim.OpIf:
		return r.ifOp(op, in, sc)
	case mim.OpCall:
		return r.call(op, in, sc)
	}
	if len(op.Outputs) != 1 {
		return nil, fmt.Errorf("a primitive has exactly one output")
	}
	v := g.Values[op.Outputs[0]]
	shape, err := r.shapeOf(op.Outputs[0])
	if err != nil {
		return nil, err
	}
	if err := r.reserve(v.DType, shape); err != nil {
		return nil, err
	}
	if r.p.FastPaths {
		if t, ok := r.fastPrimitive(op, in, v.DType, shape); ok {
			return []*ref.Tensor{t}, nil
		}
	}
	t, err := ref.Eval(op.Opcode, op.Attrs, in, v.DType, shape)
	if err != nil {
		return nil, err
	}
	return []*ref.Tensor{t}, nil
}

func subgraphAttr(op *mim.Op, key string) (int, error) {
	a := attr(op, key)
	if a == nil || a.Type != mim.AttrSubgraph {
		return 0, fmt.Errorf("missing subgraph attribute %q", key)
	}
	return int(a.Subgraph), nil
}

func scalarBool(t *ref.Tensor) (bool, error) {
	if t.DType != mim.Bool || len(t.Shape) != 0 {
		return false, fmt.Errorf("condition is %s%v, want a bool scalar", t.DType, t.Shape)
	}
	return t.Data[0] != 0, nil
}

func (r *run) while(op *mim.Op, in []*ref.Tensor, sc *scope) ([]*ref.Tensor, error) {
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
		if err := r.ctx.Err(); err != nil {
			return nil, err
		}
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

func (r *run) ifOp(op *mim.Op, in []*ref.Tensor, sc *scope) ([]*ref.Tensor, error) {
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

func (r *run) call(op *mim.Op, in []*ref.Tensor, sc *scope) ([]*ref.Tensor, error) {
	cs := r.p.calls[op]
	if cs == nil {
		cs = newCallSite(op)
	}
	if cs.err != nil {
		return nil, cs.err
	}
	key := cs.key
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
	dts := make([]mim.DType, len(op.Outputs))
	for k, id := range op.Outputs {
		v := r.p.g.Values[id]
		s, err := r.shapeOf(id)
		if err != nil {
			return nil, err
		}
		if err := r.reserve(v.DType, s); err != nil {
			return nil, err
		}
		shapes[k], dts[k] = s, v.DType
	}
	cattrs := cs.attrs
	if fast && key == "conv2d.v1" {
		if out, handled, err := r.p.convGemm(r.alloc, cattrs, in, dts, nil); err != nil || handled {
			return out, err
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

func newCallSite(op *mim.Op) *callSite {
	name, ver := attr(op, "name"), attr(op, "version")
	if name == nil || name.Type != mim.AttrString || ver == nil || ver.Type != mim.AttrInt {
		return &callSite{err: fmt.Errorf("call needs a string name and an int version")}
	}
	cs := &callSite{key: fmt.Sprintf("%s.v%d", name.String, ver.Int)}
	for _, a := range op.Attrs {
		if a.Key != "name" && a.Key != "version" {
			cs.attrs = append(cs.attrs, a)
		}
	}
	return cs
}
