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

package interp

import (
	"context"
	"fmt"
	"math"
	"sync"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/internal/kernels"
	"github.com/gsoultan/mimir-model/ref"
)

// Compiled plans. A graph without state (no variables, loops or
// conditionals, as the networks imported from ONNX are) computes the same
// values, of the same shapes, on every run with the same symbolic
// dimensions. A plan is such a graph worked out once for one binding of
// its dimensions: each op's kernel and parameters chosen, its output's
// buffer allocated, and the buffers of values the run no longer needs
// given to later values. A run of a plan then does nothing but call the
// kernels, which for a small model is most of what an interpreter's cost
// per op was.
//
// A plan's buffers belong to one run at a time; the program keeps the
// plans of each binding in a pool for runs going on at once. Values the
// caller receives, the graph's outputs, are made anew on each run.

// maxPlanBindings is the most bindings a program keeps plans for, for a
// graph whose dimensions change from run to run.
const maxPlanBindings = 8

// maxPlansPerBinding is the most plans a program keeps for one binding:
// runs beyond that many at once take the plain path.
const maxPlansPerBinding = 8

// maxReduceTable bounds the elements of a reduce whose output indices a
// plan tabulates.
const maxReduceTable = 1 << 20

// plan is the main graph compiled for one binding of its symbolic
// dimensions.
type plan struct {
	p       *Program
	slots   []*ref.Tensor // the value of each id: fixed, planned, or set by a run
	steps   []planStep
	inputs  []uint32
	outputs []uint32
}

// planStep is one compiled op (or run of ops) of the main graph.
type planStep struct {
	oi  int // the op, for errors
	opc mim.Opcode
	f   func() error
}

// planSlot holds the plans of one binding.
type planSlot struct {
	mu     sync.Mutex
	free   []*plan
	made   int
	failed bool // the graph could not be compiled for this binding
}

// planable reports whether the main graph can be compiled: it has no
// state and no control flow, and calls only standard composites.
func planable(g *mim.Graph) bool {
	if len(g.Subgraphs) != 1 || len(g.Updates) != 0 {
		return false
	}
	for _, v := range g.Values {
		if v.Kind == mim.KindVariable {
			return false
		}
	}
	for i := range g.Main().Ops {
		op := &g.Main().Ops[i]
		switch op.Opcode {
		case mim.OpWhile, mim.OpIf, mim.OpScatter:
			return false
		}
		if len(op.Outputs) != 1 && op.Opcode != mim.OpCall {
			return false
		}
	}
	return len(g.Composites) == 0
}

// planFor returns a free plan for the binding sym, compiling one when the
// program has none yet, or nil when the graph cannot be planned or too
// many runs of this binding are going on. A plan taken must be returned
// with release.
func (p *Program) planFor(sym []int) *plan {
	if !p.planable || !p.FastPaths {
		return nil
	}
	key := fmt.Sprint(sym)
	v, ok := p.plans.Load(key)
	if !ok {
		if p.planKeys.Load() >= maxPlanBindings {
			return nil
		}
		v, ok = p.plans.LoadOrStore(key, &planSlot{})
		if !ok {
			p.planKeys.Add(1)
		}
	}
	slot := v.(*planSlot)
	slot.mu.Lock()
	defer slot.mu.Unlock()
	if slot.failed {
		return nil
	}
	if n := len(slot.free); n > 0 {
		pl := slot.free[n-1]
		slot.free = slot.free[:n-1]
		return pl
	}
	if slot.made >= maxPlansPerBinding {
		return nil
	}
	pl, err := p.compile(sym)
	if err != nil {
		slot.failed = true
		return nil
	}
	slot.made++
	return pl
}

// release gives a plan back to its program's pool.
func (p *Program) release(sym []int, pl *plan) {
	v, ok := p.plans.Load(fmt.Sprint(sym))
	if !ok {
		return
	}
	slot := v.(*planSlot)
	slot.mu.Lock()
	slot.free = append(slot.free, pl)
	slot.mu.Unlock()
}

// run executes the plan on inputs, already checked against the main
// graph's inputs, in order.
func (pl *plan) run(ctx context.Context, in []*ref.Tensor) (map[string]*ref.Tensor, error) {
	sl := pl.slots
	for i, id := range pl.inputs {
		sl[id] = in[i]
	}
	defer func() {
		for _, id := range pl.inputs {
			sl[id] = nil // the caller's, not held between runs
		}
	}()
	for i := range pl.steps {
		st := &pl.steps[i]
		if i&31 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if err := st.f(); err != nil {
			return nil, fmt.Errorf("subgraph 0 op %d (%s): %w", st.oi, st.opc, err)
		}
	}
	out := make(map[string]*ref.Tensor, len(pl.outputs))
	for _, id := range pl.outputs {
		out[pl.p.g.Values[id].Name] = sl[id]
	}
	return out, nil
}

// compiler builds a plan.
type compiler struct {
	r     *run
	pl    *plan
	isOut map[uint32]bool
	// Buffers. Each planned value's data lie in a buffer; a view shares
	// its input's. A buffer whose values are all dropped goes to free, by
	// size, for a later value.
	bufOf []int // by value id, -1 for none
	refs  []int // by buffer: live values in it
	bufs  [][]byte
	free  map[int][]int
	bytes int64 // bytes of all buffers
	// Chains waiting to be compiled (see pool.go): by output value; the
	// uses of each value; the values held for them, and those of them
	// to drop when the hold ends.
	pending  map[uint32]any
	uses     []int
	held     map[uint32]int
	heldDrop map[uint32]bool
}

func (p *Program) compile(sym []int) (*plan, error) {
	g := p.g
	main := g.Main()
	pl := &plan{p: p, slots: make([]*ref.Tensor, len(g.Values)), inputs: main.Inputs, outputs: main.Outputs}
	c := &compiler{
		r:     &run{p: p, sym: sym},
		pl:    pl,
		isOut: map[uint32]bool{},
		bufOf: make([]int, len(g.Values)),
		free:  map[int][]int{},

		pending:  map[uint32]any{},
		uses:     make([]int, len(g.Values)),
		held:     map[uint32]int{},
		heldDrop: map[uint32]bool{},
	}
	for i := range main.Ops {
		for _, id := range main.Ops[i].Inputs {
			c.uses[id]++
		}
	}
	for _, id := range main.Outputs {
		c.uses[id]++
	}
	for i := range c.bufOf {
		c.bufOf[i] = -1
	}
	for _, id := range main.Outputs {
		c.isOut[id] = true
	}
	for id, t := range p.fixed {
		pl.slots[id] = t
	}
	budget := p.maxBytes()
	for id := range g.Values {
		if dt := g.Values[id].DType; budget >= 0 && g.Values[id].Kind != mim.KindWeight && g.Values[id].Kind != mim.KindConst {
			s, err := c.r.shapeOf(uint32(id))
			if err != nil {
				return nil, err
			}
			if n, err := ref.ByteSize(dt, s); err != nil || int64(n) > budget {
				return nil, fmt.Errorf("%w: value %q does not fit it", ErrBudget, g.Values[id].Name)
			}
		}
	}
	for oi := 0; oi < len(main.Ops); oi++ {
		op := &main.Ops[oi]
		first, last := oi, oi
		var err error
		switch {
		case p.convFusions[oi] != nil:
			if last, err = c.fusedConv(oi, p.convFusions[oi]); err != nil {
				return nil, err
			}
		case p.chains[oi] != nil:
			if last, err = c.chain(oi, p.chains[oi]); err != nil {
				return nil, err
			}
		default:
			if err = c.op(oi, op); err != nil {
				return nil, err
			}
		}
		for fi := first; fi <= last; fi++ {
			for _, id := range p.frees[fi] {
				c.release(id)
			}
		}
		oi = last
	}
	for _, id := range main.Outputs {
		c.materialize(len(main.Ops), id)
	}
	if budget >= 0 && c.bytes > budget {
		return nil, fmt.Errorf("%w: the plan's %d bytes exceed the %d-byte budget", ErrBudget, c.bytes, budget)
	}
	return pl, nil
}

// shape returns the resolved shape of value id.
func (c *compiler) shape(id uint32) []int {
	s, _ := c.r.shapeOf(id)
	return s
}

// tensor returns the planned tensor for value id of n bytes, with a
// buffer from the free ones when one fits; nil when the value is an
// output, which a run makes anew.
func (c *compiler) tensor(id uint32, n int) *ref.Tensor {
	if c.isOut[id] {
		return nil
	}
	v := c.r.p.g.Values[id]
	t := &ref.Tensor{DType: v.DType, Shape: c.shape(id)}
	var bi int
	if l := c.free[n]; len(l) > 0 {
		bi = l[len(l)-1]
		c.free[n] = l[:len(l)-1]
	} else {
		bi = len(c.bufs)
		c.bufs = append(c.bufs, make([]byte, n))
		c.refs = append(c.refs, 0)
		c.bytes += int64(n)
	}
	t.Data = c.bufs[bi]
	c.bufOf[id] = bi
	c.refs[bi]++
	c.pl.slots[id] = t
	return t
}

// view makes value id a view of value in's buffer, in a planned tensor
// whose data a run sets.
func (c *compiler) view(id, in uint32) *ref.Tensor {
	v := c.r.p.g.Values[id]
	t := &ref.Tensor{DType: v.DType, Shape: c.shape(id)}
	if bi := c.bufOf[in]; bi >= 0 {
		c.bufOf[id] = bi
		c.refs[bi]++
	}
	c.pl.slots[id] = t
	return t
}

// drop marks value id as no longer needed after this step.
func (c *compiler) drop(id uint32) {
	bi := c.bufOf[id]
	if bi < 0 {
		return
	}
	c.bufOf[id] = -1
	if c.refs[bi]--; c.refs[bi] == 0 {
		c.free[len(c.bufs[bi])] = append(c.free[len(c.bufs[bi])], bi)
	}
}

// fresh returns a function that makes the tensor of value id for a run:
// the planned one, or a new one each run for an output.
func (c *compiler) fresh(id uint32, n int) func() *ref.Tensor {
	if t := c.tensor(id, n); t != nil {
		return func() *ref.Tensor { return t }
	}
	v := c.r.p.g.Values[id]
	shape := c.shape(id)
	sl := c.pl.slots
	return func() *ref.Tensor {
		t := &ref.Tensor{DType: v.DType, Shape: shape, Data: make([]byte, n)}
		sl[id] = t
		return t
	}
}

func (c *compiler) add(oi int, opc mim.Opcode, f func() error) {
	c.pl.steps = append(c.pl.steps, planStep{oi: oi, opc: opc, f: f})
}

// op compiles one op of the main graph.
func (c *compiler) op(oi int, op *mim.Op) error {
	g := c.r.p.g
	sl := c.pl.slots
	if op.Opcode == mim.OpCall {
		c.materializeAll(oi, op.Inputs)
		return c.call(oi, op)
	}
	out := op.Outputs[0]
	v := g.Values[out]
	dt, shape := v.DType, c.shape(out)
	n, err := ref.ByteSize(dt, shape)
	if err != nil {
		return err
	}
	if c.treeOp(oi, op) {
		return nil
	}
	c.materializeAll(oi, op.Inputs)
	ins := make([]*ref.Tensor, len(op.Inputs)) // the inputs' static descriptions
	for k, id := range op.Inputs {
		ins[k] = &ref.Tensor{DType: g.Values[id].DType, Shape: c.shape(id)}
	}
	in := func(k int) *ref.Tensor { return sl[op.Inputs[k]] }
	switch op.Opcode {
	case mim.OpConst:
		t, err := ref.Eval(op.Opcode, op.Attrs, nil, dt, shape)
		if err != nil {
			return err
		}
		sl[out] = t // the same on every run
		return nil
	case mim.OpReshape:
		if _, ok := fastShape(op, ins[0], dt, shape); ok && !c.r.p.noView[out] {
			t := c.view(out, op.Inputs[0])
			es := n
			c.add(oi, op.Opcode, func() error {
				x := in(0)
				if len(x.Data) < es {
					return fmt.Errorf("reshape of %d bytes, need %d", len(x.Data), es)
				}
				t.Data = x.Data[:es:es]
				return nil
			})
			return nil
		}
		if !dt.IsBlock() && ins[0].DType == dt && ref.NumElems(ins[0].Shape) == ref.NumElems(shape) {
			mk := c.fresh(out, n)
			c.add(oi, op.Opcode, func() error {
				copy(mk().Data, in(0).Data[:n])
				return nil
			})
			return nil
		}
	case mim.OpAdd, mim.OpMul, mim.OpMax, mim.OpCmplt, mim.OpCmpeq, mim.OpAnd, mim.OpOr, mim.OpXor:
		if len(ins) != 2 || ins[0].DType != ins[1].DType {
			break
		}
		nan := math.Float32frombits(uint32(ref.RoundFloat(mim.F32, math.NaN(), false)))
		if dt == mim.F32 && ins[0].DType == mim.F32 && (op.Opcode == mim.OpAdd || op.Opcode == mim.OpMul || op.Opcode == mim.OpMax) &&
			sameInts(ins[0].Shape, shape) && sameInts(ins[1].Shape, shape) {
			mk := c.fresh(out, n)
			opc := op.Opcode
			c.add(oi, opc, func() error {
				binaryF32(opc, f32View(in(0)), 1, f32View(in(1)), 1, f32View(mk()), nan)
				return nil
			})
			return nil
		}
		if bc, ok := newBroadcast(op.Opcode, ins[0].DType, ins[0].Shape, ins[1].Shape, dt, shape); ok {
			mk := c.fresh(out, n)
			zero := bc.kind != 'f'
			var y *ref.Tensor // the step's output, made once per run
			rows := func(lo, hi int) { bc.rows(in(0), in(1), y, lo, hi) }
			c.add(oi, op.Opcode, func() error {
				y = mk()
				if zero {
					clear(y.Data)
				}
				parallelWork(bc.outer*bc.per, bc.n, rows)
				return nil
			})
			return nil
		}
	case mim.OpRecip:
		if dt == mim.F32 && ins[0].DType == mim.F32 {
			mk := c.fresh(out, n)
			c.add(oi, op.Opcode, func() error {
				x, y := f32View(in(0)), f32View(mk())
				for i, v := range x {
					y[i] = math.Float32frombits(uint32(ref.RoundFloat(mim.F32, 1/float64(v), false)))
				}
				return nil
			})
			return nil
		}
	case mim.OpNeg:
		if dt == mim.F32 && ins[0].DType == mim.F32 && littleEndian {
			mk := c.fresh(out, n)
			c.add(oi, op.Opcode, func() error {
				x, y := f32View(in(0)), f32View(mk())
				for i, v := range x {
					y[i] = math.Float32frombits(math.Float32bits(v) ^ 1<<31)
				}
				return nil
			})
			return nil
		}
	case mim.OpShrink:
		b := attr(op, "begin")
		if ins[0].DType != dt || dt.IsBlock() || b == nil || b.Type != mim.AttrInts || len(b.Ints) != len(shape) || len(ins[0].Shape) != len(shape) {
			break
		}
		begin := make([]int, len(shape))
		ok := true
		for i, bv := range b.Ints {
			ok = ok && bv >= 0 && int(bv)+shape[i] <= ins[0].Shape[i]
			begin[i] = int(bv)
		}
		if !ok {
			break
		}
		es := int(dt.BlockBytes())
		mk := c.fresh(out, n)
		xShape := ins[0].Shape
		zeros := make([]int, len(shape))
		c.add(oi, op.Opcode, func() error {
			copyBox(mk().Data, shape, zeros, in(0).Data, xShape, begin, shape, es)
			return nil
		})
		return nil
	case mim.OpPad:
		if pd, ok := newPad(op, ins[0].DType, ins[0].Shape, dt, shape); ok {
			mk := c.fresh(out, n)
			c.add(oi, op.Opcode, func() error {
				pd.run(in(0), mk(), true)
				return nil
			})
			return nil
		}
	case mim.OpPermute:
		p := attr(op, "perm")
		if ins[0].DType != dt || dt.IsBlock() || p == nil || p.Type != mim.AttrInts || len(p.Ints) != len(shape) || len(shape) != len(ins[0].Shape) {
			break
		}
		seen := make([]bool, len(shape))
		ok := true
		for i, a := range p.Ints {
			ok = ok && a >= 0 && int(a) < len(shape) && !seen[a] && ins[0].Shape[a] == shape[i]
			if ok {
				seen[a] = true
			}
		}
		if !ok {
			break
		}
		es := int(dt.BlockBytes())
		strides := permStrides(p.Ints, ins[0].Shape)
		mk := c.fresh(out, n)
		c.add(oi, op.Opcode, func() error {
			permuteInto(mk().Data, in(0).Data, strides, shape, es)
			return nil
		})
		return nil
	case mim.OpReduce:
		if f := c.reduce(op, ins[0], dt, shape); f != nil {
			mk := c.fresh(out, n)
			c.add(oi, op.Opcode, func() error {
				f(in(0), mk())
				return nil
			})
			return nil
		}
	}
	// Anything else runs on the reference kernels, which make a tensor of
	// their own.
	attrs := op.Attrs
	opc := op.Opcode
	args := make([]*ref.Tensor, len(op.Inputs))
	c.add(oi, opc, func() error {
		for k := range args {
			args[k] = in(k)
		}
		t, err := ref.Eval(opc, attrs, args, dt, shape)
		if err != nil {
			return err
		}
		sl[out] = t
		return nil
	})
	return nil
}

// fastShape reports whether a reshape to shape is the identity on data.
func fastShape(op *mim.Op, x *ref.Tensor, dt mim.DType, shape []int) (int, bool) {
	a := attr(op, "shape")
	if x.DType != dt || a == nil || a.Type != mim.AttrInts || len(a.Ints) != len(shape) || ref.NumElems(shape) != ref.NumElems(x.Shape) {
		return 0, false
	}
	for i, d := range a.Ints {
		if d != -1 && d != int64(shape[i]) {
			return 0, false
		}
	}
	n, err := ref.ByteSize(dt, shape)
	if err != nil {
		return 0, false
	}
	return n, true
}

// reduce compiles a float32 reduce as the reference computes it: each
// output accumulated in float64 over its inputs in order and rounded
// once. It returns nil for other reduces.
func (c *compiler) reduce(op *mim.Op, x *ref.Tensor, dt mim.DType, shape []int) func(x, y *ref.Tensor) {
	mode, axes := attr(op, "mode"), attr(op, "axes")
	if x.DType != mim.F32 || dt != mim.F32 || mode == nil || mode.Type != mim.AttrString || axes == nil || axes.Type != mim.AttrInts ||
		len(x.Shape) != len(shape) || len(op.Attrs) != 2 {
		return nil
	}
	n := ref.NumElems(x.Shape)
	if n > maxReduceTable {
		return nil
	}
	reduced := make([]bool, len(shape))
	for _, a := range axes.Ints {
		if a < 0 || int(a) >= len(shape) {
			return nil
		}
		reduced[a] = true
	}
	for i := range shape {
		if want := x.Shape[i]; reduced[i] {
			want = 1
		} else if want != shape[i] {
			return nil
		} else {
			continue
		}
		if shape[i] != 1 {
			return nil
		}
	}
	m := ref.NumElems(shape)
	// The output element of each input element.
	tab := make([]int32, n)
	ost := make([]int, len(shape))
	for i, s := len(shape)-1, 1; i >= 0; i-- {
		ost[i] = s
		s *= shape[i]
	}
	idx := make([]int, len(shape))
	for i := range n {
		o := 0
		for k := range idx {
			if !reduced[k] {
				o += idx[k] * ost[k]
			}
		}
		tab[i] = int32(o)
		for k := len(idx) - 1; k >= 0; k-- {
			if idx[k]++; idx[k] < x.Shape[k] {
				break
			}
			idx[k] = 0
		}
	}
	var start float64
	var fold func(a, v float64) float64
	switch mode.String {
	case "add":
		start, fold = 0, func(a, v float64) float64 { return a + v }
	case "mul":
		start, fold = 1, func(a, v float64) float64 { return a * v }
	case "max":
		start, fold = math.Inf(-1), func(a, v float64) float64 {
			if a != a || v != v {
				return math.NaN()
			}
			return math.Max(a, v)
		}
	default:
		return nil
	}
	acc := make([]float64, m)
	return func(x, y *ref.Tensor) {
		for i := range acc {
			acc[i] = start
		}
		xv, yv := f32View(x), f32View(y)
		for i, v := range xv[:n] {
			acc[tab[i]] = fold(acc[tab[i]], float64(v))
		}
		for i, a := range acc {
			yv[i] = math.Float32frombits(uint32(ref.RoundFloat(mim.F32, a, false)))
		}
	}
}

// chain compiles a chain of movement ops as one gather and returns its
// last op.
func (c *compiler) chain(first int, ch *chain) (int, error) {
	g := c.r.p.g
	ops := g.Main().Ops
	x, out := ops[first].Inputs[0], ops[ch.last].Outputs[0]
	xShape, shape := c.shape(x), c.shape(out)
	xdt, dt := g.Values[x].DType, g.Values[out].DType
	if xdt != dt || dt.IsBlock() || ref.NumElems(xShape) >= 1<<31 {
		// One op at a time.
		for oi := first; oi <= ch.last; oi++ {
			if err := c.op(oi, &ops[oi]); err != nil {
				return 0, err
			}
		}
		return ch.last, nil
	}
	idx, err := c.r.chainIndices(first, ch, xShape, shape)
	if err != nil {
		return 0, err
	}
	c.materialize(first, x)
	es := int(dt.BlockBytes())
	ga := &gatherOp{src: x, idx: idx, shape: shape, es: es}
	if dt == mim.F32 && c.deferrable(out) {
		ga.base, ga.strides, ga.affine = affine(idx, shape)
		c.pending[out] = ga
		c.hold(x)
		return ch.last, nil
	}
	c.gatherStep(first, out, ga)
	return ch.last, nil
}

// fusedConv compiles a convolution with its epilogue and returns its last
// op, or compiles the ops one by one when they do not fit the kernel.
func (c *compiler) fusedConv(first int, f *convFusion) (int, error) {
	g := c.r.p.g
	ops := g.Main().Ops
	call := &ops[first]
	sl := c.pl.slots
	// The consts first: they read nothing.
	for _, ci := range f.consts {
		if err := c.op(ci, &ops[ci]); err != nil {
			return 0, err
		}
	}
	c.materializeAll(first, call.Inputs)
	for _, st := range f.steps {
		c.materialize(first, st.other)
	}
	out := call.Outputs[0]
	shape := c.shape(out)
	e := &epilogue{nan: math.Float32frombits(uint32(ref.RoundFloat(mim.F32, math.NaN(), false)))}
	var live []int // steps whose operand a run sets
	ok := len(shape) == 4
	for s := range f.steps {
		st := &f.steps[s]
		ts := c.shape(st.other)
		var kind byte
		switch {
		case !ok || g.Values[st.other].DType != mim.F32 || len(ts) != 4:
			ok = false
		case sameInts(ts, []int{1, 1, 1, 1}):
			kind = 's'
		case sameInts(ts, []int{1, shape[1], 1, 1}):
			kind = 'c'
		case sameInts(ts, shape):
			kind = 'f'
		default:
			ok = false
		}
		if !ok {
			break
		}
		e.ops = append(e.ops, st.op)
		e.kinds = append(e.kinds, kind)
		var val []float32
		if t := sl[st.other]; t != nil && c.bufOf[st.other] < 0 {
			val = f32View(t) // fixed or constant
		} else {
			live = append(live, s)
		}
		e.vals = append(e.vals, val)
	}
	if ok {
		ok = c.conv(first, call, f.steps[len(f.steps)-1].out, e, func() {
			for _, s := range live {
				e.vals[s] = f32View(sl[f.steps[s].other])
			}
		})
	}
	if ok {
		return f.last, nil
	}
	// One op at a time, the consts done.
	for oi := first; oi <= f.last; oi++ {
		if ops[oi].Opcode == mim.OpConst {
			continue
		}
		if err := c.op(oi, &ops[oi]); err != nil {
			return 0, err
		}
	}
	return f.last, nil
}

// conv compiles a conv2d.v1 call on the matrix-product kernel, its output
// being value out, with the epilogue e made ready by bind before each
// run. It reports false when the kernel does not take the call, having
// compiled nothing.
func (c *compiler) conv(oi int, call *mim.Op, out uint32, e *epilogue, bind func()) bool {
	g := c.r.p.g
	cs := c.r.p.calls[call]
	if cs == nil || cs.err != nil || cs.key != "conv2d.v1" || !kernels.Sgemm() || len(call.Inputs) < 2 {
		return false
	}
	ins := make([]*ref.Tensor, len(call.Inputs))
	for k, id := range call.Inputs {
		ins[k] = &ref.Tensor{DType: g.Values[id].DType, Shape: c.shape(id)}
	}
	if ins[0].DType != mim.F32 {
		return false
	}
	cp, err := readConv(cs.attrs, ins, []mim.DType{mim.F32})
	if err != nil {
		return false
	}
	shape := c.shape(out)
	if !sameInts(shape, []int{cp.nb, cp.f, cp.ho, cp.wo}) {
		return false
	}
	n := 4 * ref.NumElems(shape)
	mk := c.fresh(out, n)
	sl := c.pl.slots
	p := c.r.p
	x, w := call.Inputs[0], call.Inputs[1]
	var cr *convRun // made on the first run when the weights are not fixed
	if sl[w] != nil && c.bufOf[w] < 0 {
		cr = p.newConvRun(cp, sl[w])
	}
	c.add(oi, call.Opcode, func() error {
		if bind != nil {
			bind()
		}
		r := cr
		if r == nil {
			r = p.newConvRun(cp, sl[w])
		}
		var bias []float32
		if len(call.Inputs) == 3 {
			bias = f32View(sl[call.Inputs[2]])
		}
		r.run(f32View(sl[x]), bias, f32View(mk()), e)
		return nil
	})
	return true
}

// call compiles a call of a standard composite: its fast path when it has
// one, else its canonical body. Either makes tensors of its own.
func (c *compiler) call(oi int, op *mim.Op) error {
	cs := c.r.p.calls[op]
	if cs == nil {
		cs = newCallSite(op)
	}
	if cs.err != nil {
		return cs.err
	}
	if cs.key == "conv2d.v1" && len(op.Outputs) == 1 {
		if c.conv(oi, op, op.Outputs[0], nil, nil) {
			return nil
		}
	}
	fn, ok := stdlib[cs.key]
	if !ok {
		return fmt.Errorf("composite %s is not supported by this executor", cs.key)
	}
	if c.planned(oi, op, cs) {
		return nil
	}
	fast := fastPaths[cs.key]
	shapes := make([][]int, len(op.Outputs))
	dts := make([]mim.DType, len(op.Outputs))
	for k, id := range op.Outputs {
		shapes[k], dts[k] = c.shape(id), c.r.p.g.Values[id].DType
	}
	sl := c.pl.slots
	args := make([]*ref.Tensor, len(op.Inputs))
	outs := op.Outputs
	attrs := cs.attrs
	c.add(oi, op.Opcode, func() error {
		for k, id := range op.Inputs {
			args[k] = sl[id]
		}
		var res []*ref.Tensor
		var err error
		handled := false
		if fast != nil {
			res, handled, err = fast(attrs, args, dts, shapes)
		}
		if err == nil && !handled {
			res, err = fn(attrs, args, dts, shapes)
		}
		if err != nil {
			return err
		}
		if len(res) != len(outs) {
			return fmt.Errorf("%d results for %d outputs", len(res), len(outs))
		}
		for k, id := range outs {
			if t := res[k]; t.DType != dts[k] || !sameInts(t.Shape, shapes[k]) {
				return fmt.Errorf("result %d is %s%v, declared %s%v", k, t.DType, t.Shape, dts[k], shapes[k])
			}
			sl[id] = res[k]
		}
		return nil
	})
	return nil
}

// planned compiles the calls of standard composites that have a fast
// path writing into a planned buffer, reporting whether op is one: the
// call makes no tensors of its own, and its work is split for the
// workers once.
func (c *compiler) planned(oi int, op *mim.Op, cs *callSite) bool {
	g := c.r.p.g
	sl := c.pl.slots
	if len(op.Outputs) != 1 {
		return false
	}
	out := op.Outputs[0]
	if g.Values[out].DType != mim.F32 {
		return false
	}
	shape := c.shape(out)
	n := ref.NumElems(shape)
	ins := make([]*ref.Tensor, len(op.Inputs))
	for k, id := range op.Inputs {
		ins[k] = &ref.Tensor{DType: g.Values[id].DType, Shape: c.shape(id)}
		if ins[k].DType != mim.F32 {
			return false
		}
	}
	in := func(k int) []float32 { return f32View(sl[op.Inputs[k]]) }
	switch cs.key {
	case "sigmoid.v1", "tanh.v1":
		if len(cs.attrs) != 0 || len(ins) != 1 || !sameInts(ins[0].Shape, shape) {
			return false
		}
		f := kernels.SigmoidF32
		if cs.key == "tanh.v1" {
			f = kernels.TanhF32
		}
		mk := c.fresh(out, 4*n)
		var y []float32
		piece := func(lo, hi int) { unaryF32Piece(y, in(0), f, lo, hi) }
		c.add(oi, op.Opcode, func() error {
			y = f32View(mk())
			parallelWork(pieces(n, unaryPiece), n, piece)
			return nil
		})
		return true
	case "linear.v1":
		if len(cs.attrs) != 0 || len(ins) < 2 || len(ins) > 3 || len(ins[0].Shape) < 2 || len(ins[1].Shape) != 2 {
			return false
		}
		k, m := ins[1].Shape[1], ins[1].Shape[0]
		rows := ref.NumElems(ins[0].Shape) / max(1, k)
		if ins[0].Shape[len(ins[0].Shape)-1] != k || !sameInts(shape, append(append([]int{}, ins[0].Shape[:len(ins[0].Shape)-1]...), m)) ||
			(len(ins) == 3 && !sameInts(ins[2].Shape, []int{m})) {
			return false
		}
		mk := c.fresh(out, 4*n)
		w := &matrixRows{k: k, rowBytes: 4 * k}
		var bias, y []float32
		cols := func(lo, hi int) { linearCols(y, in(0), rows, w, bias, lo, hi) }
		c.add(oi, op.Opcode, func() error {
			y = f32View(mk())
			w.t = sl[op.Inputs[1]]
			if len(ins) == 3 {
				bias = in(2)
			}
			parallelWork(m, rows*m*k/16, cols)
			return nil
		})
		return true
	case "matmul.v1":
		if len(cs.attrs) != 0 || len(ins) != 2 {
			return false
		}
		mm, ok := newMatmul(ins[0].Shape, ins[1].Shape)
		if !ok || !sameInts(mm.shape, shape) {
			return false
		}
		mk := c.fresh(out, 4*n)
		var y []float32
		rows := func(lo, hi int) { mm.rows(y, in(0), in(1), lo, hi) }
		c.add(oi, op.Opcode, func() error {
			y = f32View(mk())
			parallelWork(mm.batch*mm.m, mm.batch*mm.m*mm.n*mm.k/4, rows)
			return nil
		})
		return true
	case "softmax.v1":
		if len(ins) != 1 || !sameInts(ins[0].Shape, shape) {
			return false
		}
		sm, err := newSoftmax(cs.attrs, shape)
		if err != nil {
			return false
		}
		mk := c.fresh(out, 4*n)
		var y []float32
		rows := func(lo, hi int) { sm.rows(y, in(0), lo, hi) }
		c.add(oi, op.Opcode, func() error {
			y = f32View(mk())
			parallelWork(sm.outer*sm.inner, n*8, rows)
			return nil
		})
		return true
	}
	return false
}

// treeOp compiles op when it is a max or add of two pending chains or
// trees of one input, as a tree that waits in turn or runs as a step,
// reporting whether it was.
func (c *compiler) treeOp(oi int, op *mim.Op) bool {
	if (op.Opcode != mim.OpMax && op.Opcode != mim.OpAdd) || len(op.Inputs) != 2 {
		return false
	}
	g := c.r.p.g
	out := op.Outputs[0]
	a, b := c.pending[op.Inputs[0]], c.pending[op.Inputs[1]]
	if a == nil || b == nil || g.Values[out].DType != mim.F32 {
		return false
	}
	shape := c.shape(out)
	t, ok := tree(op.Opcode, a, b)
	if !ok || !sameInts(t.shape, shape) || !sameInts(c.shape(op.Inputs[0]), shape) || !sameInts(c.shape(op.Inputs[1]), shape) {
		return false
	}
	delete(c.pending, op.Inputs[0])
	delete(c.pending, op.Inputs[1])
	if c.deferrable(out) {
		c.pending[out] = t
		return true
	}
	c.poolStep(oi, out, t)
	for range t.taps {
		c.unhold(t.src)
	}
	return true
}
