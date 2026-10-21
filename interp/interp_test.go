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
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/conformance"
	"github.com/gsoultan/mimir-model/ref"
)

// gb builds graphs for the tests.
type gb struct {
	g       *mim.Graph
	tensors []tensorSpec
}

type tensorSpec struct {
	name  string
	dt    mim.DType
	shape []uint64
	data  []byte
}

func newGB() *gb {
	return &gb{g: &mim.Graph{PrimitiveVersion: mim.CurrentPrimitive, StdlibVersion: mim.CurrentStdlib,
		Subgraphs: []mim.Subgraph{{}}}}
}

func (b *gb) sym(name string, bound int64) mim.Dim {
	b.g.SymDims = append(b.g.SymDims, mim.SymDim{Name: name, Bound: bound})
	return mim.Dim(-len(b.g.SymDims))
}

func (b *gb) val(name string, dt mim.DType, kind mim.ValueKind, shape ...mim.Dim) uint32 {
	b.g.Values = append(b.g.Values, mim.Value{Name: name, DType: dt, Shape: shape, Kind: kind})
	return uint32(len(b.g.Values) - 1)
}

func (b *gb) input(si int, name string, dt mim.DType, shape ...mim.Dim) uint32 {
	id := b.val(name, dt, mim.KindInput, shape...)
	b.g.Subgraphs[si].Inputs = append(b.g.Subgraphs[si].Inputs, id)
	return id
}

// stored adds a weight, const or variable value backed by a file tensor.
func (b *gb) stored(name string, kind mim.ValueKind, dt mim.DType, shape []uint64, data []byte) uint32 {
	dims := make([]mim.Dim, len(shape))
	for i, s := range shape {
		dims[i] = mim.Dim(s)
	}
	id := b.val(name, dt, kind, dims...)
	b.g.Values[id].Ref = uint32(len(b.tensors))
	b.tensors = append(b.tensors, tensorSpec{name, dt, shape, data})
	return id
}

func (b *gb) op(si int, opc mim.Opcode, in []uint32, out []uint32, attrs ...mim.Attr) {
	b.g.Subgraphs[si].Ops = append(b.g.Subgraphs[si].Ops, mim.Op{Opcode: opc, Inputs: in, Outputs: out, Attrs: attrs})
}

func (b *gb) output(si int, id uint32) {
	if si == 0 {
		b.g.Values[id].Kind = mim.KindOutput
	}
	b.g.Subgraphs[si].Outputs = append(b.g.Subgraphs[si].Outputs, id)
}

// load encodes the graph and tensors into a mim file and loads it, so the
// tests go through the real reader.
func (b *gb) load(t testing.TB) *Program {
	t.Helper()
	// The file's tensor index is sorted by name, and a value's Ref is a
	// position in that index.
	order := make([]int, len(b.tensors))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool { return b.tensors[order[i]].name < b.tensors[order[j]].name })
	pos := make([]uint32, len(order))
	for p, i := range order {
		pos[i] = uint32(p)
	}
	for i := range b.g.Values {
		if k := b.g.Values[i].Kind; (k == mim.KindWeight || k == mim.KindConst || k == mim.KindVariable) && !b.g.Values[i].Zero {
			b.g.Values[i].Ref = pos[b.g.Values[i].Ref]
		}
	}
	w := mim.NewWriter()
	for _, ts := range b.tensors {
		if err := w.AddTensor(ts.name, ts.dt, ts.shape, ts.data); err != nil {
			t.Fatal(err)
		}
	}
	enc, err := b.g.Encode()
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AddSection(mim.SectionGraph, true, enc); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := w.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	f, err := mim.Parse(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	p, err := Load(f)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func sAttr(k, v string) mim.Attr { return mim.Attr{Key: k, Type: mim.AttrString, String: v} }
func iAttr(k string, v int64) mim.Attr {
	return mim.Attr{Key: k, Type: mim.AttrInt, Int: v}
}
func fAttr(k string, v float64) mim.Attr { return mim.Attr{Key: k, Type: mim.AttrFloat, Float: v} }
func gAttr(k string, v int) mim.Attr {
	return mim.Attr{Key: k, Type: mim.AttrSubgraph, Subgraph: uint32(v)}
}
func callAttrs(name string, extra ...mim.Attr) []mim.Attr {
	return append([]mim.Attr{sAttr("name", name), iAttr("version", 1)}, extra...)
}

func f32t(shape []int, v ...float32) *ref.Tensor {
	t, _ := ref.New(mim.F32, shape)
	for i, x := range v {
		t.SetBits(i, uint64(math.Float32bits(x)))
	}
	return t
}

func f32bytes(v ...float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

func i64t(shape []int, v ...int64) *ref.Tensor {
	t, _ := ref.New(mim.I64, shape)
	for i, x := range v {
		t.SetBits(i, uint64(x))
	}
	return t
}

func f32s(t *ref.Tensor) []float64 {
	r := make([]float64, t.NumElems())
	for i := range r {
		r[i] = float64(math.Float32frombits(uint32(t.Bits(i))))
	}
	return r
}

// TestConformanceThroughGraphs runs every corpus case as a one-op graph
// written to a mim file, so the executor, the graph encoding and the
// kernels are checked together against the oracle.
func TestConformanceThroughGraphs(t *testing.T) {
	cases, err := conformance.LoadDir(filepath.Join("..", "testdata", "conformance", "primitives"))
	if err != nil {
		t.Fatal(err)
	}
	for i := range cases {
		c := &cases[i]
		b := newGB()
		in := map[string]*ref.Tensor{}
		var ids []uint32
		for k, x := range c.Inputs {
			dims := make([]mim.Dim, len(x.Shape))
			for j, d := range x.Shape {
				dims[j] = mim.Dim(d)
			}
			name := string(rune('a' + k))
			ids = append(ids, b.input(0, name, x.DType, dims...))
			in[name] = x
		}
		w := c.Output.Tensor
		dims := make([]mim.Dim, len(w.Shape))
		for j, d := range w.Shape {
			dims[j] = mim.Dim(d)
		}
		out := b.val("y", w.DType, mim.KindOutput, dims...)
		b.op(0, c.Opcode, ids, []uint32{out}, c.Attrs...)
		b.output(0, out)
		p := b.load(t)
		res, err := p.NewSession().Run(in)
		if err != nil {
			t.Errorf("%s: %v", c.ID(), err)
			continue
		}
		if err := c.Verify(res["y"], conformance.IEEE); err != nil {
			t.Errorf("%s: %v", c.ID(), err)
		}
	}
}

// TestKVCache follows the decode-step example of spec/primitives.md
// section 7: a variable cache updated by scatter, persisting across runs.
func TestKVCache(t *testing.T) {
	b := newGB()
	cache := b.stored("k_cache", mim.KindVariable, mim.F32, []uint64{1, 4, 2}, make([]byte, 32))
	pos := b.input(0, "pos", mim.I64, 1)
	kt := b.input(0, "k_t", mim.F32, 1, 1, 2)
	knew := b.val("k_new", mim.F32, mim.KindIntermediate, 1, 4, 2)
	sum := b.val("sum", mim.F32, mim.KindIntermediate, 1, 1, 1)
	b.op(0, mim.OpScatter, []uint32{cache, pos, kt}, []uint32{knew}, iAttr("axis", 1))
	b.op(0, mim.OpReduce, []uint32{knew}, []uint32{sum}, sAttr("mode", "add"), mim.Attr{Key: "axes", Type: mim.AttrInts, Ints: []int64{1, 2}})
	b.output(0, sum)
	b.g.Updates = []mim.Update{{Variable: cache, Value: knew}}
	p := b.load(t)

	s := p.NewSession()
	for step := 0; step < 3; step++ {
		v := float32(step + 1)
		out, err := s.Run(map[string]*ref.Tensor{"pos": i64t([]int{1}, int64(step)), "k_t": f32t([]int{1, 1, 2}, v, 10*v)})
		if err != nil {
			t.Fatal(err)
		}
		// Each run sees the cache as it was at the start of the run, plus
		// this step's row: the sum of 11 * (1 + ... + step+1).
		want := 11 * float64((step+1)*(step+2)/2)
		if got := f32s(out["sum"])[0]; got != want {
			t.Errorf("step %d: sum %v, want %v", step, got, want)
		}
	}
	c, _ := s.Variable("k_cache")
	if got := f32s(c); got[0] != 1 || got[1] != 10 || got[4] != 3 || got[6] != 0 {
		t.Errorf("cache = %v", got)
	}
	// A failed run changes no variable.
	if _, err := s.Run(map[string]*ref.Tensor{"pos": i64t([]int{1}, 3), "k_t": f32t([]int{1, 2, 2})}); err == nil {
		t.Fatal("accepted a wrongly shaped input")
	}
	if c2, _ := s.Variable("k_cache"); c2 != c {
		t.Error("a failed run changed the cache")
	}
	// A new session starts from the initial value.
	c3, _ := p.NewSession().Variable("k_cache")
	if f32s(c3)[0] != 0 {
		t.Error("a new session does not start from the initial value")
	}
}

// TestWhileIf sums 0..n-1 in a while loop, then picks the sum or its
// negation with an if on a predicate input.
func TestWhileIf(t *testing.T) {
	b := newGB()
	b.g.Subgraphs = make([]mim.Subgraph, 5)
	n := b.input(0, "n", mim.I64)
	neg := b.input(0, "negate", mim.Bool)
	zero := b.val("zero", mim.I64, mim.KindIntermediate)
	b.op(0, mim.OpConst, nil, []uint32{zero}, iAttr("value", 0))
	i1, acc1 := b.val("i1", mim.I64, 0), b.val("acc1", mim.I64, 0)
	b.op(0, mim.OpWhile, []uint32{zero, zero}, []uint32{i1, acc1}, gAttr("cond", 1), gAttr("body", 2))
	res := b.val("result", mim.I64, 0)
	b.op(0, mim.OpIf, []uint32{neg, acc1}, []uint32{res}, gAttr("then", 3), gAttr("else", 4))
	b.output(0, res)

	// cond: i < n, reading n from the enclosing graph
	ci, ca := b.val("ci", mim.I64, 0), b.val("ca", mim.I64, 0)
	b.g.Subgraphs[1].Inputs = []uint32{ci, ca}
	lt := b.val("lt", mim.Bool, 0)
	b.op(1, mim.OpCmplt, []uint32{ci, n}, []uint32{lt})
	b.output(1, lt)
	// body: (i + 1, acc + i)
	bi, ba := b.val("bi", mim.I64, 0), b.val("ba", mim.I64, 0)
	b.g.Subgraphs[2].Inputs = []uint32{bi, ba}
	one, ni, na := b.val("one", mim.I64, 0), b.val("ni", mim.I64, 0), b.val("na", mim.I64, 0)
	b.op(2, mim.OpConst, nil, []uint32{one}, iAttr("value", 1))
	b.op(2, mim.OpAdd, []uint32{bi, one}, []uint32{ni})
	b.op(2, mim.OpAdd, []uint32{ba, bi}, []uint32{na})
	b.output(2, ni)
	b.output(2, na)
	// then: -x; else: x + 0
	tx, ty := b.val("tx", mim.I64, 0), b.val("ty", mim.I64, 0)
	b.g.Subgraphs[3].Inputs = []uint32{tx}
	b.op(3, mim.OpNeg, []uint32{tx}, []uint32{ty})
	b.output(3, ty)
	ex, ez, ey := b.val("ex", mim.I64, 0), b.val("ez", mim.I64, 0), b.val("ey", mim.I64, 0)
	b.g.Subgraphs[4].Inputs = []uint32{ex}
	b.op(4, mim.OpConst, nil, []uint32{ez}, iAttr("value", 0))
	b.op(4, mim.OpAdd, []uint32{ex, ez}, []uint32{ey})
	b.output(4, ey)
	p := b.load(t)

	boolT := func(v bool) *ref.Tensor {
		x, _ := ref.New(mim.Bool, nil)
		if v {
			x.Data[0] = 1
		}
		return x
	}
	for _, c := range []struct {
		n    int64
		neg  bool
		want int64
	}{{10, false, 45}, {10, true, -45}, {0, false, 0}, {1, true, 0}} {
		out, err := p.NewSession().Run(map[string]*ref.Tensor{"n": i64t(nil, c.n), "negate": boolT(c.neg)})
		if err != nil {
			t.Fatal(err)
		}
		if got := int64(out["result"].Bits(0)); got != c.want {
			t.Errorf("n=%d negate=%v: %d, want %d", c.n, c.neg, got, c.want)
		}
	}
	p.MaxIterations = 5
	if _, err := p.NewSession().Run(map[string]*ref.Tensor{"n": i64t(nil, 100), "negate": boolT(false)}); err == nil ||
		!strings.Contains(err.Error(), "iterations") {
		t.Errorf("iteration limit: %v", err)
	}

	// A long loop stops when its context ends.
	p.MaxIterations = 0
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, err := p.NewSession().RunContext(ctx, map[string]*ref.Tensor{"n": i64t(nil, 1<<23), "negate": boolT(false)}); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("cancelled loop: %v", err)
	}
	if d := time.Since(start); d > 2*time.Second {
		t.Errorf("the cancelled loop ran on for %v", d)
	}
	done, stop := context.WithCancel(context.Background())
	stop()
	if _, err := p.NewSession().RunContext(done, map[string]*ref.Tensor{"n": i64t(nil, 1), "negate": boolT(false)}); !errors.Is(err, context.Canceled) {
		t.Errorf("run on a done context: %v", err)
	}
}

// TestMLP runs a small shape-polymorphic network with stdlib calls:
// softmax(rms_norm(silu(linear(x, w, bias)))) over a symbolic batch.
func TestMLP(t *testing.T) {
	b := newGB()
	B := b.sym("batch", 16)
	x := b.input(0, "x", mim.F32, B, 3)
	wv := []float32{0.5, -1, 2, 1.5, 0.25, -0.75}
	bv := []float32{0.1, -0.2}
	w := b.stored("w", mim.KindWeight, mim.F32, []uint64{2, 3}, f32bytes(wv...))
	bias := b.stored("bias", mim.KindWeight, mim.F32, []uint64{2}, f32bytes(bv...))
	h1, h2, h3, y := b.val("h1", mim.F32, 0, B, 2), b.val("h2", mim.F32, 0, B, 2), b.val("h3", mim.F32, 0, B, 2), b.val("y", mim.F32, 0, B, 2)
	b.op(0, mim.OpCall, []uint32{x, w, bias}, []uint32{h1}, callAttrs("linear")...)
	b.op(0, mim.OpCall, []uint32{h1}, []uint32{h2}, callAttrs("silu")...)
	b.op(0, mim.OpCall, []uint32{h2}, []uint32{h3}, callAttrs("rms_norm", fAttr("epsilon", 1e-5))...)
	b.op(0, mim.OpCall, []uint32{h3}, []uint32{y}, callAttrs("softmax", iAttr("axis", -1))...)
	b.output(0, y)
	p := b.load(t)

	for _, batch := range []int{1, 4} {
		xs := make([]float32, 3*batch)
		for i := range xs {
			xs[i] = float32(i%5) - 1.5
		}
		out, err := p.NewSession().Run(map[string]*ref.Tensor{"x": f32t([]int{batch, 3}, xs...)})
		if err != nil {
			t.Fatal(err)
		}
		got := f32s(out["y"])
		for r := 0; r < batch; r++ {
			var h [2]float64
			for j := 0; j < 2; j++ {
				s := float64(bv[j])
				for k := 0; k < 3; k++ {
					s += float64(xs[3*r+k]) * float64(wv[3*j+k])
				}
				h[j] = s / (1 + math.Exp(-s))
			}
			ms := (h[0]*h[0] + h[1]*h[1]) / 2
			for j := range h {
				h[j] /= math.Sqrt(ms + 1e-5)
			}
			m := math.Max(h[0], h[1])
			e0, e1 := math.Exp(h[0]-m), math.Exp(h[1]-m)
			want := []float64{e0 / (e0 + e1), e1 / (e0 + e1)}
			for j := 0; j < 2; j++ {
				if d := math.Abs(got[2*r+j] - want[j]); d > 1e-6 {
					t.Errorf("batch %d row %d col %d: %v, want %v", batch, r, j, got[2*r+j], want[j])
				}
			}
		}
	}
	if _, err := p.NewSession().Run(map[string]*ref.Tensor{"x": f32t([]int{17, 3})}); err == nil {
		t.Error("accepted a batch above the symbolic bound")
	}
}

// TestMatmulBatchBroadcast checks matmul.v1 with a broadcast batch dimension
// and integer operands, which wrap.
func TestMatmulBatchBroadcast(t *testing.T) {
	b := newGB()
	a := b.input(0, "a", mim.I64, 2, 1, 2)
	m := b.input(0, "m", mim.I64, 1, 2, 2)
	y := b.val("y", mim.I64, 0, 2, 1, 2)
	b.op(0, mim.OpCall, []uint32{a, m}, []uint32{y}, callAttrs("matmul")...)
	b.output(0, y)
	p := b.load(t)
	out, err := p.NewSession().Run(map[string]*ref.Tensor{
		"a": i64t([]int{2, 1, 2}, 1, 2, 3, math.MaxInt64),
		"m": i64t([]int{1, 2, 2}, 5, 6, 7, 2),
	})
	if err != nil {
		t.Fatal(err)
	}
	mx := int64(math.MaxInt64)
	want := []int64{19, 10, 15 + 7*mx, 18 + 2*mx} // wraps modulo 2^64
	for i, w := range want {
		if got := int64(out["y"].Bits(i)); got != w {
			t.Errorf("y[%d] = %d, want %d", i, got, w)
		}
	}
}

// TestLocalComposite calls a model-local composite from the composites
// table.
func TestLocalComposite(t *testing.T) {
	b := newGB()
	b.g.Subgraphs = make([]mim.Subgraph, 2)
	x := b.input(0, "x", mim.F32, 3)
	y := b.val("y", mim.F32, 0, 3)
	b.op(0, mim.OpCall, []uint32{x}, []uint32{y}, sAttr("name", "acme/double"), iAttr("version", 2))
	b.output(0, y)
	cx, cy := b.val("cx", mim.F32, 0, 3), b.val("cy", mim.F32, 0, 3)
	b.g.Subgraphs[1].Inputs = []uint32{cx}
	b.op(1, mim.OpAdd, []uint32{cx, cx}, []uint32{cy})
	b.output(1, cy)
	b.g.Composites = []mim.Composite{{Name: "acme/double", Version: 2, Subgraph: 1}}
	p := b.load(t)
	out, err := p.NewSession().Run(map[string]*ref.Tensor{"x": f32t([]int{3}, 1, -2, 3.5)})
	if err != nil {
		t.Fatal(err)
	}
	if got := f32s(out["y"]); got[0] != 2 || got[1] != -4 || got[2] != 7 {
		t.Errorf("y = %v", got)
	}
}

func TestRunErrors(t *testing.T) {
	b := newGB()
	S := b.sym("seq", -1)
	x := b.input(0, "x", mim.F32, S)
	z := b.input(0, "z", mim.F32, S)
	y, u := b.val("y", mim.F32, 0, S), b.val("u", mim.F32, 0, S)
	b.op(0, mim.OpAdd, []uint32{x, z}, []uint32{y})
	b.op(0, mim.OpCall, []uint32{y}, []uint32{u}, callAttrs("softplus")...)
	b.output(0, u)
	p := b.load(t)
	s := p.NewSession()
	cases := map[string]map[string]*ref.Tensor{
		"missing input":      {"x": f32t([]int{2})},
		"extra input":        {"x": f32t([]int{2}), "z": f32t([]int{2}), "q": f32t([]int{2})},
		"inconsistent dims":  {"x": f32t([]int{2}), "z": f32t([]int{3})},
		"wrong dtype":        {"x": f32t([]int{2}), "z": i64t([]int{2})},
		"unsupported stdlib": {"x": f32t([]int{2}), "z": f32t([]int{2})},
	}
	for name, in := range cases {
		if _, err := s.Run(in); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// TestZeroVariable runs a cache that starts as zeros and is not stored in
// the file: each session gets its own zeroed buffer.
func TestZeroVariable(t *testing.T) {
	b := newGB()
	cache := b.val("k_cache", mim.F32, mim.KindVariable, 1, 4, 2)
	b.g.Values[cache].Zero = true
	pos := b.input(0, "pos", mim.I64, 1)
	kt := b.input(0, "k_t", mim.F32, 1, 1, 2)
	knew := b.val("k_new", mim.F32, mim.KindIntermediate, 1, 4, 2)
	b.op(0, mim.OpScatter, []uint32{cache, pos, kt}, []uint32{knew}, iAttr("axis", 1))
	b.g.Updates = []mim.Update{{Variable: cache, Value: knew}}
	p := b.load(t)
	p.FastPaths = true

	s1, s2 := p.NewSession(), p.NewSession()
	if _, err := s1.Run(map[string]*ref.Tensor{"pos": i64t([]int{1}, 2), "k_t": f32t([]int{1, 1, 2}, 5, 6)}); err != nil {
		t.Fatal(err)
	}
	c1, _ := s1.Variable("k_cache")
	c2, _ := s2.Variable("k_cache")
	if got := f32s(c1); got[4] != 5 || got[5] != 6 || got[0] != 0 {
		t.Errorf("cache = %v", got)
	}
	for _, v := range f32s(c2) {
		if v != 0 {
			t.Fatalf("another session's cache changed: %v", f32s(c2))
		}
	}
}

// TestMemoryBudget checks that MaxBytes bounds the zero variables of a
// session and the values a run computes, failing before allocating.
func TestMemoryBudget(t *testing.T) {
	// 8 GiB of zero variable is over the default budget.
	b := newGB()
	cache := b.val("k_cache", mim.F32, mim.KindVariable, 1<<31)
	b.g.Values[cache].Zero = true
	x := b.input(0, "x", mim.F32, 1<<31)
	b.g.Updates = []mim.Update{{Variable: cache, Value: x}}
	p, err := New(b.g, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.StartSession(); err == nil || !strings.Contains(err.Error(), "budget") {
		t.Fatalf("StartSession: got %v", err)
	}
	if _, err := p.NewSession().Run(map[string]*ref.Tensor{}); err == nil || !strings.Contains(err.Error(), "budget") {
		t.Fatalf("Run: got %v", err)
	}

	// A configured budget applies to small variables too, and a negative
	// one lifts it.
	b = newGB()
	cache = b.val("k_cache", mim.F32, mim.KindVariable, 1<<10)
	b.g.Values[cache].Zero = true
	x = b.input(0, "x", mim.F32, 1<<10)
	b.g.Updates = []mim.Update{{Variable: cache, Value: x}}
	p = b.load(t)
	p.MaxBytes = 1 << 10
	if _, err := p.StartSession(); err == nil {
		t.Fatal("a 4 KiB variable fit in a 1 KiB budget")
	}
	p.MaxBytes = -1
	if _, err := p.StartSession(); err != nil {
		t.Fatal(err)
	}

	// An intermediate value declared far larger than its input is refused
	// before it is allocated.
	b = newGB()
	x = b.input(0, "x", mim.F32, 1)
	y := b.val("y", mim.F32, mim.KindIntermediate, 1<<31)
	b.op(0, mim.OpExpand, []uint32{x}, []uint32{y})
	b.output(0, y)
	p = b.load(t)
	if _, err := p.NewSession().Run(map[string]*ref.Tensor{"x": f32t([]int{1}, 1)}); err == nil || !strings.Contains(err.Error(), "budget") {
		t.Fatalf("expand: got %v", err)
	}

	// Values the run still holds count against the budget: two 1 KiB
	// outputs do not fit in 1.5 KiB, though each does.
	b = newGB()
	x = b.input(0, "x", mim.F32, 256)
	a := b.val("a", mim.F32, mim.KindIntermediate, 256)
	c := b.val("c", mim.F32, mim.KindIntermediate, 256)
	b.op(0, mim.OpAdd, []uint32{x, x}, []uint32{a})
	b.op(0, mim.OpAdd, []uint32{a, a}, []uint32{c})
	b.output(0, a)
	b.output(0, c)
	p = b.load(t)
	in := map[string]*ref.Tensor{"x": f32t([]int{256})}
	p.MaxBytes = 3 << 9
	if _, err := p.NewSession().Run(in); err == nil || !strings.Contains(err.Error(), "budget") {
		t.Fatalf("held values: got %v", err)
	}
	p.MaxBytes = 2 << 10
	if _, err := p.NewSession().Run(in); err != nil {
		t.Fatal(err)
	}
}

func TestZeroVariableTooLarge(t *testing.T) {
	b := newGB()
	cache := b.val("k_cache", mim.F32, mim.KindVariable, 1<<30, 1<<30)
	b.g.Values[cache].Zero = true
	x := b.input(0, "x", mim.F32, 1<<30, 1<<30)
	b.g.Updates = []mim.Update{{Variable: cache, Value: x}}
	if _, err := New(b.g, nil); err == nil || !strings.Contains(err.Error(), "larger than") {
		t.Fatalf("got %v", err)
	}
}

// BenchmarkRunMLP runs one 256-wide linear and silu over a batch of 16:
// the interpreter's cost per op on a small model, on its fast paths.
func BenchmarkRunMLP(b *testing.B) {
	g := newGB()
	B := g.sym("batch", 16)
	x := g.input(0, "x", mim.F32, B, 256)
	wv := make([]float32, 256*256)
	for i := range wv {
		wv[i] = float32(i%7) / 7
	}
	w := g.stored("w", mim.KindWeight, mim.F32, []uint64{256, 256}, f32bytes(wv...))
	bias := g.stored("bias", mim.KindWeight, mim.F32, []uint64{256}, f32bytes(make([]float32, 256)...))
	h, y := g.val("h", mim.F32, 0, B, 256), g.val("y", mim.F32, 0, B, 256)
	g.op(0, mim.OpCall, []uint32{x, w, bias}, []uint32{h}, callAttrs("linear")...)
	g.op(0, mim.OpCall, []uint32{h}, []uint32{y}, callAttrs("silu")...)
	g.output(0, y)
	p := g.load(b)
	p.FastPaths = true
	in := map[string]*ref.Tensor{"x": f32t([]int{16, 256}, make([]float32, 16*256)...)}
	s := p.NewSession()
	b.ResetTimer()
	for range b.N {
		if _, err := s.Run(in); err != nil {
			b.Fatal(err)
		}
	}
}
