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
	"math"
	"math/rand"
	"testing"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/ref"
)

// TestScatterInPlace runs a cache update with and without the primitive
// fast paths, which write the cache in place from the second run on, and
// checks that a run failing after the scatter leaves the cache unchanged.
func TestScatterInPlace(t *testing.T) {
	b := newGB()
	b.g.Subgraphs = make([]mim.Subgraph, 3)
	cache := b.stored("k_cache", mim.KindVariable, mim.F32, []uint64{1, 4, 2}, make([]byte, 32))
	pos := b.input(0, "pos", mim.I64, 1)
	kt := b.input(0, "k_t", mim.F32, 1, 1, 2)
	fail := b.input(0, "fail", mim.Bool)
	knew := b.val("k_new", mim.F32, mim.KindIntermediate, 1, 4, 2)
	sum := b.val("sum", mim.F32, mim.KindIntermediate, 1, 1, 1)
	res := b.val("res", mim.F32, mim.KindIntermediate, 1, 1, 1)
	b.op(0, mim.OpScatter, []uint32{cache, pos, kt}, []uint32{knew}, iAttr("axis", 1))
	b.op(0, mim.OpReduce, []uint32{knew}, []uint32{sum}, sAttr("mode", "add"), mim.Attr{Key: "axes", Type: mim.AttrInts, Ints: []int64{1, 2}})
	b.op(0, mim.OpIf, []uint32{fail, sum}, []uint32{res}, gAttr("then", 1), gAttr("else", 2))
	b.output(0, res)
	// then: a composite this executor does not have, so the run fails
	// after the scatter. else: pass the sum through.
	subIn := func(si int, name string) uint32 {
		id := b.val(name, mim.F32, mim.KindIntermediate, 1, 1, 1)
		b.g.Subgraphs[si].Inputs = append(b.g.Subgraphs[si].Inputs, id)
		return id
	}
	x1 := subIn(1, "x1")
	y1 := b.val("y1", mim.F32, mim.KindIntermediate, 1, 1, 1)
	b.op(1, mim.OpCall, []uint32{x1}, []uint32{y1}, callAttrs("no_such_composite")...)
	b.output(1, y1)
	x2 := subIn(2, "x2")
	b.output(2, x2)
	b.g.Updates = []mim.Update{{Variable: cache, Value: knew}}
	p := b.load(t)
	if len(p.inplace) != 1 {
		t.Fatalf("%d scatters may run in place, want 1", len(p.inplace))
	}

	no := &ref.Tensor{DType: mim.Bool, Shape: []int{}, Data: []byte{0}}
	yes := &ref.Tensor{DType: mim.Bool, Shape: []int{}, Data: []byte{1}}
	step := func(s *Session, i int64, f *ref.Tensor) (map[string]*ref.Tensor, error) {
		v := float32(i + 1)
		return s.Run(map[string]*ref.Tensor{"pos": i64t([]int{1}, i%4), "k_t": f32t([]int{1, 1, 2}, v, -v), "fail": f})
	}
	p.FastPaths = false
	slow := p.NewSession()
	p.FastPaths = true
	fast := p.NewSession()
	for i := int64(0); i < 6; i++ {
		p.FastPaths = false
		want, err := step(slow, i, no)
		if err != nil {
			t.Fatal(err)
		}
		p.FastPaths = true
		before, _ := fast.Variable("k_cache")
		saved := append([]byte{}, before.Data...)
		if _, err := step(fast, i+100, yes); err == nil {
			t.Fatal("the failing branch did not fail")
		}
		if after, _ := fast.Variable("k_cache"); after != before || !bytes.Equal(after.Data, saved) {
			t.Fatalf("step %d: a failed run changed the cache", i)
		}
		got, err := step(fast, i, no)
		if err != nil {
			t.Fatal(err)
		}
		if f32s(got["res"])[0] != f32s(want["res"])[0] {
			t.Errorf("step %d: in place %v, reference %v", i, f32s(got["res"]), f32s(want["res"]))
		}
	}
	a, _ := slow.Variable("k_cache")
	c, _ := fast.Variable("k_cache")
	if !bytes.Equal(a.Data, c.Data) {
		t.Errorf("caches differ: %v and %v", f32s(a), f32s(c))
	}
	// The file's initial value was never written.
	if init, _ := p.NewSession().Variable("k_cache"); !bytes.Equal(init.Data, make([]byte, 32)) {
		t.Error("the initial value changed")
	}
}

func TestScatterInPlaceNeedsSoleUse(t *testing.T) {
	b := newGB()
	cache := b.stored("k_cache", mim.KindVariable, mim.F32, []uint64{1, 4, 2}, make([]byte, 32))
	pos := b.input(0, "pos", mim.I64, 1)
	kt := b.input(0, "k_t", mim.F32, 1, 1, 2)
	knew := b.val("k_new", mim.F32, mim.KindIntermediate, 1, 4, 2)
	old := b.val("old", mim.F32, mim.KindIntermediate, 1, 4, 2)
	b.op(0, mim.OpScatter, []uint32{cache, pos, kt}, []uint32{knew}, iAttr("axis", 1))
	b.op(0, mim.OpAdd, []uint32{cache, knew}, []uint32{old}) // a second use of the old value
	b.output(0, old)
	b.g.Updates = []mim.Update{{Variable: cache, Value: knew}}
	if p := b.load(t); len(p.inplace) != 0 {
		t.Error("a scatter whose input has another use may run in place")
	}
}

func TestFastPermute(t *testing.T) {
	perm := func(p ...int64) *mim.Op {
		return &mim.Op{Opcode: mim.OpPermute, Attrs: []mim.Attr{{Key: "perm", Type: mim.AttrInts, Ints: p}}}
	}
	// Every permutation of a rank-4 shape, with axes of size 1 among
	// them, for elements of 4, 2 and 1 bytes; and the empty tensor.
	var perms [][]int64
	var gen func(p []int64, used int)
	gen = func(p []int64, used int) {
		if len(p) == 4 {
			perms = append(perms, append([]int64{}, p...))
			return
		}
		for a := int64(0); a < 4; a++ {
			if used&(1<<a) == 0 {
				gen(append(p, a), used|1<<a)
			}
		}
	}
	gen(nil, 0)
	for _, dt := range []mim.DType{mim.F32, mim.F16, mim.Bool} {
		for _, xs := range [][]int{{1, 3, 1, 2}, {2, 3, 4, 5}, {3, 0, 2, 1}} {
			x, err := ref.New(dt, xs)
			if err != nil {
				t.Fatal(err)
			}
			for i := range x.Data {
				x.Data[i] = byte(i*7 + 1)
				if dt == mim.Bool {
					x.Data[i] &= 1
				}
			}
			for _, p := range perms {
				shape := []int{xs[p[0]], xs[p[1]], xs[p[2]], xs[p[3]]}
				got, ok := fastPermute(perm(p...), []*ref.Tensor{x}, dt, shape)
				if !ok {
					t.Errorf("%s %v perm %v: not handled", dt, xs, p)
					continue
				}
				want, err := ref.Eval(mim.OpPermute, perm(p...).Attrs, []*ref.Tensor{x}, dt, shape)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got.Data, want.Data) || !sameInts(got.Shape, want.Shape) {
					t.Errorf("%s %v perm %v: wrong result", dt, xs, p)
				}
			}
		}
	}
	q, err := ref.New(mim.Q8_0, []int{2, 32})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := fastPermute(perm(1, 0), []*ref.Tensor{q}, mim.Q8_0, []int{32, 2}); ok {
		t.Error("handled a block dtype")
	}
}

// TestFastCastF16 checks the cast from f32 to f16 against the reference,
// bit for bit, NaNs and halfway cases included.
func TestFastCastF16(t *testing.T) {
	bits := []uint32{0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFC12345, 0x7F802015,
		0x477FEFFF, 0x477FF000, 0x47800000, 0x33000000, 0x33000001, 0x387FC000, 0x38800000, 0x3F801000, 0x3F803000}
	rng := rand.New(rand.NewSource(5))
	for len(bits) < 10000 {
		bits = append(bits, rng.Uint32())
	}
	x, _ := ref.New(mim.F32, []int{len(bits)})
	for i, b := range bits {
		x.SetBits(i, uint64(b))
	}
	attrs := []mim.Attr{{Key: "to", Type: mim.AttrDType, DType: mim.F16}}
	want, err := ref.Eval(mim.OpCast, attrs, []*ref.Tensor{x}, mim.F16, x.Shape)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := fastCastF16([]*ref.Tensor{x}, mim.F16, x.Shape)
	if !ok {
		t.Fatal("fastCastF16 refused f32 to f16")
	}
	for i := range bits {
		if got.Bits(i) != want.Bits(i) {
			t.Errorf("%08x: got %04x, reference %04x", bits[i], got.Bits(i), want.Bits(i))
		}
	}
}

// TestFastBroadcast checks the broadcasting binary fast paths against the
// reference kernels, bit for bit, on operands that broadcast along
// various axes.
func TestFastBroadcast(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	shapes := [][3][]int{
		{{1, 1, 5, 1}, {1, 1, 1, 7}, {1, 1, 5, 7}},
		{{2, 3, 4, 5}, {1, 3, 1, 1}, {2, 3, 4, 5}},
		{{1, 3, 1, 1}, {2, 3, 4, 5}, {2, 3, 4, 5}},
		{{2, 1, 4}, {2, 3, 1}, {2, 3, 4}},
		{{1, 1, 1, 1}, {1, 1, 6, 9}, {1, 1, 6, 9}},
		{{3, 1}, {1, 1}, {3, 1}},
		{{4, 6}, {4, 6}, {4, 6}},
		{{0, 3}, {1, 3}, {0, 3}},
		// Large enough to be cut into pieces.
		{{2, 100, 100}, {2, 100, 100}, {2, 100, 100}},
		{{1, 3, 90, 90}, {1, 3, 1, 1}, {1, 3, 90, 90}},
		{{1, 1}, {9000, 1}, {9000, 1}},
	}
	specials := []float32{0, float32(math.Copysign(0, -1)), float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), 1, -1}
	fill := func(dt mim.DType, shape []int) *ref.Tensor {
		x, _ := ref.New(dt, shape)
		for i := range x.NumElems() {
			switch dt {
			case mim.F32:
				v := float32(rng.NormFloat64())
				if rng.Intn(3) == 0 {
					v = specials[rng.Intn(len(specials))]
				}
				x.SetBits(i, uint64(math.Float32bits(v)))
			case mim.Bool:
				x.SetBits(i, uint64(rng.Intn(2)))
			default:
				x.SetBits(i, uint64(int64(rng.Intn(9)-4)))
			}
		}
		return x
	}
	cases := []struct {
		op      mim.Opcode
		in, out mim.DType
	}{
		{mim.OpAdd, mim.F32, mim.F32}, {mim.OpMul, mim.F32, mim.F32}, {mim.OpMax, mim.F32, mim.F32},
		{mim.OpCmplt, mim.I64, mim.Bool}, {mim.OpCmpeq, mim.I64, mim.Bool},
		{mim.OpCmplt, mim.I32, mim.Bool}, {mim.OpCmpeq, mim.I32, mim.Bool},
		{mim.OpAnd, mim.Bool, mim.Bool}, {mim.OpOr, mim.Bool, mim.Bool},
		{mim.OpXor, mim.Bool, mim.Bool}, {mim.OpCmpeq, mim.Bool, mim.Bool},
	}
	for _, c := range cases {
		for _, s := range shapes {
			a, b := fill(c.in, s[0]), fill(c.in, s[1])
			got, ok := fastBroadcast(c.op, []*ref.Tensor{a, b}, c.out, s[2])
			if !ok {
				t.Fatalf("%s %s %v %v: not handled", c.op, c.in, s[0], s[1])
			}
			want, err := ref.Eval(c.op, nil, []*ref.Tensor{a, b}, c.out, s[2])
			if err != nil {
				t.Fatal(err)
			}
			if !sameInts(got.Shape, want.Shape) || !bytes.Equal(got.Data, want.Data) {
				t.Errorf("%s %s %v %v: got %v, want %v", c.op, c.in, s[0], s[1], got.Data, want.Data)
			}
		}
	}
	// Shapes that do not broadcast and dtypes without a fast path fall back.
	x := fill(mim.F32, []int{2, 3})
	if _, ok := fastBroadcast(mim.OpAdd, []*ref.Tensor{x, fill(mim.F32, []int{3, 2})}, mim.F32, []int{2, 3}); ok {
		t.Error("mismatched shapes were handled")
	}
	if _, ok := fastBroadcast(mim.OpPow, []*ref.Tensor{x, x}, mim.F32, []int{2, 3}); ok {
		t.Error("pow was handled")
	}
}

// TestReshapeView checks that a reshape shares its input's bytes only when
// nothing can write them later: here the reshape of a cache that a scatter
// updates in place is an output, and must keep its value after the next
// run writes the cache.
func TestReshapeView(t *testing.T) {
	b := newGB()
	cache := b.stored("k_cache", mim.KindVariable, mim.F32, []uint64{1, 4, 2}, make([]byte, 32))
	pos := b.input(0, "pos", mim.I64, 1)
	kt := b.input(0, "k_t", mim.F32, 1, 1, 2)
	knew := b.val("k_new", mim.F32, mim.KindIntermediate, 1, 4, 2)
	flat := b.val("flat", mim.F32, mim.KindIntermediate, 8)
	back := b.val("back", mim.F32, mim.KindIntermediate, 4, 2)
	twice := b.val("twice", mim.F32, mim.KindIntermediate, 4, 2)
	shape := func(d ...int64) mim.Attr { return mim.Attr{Key: "shape", Type: mim.AttrInts, Ints: d} }
	b.op(0, mim.OpScatter, []uint32{cache, pos, kt}, []uint32{knew}, iAttr("axis", 1))
	b.op(0, mim.OpReshape, []uint32{knew}, []uint32{flat}, shape(8))
	b.op(0, mim.OpReshape, []uint32{flat}, []uint32{back}, shape(4, -1))
	b.op(0, mim.OpAdd, []uint32{back, back}, []uint32{twice})
	b.output(0, flat)
	b.output(0, twice)
	b.g.Updates = []mim.Update{{Variable: cache, Value: knew}}
	p := b.load(t)
	p.FastPaths = true
	if len(p.inplace) != 1 {
		t.Fatalf("%d scatters may run in place, want 1", len(p.inplace))
	}
	s := p.NewSession()
	var outs []*ref.Tensor
	for i := int64(0); i < 4; i++ {
		v := float32(i + 1)
		got, err := s.Run(map[string]*ref.Tensor{"pos": i64t([]int{1}, i), "k_t": f32t([]int{1, 1, 2}, v, -v)})
		if err != nil {
			t.Fatal(err)
		}
		outs = append(outs, got["flat"])
		want := make([]float64, 8)
		for j := int64(0); j <= i; j++ {
			want[2*j], want[2*j+1] = float64(j+1), -float64(j+1)
		}
		for r, o := range outs {
			w := want
			if r < int(i) {
				w = make([]float64, 8)
				for j := 0; j <= r; j++ {
					w[2*j], w[2*j+1] = float64(j+1), -float64(j+1)
				}
			}
			for k, x := range f32s(o) {
				if x != w[k] {
					t.Fatalf("after run %d, the output of run %d is %v, want %v", i, r, f32s(o), w)
				}
			}
		}
		for k, x := range f32s(got["twice"]) {
			if x != 2*want[k] {
				t.Fatalf("run %d: twice is %v", i, f32s(got["twice"]))
			}
		}
	}
}

// TestFastShrinkPad checks shrink and pad against the reference kernels,
// bit for bit, over boxes that cover whole trailing axes or not.
func TestFastShrinkPad(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	for iter := range 300 {
		rank := 1 + rng.Intn(5)
		shape := make([]int, rank)
		for i := range shape {
			shape[i] = 1 + rng.Intn(5)
		}
		dt := []mim.DType{mim.F32, mim.Bool, mim.I64, mim.F16}[iter%4]
		x, _ := ref.New(dt, shape)
		rng.Read(x.Data)
		if dt == mim.Bool {
			for i := range x.Data {
				x.Data[i] &= 1
			}
		}
		begin, end, out := make([]int64, rank), make([]int64, rank), make([]int, rank)
		pads, padded := make([]int64, 2*rank), make([]int, rank)
		for i, d := range shape {
			b := rng.Intn(d + 1)
			e := b + rng.Intn(d-b+1)
			if rng.Intn(3) == 0 {
				b, e = 0, d
			}
			begin[i], end[i], out[i] = int64(b), int64(e), e-b
			if e == d && rng.Intn(2) == 0 {
				end[i] = -1
			}
			lo, hi := rng.Intn(3), rng.Intn(3)
			if rng.Intn(3) == 0 {
				lo, hi = 0, 0
			}
			pads[2*i], pads[2*i+1], padded[i] = int64(lo), int64(hi), d+lo+hi
		}
		sh := &mim.Op{Opcode: mim.OpShrink, Attrs: []mim.Attr{
			{Key: "begin", Type: mim.AttrInts, Ints: begin}, {Key: "end", Type: mim.AttrInts, Ints: end}}}
		got, ok := fastShrink(sh, []*ref.Tensor{x}, dt, out)
		want, err := ref.Eval(mim.OpShrink, sh.Attrs, []*ref.Tensor{x}, dt, out)
		if err != nil || !ok || !bytes.Equal(got.Data, want.Data) {
			t.Fatalf("shrink %v of %v: handled %v, %v", begin, shape, ok, err)
		}
		pa := []mim.Attr{{Key: "pads", Type: mim.AttrInts, Ints: pads}}
		if dt == mim.F32 && iter%8 == 0 {
			pa = append(pa, mim.Attr{Key: "value", Type: mim.AttrFloat, Float: math.Inf(-1)})
		}
		pd := &mim.Op{Opcode: mim.OpPad, Attrs: pa}
		got, ok = fastPad(pd, []*ref.Tensor{x}, dt, padded)
		want, err = ref.Eval(mim.OpPad, pd.Attrs, []*ref.Tensor{x}, dt, padded)
		if err != nil || !ok || !bytes.Equal(got.Data, want.Data) {
			t.Fatalf("pad %v of %v: handled %v, %v", pads, shape, ok, err)
		}
	}
}
