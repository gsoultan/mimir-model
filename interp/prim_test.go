// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mlm, created and owned by Gembit Soultan Shirazi.
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
	"math/rand"
	"testing"

	"github.com/gsoultan/mlm"
	"github.com/gsoultan/mlm/ref"
)

// TestScatterInPlace runs a cache update with and without the primitive
// fast paths, which write the cache in place from the second run on, and
// checks that a run failing after the scatter leaves the cache unchanged.
func TestScatterInPlace(t *testing.T) {
	b := newGB()
	b.g.Subgraphs = make([]mlm.Subgraph, 3)
	cache := b.stored("k_cache", mlm.KindVariable, mlm.F32, []uint64{1, 4, 2}, make([]byte, 32))
	pos := b.input(0, "pos", mlm.I64, 1)
	kt := b.input(0, "k_t", mlm.F32, 1, 1, 2)
	fail := b.input(0, "fail", mlm.Bool)
	knew := b.val("k_new", mlm.F32, mlm.KindIntermediate, 1, 4, 2)
	sum := b.val("sum", mlm.F32, mlm.KindIntermediate, 1, 1, 1)
	res := b.val("res", mlm.F32, mlm.KindIntermediate, 1, 1, 1)
	b.op(0, mlm.OpScatter, []uint32{cache, pos, kt}, []uint32{knew}, iAttr("axis", 1))
	b.op(0, mlm.OpReduce, []uint32{knew}, []uint32{sum}, sAttr("mode", "add"), mlm.Attr{Key: "axes", Type: mlm.AttrInts, Ints: []int64{1, 2}})
	b.op(0, mlm.OpIf, []uint32{fail, sum}, []uint32{res}, gAttr("then", 1), gAttr("else", 2))
	b.output(0, res)
	// then: a composite this executor does not have, so the run fails
	// after the scatter. else: pass the sum through.
	subIn := func(si int, name string) uint32 {
		id := b.val(name, mlm.F32, mlm.KindIntermediate, 1, 1, 1)
		b.g.Subgraphs[si].Inputs = append(b.g.Subgraphs[si].Inputs, id)
		return id
	}
	x1 := subIn(1, "x1")
	y1 := b.val("y1", mlm.F32, mlm.KindIntermediate, 1, 1, 1)
	b.op(1, mlm.OpCall, []uint32{x1}, []uint32{y1}, callAttrs("no_such_composite")...)
	b.output(1, y1)
	x2 := subIn(2, "x2")
	b.output(2, x2)
	b.g.Updates = []mlm.Update{{Variable: cache, Value: knew}}
	p := b.load(t)
	if len(p.inplace) != 1 {
		t.Fatalf("%d scatters may run in place, want 1", len(p.inplace))
	}

	no := &ref.Tensor{DType: mlm.Bool, Shape: []int{}, Data: []byte{0}}
	yes := &ref.Tensor{DType: mlm.Bool, Shape: []int{}, Data: []byte{1}}
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
	cache := b.stored("k_cache", mlm.KindVariable, mlm.F32, []uint64{1, 4, 2}, make([]byte, 32))
	pos := b.input(0, "pos", mlm.I64, 1)
	kt := b.input(0, "k_t", mlm.F32, 1, 1, 2)
	knew := b.val("k_new", mlm.F32, mlm.KindIntermediate, 1, 4, 2)
	old := b.val("old", mlm.F32, mlm.KindIntermediate, 1, 4, 2)
	b.op(0, mlm.OpScatter, []uint32{cache, pos, kt}, []uint32{knew}, iAttr("axis", 1))
	b.op(0, mlm.OpAdd, []uint32{cache, knew}, []uint32{old}) // a second use of the old value
	b.output(0, old)
	b.g.Updates = []mlm.Update{{Variable: cache, Value: knew}}
	if p := b.load(t); len(p.inplace) != 0 {
		t.Error("a scatter whose input has another use may run in place")
	}
}

func TestFastPermute(t *testing.T) {
	perm := func(p ...int64) *mlm.Op {
		return &mlm.Op{Opcode: mlm.OpPermute, Attrs: []mlm.Attr{{Key: "perm", Type: mlm.AttrInts, Ints: p}}}
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
	for _, dt := range []mlm.DType{mlm.F32, mlm.F16, mlm.Bool} {
		for _, xs := range [][]int{{1, 3, 1, 2}, {2, 3, 4, 5}, {3, 0, 2, 1}} {
			x, err := ref.New(dt, xs)
			if err != nil {
				t.Fatal(err)
			}
			for i := range x.Data {
				x.Data[i] = byte(i*7 + 1)
				if dt == mlm.Bool {
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
				want, err := ref.Eval(mlm.OpPermute, perm(p...).Attrs, []*ref.Tensor{x}, dt, shape)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got.Data, want.Data) || !sameInts(got.Shape, want.Shape) {
					t.Errorf("%s %v perm %v: wrong result", dt, xs, p)
				}
			}
		}
	}
	q, err := ref.New(mlm.Q8_0, []int{2, 32})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := fastPermute(perm(1, 0), []*ref.Tensor{q}, mlm.Q8_0, []int{32, 2}); ok {
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
	x, _ := ref.New(mlm.F32, []int{len(bits)})
	for i, b := range bits {
		x.SetBits(i, uint64(b))
	}
	attrs := []mlm.Attr{{Key: "to", Type: mlm.AttrDType, DType: mlm.F16}}
	want, err := ref.Eval(mlm.OpCast, attrs, []*ref.Tensor{x}, mlm.F16, x.Shape)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := fastCastF16([]*ref.Tensor{x}, mlm.F16, x.Shape)
	if !ok {
		t.Fatal("fastCastF16 refused f32 to f16")
	}
	for i := range bits {
		if got.Bits(i) != want.Bits(i) {
			t.Errorf("%08x: got %04x, reference %04x", bits[i], got.Bits(i), want.Bits(i))
		}
	}
}

// TestBoolBinary checks the comparison and logical fast paths against the
// reference kernels, bit for bit, with operands broadcast along every
// combination of axes.
func TestBoolBinary(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	out := []int{2, 3, 5}
	cases := []struct {
		opc mlm.Opcode
		dt  mlm.DType
	}{
		{mlm.OpCmplt, mlm.I64}, {mlm.OpCmpeq, mlm.I64}, {mlm.OpCmplt, mlm.I32}, {mlm.OpCmpeq, mlm.I32},
		{mlm.OpAnd, mlm.Bool}, {mlm.OpOr, mlm.Bool}, {mlm.OpXor, mlm.Bool}, {mlm.OpCmpeq, mlm.Bool},
	}
	random := func(dt mlm.DType, shape []int) *ref.Tensor {
		x, err := ref.New(dt, shape)
		if err != nil {
			t.Fatal(err)
		}
		for i := range x.NumElems() {
			switch dt {
			case mlm.Bool:
				x.SetBits(i, uint64(rng.Intn(2)))
			default:
				x.SetBits(i, uint64(int64(rng.Intn(7)-3)))
			}
		}
		return x
	}
	for _, c := range cases {
		for ma := range 8 {
			for mb := range 8 {
				if ma&mb != 0 {
					continue // the output would have size 1 there
				}
				sa, sb := append([]int{}, out...), append([]int{}, out...)
				for k := range out {
					if ma>>k&1 != 0 {
						sa[k] = 1
					}
					if mb>>k&1 != 0 {
						sb[k] = 1
					}
				}
				in := []*ref.Tensor{random(c.dt, sa), random(c.dt, sb)}
				want, err := ref.Eval(c.opc, nil, in, mlm.Bool, out)
				if err != nil {
					t.Fatal(err)
				}
				got, ok := fastBoolBinary(c.opc, in, mlm.Bool, out)
				if !ok {
					t.Fatalf("%s %s %v %v: not handled", c.opc, c.dt, sa, sb)
				}
				if !bytes.Equal(got.Data, want.Data) || !sameInts(got.Shape, want.Shape) {
					t.Errorf("%s %s %v %v: %v, want %v", c.opc, c.dt, sa, sb, got.Data, want.Data)
				}
			}
		}
	}
	if _, ok := fastBoolBinary(mlm.OpCmplt, []*ref.Tensor{random(mlm.Bool, out), random(mlm.Bool, out)}, mlm.Bool, out); ok {
		t.Error("cmplt on bool tensors handled")
	}
}
