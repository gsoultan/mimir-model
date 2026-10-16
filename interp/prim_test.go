// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mmf, created and owned by Gembit Soultan Shirazi.
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
	"testing"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/ref"
)

// TestScatterInPlace runs a cache update with and without the primitive
// fast paths, which write the cache in place from the second run on, and
// checks that a run failing after the scatter leaves the cache unchanged.
func TestScatterInPlace(t *testing.T) {
	b := newGB()
	b.g.Subgraphs = make([]mmf.Subgraph, 3)
	cache := b.stored("k_cache", mmf.KindVariable, mmf.F32, []uint64{1, 4, 2}, make([]byte, 32))
	pos := b.input(0, "pos", mmf.I64, 1)
	kt := b.input(0, "k_t", mmf.F32, 1, 1, 2)
	fail := b.input(0, "fail", mmf.Bool)
	knew := b.val("k_new", mmf.F32, mmf.KindIntermediate, 1, 4, 2)
	sum := b.val("sum", mmf.F32, mmf.KindIntermediate, 1, 1, 1)
	res := b.val("res", mmf.F32, mmf.KindIntermediate, 1, 1, 1)
	b.op(0, mmf.OpScatter, []uint32{cache, pos, kt}, []uint32{knew}, iAttr("axis", 1))
	b.op(0, mmf.OpReduce, []uint32{knew}, []uint32{sum}, sAttr("mode", "add"), mmf.Attr{Key: "axes", Type: mmf.AttrInts, Ints: []int64{1, 2}})
	b.op(0, mmf.OpIf, []uint32{fail, sum}, []uint32{res}, gAttr("then", 1), gAttr("else", 2))
	b.output(0, res)
	// then: a composite this executor does not have, so the run fails
	// after the scatter. else: pass the sum through.
	subIn := func(si int, name string) uint32 {
		id := b.val(name, mmf.F32, mmf.KindIntermediate, 1, 1, 1)
		b.g.Subgraphs[si].Inputs = append(b.g.Subgraphs[si].Inputs, id)
		return id
	}
	x1 := subIn(1, "x1")
	y1 := b.val("y1", mmf.F32, mmf.KindIntermediate, 1, 1, 1)
	b.op(1, mmf.OpCall, []uint32{x1}, []uint32{y1}, callAttrs("no_such_composite")...)
	b.output(1, y1)
	x2 := subIn(2, "x2")
	b.output(2, x2)
	b.g.Updates = []mmf.Update{{Variable: cache, Value: knew}}
	p := b.load(t)
	if len(p.inplace) != 1 {
		t.Fatalf("%d scatters may run in place, want 1", len(p.inplace))
	}

	no := &ref.Tensor{DType: mmf.Bool, Shape: []int{}, Data: []byte{0}}
	yes := &ref.Tensor{DType: mmf.Bool, Shape: []int{}, Data: []byte{1}}
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
	cache := b.stored("k_cache", mmf.KindVariable, mmf.F32, []uint64{1, 4, 2}, make([]byte, 32))
	pos := b.input(0, "pos", mmf.I64, 1)
	kt := b.input(0, "k_t", mmf.F32, 1, 1, 2)
	knew := b.val("k_new", mmf.F32, mmf.KindIntermediate, 1, 4, 2)
	old := b.val("old", mmf.F32, mmf.KindIntermediate, 1, 4, 2)
	b.op(0, mmf.OpScatter, []uint32{cache, pos, kt}, []uint32{knew}, iAttr("axis", 1))
	b.op(0, mmf.OpAdd, []uint32{cache, knew}, []uint32{old}) // a second use of the old value
	b.output(0, old)
	b.g.Updates = []mmf.Update{{Variable: cache, Value: knew}}
	if p := b.load(t); len(p.inplace) != 0 {
		t.Error("a scatter whose input has another use may run in place")
	}
}

func TestFastPermute(t *testing.T) {
	perm := func(p ...int64) *mmf.Op {
		return &mmf.Op{Opcode: mmf.OpPermute, Attrs: []mmf.Attr{{Key: "perm", Type: mmf.AttrInts, Ints: p}}}
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
	for _, dt := range []mmf.DType{mmf.F32, mmf.F16, mmf.Bool} {
		for _, xs := range [][]int{{1, 3, 1, 2}, {2, 3, 4, 5}, {3, 0, 2, 1}} {
			x, err := ref.New(dt, xs)
			if err != nil {
				t.Fatal(err)
			}
			for i := range x.Data {
				x.Data[i] = byte(i*7 + 1)
				if dt == mmf.Bool {
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
				want, err := ref.Eval(mmf.OpPermute, perm(p...).Attrs, []*ref.Tensor{x}, dt, shape)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got.Data, want.Data) || !sameInts(got.Shape, want.Shape) {
					t.Errorf("%s %v perm %v: wrong result", dt, xs, p)
				}
			}
		}
	}
	q, err := ref.New(mmf.Q8_0, []int{2, 32})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := fastPermute(perm(1, 0), []*ref.Tensor{q}, mmf.Q8_0, []int{32, 2}); ok {
		t.Error("handled a block dtype")
	}
}
