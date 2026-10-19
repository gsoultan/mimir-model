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

// TestConvFusion runs a convolution followed by a batch norm (mul and add
// by per-channel weights), a residual add, a const and a ReLU (max with
// the const, as its first operand), as an ONNX CNN is imported, and checks
// that the run is one fusion whose results are those of the ops one by
// one, bit for bit, NaNs included; and that a convolution whose result is
// read twice is not fused.
func TestConvFusion(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	b := newGB()
	n := b.sym("N", 4)
	x := b.input(0, "x", mim.F32, n, 4, 7, 6)
	res := b.input(0, "res", mim.F32, n, 13, 7, 6)
	w := b.stored("w", mim.KindWeight, mim.F32, []uint64{13, 4, 3, 3}, randF32(rng, []int{13, 4, 3, 3}, 1).Data)
	sc, sh := randF32(rng, []int{1, 13, 1, 1}, 1), randF32(rng, []int{1, 13, 1, 1}, 1)
	scale := b.stored("scale", mim.KindWeight, mim.F32, []uint64{1, 13, 1, 1}, sc.Data)
	shift := b.stored("shift", mim.KindWeight, mim.F32, []uint64{1, 13, 1, 1}, sh.Data)
	full := []mim.Dim{n, 13, 7, 6}
	c := b.val("c", mim.F32, mim.KindIntermediate, full...)
	conv := callAttrs("conv2d", intsAttr("pads", 1, 1, 1, 1))
	b.op(0, mim.OpCall, []uint32{x, w}, []uint32{c}, conv...)
	m := b.val("m", mim.F32, mim.KindIntermediate, full...)
	b.op(0, mim.OpMul, []uint32{c, scale}, []uint32{m})
	a := b.val("a", mim.F32, mim.KindIntermediate, full...)
	b.op(0, mim.OpAdd, []uint32{m, shift}, []uint32{a})
	r := b.val("r", mim.F32, mim.KindIntermediate, full...)
	b.op(0, mim.OpAdd, []uint32{res, a}, []uint32{r})
	zero := b.val("zero", mim.F32, mim.KindIntermediate, 1, 1, 1, 1)
	b.op(0, mim.OpConst, nil, []uint32{zero}, fAttr("value", 0))
	y := b.val("y", mim.F32, mim.KindIntermediate, full...)
	b.op(0, mim.OpMax, []uint32{zero, r}, []uint32{y})
	b.output(0, y)
	// A second convolution, whose result is also an output.
	c2 := b.val("c2", mim.F32, mim.KindIntermediate, full...)
	b.op(0, mim.OpCall, []uint32{x, w}, []uint32{c2}, conv...)
	b.output(0, c2)
	y2 := b.val("y2", mim.F32, mim.KindIntermediate, full...)
	b.op(0, mim.OpMul, []uint32{c2, scale}, []uint32{y2})
	b.output(0, y2)
	p := b.load(t)
	if f := p.convFusions[0]; len(p.convFusions) != 1 || f == nil || f.last != 5 || len(f.steps) != 4 || len(f.consts) != 1 {
		t.Fatalf("fusions %v, want ops 0 to 5", p.convFusions)
	}
	for _, nb := range []int{1, 3} {
		xs := randF32(rng, []int{nb, 4, 7, 6}, 1)
		rs := randF32(rng, []int{nb, 13, 7, 6}, 2)
		xv, rv := f32View(xs), f32View(rs)
		xv[5] = float32(math.NaN())
		rv[7] = float32(math.Inf(1))
		rv[100] = float32(math.Inf(-1))
		in := map[string]*ref.Tensor{"x": xs, "res": rs}
		p.FastPaths = true
		got, err := p.NewSession().Run(in)
		if err != nil {
			t.Fatal(err)
		}
		p.FastPaths = false
		want, err := p.NewSession().Run(in)
		if err != nil {
			t.Fatal(err)
		}
		// The reference's convolution differs from the fast one within
		// the tolerance, so the fused run is checked against the fast
		// convolution followed by the reference's element-wise ops.
		cw := got["c2"]
		bin := func(op mim.Opcode, a, b *ref.Tensor) *ref.Tensor {
			shape := a.Shape
			if len(b.Data) > len(a.Data) {
				shape = b.Shape
			}
			out, err := ref.Eval(op, nil, []*ref.Tensor{a, b}, mim.F32, shape)
			if err != nil {
				t.Fatal(err)
			}
			return out
		}
		e := bin(mim.OpMax, f32t([]int{1, 1, 1, 1}, 0), bin(mim.OpAdd, rs, bin(mim.OpAdd, bin(mim.OpMul, cw, sc), sh)))
		if !bytes.Equal(got["y"].Data, e.Data) {
			t.Errorf("batch %d: fused run differs from the ops one by one", nb)
		}
		if !bytes.Equal(got["y2"].Data, bin(mim.OpMul, cw, sc).Data) {
			t.Errorf("batch %d: y2 differs", nb)
		}
		if !sameInts(got["y"].Shape, want["y"].Shape) {
			t.Errorf("batch %d: shape %v, want %v", nb, got["y"].Shape, want["y"].Shape)
		}
	}
}
