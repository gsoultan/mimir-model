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
	"fmt"
	"math/rand"
	"testing"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/ref"
)

// TestMoveChain runs a strided take on both spatial axes, as an ONNX
// max pool is imported, followed by a permute, with a symbolic batch
// size, and checks that the chain runs as one gather and gives the
// reference's result bit for bit; and that a run of movement ops whose
// intermediate value is read elsewhere is not a chain.
func TestMoveChain(t *testing.T) {
	b := newGB()
	n := b.sym("N", 8)
	x := b.input(0, "x", mim.F32, n, 3, 9, 8)
	ints := func(k string, v ...int64) mim.Attr { return mim.Attr{Key: k, Type: mim.AttrInts, Ints: v} }
	cur := x
	step := func(opc mim.Opcode, shape []mim.Dim, attrs ...mim.Attr) {
		id := b.val(fmt.Sprint("v", len(b.g.Values)), mim.F32, mim.KindIntermediate, shape...)
		b.op(0, opc, []uint32{cur}, []uint32{id}, attrs...)
		cur = id
	}
	// take_strided(x, 2, 1, 4, 2): rows 1, 3, 5, 7.
	step(mim.OpShrink, []mim.Dim{n, 3, 8, 8}, ints("begin", 0, 0, 1, 0), ints("end", -1, -1, 9, -1))
	step(mim.OpReshape, []mim.Dim{n, 3, 4, 2, 8}, ints("shape", -1, 3, 4, 2, 8))
	step(mim.OpShrink, []mim.Dim{n, 3, 4, 1, 8}, ints("begin", 0, 0, 0, 0, 0), ints("end", -1, -1, -1, 1, -1))
	step(mim.OpReshape, []mim.Dim{n, 3, 4, 8}, ints("shape", -1, 3, 4, 8))
	// take_strided(., 3, 0, 3, 2): columns 0, 2, 4.
	step(mim.OpShrink, []mim.Dim{n, 3, 4, 6}, ints("begin", 0, 0, 0, 0), ints("end", -1, -1, -1, 6))
	step(mim.OpReshape, []mim.Dim{n, 3, 4, 3, 2}, ints("shape", -1, 3, 4, 3, 2))
	step(mim.OpShrink, []mim.Dim{n, 3, 4, 3, 1}, ints("begin", 0, 0, 0, 0, 0), ints("end", -1, -1, -1, -1, 1))
	step(mim.OpReshape, []mim.Dim{n, 3, 4, 3}, ints("shape", -1, 3, 4, 3))
	step(mim.OpPermute, []mim.Dim{n, 4, 3, 3}, ints("perm", 0, 2, 3, 1))
	b.output(0, cur)
	// A second run: its shrink's result is also an output.
	cur = x
	step(mim.OpShrink, []mim.Dim{n, 3, 2, 8}, ints("begin", 0, 0, 3, 0), ints("end", -1, -1, 5, -1))
	b.output(0, cur)
	step(mim.OpReshape, []mim.Dim{n, 48}, ints("shape", -1, 48))
	b.output(0, cur)
	p := b.load(t)
	if len(p.chains) != 1 || p.chains[0] == nil || p.chains[0].last != 8 {
		t.Fatalf("chains %v, want ops 0 to 8", p.chains)
	}
	rng := rand.New(rand.NewSource(2))
	for _, nb := range []int{1, 3, 2} {
		in := map[string]*ref.Tensor{"x": randF32(rng, []int{nb, 3, 9, 8}, 1)}
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
		if len(want) != 3 {
			t.Fatalf("%d outputs", len(want))
		}
		for k, w := range want {
			if g := got[k]; g == nil || !sameInts(g.Shape, w.Shape) || !bytes.Equal(g.Data, w.Data) {
				t.Errorf("batch %d: output %q differs from the reference", nb, k)
			}
		}
	}
}
