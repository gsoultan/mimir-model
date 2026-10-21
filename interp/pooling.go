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
	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/internal/kernels"
	"github.com/gsoultan/mimir-model/ref"
)

// Pooling in a plan. A max or average pool comes out of an importer as a
// pad, movement chains that take each window element of every output
// position (a strided slice), and a tree of maxes or adds over them. A
// plan compiles the chains lazily: a chain whose output is read once, by
// a max or add of another chain of the same input, becomes a tap of a
// gatherTree, and the tree runs as one step over the windows, a row of
// output positions at a time on a vector kernel, with no element of a
// chain stored. A chain read by anything else is stored as it is reached.

// A gatherOp is a movement chain waiting to be compiled: it copies
// element idx[i] of value src to element i of its output.
type gatherOp struct {
	src   uint32
	idx   []int32
	shape []int
	es    int
	// With affine set, idx[i] = base + the sum over the output's axes of
	// i_d * strides[d], and the chain is a strided slice.
	affine  bool
	base    int
	strides []int
}

// A gatherTree is a left-deep tree of opc over chains of one input, the
// taps: tap 0, then opc with each tap after it in turn.
type gatherTree struct {
	opc   mim.Opcode
	src   uint32
	shape []int
	taps  []*gatherOp
}

// affine finds the base and strides of idx over shape when the indices
// are a strided slice.
func affine(idx []int32, shape []int) (base int, strides []int, ok bool) {
	if len(idx) == 0 {
		return 0, nil, false
	}
	base = int(idx[0])
	strides = make([]int, len(shape))
	for d, st := len(shape)-1, 1; d >= 0; d-- {
		if shape[d] > 1 {
			strides[d] = int(idx[st]) - base
			if strides[d] < 0 {
				return 0, nil, false
			}
		}
		st *= shape[d]
	}
	// Every index, counting up through the multi-index.
	pos := make([]int, len(shape))
	want := base
	for _, j := range idx {
		if int(j) != want {
			return 0, nil, false
		}
		for d := len(shape) - 1; d >= 0; d-- {
			pos[d]++
			want += strides[d]
			if pos[d] < shape[d] {
				break
			}
			want -= pos[d] * strides[d]
			pos[d] = 0
		}
	}
	return base, strides, true
}

// binOp is the kernels' code for the binary op opc, one of add, mul and
// max.
func binOp(opc mim.Opcode) int {
	switch opc {
	case mim.OpMul:
		return kernels.BinMul
	case mim.OpMax:
		return kernels.BinMax
	}
	return kernels.BinAdd
}

// combine is y opc v as the kernels of a binary op would give it.
func combine(opc mim.Opcode, y, v float32) float32 {
	if opc == mim.OpMax {
		return maxF32(y, v, nanF32)
	}
	s := y + v
	if opc == mim.OpMul {
		s = y * v
	}
	if s != s {
		s = nanF32
	}
	return s
}

// poolPlane computes rows rows of w outputs: y[r*w+j] = the opc over the
// taps t, in order, of x[offs[t]+r*xrs+j*stride], as the kernels of a
// binary op would.
func poolPlane(opc mim.Opcode, y, x []float32, offs []int, w, rows, stride, xrs int) {
	if kernels.PoolF32(binOp(opc), y, x, offs, w, rows, stride, xrs, w, nanF32) {
		return
	}
	for ti, off := range offs {
		for r := range rows {
			yr, xr := y[r*w:(r+1)*w], x[off+r*xrs:]
			for j := range yr {
				v := xr[j*stride]
				if ti == 0 {
					yr[j] = v
				} else {
					yr[j] = combine(opc, yr[j], v)
				}
			}
		}
	}
}

// hold keeps value id's buffer from being dropped until unhold, for a
// gather waiting to be compiled.
func (c *compiler) hold(id uint32) { c.held[id]++ }

func (c *compiler) unhold(id uint32) {
	if c.held[id]--; c.held[id] == 0 && c.heldDrop[id] {
		delete(c.heldDrop, id)
		c.drop(id)
	}
}

// release is drop for the compile loop: a held value is dropped when
// its hold ends.
func (c *compiler) release(id uint32) {
	if c.held[id] > 0 {
		c.heldDrop[id] = true
		return
	}
	c.drop(id)
}

// deferrable reports whether value id, the output of a chain or a tree,
// may wait to be compiled: it is read once and is not an output.
func (c *compiler) deferrable(id uint32) bool {
	return c.uses[id] == 1 && !c.isOut[id]
}

// materialize compiles the pending gather or tree of value id, if any,
// as a step at op oi.
func (c *compiler) materialize(oi int, id uint32) {
	switch pd := c.pending[id].(type) {
	case *gatherOp:
		delete(c.pending, id)
		c.gatherStep(oi, id, pd)
		c.unhold(pd.src)
	case *gatherTree:
		delete(c.pending, id)
		c.poolStep(oi, id, pd)
		for range pd.taps {
			c.unhold(pd.src)
		}
	}
}

// materializeAll compiles what is pending of each of ids.
func (c *compiler) materializeAll(oi int, ids []uint32) {
	for _, id := range ids {
		c.materialize(oi, id)
	}
}

// gatherStep compiles g as a step storing value out.
func (c *compiler) gatherStep(oi int, out uint32, g *gatherOp) {
	mk := c.fresh(out, len(g.idx)*g.es)
	sl := c.pl.slots
	src, idx, es := g.src, g.idx, g.es
	var y []byte // the step's output, made once per run
	gath := func(lo, hi int) { gatherPieces(y, sl[src].Data, idx, es, lo, hi) }
	c.add(oi, mim.OpShrink, func() error {
		y = mk().Data
		parallelWork(pieces(len(idx), gatherPiece), len(idx), gath)
		return nil
	})
}

// tree combines a and b, the pending gathers or trees of the inputs of a
// binary op opc, into one tree, or reports that it cannot: the ops are
// applied in the graph's order, so a tree may only grow by one chain at a
// time, except under max, which is associative.
func tree(opc mim.Opcode, a, b any) (*gatherTree, bool) {
	taps := func(v any) (uint32, []*gatherOp, bool) {
		switch v := v.(type) {
		case *gatherOp:
			return v.src, []*gatherOp{v}, v.affine && v.es == 4
		case *gatherTree:
			return v.src, v.taps, v.opc == opc
		}
		return 0, nil, false
	}
	sa, ta, oka := taps(a)
	sb, tb, okb := taps(b)
	if !oka || !okb || sa != sb || (len(ta) > 1 && len(tb) > 1 && opc != mim.OpMax) {
		return nil, false
	}
	if len(tb) > 1 && len(ta) == 1 {
		ta, tb = tb, ta // opc is commutative: the chain joins the tree
	}
	t := &gatherTree{opc: opc, src: sa, shape: ta[0].shape}
	t.taps = append(append(t.taps, ta...), tb...)
	return t, true
}

// poolStep compiles t as a step storing value out: for each plane of
// output positions (the last two axes), the taps' windows combined by
// opc.
func (c *compiler) poolStep(oi int, out uint32, t *gatherTree) {
	n := ref.NumElems(t.shape)
	mk := c.fresh(out, 4*n)
	sl := c.pl.slots
	r := len(t.shape)
	w, h := t.shape[r-1], 1
	if r > 1 {
		h = t.shape[r-2]
	}
	plane := w * h
	planes := n / max(1, plane)
	outer := t.shape[:max(0, r-2)]
	taps, opc, src := t.taps, t.opc, t.src
	// The taps of a pool step through a plane alike; otherwise, and for a
	// narrow plane, each is gathered.
	stride, xrs := taps[0].strides[r-1], 0
	if r > 1 {
		xrs = taps[0].strides[r-2]
	}
	alike := plane >= gatherPlane
	for _, tp := range taps {
		alike = alike && sameInts(tp.strides, taps[0].strides)
	}
	if !alike {
		c.gatherPoolStep(oi, out, t, mk)
		return
	}
	// The taps step through the outer axes alike too: a plane's offset
	// is worked out once, and each tap's is its base on.
	ostrides := taps[0].strides[:len(outer)]
	bases := make([]int, len(taps))
	for ti, tp := range taps {
		bases[ti] = tp.base
	}
	// per planes of output go to a worker at a time.
	per := max(1, 4096/max(1, plane))
	var y []float32 // the step's output, made once per run
	body := func(lo, hi int) {
		x := f32View(sl[src])
		var posa, offa [8]int
		pos, offs := posa[:0], offa[:0]
		if len(outer) > len(posa) {
			pos = make([]int, 0, len(outer))
		}
		if len(taps) > len(offa) {
			offs = make([]int, 0, len(taps))
		}
		pos, offs = pos[:len(outer)], offs[:len(taps)]
		lo, hi = lo*per, min(hi*per, planes)
		for d, q := len(outer)-1, lo; d >= 0; d-- {
			pos[d] = q % outer[d]
			q /= outer[d]
		}
		for p := lo; p < hi; p++ {
			off := 0
			for d, i := range pos {
				off += i * ostrides[d]
			}
			for ti, b := range bases {
				offs[ti] = b + off
			}
			poolPlane(opc, y[p*plane:(p+1)*plane], x, offs, w, h, stride, xrs)
			for d := len(outer) - 1; d >= 0; d-- {
				if pos[d]++; pos[d] < outer[d] {
					break
				}
				pos[d] = 0
			}
		}
	}
	c.add(oi, opc, func() error {
		y = f32View(mk())
		parallelWork(pieces(planes, per), n*len(taps), body)
		return nil
	})
}

// gatherPlane is the plane size below which a tree's taps are gathered
// through their indices rather than read as rows: a narrow plane leaves
// most of a vector's lanes idle, and the planes are many.
const gatherPlane = 64

// gatherPoolStep compiles t as a step storing value out, each output
// position the taps' elements, gathered through their indices, combined
// by opc.
func (c *compiler) gatherPoolStep(oi int, out uint32, t *gatherTree, mk func() *ref.Tensor) {
	n := ref.NumElems(t.shape)
	// The kernel reads indices sixteen at a time: each tap's are padded
	// with a valid index.
	np := (n + 15) &^ 15
	idx := make([]int32, np*len(t.taps))
	for ti, tp := range t.taps {
		copy(idx[ti*np:], tp.idx)
		for i := n; i < np; i++ {
			idx[ti*np+i] = tp.idx[n-1]
		}
	}
	sl := c.pl.slots
	src, opc, taps := t.src, t.opc, len(t.taps)
	var y []float32 // the step's output, made once per run
	body := func(lo, hi int) {
		x := f32View(sl[src])
		lo, hi = lo*gatherPiece, min(hi*gatherPiece, n)
		if kernels.PoolGatherF32(binOp(opc), y[lo:hi], x, idx[lo:], np, taps, nanF32) {
			return
		}
		for ti := range taps {
			ix := idx[ti*np:]
			for i := lo; i < hi; i++ {
				v := x[ix[i]]
				if ti == 0 {
					y[i] = v
				} else {
					y[i] = combine(opc, y[i], v)
				}
			}
		}
	}
	c.add(oi, opc, func() error {
		y = f32View(mk())
		parallelWork(pieces(n, gatherPiece), n*taps, body)
		return nil
	})
}
