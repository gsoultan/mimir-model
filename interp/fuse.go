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
	"math"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/ref"
)

// Convolutions with their element-wise epilogue. An imported CNN follows
// most convolutions with a batch norm (a mul and an add by per-channel
// tensors), often a residual add, and a ReLU (a max with 0). When each
// value in such a run is read only by the next op, the convolution kernel
// applies the ops to each tile of its output while the tile is in the
// cache, instead of each op reading and writing the whole tensor. Every
// op is the same float32 operation as on its own, so the results are
// those of the separate ops, bit for bit. Const ops between them, which
// read nothing, run first.

// convFusion is a conv2d.v1 call of the main graph and the element-wise
// ops after it, up to op last.
type convFusion struct {
	last   int
	consts []int     // the const ops in the run
	steps  []epiStep // the element-wise ops, in order
}

// epiStep is one element-wise op of an epilogue: the running value op
// other (add, mul or max).
type epiStep struct {
	op    mim.Opcode
	other uint32
	out   uint32
}

// convFusions finds the convolutions of the main graph with an
// element-wise epilogue.
func convFusions(g *mim.Graph) map[int]*convFusion {
	out := map[int]*convFusion{}
	if len(g.Subgraphs) == 0 {
		return out
	}
	uses := map[uint32]int{}
	for si := range g.Subgraphs {
		for _, op := range g.Subgraphs[si].Ops {
			for _, id := range op.Inputs {
				uses[id]++
			}
		}
		for _, id := range g.Subgraphs[si].Outputs {
			uses[id] += 2
		}
	}
	for _, u := range g.Updates {
		uses[u.Value] += 2
	}
	ops := g.Main().Ops
	for i := range ops {
		op := &ops[i]
		name, ver := attr(op, "name"), attr(op, "version")
		if op.Opcode != mim.OpCall || name == nil || name.String != "conv2d" || ver == nil || ver.Int != 1 ||
			len(op.Outputs) != 1 || g.Values[op.Outputs[0]].DType != mim.F32 {
			continue
		}
		f := &convFusion{}
		cur := op.Outputs[0]
		for j := i + 1; j < len(ops); j++ {
			o := &ops[j]
			if o.Opcode == mim.OpConst && len(o.Inputs) == 0 {
				f.consts = append(f.consts, j)
				continue
			}
			if (o.Opcode != mim.OpAdd && o.Opcode != mim.OpMul && o.Opcode != mim.OpMax) || len(o.Inputs) != 2 ||
				len(o.Outputs) != 1 || uses[cur] != 1 || g.Values[cur].Kind != mim.KindIntermediate ||
				g.Values[o.Outputs[0]].DType != mim.F32 {
				break
			}
			other := o.Inputs[1]
			if o.Inputs[1] == cur {
				other = o.Inputs[0]
			}
			if other == cur || (o.Inputs[0] != cur && o.Inputs[1] != cur) {
				break
			}
			f.steps = append(f.steps, epiStep{op: o.Opcode, other: other, out: o.Outputs[0]})
			f.last = j
			cur = o.Outputs[0]
		}
		for len(f.consts) > 0 && f.consts[len(f.consts)-1] > f.last {
			f.consts = f.consts[:len(f.consts)-1]
		}
		if len(f.steps) > 0 {
			out[i] = f
		}
	}
	return out
}

// epilogue is an element-wise epilogue bound to its operands for a run.
type epilogue struct {
	ops   []mim.Opcode
	vals  [][]float32 // each step's other operand
	kinds []byte      // 's' scalar, 'c' one per channel, 'f' the output's shape
	nan   float32
}

// apply runs the epilogue on y, the outputs of channel f of image n at
// positions p0 to p0+len(y)-1, in an output of nf channels of plane
// positions each.
func (e *epilogue) apply(y []float32, n, f, nf, plane, p0 int) {
	for s, opc := range e.ops {
		v := e.vals[s]
		switch e.kinds[s] {
		case 's':
			binaryF32(opc, y, 1, v, 0, y, e.nan)
		case 'c':
			binaryF32(opc, y, 1, v[f:], 0, y, e.nan)
		default:
			binaryF32(opc, y, 1, v[(n*nf+f)*plane+p0:], 1, y, e.nan)
		}
	}
}

// fusedConv runs the convolution at op first of the main graph with its
// epilogue f, binding the const ops' values in sc. It returns false,
// having run nothing, when the operands do not fit an epilogue; the
// caller then runs the ops one by one.
func (r *run) fusedConv(first int, f *convFusion, in []*ref.Tensor, sc *scope) (*ref.Tensor, bool, error) {
	ops := r.p.g.Main().Ops
	call := &ops[first]
	outV := r.p.g.Values[call.Outputs[0]]
	shape, err := r.shape(outV)
	if err != nil || len(shape) != 4 || len(in) < 2 || in[0].DType != mim.F32 {
		return nil, false, nil
	}
	nf := shape[1]
	// The consts first: they read nothing.
	consts := map[uint32]*ref.Tensor{}
	for _, ci := range f.consts {
		outs, err := r.op(&ops[ci], nil, sc)
		if err != nil {
			return nil, false, err
		}
		consts[ops[ci].Outputs[0]] = outs[0]
	}
	e := &epilogue{nan: math.Float32frombits(uint32(ref.RoundFloat(mim.F32, math.NaN(), false)))}
	ok := true
	for _, s := range f.steps {
		t := consts[s.other]
		if t == nil {
			t, _ = sc.get(s.other)
		}
		if t == nil || t.DType != mim.F32 || len(t.Shape) != 4 {
			ok = false
			break
		}
		var kind byte
		switch {
		case sameInts(t.Shape, []int{1, 1, 1, 1}):
			kind = 's'
		case sameInts(t.Shape, []int{1, nf, 1, 1}):
			kind = 'c'
		case sameInts(t.Shape, shape):
			kind = 'f'
		default:
			ok = false
		}
		if !ok {
			break
		}
		e.ops = append(e.ops, s.op)
		e.vals = append(e.vals, f32View(t))
		e.kinds = append(e.kinds, kind)
	}
	if ok {
		var cattrs []mim.Attr
		for _, a := range call.Attrs {
			if a.Key != "name" && a.Key != "version" {
				cattrs = append(cattrs, a)
			}
		}
		if err := r.reserve(mim.F32, shape); err != nil {
			return nil, false, err
		}
		out, handled, err := r.p.convGemm(cattrs, in, []mim.DType{mim.F32}, e)
		if err != nil {
			return nil, false, err
		}
		if handled {
			for id, t := range consts {
				sc.vals[id] = t
				r.live += int64(len(t.Data))
			}
			return out[0], true, nil
		}
	}
	return nil, false, nil
}
