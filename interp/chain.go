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
	"encoding/binary"
	"fmt"
	"strings"
	"unsafe"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/ref"
)

// Chains of movement ops. An importer writes some operations as runs of
// primitives that only move elements: an ONNX max pool takes each window
// position with a shrink, a reshape, a shrink and a reshape per spatial
// axis (spec/stdlib.md section 2.2, take_strided). Each op copies its
// input. When every value inside such a run is read only by the next op,
// the run's output is a gather of its input's elements, and which
// elements depends only on the input's shape: the executor works the
// indices out once per shape, by running the ops on a tensor of indices,
// and then copies each output element straight from the input.

// maxGatherElems bounds the indices a program keeps, for inputs whose
// shapes change from run to run.
const maxGatherElems = 64 << 20

// chain is a run of movement ops of the main graph, from its key in
// Program.chains to last.
type chain struct {
	last int
}

// movement reports whether op only moves elements of its one input.
func movement(op *mim.Op) bool {
	switch op.Opcode {
	case mim.OpShrink, mim.OpReshape, mim.OpPermute, mim.OpExpand:
		return len(op.Inputs) == 1 && len(op.Outputs) == 1
	}
	return false
}

// moveChains finds the runs of at least two movement ops in the main graph,
// one of them a shrink, in which each op reads the previous one's output,
// and nothing else reads it: it is not an output, a variable update or the
// input of any other op of any subgraph.
func moveChains(g *mim.Graph) map[int]*chain {
	out := map[int]*chain{}
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
	for i := 0; i < len(ops); i++ {
		if !movement(&ops[i]) {
			continue
		}
		j := i
		for j+1 < len(ops) && movement(&ops[j+1]) && ops[j+1].Inputs[0] == ops[j].Outputs[0] &&
			uses[ops[j].Outputs[0]] == 1 && g.Values[ops[j].Outputs[0]].Kind == mim.KindIntermediate {
			j++
		}
		// Reshapes and permutes alone are cheaper as views and strided
		// copies than through a table of indices.
		shrinks := false
		for k := i; k <= j; k++ {
			shrinks = shrinks || ops[k].Opcode == mim.OpShrink
		}
		if j > i && shrinks {
			out[i] = &chain{last: j}
		}
		i = j
	}
	return out
}

// moveChain runs the chain that starts at op first of the main graph on
// x, the first op's input.
func (r *run) moveChain(first int, ch *chain, x *ref.Tensor) (*ref.Tensor, error) {
	ops := r.p.g.Main().Ops
	last := r.p.g.Values[ops[ch.last].Outputs[0]]
	shape, err := r.shapeOf(ops[ch.last].Outputs[0])
	if err != nil {
		return nil, err
	}
	if err := r.reserve(last.DType, shape); err != nil {
		return nil, err
	}
	n := x.NumElems()
	if x.DType != last.DType || x.DType.IsBlock() || n >= 1<<31 || len(x.Data) < n*int(x.DType.BlockBytes()) {
		return r.chainOps(first, ch, x)
	}
	idx, err := r.chainIndices(first, ch, x.Shape, shape)
	if err != nil {
		return nil, err
	}
	es := int(x.DType.BlockBytes())
	t := &ref.Tensor{DType: x.DType, Shape: append([]int{}, shape...), Data: r.alloc.bytes(len(idx)*es, false)}
	gather(t.Data, x.Data, idx, es)
	return t, nil
}

// chainIndices returns, for each element of the chain's output of the
// given shape, the index of the element of its input, of shape xShape,
// that it copies. The indices are kept with the program.
func (r *run) chainIndices(first int, ch *chain, xShape, shape []int) ([]int32, error) {
	ops := r.p.g.Main().Ops
	n := ref.NumElems(xShape)
	var key strings.Builder
	fmt.Fprint(&key, first, xShape)
	if v, ok := r.p.gathers.Load(key.String()); ok {
		return v.([]int32), nil
	}
	it := &ref.Tensor{DType: mim.I32, Shape: append([]int{}, xShape...), Data: make([]byte, 4*n)}
	for i := range n {
		binary.LittleEndian.PutUint32(it.Data[4*i:], uint32(i))
	}
	for oi := first; oi <= ch.last; oi++ {
		op := &ops[oi]
		s, err := r.shape(r.p.g.Values[op.Outputs[0]])
		if err != nil {
			return nil, err
		}
		if t, ok := r.fastPrimitive(op, []*ref.Tensor{it}, mim.I32, s); ok {
			it = t
		} else if it, err = ref.Eval(op.Opcode, op.Attrs, []*ref.Tensor{it}, mim.I32, s); err != nil {
			return nil, err
		}
	}
	if !sameInts(it.Shape, shape) {
		return nil, fmt.Errorf("a chain of movement ops gives %v, declared %v", it.Shape, shape)
	}
	idx := make([]int32, it.NumElems())
	for i := range idx {
		idx[i] = int32(binary.LittleEndian.Uint32(it.Data[4*i:]))
	}
	if r.p.gatherElems.Add(int64(len(idx))) <= maxGatherElems {
		r.p.gathers.Store(key.String(), idx)
	}
	return idx, nil
}

// gather copies element idx[i] of src, of es bytes each, to element i of
// dst.
func gather(dst, src []byte, idx []int32, es int) {
	parallelWork(pieces(len(idx), gatherPiece), len(idx), func(lo, hi int) { gatherPieces(dst, src, idx, es, lo, hi) })
}

// gatherPiece is the elements of a piece of a gather's work.
const gatherPiece = 16 << 10

// gatherPieces does pieces lo to hi-1 of gather.
func gatherPieces(dst, src []byte, idx []int32, es, lo, hi int) {
	lo, hi = lo*gatherPiece, min(hi*gatherPiece, len(idx))
	if es == 4 && littleEndian {
		s := unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(src))), len(src)/4)
		d := unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(dst))), len(idx))
		for i, j := range idx[lo:hi] {
			d[lo+i] = s[j]
		}
		return
	}
	for i := lo; i < hi; i++ {
		j := int(idx[i])
		copy(dst[i*es:(i+1)*es], src[j*es:(j+1)*es])
	}
}

// chainOps runs the ops of a chain one by one.
func (r *run) chainOps(first int, ch *chain, x *ref.Tensor) (*ref.Tensor, error) {
	ops := r.p.g.Main().Ops
	for oi := first; oi <= ch.last; oi++ {
		outs, err := r.op(&ops[oi], []*ref.Tensor{x}, nil)
		if err != nil {
			return nil, err
		}
		x = outs[0]
	}
	return x, nil
}
