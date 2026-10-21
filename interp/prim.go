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
	"math"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/internal/kernels"
	"github.com/gsoultan/mimir-model/ref"
)

// Primitive fast paths. With FastPaths set, a few primitives that a
// transformer runs at every step skip the reference kernels' generic
// element loops. Their results are bit-identical to the reference
// kernels'; anything unusual falls back to them.

// fastPrimitive returns handled = false when the reference kernel must run.
func (r *run) fastPrimitive(op *mim.Op, in []*ref.Tensor, dt mim.DType, shape []int) (*ref.Tensor, bool) {
	switch op.Opcode {
	case mim.OpAdd, mim.OpMul:
		if t, ok := fastAddMul(r.alloc, op.Opcode, in, dt, shape); ok {
			return t, true
		}
		return fastBroadcast(r.alloc, op.Opcode, in, dt, shape)
	case mim.OpMax, mim.OpCmplt, mim.OpCmpeq, mim.OpAnd, mim.OpOr, mim.OpXor:
		return fastBroadcast(r.alloc, op.Opcode, in, dt, shape)
	case mim.OpPermute:
		return fastPermute(r.alloc, op, in, dt, shape)
	case mim.OpReshape:
		return r.reshapeView(op, in, dt, shape)
	case mim.OpShrink:
		return fastShrink(r.alloc, op, in, dt, shape)
	case mim.OpPad:
		return fastPad(r.alloc, op, in, dt, shape)
	case mim.OpScatter:
		return r.scatterInPlace(op, in, dt, shape)
	case mim.OpCast:
		return fastCastF16(in, dt, shape)
	case mim.OpNeg:
		return fastNeg(r.alloc, in, dt)
	}
	return nil, false
}

// fastNeg negates an f32 tensor by flipping each sign bit, as the
// reference does, NaNs included.
func fastNeg(a *arena, in []*ref.Tensor, dt mim.DType) (*ref.Tensor, bool) {
	if len(in) != 1 || dt != mim.F32 || in[0].DType != mim.F32 || !littleEndian {
		return nil, false
	}
	x := f32View(in[0])
	y := a.f32(len(x), false)
	for i, v := range x {
		y[i] = math.Float32frombits(math.Float32bits(v) ^ 1<<31)
	}
	return f32Tensor(in[0].Shape, y), true
}

// fastCastF16 casts an f32 tensor to f16, as a key-value cache in f16
// needs for every new key and value, in parallel. mim.F32ToF16 rounds as
// the reference does except in the payload of a NaN, which goes through
// the reference's rounding.
func fastCastF16(in []*ref.Tensor, dt mim.DType, shape []int) (*ref.Tensor, bool) {
	if len(in) != 1 || in[0].DType != mim.F32 || dt != mim.F16 || !sameInts(in[0].Shape, shape) {
		return nil, false
	}
	x := f32View(in[0])
	t := &ref.Tensor{DType: mim.F16, Shape: append([]int{}, shape...), Data: make([]byte, 2*len(x))}
	parallel(len(x), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			h := mim.F32ToF16(x[i])
			if x[i] != x[i] {
				h = uint16(ref.RoundFloat(mim.F16, float64(x[i]), false))
			}
			binary.LittleEndian.PutUint16(t.Data[2*i:], h)
		}
	})
	return t, true
}

// fastAddMul adds or multiplies two f32 tensors of the output's shape.
// Float32 addition and multiplication are correctly rounded, as the
// primitives require, and a NaN result is the reference's NaN.
func fastAddMul(al *arena, opc mim.Opcode, in []*ref.Tensor, dt mim.DType, shape []int) (*ref.Tensor, bool) {
	if len(in) != 2 || dt != mim.F32 || in[0].DType != mim.F32 || in[1].DType != mim.F32 ||
		!sameInts(in[0].Shape, shape) || !sameInts(in[1].Shape, shape) {
		return nil, false
	}
	a, b := f32View(in[0]), f32View(in[1])
	y := al.f32(len(a), false)
	b = b[:len(a)]
	nan := math.Float32frombits(uint32(ref.RoundFloat(mim.F32, math.NaN(), false)))
	// Pieces of 4096 elements, so that a small tensor stays on one core.
	const piece = 4096
	parallelWork((len(y)+piece-1)/piece, len(y)/4, func(lo, hi int) {
		lo, hi = lo*piece, min(hi*piece, len(y))
		binaryF32(opc, a[lo:hi], 1, b[lo:hi], 1, y[lo:hi], nan)
	})
	return f32Tensor(shape, y), true
}

// fastPermute handles permutations of tensors whose elements are whole
// bytes. One that only moves axes of size 1 leaves the elements in the
// same order, so the result is a copy; any other is copied in runs along
// the last output axis, walking the input with strides instead of
// recomputing every index as the reference does.
func fastPermute(al *arena, op *mim.Op, in []*ref.Tensor, dt mim.DType, shape []int) (*ref.Tensor, bool) {
	p := attr(op, "perm")
	if len(in) != 1 || in[0].DType != dt || p == nil || p.Type != mim.AttrInts ||
		len(p.Ints) != len(shape) || len(shape) != len(in[0].Shape) {
		return nil, false
	}
	x := in[0]
	seen := make([]bool, len(shape))
	last, moved := -1, false
	for i, a := range p.Ints {
		if a < 0 || int(a) >= len(shape) || seen[a] || x.Shape[a] != shape[i] {
			return nil, false
		}
		seen[a] = true
		if shape[i] != 1 {
			moved = moved || int(a) < last
			last = int(a)
		}
	}
	if !moved {
		return &ref.Tensor{DType: dt, Shape: append([]int{}, shape...), Data: append([]byte{}, x.Data...)}, true
	}
	return permuteStrided(al, x, p.Ints, shape)
}

// permuteStrided permutes x, of a dtype whose elements are whole bytes,
// into shape.
func permuteStrided(al *arena, x *ref.Tensor, perm []int64, shape []int) (*ref.Tensor, bool) {
	if x.DType.IsBlock() {
		return nil, false
	}
	es, err := ref.ByteSize(x.DType, []int{1})
	if n8, err8 := ref.ByteSize(x.DType, []int{8}); err != nil || err8 != nil || n8 != 8*es {
		return nil, false
	}
	in := permStrides(perm, x.Shape)
	total := ref.NumElems(shape)
	y := al.bytes(total*es, false)
	permuteInto(y, x.Data, in, shape, es)
	return &ref.Tensor{DType: x.DType, Shape: append([]int{}, shape...), Data: y}, true
}

// permStrides returns the input stride, in elements, of each output axis
// of a permutation of xShape by perm.
func permStrides(perm []int64, xShape []int) []int {
	r := len(xShape)
	in := make([]int, r)
	acc := 1
	for a := r - 1; a >= 0; a-- {
		for k, pa := range perm {
			if int(pa) == a {
				in[k] = acc
			}
		}
		acc *= xShape[a]
	}
	return in
}

// permuteInto writes the permutation of x into y, in holding the input
// stride of each output axis (permStrides), elements of es bytes.
func permuteInto(y, x []byte, in, shape []int, es int) {
	r := len(shape)
	total := ref.NumElems(shape)
	if total == 0 {
		return
	}
	inner, step := shape[r-1], in[r-1]
	idx := make([]int, r-1)
	src := 0
	for dst := 0; dst < total; dst += inner {
		if step == 1 {
			copy(y[dst*es:(dst+inner)*es], x[src*es:(src+inner)*es])
		} else {
			for i := 0; i < inner; i++ {
				copy(y[(dst+i)*es:(dst+i+1)*es], x[(src+i*step)*es:])
			}
		}
		// Advance the outer index, last axis first.
		for k := r - 2; k >= 0; k-- {
			idx[k]++
			src += in[k]
			if idx[k] < shape[k] {
				break
			}
			src -= idx[k] * in[k]
			idx[k] = 0
		}
	}
}

// In-place scatter. A key-value cache is a variable that each step
// scatters new rows into; the reference scatter copies the whole cache to
// do so. When the variable's old value is used by nothing but that
// scatter, and the scatter's result is the variable's next value and
// nothing else's, the copy is not observable and the rows can be written
// into the session's own buffer (spec/primitives.md section 7). The
// overwritten bytes are kept until the run ends, so that a failed run
// leaves the variable unchanged.

// inplaceScatters finds the scatters of the main graph that may run in
// place, and returns the variable each one updates.
func inplaceScatters(g *mim.Graph) map[*mim.Op]uint32 {
	uses := map[uint32]int{}
	for si := range g.Subgraphs {
		for _, op := range g.Subgraphs[si].Ops {
			for _, id := range op.Inputs {
				uses[id]++
			}
		}
		for _, id := range g.Subgraphs[si].Outputs {
			uses[id]++
		}
	}
	updates := map[uint32]int{} // variable -> number of updates
	valueUpdates := map[uint32]int{}
	for _, u := range g.Updates {
		updates[u.Variable]++
		valueUpdates[u.Value]++
	}
	out := map[*mim.Op]uint32{}
	if len(g.Subgraphs) == 0 {
		return out
	}
	main := &g.Subgraphs[0]
	for oi := range main.Ops {
		op := &main.Ops[oi]
		if op.Opcode != mim.OpScatter || len(op.Inputs) != 3 || len(op.Outputs) != 1 {
			continue
		}
		v, o := op.Inputs[0], op.Outputs[0]
		if g.Values[v].Kind != mim.KindVariable || uses[v] != 1 || updates[v] != 1 || valueUpdates[v] != 0 ||
			isOutput(main, o) || valueUpdates[o] != 1 {
			continue
		}
		for _, u := range g.Updates {
			if u.Variable == v && u.Value == o {
				out[op] = v
			}
		}
	}
	return out
}

func isOutput(sg *mim.Subgraph, id uint32) bool {
	for _, o := range sg.Outputs {
		if o == id {
			return true
		}
	}
	return false
}

// undo is the old content of bytes an in-place scatter overwrote.
type undo struct {
	dst, old []byte
}

func (r *run) scatterInPlace(op *mim.Op, in []*ref.Tensor, dt mim.DType, shape []int) (*ref.Tensor, bool) {
	v, ok := r.p.inplace[op]
	if !ok || r.s == nil || !r.s.owned[v] || len(in) != 3 {
		return nil, false
	}
	data, ind, upd := in[0], in[1], in[2]
	ax := attr(op, "axis")
	if ax == nil || ax.Type != mim.AttrInt || data.DType != dt || upd.DType != dt || dt.IsBlock() ||
		!sameInts(data.Shape, shape) || (ind.DType != mim.I32 && ind.DType != mim.I64) {
		return nil, false
	}
	rank := len(shape)
	axis := int(ax.Int)
	if axis < 0 || axis >= rank {
		return nil, false
	}
	want := append(append(append([]int{}, shape[:axis]...), ind.Shape...), shape[axis+1:]...)
	if !sameInts(upd.Shape, want) {
		return nil, false
	}
	es := int(dt.BlockBytes())
	outer := ref.NumElems(shape[:axis])
	inner := ref.NumElems(shape[axis+1:]) * es
	n, nj := shape[axis], ind.NumElems()
	for a := 0; a < outer; a++ {
		for j := 0; j < nj; j++ {
			k := int64(position(ind, j))
			if k < 0 || k >= int64(n) {
				continue // unspecified effect; the reference skips them too
			}
			dst := data.Data[(a*n+int(k))*inner:][:inner]
			r.undo = append(r.undo, undo{dst: dst, old: append([]byte{}, dst...)})
			copy(dst, upd.Data[(a*nj+j)*inner:][:inner])
		}
	}
	return data, true
}

// fastBroadcast runs the binary primitives a graph uses on broadcast
// operands, such as an attention mask built from positions or a
// per-channel scale, without the reference's per-element index map: f32
// add, mul and max; cmplt and cmpeq of i32 or i64; and, or, xor and cmpeq
// of bools. Each element is computed as the reference computes it.
func fastBroadcast(al *arena, opc mim.Opcode, in []*ref.Tensor, dt mim.DType, shape []int) (*ref.Tensor, bool) {
	if len(in) != 2 {
		return nil, false
	}
	bc, ok := newBroadcast(opc, in[0].DType, in[0].Shape, in[1].Shape, dt, shape)
	if !ok {
		return nil, false
	}
	var t *ref.Tensor
	if bc.kind == 'f' {
		t = f32Tensor(shape, al.f32(bc.n, false))
	} else {
		// Only the true results are written.
		t = &ref.Tensor{DType: dt, Shape: append([]int{}, shape...), Data: al.bytes(bc.n, true)}
	}
	bc.run(in[0], in[1], t)
	return t, true
}

// bcastAxis is one merged axis of a broadcast: its length and each
// operand's stride along it, 0 where the operand is broadcast.
type bcastAxis struct{ n, sa, sb int }

// A broadcast is a binary primitive on two operands broadcast to a shape,
// worked out once for the shapes.
type broadcast struct {
	opc   mim.Opcode
	kind  byte // 'f' f32 arithmetic, 'i' integer compare, 'b' boolean
	es    int  // bytes per input element
	oes   int  // bytes per output element
	n     int  // elements of the output
	axes  []bcastAxis
	outer int // rows: the product of the axes but the last
	inner int // the last axis
	per   int // pieces per row
	nan   float32
}

// newBroadcast plans opc on operands of dtype idt and shapes as and bs,
// giving dt of the given shape. See fastBroadcast for what it handles.
func newBroadcast(opc mim.Opcode, idt mim.DType, as, bs []int, dt mim.DType, shape []int) (*broadcast, bool) {
	if len(as) != len(shape) || len(bs) != len(shape) || len(shape) == 0 {
		return nil, false
	}
	for i, d := range shape {
		if (as[i] != d && as[i] != 1) || (bs[i] != d && bs[i] != 1) {
			return nil, false
		}
	}
	bc := &broadcast{opc: opc, es: int(idt.BlockBytes()), oes: int(dt.BlockBytes()), n: ref.NumElems(shape)}
	switch {
	case idt == mim.F32 && dt == mim.F32 && (opc == mim.OpAdd || opc == mim.OpMul || opc == mim.OpMax):
		bc.kind = 'f'
	case (idt == mim.I64 || idt == mim.I32) && dt == mim.Bool && (opc == mim.OpCmplt || opc == mim.OpCmpeq):
		bc.kind = 'i'
	case idt == mim.Bool && dt == mim.Bool && (opc == mim.OpAnd || opc == mim.OpOr || opc == mim.OpXor || opc == mim.OpCmpeq):
		bc.kind = 'b'
	default:
		return nil, false
	}
	// Merge adjacent axes along which both operands are broadcast the
	// same way, then run the last merged axis as the inner loop.
	sa, sb := 1, 1
	stA, stB := make([]int, len(shape)), make([]int, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		if as[i] != 1 {
			stA[i] = sa
		}
		if bs[i] != 1 {
			stB[i] = sb
		}
		sa *= as[i]
		sb *= bs[i]
	}
	for i, d := range shape {
		if d == 1 {
			continue
		}
		if k := len(bc.axes) - 1; k >= 0 && bc.axes[k].sa == stA[i]*d && bc.axes[k].sb == stB[i]*d {
			bc.axes[k] = bcastAxis{bc.axes[k].n * d, stA[i], stB[i]}
			continue
		}
		bc.axes = append(bc.axes, bcastAxis{d, stA[i], stB[i]})
	}
	if len(bc.axes) == 0 {
		bc.axes = []bcastAxis{{1, 0, 0}}
	}
	last := bc.axes[len(bc.axes)-1]
	bc.inner = last.n
	bc.outer = 0
	if bc.n > 0 {
		bc.outer = bc.n / last.n
	}
	// Work goes in pieces of at most piece elements of a row, so that
	// operands of one shape, merged into a single row, still spread over
	// the cores.
	const piece = 8192
	bc.per = (bc.inner + piece - 1) / piece
	bc.nan = math.Float32frombits(uint32(ref.RoundFloat(mim.F32, math.NaN(), false)))
	return bc, true
}

// run computes the broadcast of a and b into t, whose data must hold the
// output and, for a compare or boolean result, be all zero.
// rows computes pieces lo to hi-1 of t = a op b: each is up to piece
// elements of one row of the output (the last broadcast axis).
func (bc *broadcast) rows(a, b, t *ref.Tensor, lo, hi int) {
	if bc.n == 0 || len(a.Data) < bc.es*ref.NumElems(a.Shape) || len(b.Data) < bc.es*ref.NumElems(b.Shape) {
		return
	}
	opc, es, axes, inner, nan := bc.opc, bc.es, bc.axes, bc.inner, bc.nan
	last := axes[len(axes)-1]
	var fa, fb, fy []float32
	if bc.kind == 'f' {
		fa, fb, fy = f32View(a), f32View(b), f32View(t)
	}
	const piece = 8192
	row := func(o, j0, j1 int) {
		ia, ib := 0, 0
		r := o
		for k := len(axes) - 2; k >= 0; k-- {
			i := r % axes[k].n
			r /= axes[k].n
			ia += i * axes[k].sa
			ib += i * axes[k].sb
		}
		ia += j0 * last.sa
		ib += j0 * last.sb
		y := t.Data[o*inner*bc.oes:]
		switch bc.kind {
		case 'f':
			binaryF32(opc, fa[ia:], last.sa, fb[ib:], last.sb, fy[o*inner+j0:o*inner+j1], nan)
		case 'i':
			ja, jb := ia, ib
			for j := j0; j < j1; j++ {
				var u, v int64
				if es == 8 {
					u = int64(binary.LittleEndian.Uint64(a.Data[8*ja:]))
					v = int64(binary.LittleEndian.Uint64(b.Data[8*jb:]))
				} else {
					u = int64(int32(binary.LittleEndian.Uint32(a.Data[4*ja:])))
					v = int64(int32(binary.LittleEndian.Uint32(b.Data[4*jb:])))
				}
				if (opc == mim.OpCmplt && u < v) || (opc == mim.OpCmpeq && u == v) {
					y[j] = 1
				}
				ja += last.sa
				jb += last.sb
			}
		case 'b':
			ja, jb := ia, ib
			for j := j0; j < j1; j++ {
				u, v := a.Data[ja] != 0, b.Data[jb] != 0
				var r bool
				switch opc {
				case mim.OpAnd:
					r = u && v
				case mim.OpOr:
					r = u || v
				case mim.OpXor:
					r = u != v
				default:
					r = u == v
				}
				if r {
					y[j] = 1
				}
				ja += last.sa
				jb += last.sb
			}
		}
	}
	per := bc.per
	for w := lo; w < hi; w++ {
		o, p := w/per, w%per
		row(o, p*piece, min((p+1)*piece, inner))
	}
}

// run computes t = a op b over the workers.
func (bc *broadcast) run(a, b, t *ref.Tensor) {
	parallelWork(bc.outer*bc.per, bc.n, func(lo, hi int) { bc.rows(a, b, t, lo, hi) })
}

// binaryF32 sets y[j] to x[j*sx] op z[j*sz] for f32 add, mul or max, with
// the reference's NaN for a NaN result. Contiguous x with contiguous or
// scalar z run on the vector kernel when there is one, and the other
// common strides get their own loops. y is x, z or neither.
func binaryF32(opc mim.Opcode, x []float32, sx int, z []float32, sz int, y []float32, nan float32) {
	if sx == 1 && (sz == 0 || sz == 1) {
		op := kernels.BinAdd
		switch opc {
		case mim.OpMul:
			op = kernels.BinMul
		case mim.OpMax:
			op = kernels.BinMax
		}
		if kernels.BinaryF32(op, y, x, z, sz == 0, nan) {
			return
		}
	}
	n := len(y)
	switch {
	case opc == mim.OpMax && sx == 1 && sz == 1:
		x, z = x[:n], z[:n]
		for j := range y {
			y[j] = maxF32(x[j], z[j], nan)
		}
	case opc == mim.OpMax && sx == 1 && sz == 0:
		v := z[0]
		x = x[:n]
		for j := range y {
			y[j] = maxF32(x[j], v, nan)
		}
	case opc == mim.OpAdd && sx == 1 && sz == 1:
		x, z = x[:n], z[:n]
		for j := range y {
			y[j] = x[j] + z[j]
		}
	case opc == mim.OpAdd && sx == 1 && sz == 0:
		v := z[0]
		x = x[:n]
		for j := range y {
			y[j] = x[j] + v
		}
	case opc == mim.OpMul && sx == 1 && sz == 1:
		x, z = x[:n], z[:n]
		for j := range y {
			y[j] = x[j] * z[j]
		}
	case opc == mim.OpMul && sx == 1 && sz == 0:
		v := z[0]
		x = x[:n]
		for j := range y {
			y[j] = x[j] * v
		}
	default:
		ja, jb := 0, 0
		for j := range y {
			u, v := x[ja], z[jb]
			switch opc {
			case mim.OpAdd:
				y[j] = u + v
			case mim.OpMul:
				y[j] = u * v
			default:
				y[j] = maxF32(u, v, nan)
			}
			ja += sx
			jb += sz
		}
	}
	if opc != mim.OpMax {
		for j, r := range y {
			if r != r {
				y[j] = nan // the reference's NaN
			}
		}
	}
}

// maxF32 is the reference's float max (ref.floatMax) in float32: NaN if
// either operand is NaN, -0 only when both are -0.
func maxF32(a, b, nan float32) float32 {
	if a != a || b != b {
		return nan
	}
	if a == 0 && b == 0 {
		if math.Signbit(float64(a)) && math.Signbit(float64(b)) {
			return a
		}
		return 0
	}
	if a > b {
		return a
	}
	return b
}

// reshapeView reshapes without copying: the result shares its input's
// bytes. Values are never written once computed, with two exceptions that
// noViews keeps out: a variable's tensor, which an in-place scatter
// writes, and the values a run hands back (outputs and variable updates),
// which must not share bytes with anything a later run may write.
func (r *run) reshapeView(op *mim.Op, in []*ref.Tensor, dt mim.DType, shape []int) (*ref.Tensor, bool) {
	if len(in) != 1 || len(op.Inputs) != 1 || len(op.Outputs) != 1 || in[0].DType != dt ||
		r.p.noView[op.Inputs[0]] || r.p.noView[op.Outputs[0]] {
		return nil, false
	}
	x := in[0]
	a := attr(op, "shape")
	if a == nil || a.Type != mim.AttrInts || len(a.Ints) != len(shape) || ref.NumElems(shape) != x.NumElems() {
		return nil, false
	}
	for i, d := range a.Ints {
		if d != -1 && d != int64(shape[i]) {
			return nil, false
		}
	}
	n := len(x.Data)
	if !dt.IsBlock() {
		n = x.NumElems() * int(dt.BlockBytes())
	}
	if len(x.Data) < n {
		return nil, false
	}
	return &ref.Tensor{DType: dt, Shape: append([]int{}, shape...), Data: x.Data[:n:n]}, true
}

// noViews returns the values reshapeView must copy: variables and the
// results of scatters, whose bytes may be a variable's, as inputs; the
// outputs of every subgraph and the variable updates, as results.
func noViews(g *mim.Graph) map[uint32]bool {
	no := map[uint32]bool{}
	for id, v := range g.Values {
		if v.Kind == mim.KindVariable {
			no[uint32(id)] = true
		}
	}
	for si := range g.Subgraphs {
		sg := &g.Subgraphs[si]
		for _, op := range sg.Ops {
			if op.Opcode == mim.OpScatter {
				for _, id := range op.Outputs {
					no[id] = true
				}
			}
		}
		for _, id := range sg.Outputs {
			no[id] = true
		}
	}
	for _, u := range g.Updates {
		no[u.Value] = true
	}
	return no
}

// fastShrink copies the box of a shrink as runs along the last axes, in
// parallel, instead of one element at a time.
func fastShrink(a *arena, op *mim.Op, in []*ref.Tensor, dt mim.DType, shape []int) (*ref.Tensor, bool) {
	b := attr(op, "begin")
	if len(in) != 1 || in[0].DType != dt || dt.IsBlock() || b == nil || b.Type != mim.AttrInts ||
		len(b.Ints) != len(shape) || len(in[0].Shape) != len(shape) {
		return nil, false
	}
	x := in[0]
	begin := make([]int, len(shape))
	for i, v := range b.Ints {
		if v < 0 || int(v)+shape[i] > x.Shape[i] {
			return nil, false
		}
		begin[i] = int(v)
	}
	es := int(dt.BlockBytes())
	if len(x.Data) < x.NumElems()*es {
		return nil, false
	}
	t := &ref.Tensor{DType: dt, Shape: append([]int{}, shape...), Data: a.bytes(ref.NumElems(shape)*es, false)}
	copyBox(t.Data, shape, make([]int, len(shape)), x.Data, x.Shape, begin, shape, es)
	return t, true
}

// fastPad fills the padding with the pad value and copies the input into
// the middle, in parallel.
func fastPad(a *arena, op *mim.Op, in []*ref.Tensor, dt mim.DType, shape []int) (*ref.Tensor, bool) {
	if len(in) != 1 {
		return nil, false
	}
	pd, ok := newPad(op, in[0].DType, in[0].Shape, dt, shape)
	if !ok {
		return nil, false
	}
	t := &ref.Tensor{DType: dt, Shape: append([]int{}, shape...), Data: a.bytes(pd.n*pd.es, pd.zero)}
	pd.run(in[0], t, !pd.zero)
	return t, true
}

// A pad is a pad op worked out for its shapes.
type pad struct {
	off   []int  // where the input goes in the output
	zeros []int  // a zero offset per axis
	value []byte // the pad value's bytes
	zero  bool   // the value is all zero bytes
	es, n int
	shape []int
	// With planes set, only the last two axes are padded, and the input's
	// planes of h by w go to the output's of hp by wp, at top, left.
	planes                  bool
	h, w, hp, wp, top, left int
	slab                    []float32 // a run of the pad value, to copy from
}

func newPad(op *mim.Op, idt mim.DType, xShape []int, dt mim.DType, shape []int) (*pad, bool) {
	p := attr(op, "pads")
	if idt != dt || dt.IsBlock() || p == nil || p.Type != mim.AttrInts ||
		len(p.Ints) != 2*len(shape) || len(xShape) != len(shape) {
		return nil, false
	}
	pd := &pad{off: make([]int, len(shape)), zeros: make([]int, len(shape)), es: int(dt.BlockBytes()), n: ref.NumElems(shape), shape: shape}
	for i := range shape {
		lo, hi := p.Ints[2*i], p.Ints[2*i+1]
		if lo < 0 || hi < 0 || int(lo)+xShape[i]+int(hi) != shape[i] {
			return nil, false
		}
		pd.off[i] = int(lo)
	}
	// The pad value, as the reference computes it: pad one element.
	var va []mim.Attr
	if v := attr(op, "value"); v != nil {
		va = append(va, *v)
	}
	one := &ref.Tensor{DType: dt, Shape: []int{1}, Data: make([]byte, pd.es)}
	pv, err := ref.Eval(mim.OpPad, append(va, mim.Attr{Key: "pads", Type: mim.AttrInts, Ints: []int64{1, 0}}), []*ref.Tensor{one}, dt, []int{2})
	if err != nil {
		return nil, false
	}
	pd.value = pv.Data[:pd.es]
	pd.zero = true
	for _, c := range pd.value {
		pd.zero = pd.zero && c == 0
	}
	if r := len(shape); r >= 2 && pd.es == 4 {
		pd.planes = true
		for i := range r - 2 {
			pd.planes = pd.planes && xShape[i] == shape[i]
		}
		pd.h, pd.w, pd.hp, pd.wp = xShape[r-2], xShape[r-1], shape[r-2], shape[r-1]
		pd.top, pd.left = pd.off[r-2], pd.off[r-1]
		pd.slab = make([]float32, min(pd.hp*pd.wp, padSlab))
		fillF32(pd.slab, math.Float32frombits(binary.LittleEndian.Uint32(pd.value)))
	}
	return pd, true
}

// run pads x into t, filling the padding first when fill is set (a fresh
// zero buffer with a zero value needs no fill).
func (pd *pad) run(x, t *ref.Tensor, fill bool) {
	if pd.planes && len(x.Data) >= x.NumElems()*pd.es {
		pd.runPlanes(f32View(x), f32View(t), fill)
		return
	}
	if fill {
		parallelWork(pieces(pd.n, padPiece), pd.n/8, func(lo, hi int) { pd.fill(t, lo, hi) })
	}
	pd.copy(x, t)
}

// runPlanes pads x into t plane by plane, with the border filled when
// fill is set.
func (pd *pad) runPlanes(x, y []float32, fill bool) {
	v := math.Float32frombits(binary.LittleEndian.Uint32(pd.value))
	plane, planes := pd.hp*pd.wp, pd.n/max(1, pd.hp*pd.wp)
	per := max(1, 4096/max(1, plane))
	var one [1]int
	body := func(lo, hi int) {
		for p := lo * per; p < min(hi*per, planes); p++ {
			dst, src := y[p*plane:(p+1)*plane], x[p*pd.h*pd.w:(p+1)*pd.h*pd.w]
			if kernels.PadPlaneF32(dst, src, pd.h, pd.w, pd.hp, pd.wp, pd.top, pd.left, v) {
				continue
			}
			if fill {
				// The whole plane: a border's many short runs cost more.
				for done := 0; done < len(dst); {
					done += copy(dst[done:], pd.slab)
				}
			}
			if pd.h == 0 || pd.w == 0 {
				continue
			}
			at := dst[pd.top*pd.wp+pd.left:]
			if kernels.PoolF32(kernels.BinAdd, at, src, one[:], pd.w, pd.h, 1, pd.w, pd.wp, 0) {
				continue
			}
			for r := range pd.h {
				copy(at[r*pd.wp:r*pd.wp+pd.w], src[r*pd.w:(r+1)*pd.w])
			}
		}
	}
	parallelWork(pieces(planes, per), pd.n/8, body)
}

// padSlab is the most elements of a pad's slab.
const padSlab = 4096

// fillF32 sets every element of d to v.
func fillF32(d []float32, v float32) {
	if len(d) == 0 {
		return
	}
	if v == 0 && !math.Signbit(float64(v)) {
		clear(d)
		return
	}
	d[0] = v
	for done := 1; done < len(d); {
		done += copy(d[done:], d[:done])
	}
}

// padPiece is the elements of a piece of a pad's fill.
const padPiece = 4096

// fill sets pieces lo to hi-1 of t to the pad value.
func (pd *pad) fill(t *ref.Tensor, lo, hi int) {
	// One element, then the filled part copied after itself.
	d := t.Data[lo*padPiece*pd.es : min(hi*padPiece, pd.n)*pd.es]
	done := copy(d, pd.value)
	for done < len(d) {
		done += copy(d[done:], d[:done])
	}
}

// copy puts x in its place in t.
func (pd *pad) copy(x, t *ref.Tensor) {
	if len(x.Data) < x.NumElems()*pd.es {
		return
	}
	copyBox(t.Data, pd.shape, pd.off, x.Data, x.Shape, pd.zeros, x.Shape, pd.es)
}

// copyBox copies the box of the given shape at srcOff in src (of shape
// srcShape) to dstOff in dst (of shape dstShape), elements of es bytes.
// The trailing axes that both cover whole make one run.
func copyBox(dst []byte, dstShape, dstOff []int, src []byte, srcShape, srcOff, box []int, es int) {
	r := len(box)
	n := ref.NumElems(box)
	if n == 0 {
		return
	}
	if r == 0 {
		copy(dst[:es], src[:es])
		return
	}
	// The run: the last axis, and the axes before it as long as the box
	// covers the axes after them in both tensors.
	inner, k := box[r-1], r-1
	for k > 0 && box[k] == srcShape[k] && box[k] == dstShape[k] {
		k--
		inner *= box[k]
	}
	// Axes 0..k-1 are walked; axis k is part of the run from its offset.
	var ssa, dsa [8]int
	ss, ds := ssa[:0], dsa[:0]
	if r > len(ssa) {
		ss, ds = make([]int, 0, r), make([]int, 0, r)
	}
	ss, ds = ss[:r], ds[:r]
	s, d := es, es
	for i := r - 1; i >= 0; i-- {
		ss[i], ds[i] = s, d
		s *= srcShape[i]
		d *= dstShape[i]
	}
	sb, db := 0, 0
	for i := range r {
		sb += srcOff[i] * ss[i]
		db += dstOff[i] * ds[i]
	}
	runBytes := inner * es
	rows := n / inner
	if n*es/16 < minParallelWork {
		copyRows(dst, src, db, sb, box[:k], ss, ds, runBytes, 0, rows)
		return
	}
	const piece = 64 << 10 // bytes per parallel piece, at least
	per := max(1, piece/max(1, runBytes))
	parallel((rows+per-1)/per, func(lo, hi int) {
		copyRows(dst, src, db, sb, box[:k], ss, ds, runBytes, lo*per, min(hi*per, rows))
	})
}

// copyRows copies rows lo to hi-1 of a box for copyBox: the runs of
// runBytes at the positions walked over the axes of box (the strides of
// the source and destination ss and ds), from sb and db for row 0.
func copyRows(dst, src []byte, db, sb int, box, ss, ds []int, runBytes, lo, hi int) {
	k := len(box)
	var idxa [8]int
	idx := idxa[:0]
	if k > len(idxa) {
		idx = make([]int, 0, k)
	}
	idx = idx[:k]
	// Where row lo is.
	so, do := sb, db
	for i, q := k-1, lo; i >= 0; i-- {
		idx[i] = q % box[i]
		q /= box[i]
		so += idx[i] * ss[i]
		do += idx[i] * ds[i]
	}
	for row := lo; row < hi; row++ {
		copy(dst[do:do+runBytes], src[so:so+runBytes])
		// The next row: count up, carrying.
		for i := k - 1; i >= 0; i-- {
			idx[i]++
			so += ss[i]
			do += ds[i]
			if idx[i] < box[i] {
				break
			}
			so -= idx[i] * ss[i]
			do -= idx[i] * ds[i]
			idx[i] = 0
		}
	}
}
