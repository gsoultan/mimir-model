package interp

import (
	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/ref"
)

// Primitive fast paths. With FastPaths set, a few primitives that a
// transformer runs at every step skip the reference kernels' generic
// element loops. Their results are bit-identical to the reference
// kernels'; anything unusual falls back to them.

// fastPrimitive returns handled = false when the reference kernel must run.
func (r *run) fastPrimitive(op *mmf.Op, in []*ref.Tensor, dt mmf.DType, shape []int) (*ref.Tensor, bool) {
	switch op.Opcode {
	case mmf.OpAdd, mmf.OpMul:
		return fastAddMul(op.Opcode, in, dt, shape)
	case mmf.OpPermute:
		return fastPermute(op, in, dt, shape)
	case mmf.OpScatter:
		return r.scatterInPlace(op, in, dt, shape)
	}
	return nil, false
}

// fastAddMul adds or multiplies two f32 tensors of the output's shape.
// Float32 addition and multiplication are correctly rounded, as the
// primitives require.
func fastAddMul(opc mmf.Opcode, in []*ref.Tensor, dt mmf.DType, shape []int) (*ref.Tensor, bool) {
	if len(in) != 2 || dt != mmf.F32 || in[0].DType != mmf.F32 || in[1].DType != mmf.F32 ||
		!sameInts(in[0].Shape, shape) || !sameInts(in[1].Shape, shape) {
		return nil, false
	}
	a, b := f32View(in[0]), f32View(in[1])
	y := make([]float32, len(a))
	b = b[:len(a)]
	if opc == mmf.OpAdd {
		for i := range y {
			y[i] = a[i] + b[i]
		}
	} else {
		for i := range y {
			y[i] = a[i] * b[i]
		}
	}
	return f32Tensor(shape, y), true
}

// fastPermute handles permutations that only move axes of size 1, which
// leave the elements in the same order: the result is a copy.
func fastPermute(op *mmf.Op, in []*ref.Tensor, dt mmf.DType, shape []int) (*ref.Tensor, bool) {
	p := attr(op, "perm")
	if len(in) != 1 || in[0].DType != dt || p == nil || p.Type != mmf.AttrInts ||
		len(p.Ints) != len(shape) || len(shape) != len(in[0].Shape) {
		return nil, false
	}
	x := in[0]
	seen := make([]bool, len(shape))
	last := -1
	for i, a := range p.Ints {
		if a < 0 || int(a) >= len(shape) || seen[a] || x.Shape[a] != shape[i] {
			return nil, false
		}
		seen[a] = true
		if shape[i] != 1 {
			if int(a) < last {
				return nil, false
			}
			last = int(a)
		}
	}
	return &ref.Tensor{DType: dt, Shape: append([]int{}, shape...), Data: append([]byte{}, x.Data...)}, true
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
func inplaceScatters(g *mmf.Graph) map[*mmf.Op]uint32 {
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
	out := map[*mmf.Op]uint32{}
	if len(g.Subgraphs) == 0 {
		return out
	}
	main := &g.Subgraphs[0]
	for oi := range main.Ops {
		op := &main.Ops[oi]
		if op.Opcode != mmf.OpScatter || len(op.Inputs) != 3 || len(op.Outputs) != 1 {
			continue
		}
		v, o := op.Inputs[0], op.Outputs[0]
		if g.Values[v].Kind != mmf.KindVariable || uses[v] != 1 || updates[v] != 1 || valueUpdates[v] != 0 ||
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

func isOutput(sg *mmf.Subgraph, id uint32) bool {
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

func (r *run) scatterInPlace(op *mmf.Op, in []*ref.Tensor, dt mmf.DType, shape []int) (*ref.Tensor, bool) {
	v, ok := r.p.inplace[op]
	if !ok || r.s == nil || !r.s.owned[v] || len(in) != 3 {
		return nil, false
	}
	data, ind, upd := in[0], in[1], in[2]
	ax := attr(op, "axis")
	if ax == nil || ax.Type != mmf.AttrInt || data.DType != dt || upd.DType != dt || dt.IsBlock() ||
		!sameInts(data.Shape, shape) || (ind.DType != mmf.I32 && ind.DType != mmf.I64) {
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
