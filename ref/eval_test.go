package ref

import (
	"math"
	"testing"

	"github.com/gsoultan/mmf"
)

func f32s(shape []int, v ...float32) *Tensor {
	t, _ := New(mmf.F32, shape)
	for i, x := range v {
		t.SetBits(i, uint64(math.Float32bits(x)))
	}
	return t
}

// TestEvalRejects covers the type, shape and attribute rules that the
// conformance corpus, which holds only valid operations, does not reach.
func TestEvalRejects(t *testing.T) {
	x := f32s([]int{2, 3}, 1, 2, 3, 4, 5, 6)
	i32, _ := New(mmf.I32, []int{2, 3})
	q, _ := New(mmf.Q8_0, []int{32})
	f8, _ := New(mmf.F8E4M3, []int{2, 3})
	cases := []struct {
		name  string
		op    mmf.Opcode
		attrs []mmf.Attr
		in    []*Tensor
		dt    mmf.DType
		shape []int
	}{
		{"not a primitive", mmf.OpCall, nil, nil, mmf.F32, nil},
		{"wrong input count", mmf.OpAdd, nil, []*Tensor{x}, mmf.F32, []int{2, 3}},
		{"mixed dtypes", mmf.OpAdd, nil, []*Tensor{x, i32}, mmf.F32, []int{2, 3}},
		{"float op on ints", mmf.OpSqrt, nil, []*Tensor{i32}, mmf.I32, []int{2, 3}},
		{"int op on floats", mmf.OpIdiv, nil, []*Tensor{x, x}, mmf.F32, []int{2, 3}},
		{"bad broadcast", mmf.OpAdd, nil, []*Tensor{x, f32s([]int{3, 3})}, mmf.F32, []int{3, 3}},
		{"wrong declared shape", mmf.OpNeg, nil, []*Tensor{x}, mmf.F32, []int{3, 2}},
		{"wrong declared dtype", mmf.OpCmplt, nil, []*Tensor{x, x}, mmf.F32, []int{2, 3}},
		{"block outside dequantize", mmf.OpReshape, []mmf.Attr{{Key: "shape", Type: mmf.AttrInts, Ints: []int64{32}}}, []*Tensor{q}, mmf.Q8_0, []int{32}},
		{"storage float in add", mmf.OpAdd, nil, []*Tensor{f8, f8}, mmf.F8E4M3, []int{2, 3}},
		{"missing attribute", mmf.OpCast, nil, []*Tensor{x}, mmf.I32, []int{2, 3}},
		{"unknown attribute", mmf.OpNeg, []mmf.Attr{{Key: "x", Type: mmf.AttrInt}}, []*Tensor{x}, mmf.F32, []int{2, 3}},
		{"mistyped attribute", mmf.OpGather, []mmf.Attr{{Key: "axis", Type: mmf.AttrFloat}}, []*Tensor{x, i32}, mmf.F32, nil},
		{"reshape count", mmf.OpReshape, []mmf.Attr{{Key: "shape", Type: mmf.AttrInts, Ints: []int64{5}}}, []*Tensor{x}, mmf.F32, []int{5}},
		{"bad perm", mmf.OpPermute, []mmf.Attr{{Key: "perm", Type: mmf.AttrInts, Ints: []int64{0, 0}}}, []*Tensor{x}, mmf.F32, []int{2, 2}},
		{"negative pad", mmf.OpPad, []mmf.Attr{{Key: "pads", Type: mmf.AttrInts, Ints: []int64{-1, 0, 0, 0}}}, []*Tensor{x}, mmf.F32, []int{1, 3}},
		{"shrink past end", mmf.OpShrink, []mmf.Attr{{Key: "begin", Type: mmf.AttrInts, Ints: []int64{0, 0}}, {Key: "end", Type: mmf.AttrInts, Ints: []int64{3, 3}}}, []*Tensor{x}, mmf.F32, []int{3, 3}},
		{"reduce bad mode", mmf.OpReduce, []mmf.Attr{{Key: "mode", Type: mmf.AttrString, String: "min"}, {Key: "axes", Type: mmf.AttrInts}}, []*Tensor{x}, mmf.F32, []int{2, 3}},
		{"reduce repeated axis", mmf.OpReduce, []mmf.Attr{{Key: "mode", Type: mmf.AttrString, String: "add"}, {Key: "axes", Type: mmf.AttrInts, Ints: []int64{1, 1}}}, []*Tensor{x}, mmf.F32, []int{2, 1}},
		{"gather float indices", mmf.OpGather, []mmf.Attr{{Key: "axis", Type: mmf.AttrInt}}, []*Tensor{x, x}, mmf.F32, []int{2, 3, 3}},
		{"const out of range", mmf.OpConst, []mmf.Attr{{Key: "value", Type: mmf.AttrInt, Int: 300}}, nil, mmf.U8, []int{1}},
		{"const non-integral int", mmf.OpConst, []mmf.Attr{{Key: "value", Type: mmf.AttrFloat, Float: 1.5}}, nil, mmf.I32, []int{1}},
		{"bitcast width", mmf.OpBitcast, []mmf.Attr{{Key: "to", Type: mmf.AttrDType, DType: mmf.F64}}, []*Tensor{x}, mmf.F64, []int{2, 3}},
		{"dequantize to f16", mmf.OpDequantize, nil, []*Tensor{q}, mmf.F16, []int{32}},
		{"truncated data", mmf.OpNeg, nil, []*Tensor{{DType: mmf.F32, Shape: []int{4}, Data: make([]byte, 8)}}, mmf.F32, []int{4}},
	}
	for _, c := range cases {
		if _, err := Eval(c.op, c.attrs, c.in, c.dt, c.shape); err == nil {
			t.Errorf("%s: accepted", c.name)
		}
	}
}

// TestRoundExactCarry checks rounding at the boundaries the corpus reaches
// only indirectly: a carry into the next binade and into overflow.
func TestRoundExactCarry(t *testing.T) {
	cases := []struct {
		dt   mmf.DType
		x    float64
		sat  bool
		want uint64
	}{
		{mmf.F16, 2047.0 / 1024 * 0.9999999, false, 0x3FFF},
		{mmf.F16, 65520, false, 0x7C00},    // ties up to infinity
		{mmf.F16, 65519.99, false, 0x7BFF}, // just below the tie
		{mmf.F8E4M3, 464, true, 0x7E},      // tie between 448 and 480 goes to even 448
		{mmf.F8E4M3, 1e9, true, 0x7E},
		{mmf.F8E4M3, math.Inf(-1), true, 0xFE},
		{mmf.F8E5M2, 1e9, true, 0x7B},
		{mmf.F8E5M2, math.Inf(1), true, 0x7C},
		{mmf.BF16, math.Ldexp(1, -134), false, 0x0000}, // half the smallest subnormal ties to even 0
		{mmf.BF16, math.Ldexp(3, -135), false, 0x0001},
		{mmf.F32, math.Ldexp(1, 128), false, 0x7F800000},
	}
	for _, c := range cases {
		if got := RoundFloat(c.dt, c.x, c.sat); got != c.want {
			t.Errorf("RoundFloat(%s, %v) = %#x, want %#x", c.dt, c.x, got, c.want)
		}
	}
	// Integers beyond 2^53 round directly, without a double rounding
	// through float64: 2^62 + 2^38 + 1 is just above a tie in f32.
	if got := roundInt(mmf.F32, false, 1<<62+1<<38+1, false); got != 0x5E800001 {
		t.Errorf("roundInt = %#x, want 0x5e800001", got)
	}
}
