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

package ref

import (
	"math"
	"testing"

	mim "github.com/gsoultan/mimir-model"
)

func f32s(shape []int, v ...float32) *Tensor {
	t, _ := New(mim.F32, shape)
	for i, x := range v {
		t.SetBits(i, uint64(math.Float32bits(x)))
	}
	return t
}

// TestEvalRejects covers the type, shape and attribute rules that the
// conformance corpus, which holds only valid operations, does not reach.
func TestEvalRejects(t *testing.T) {
	x := f32s([]int{2, 3}, 1, 2, 3, 4, 5, 6)
	i32, _ := New(mim.I32, []int{2, 3})
	q, _ := New(mim.Q8_0, []int{32})
	f8, _ := New(mim.F8E4M3, []int{2, 3})
	cases := []struct {
		name  string
		op    mim.Opcode
		attrs []mim.Attr
		in    []*Tensor
		dt    mim.DType
		shape []int
	}{
		{"not a primitive", mim.OpCall, nil, nil, mim.F32, nil},
		{"wrong input count", mim.OpAdd, nil, []*Tensor{x}, mim.F32, []int{2, 3}},
		{"mixed dtypes", mim.OpAdd, nil, []*Tensor{x, i32}, mim.F32, []int{2, 3}},
		{"float op on ints", mim.OpSqrt, nil, []*Tensor{i32}, mim.I32, []int{2, 3}},
		{"int op on floats", mim.OpIdiv, nil, []*Tensor{x, x}, mim.F32, []int{2, 3}},
		{"bad broadcast", mim.OpAdd, nil, []*Tensor{x, f32s([]int{3, 3})}, mim.F32, []int{3, 3}},
		{"wrong declared shape", mim.OpNeg, nil, []*Tensor{x}, mim.F32, []int{3, 2}},
		{"wrong declared dtype", mim.OpCmplt, nil, []*Tensor{x, x}, mim.F32, []int{2, 3}},
		{"block outside dequantize", mim.OpReshape, []mim.Attr{{Key: "shape", Type: mim.AttrInts, Ints: []int64{32}}}, []*Tensor{q}, mim.Q8_0, []int{32}},
		{"storage float in add", mim.OpAdd, nil, []*Tensor{f8, f8}, mim.F8E4M3, []int{2, 3}},
		{"missing attribute", mim.OpCast, nil, []*Tensor{x}, mim.I32, []int{2, 3}},
		{"unknown attribute", mim.OpNeg, []mim.Attr{{Key: "x", Type: mim.AttrInt}}, []*Tensor{x}, mim.F32, []int{2, 3}},
		{"mistyped attribute", mim.OpGather, []mim.Attr{{Key: "axis", Type: mim.AttrFloat}}, []*Tensor{x, i32}, mim.F32, nil},
		{"reshape count", mim.OpReshape, []mim.Attr{{Key: "shape", Type: mim.AttrInts, Ints: []int64{5}}}, []*Tensor{x}, mim.F32, []int{5}},
		{"bad perm", mim.OpPermute, []mim.Attr{{Key: "perm", Type: mim.AttrInts, Ints: []int64{0, 0}}}, []*Tensor{x}, mim.F32, []int{2, 2}},
		{"negative pad", mim.OpPad, []mim.Attr{{Key: "pads", Type: mim.AttrInts, Ints: []int64{-1, 0, 0, 0}}}, []*Tensor{x}, mim.F32, []int{1, 3}},
		{"shrink past end", mim.OpShrink, []mim.Attr{{Key: "begin", Type: mim.AttrInts, Ints: []int64{0, 0}}, {Key: "end", Type: mim.AttrInts, Ints: []int64{3, 3}}}, []*Tensor{x}, mim.F32, []int{3, 3}},
		{"reduce bad mode", mim.OpReduce, []mim.Attr{{Key: "mode", Type: mim.AttrString, String: "min"}, {Key: "axes", Type: mim.AttrInts}}, []*Tensor{x}, mim.F32, []int{2, 3}},
		{"reduce repeated axis", mim.OpReduce, []mim.Attr{{Key: "mode", Type: mim.AttrString, String: "add"}, {Key: "axes", Type: mim.AttrInts, Ints: []int64{1, 1}}}, []*Tensor{x}, mim.F32, []int{2, 1}},
		{"gather float indices", mim.OpGather, []mim.Attr{{Key: "axis", Type: mim.AttrInt}}, []*Tensor{x, x}, mim.F32, []int{2, 3, 3}},
		{"const out of range", mim.OpConst, []mim.Attr{{Key: "value", Type: mim.AttrInt, Int: 300}}, nil, mim.U8, []int{1}},
		{"const non-integral int", mim.OpConst, []mim.Attr{{Key: "value", Type: mim.AttrFloat, Float: 1.5}}, nil, mim.I32, []int{1}},
		{"bitcast width", mim.OpBitcast, []mim.Attr{{Key: "to", Type: mim.AttrDType, DType: mim.F64}}, []*Tensor{x}, mim.F64, []int{2, 3}},
		{"dequantize to f16", mim.OpDequantize, nil, []*Tensor{q}, mim.F16, []int{32}},
		{"truncated data", mim.OpNeg, nil, []*Tensor{{DType: mim.F32, Shape: []int{4}, Data: make([]byte, 8)}}, mim.F32, []int{4}},
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
		dt   mim.DType
		x    float64
		sat  bool
		want uint64
	}{
		{mim.F16, 2047.0 / 1024 * 0.9999999, false, 0x3FFF},
		{mim.F16, 65520, false, 0x7C00},    // ties up to infinity
		{mim.F16, 65519.99, false, 0x7BFF}, // just below the tie
		{mim.F8E4M3, 464, true, 0x7E},      // tie between 448 and 480 goes to even 448
		{mim.F8E4M3, 1e9, true, 0x7E},
		{mim.F8E4M3, math.Inf(-1), true, 0xFE},
		{mim.F8E5M2, 1e9, true, 0x7B},
		{mim.F8E5M2, math.Inf(1), true, 0x7C},
		{mim.BF16, math.Ldexp(1, -134), false, 0x0000}, // half the smallest subnormal ties to even 0
		{mim.BF16, math.Ldexp(3, -135), false, 0x0001},
		{mim.F32, math.Ldexp(1, 128), false, 0x7F800000},
	}
	for _, c := range cases {
		if got := RoundFloat(c.dt, c.x, c.sat); got != c.want {
			t.Errorf("RoundFloat(%s, %v) = %#x, want %#x", c.dt, c.x, got, c.want)
		}
	}
	// Integers beyond 2^53 round directly, without a double rounding
	// through float64: 2^62 + 2^38 + 1 is just above a tie in f32.
	if got := roundInt(mim.F32, false, 1<<62+1<<38+1, false); got != 0x5E800001 {
		t.Errorf("roundInt = %#x, want 0x5e800001", got)
	}
}
