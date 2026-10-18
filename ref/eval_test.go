// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mlm, created and owned by Gembit Soultan Shirazi.
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

	"github.com/gsoultan/mlm"
)

func f32s(shape []int, v ...float32) *Tensor {
	t, _ := New(mlm.F32, shape)
	for i, x := range v {
		t.SetBits(i, uint64(math.Float32bits(x)))
	}
	return t
}

// TestEvalRejects covers the type, shape and attribute rules that the
// conformance corpus, which holds only valid operations, does not reach.
func TestEvalRejects(t *testing.T) {
	x := f32s([]int{2, 3}, 1, 2, 3, 4, 5, 6)
	i32, _ := New(mlm.I32, []int{2, 3})
	q, _ := New(mlm.Q8_0, []int{32})
	f8, _ := New(mlm.F8E4M3, []int{2, 3})
	cases := []struct {
		name  string
		op    mlm.Opcode
		attrs []mlm.Attr
		in    []*Tensor
		dt    mlm.DType
		shape []int
	}{
		{"not a primitive", mlm.OpCall, nil, nil, mlm.F32, nil},
		{"wrong input count", mlm.OpAdd, nil, []*Tensor{x}, mlm.F32, []int{2, 3}},
		{"mixed dtypes", mlm.OpAdd, nil, []*Tensor{x, i32}, mlm.F32, []int{2, 3}},
		{"float op on ints", mlm.OpSqrt, nil, []*Tensor{i32}, mlm.I32, []int{2, 3}},
		{"int op on floats", mlm.OpIdiv, nil, []*Tensor{x, x}, mlm.F32, []int{2, 3}},
		{"bad broadcast", mlm.OpAdd, nil, []*Tensor{x, f32s([]int{3, 3})}, mlm.F32, []int{3, 3}},
		{"wrong declared shape", mlm.OpNeg, nil, []*Tensor{x}, mlm.F32, []int{3, 2}},
		{"wrong declared dtype", mlm.OpCmplt, nil, []*Tensor{x, x}, mlm.F32, []int{2, 3}},
		{"block outside dequantize", mlm.OpReshape, []mlm.Attr{{Key: "shape", Type: mlm.AttrInts, Ints: []int64{32}}}, []*Tensor{q}, mlm.Q8_0, []int{32}},
		{"storage float in add", mlm.OpAdd, nil, []*Tensor{f8, f8}, mlm.F8E4M3, []int{2, 3}},
		{"missing attribute", mlm.OpCast, nil, []*Tensor{x}, mlm.I32, []int{2, 3}},
		{"unknown attribute", mlm.OpNeg, []mlm.Attr{{Key: "x", Type: mlm.AttrInt}}, []*Tensor{x}, mlm.F32, []int{2, 3}},
		{"mistyped attribute", mlm.OpGather, []mlm.Attr{{Key: "axis", Type: mlm.AttrFloat}}, []*Tensor{x, i32}, mlm.F32, nil},
		{"reshape count", mlm.OpReshape, []mlm.Attr{{Key: "shape", Type: mlm.AttrInts, Ints: []int64{5}}}, []*Tensor{x}, mlm.F32, []int{5}},
		{"bad perm", mlm.OpPermute, []mlm.Attr{{Key: "perm", Type: mlm.AttrInts, Ints: []int64{0, 0}}}, []*Tensor{x}, mlm.F32, []int{2, 2}},
		{"negative pad", mlm.OpPad, []mlm.Attr{{Key: "pads", Type: mlm.AttrInts, Ints: []int64{-1, 0, 0, 0}}}, []*Tensor{x}, mlm.F32, []int{1, 3}},
		{"shrink past end", mlm.OpShrink, []mlm.Attr{{Key: "begin", Type: mlm.AttrInts, Ints: []int64{0, 0}}, {Key: "end", Type: mlm.AttrInts, Ints: []int64{3, 3}}}, []*Tensor{x}, mlm.F32, []int{3, 3}},
		{"reduce bad mode", mlm.OpReduce, []mlm.Attr{{Key: "mode", Type: mlm.AttrString, String: "min"}, {Key: "axes", Type: mlm.AttrInts}}, []*Tensor{x}, mlm.F32, []int{2, 3}},
		{"reduce repeated axis", mlm.OpReduce, []mlm.Attr{{Key: "mode", Type: mlm.AttrString, String: "add"}, {Key: "axes", Type: mlm.AttrInts, Ints: []int64{1, 1}}}, []*Tensor{x}, mlm.F32, []int{2, 1}},
		{"gather float indices", mlm.OpGather, []mlm.Attr{{Key: "axis", Type: mlm.AttrInt}}, []*Tensor{x, x}, mlm.F32, []int{2, 3, 3}},
		{"const out of range", mlm.OpConst, []mlm.Attr{{Key: "value", Type: mlm.AttrInt, Int: 300}}, nil, mlm.U8, []int{1}},
		{"const non-integral int", mlm.OpConst, []mlm.Attr{{Key: "value", Type: mlm.AttrFloat, Float: 1.5}}, nil, mlm.I32, []int{1}},
		{"bitcast width", mlm.OpBitcast, []mlm.Attr{{Key: "to", Type: mlm.AttrDType, DType: mlm.F64}}, []*Tensor{x}, mlm.F64, []int{2, 3}},
		{"dequantize to f16", mlm.OpDequantize, nil, []*Tensor{q}, mlm.F16, []int{32}},
		{"truncated data", mlm.OpNeg, nil, []*Tensor{{DType: mlm.F32, Shape: []int{4}, Data: make([]byte, 8)}}, mlm.F32, []int{4}},
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
		dt   mlm.DType
		x    float64
		sat  bool
		want uint64
	}{
		{mlm.F16, 2047.0 / 1024 * 0.9999999, false, 0x3FFF},
		{mlm.F16, 65520, false, 0x7C00},    // ties up to infinity
		{mlm.F16, 65519.99, false, 0x7BFF}, // just below the tie
		{mlm.F8E4M3, 464, true, 0x7E},      // tie between 448 and 480 goes to even 448
		{mlm.F8E4M3, 1e9, true, 0x7E},
		{mlm.F8E4M3, math.Inf(-1), true, 0xFE},
		{mlm.F8E5M2, 1e9, true, 0x7B},
		{mlm.F8E5M2, math.Inf(1), true, 0x7C},
		{mlm.BF16, math.Ldexp(1, -134), false, 0x0000}, // half the smallest subnormal ties to even 0
		{mlm.BF16, math.Ldexp(3, -135), false, 0x0001},
		{mlm.F32, math.Ldexp(1, 128), false, 0x7F800000},
	}
	for _, c := range cases {
		if got := RoundFloat(c.dt, c.x, c.sat); got != c.want {
			t.Errorf("RoundFloat(%s, %v) = %#x, want %#x", c.dt, c.x, got, c.want)
		}
	}
	// Integers beyond 2^53 round directly, without a double rounding
	// through float64: 2^62 + 2^38 + 1 is just above a tie in f32.
	if got := roundInt(mlm.F32, false, 1<<62+1<<38+1, false); got != 0x5E800001 {
		t.Errorf("roundInt = %#x, want 0x5e800001", got)
	}
}
