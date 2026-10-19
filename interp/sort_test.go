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
	"slices"
	"testing"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/ref"
)

func i64s(t *ref.Tensor) []int64 {
	out := make([]int64, t.NumElems())
	for i := range out {
		out[i] = int64(t.Bits(i))
	}
	return out
}

// TestTopKAndArgsortOrder checks the order of spec/stdlib.md section 7:
// NaN above +inf, -0 equal to +0, ties by lower index.
func TestTopKAndArgsortOrder(t *testing.T) {
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	x := f32t([]int{1, 7}, 1, nan, float32(math.Copysign(0, -1)), 3, 0, -inf, 3)
	k := func(n int64, largest int64) ([]float64, []int64) {
		t.Helper()
		out, err := topKV1([]mim.Attr{iAttr("k", n), iAttr("largest", largest)}, []*ref.Tensor{x}, []mim.DType{mim.F32, mim.I64}, [][]int{{1, int(n)}, {1, int(n)}})
		if err != nil {
			t.Fatal(err)
		}
		return f32s(out[0]), i64s(out[1])
	}
	vals, idx := k(4, 1)
	if !slices.Equal(idx, []int64{1, 3, 6, 0}) || !math.IsNaN(float64(vals[0])) || vals[1] != 3 || vals[3] != 1 {
		t.Errorf("top 4: %v %v", vals, idx)
	}
	if _, idx := k(3, 0); !slices.Equal(idx, []int64{5, 2, 4}) {
		t.Errorf("bottom 3: %v", idx)
	}
	sorted := func(x *ref.Tensor, attrs ...mim.Attr) []int64 {
		t.Helper()
		out, err := argsortV1(attrs, []*ref.Tensor{x}, []mim.DType{mim.I64}, [][]int{x.Shape})
		if err != nil {
			t.Fatal(err)
		}
		return i64s(out[0])
	}
	if got := sorted(x, iAttr("descending", 1)); !slices.Equal(got, []int64{1, 3, 6, 0, 2, 4, 5}) {
		t.Errorf("argsort descending: %v", got)
	}
	if got := sorted(x); !slices.Equal(got, []int64{5, 2, 4, 0, 3, 6, 1}) {
		t.Errorf("argsort ascending: %v", got)
	}
	u, _ := ref.New(mim.U64, []int{3})
	for i, v := range []uint64{math.MaxUint64, 0, math.MaxUint64} {
		u.SetBits(i, v)
	}
	if got := sorted(u); !slices.Equal(got, []int64{1, 0, 2}) {
		t.Errorf("argsort of u64: %v", got)
	}
	s, _ := ref.New(mim.I8, []int{3})
	for i, v := range []int8{-1, 5, -128} {
		s.SetBits(i, uint64(uint8(v)))
	}
	if got := sorted(s); !slices.Equal(got, []int64{2, 0, 1}) {
		t.Errorf("argsort of i8: %v", got)
	}
}

// TestTopKAlongAnInnerAxis selects along axis 0 of a [3 2] tensor.
func TestTopKAlongAnInnerAxis(t *testing.T) {
	x := f32t([]int{3, 2}, 1, 6, 5, 2, 3, 4)
	out, err := topKV1([]mim.Attr{iAttr("k", 2), iAttr("axis", 0)}, []*ref.Tensor{x}, []mim.DType{mim.F32, mim.I64}, [][]int{{2, 2}, {2, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if v, i := f32s(out[0]), i64s(out[1]); !slices.Equal(v, []float64{5, 6, 3, 4}) || !slices.Equal(i, []int64{1, 0, 2, 2}) {
		t.Errorf("values %v indices %v", v, i)
	}
	out, err = argsortV1([]mim.Attr{iAttr("axis", 0)}, []*ref.Tensor{x}, []mim.DType{mim.I64}, [][]int{{3, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if got := i64s(out[0]); !slices.Equal(got, []int64{0, 1, 2, 2, 1, 0}) {
		t.Errorf("argsort along axis 0: %v", got)
	}
	for name, attrs := range map[string][]mim.Attr{
		"k above the axis": {iAttr("k", 4), iAttr("axis", 0)},
		"no k":             {iAttr("axis", 0)},
	} {
		if _, err := topKV1(attrs, []*ref.Tensor{x}, []mim.DType{mim.F32, mim.I64}, [][]int{{2, 2}, {2, 2}}); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
