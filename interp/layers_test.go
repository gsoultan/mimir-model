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
	"testing"

	"github.com/gsoultan/mim"
	"github.com/gsoultan/mim/ref"
)

func TestCumsum(t *testing.T) {
	x := f32t([]int{2, 3}, 1, 2, 3, 4, 5, 6)
	for _, c := range []struct {
		exclusive, reverse int64
		want               []float64
	}{
		{0, 0, []float64{1, 3, 6, 4, 9, 15}},
		{1, 0, []float64{0, 1, 3, 0, 4, 9}},
		{0, 1, []float64{6, 5, 3, 15, 11, 6}},
		{1, 1, []float64{5, 3, 0, 11, 6, 0}},
	} {
		y, err := cumsumV1([]mim.Attr{iAttr("axis", -1), iAttr("exclusive", c.exclusive), iAttr("reverse", c.reverse)},
			[]*ref.Tensor{x}, []mim.DType{mim.F32}, [][]int{{2, 3}})
		if err != nil {
			t.Fatal(err)
		}
		if got := f32s(y[0]); !sameFloats(got, c.want) {
			t.Errorf("exclusive %d reverse %d: %v, want %v", c.exclusive, c.reverse, got, c.want)
		}
	}
	// Integers wrap, and the axis is required.
	n := i64t([]int{3}, math.MaxInt64, 1, 1)
	y, err := cumsumV1([]mim.Attr{iAttr("axis", 0)}, []*ref.Tensor{n}, []mim.DType{mim.I64}, [][]int{{3}})
	if err != nil {
		t.Fatal(err)
	}
	if v := int64(y[0].Bits(2)); v != math.MinInt64+1 {
		t.Errorf("wrapped sum %d", v)
	}
	if _, err := cumsumV1(nil, []*ref.Tensor{n}, []mim.DType{mim.I64}, [][]int{{3}}); err == nil {
		t.Error("cumsum.v1 without an axis was accepted")
	}
}

func TestErfAndGelu(t *testing.T) {
	x := f32t([]int{5}, -2, -0.5, 0, 0.5, 2)
	run := func(f compositeFunc) []float64 {
		y, err := f(nil, []*ref.Tensor{x}, []mim.DType{mim.F32}, [][]int{{5}})
		if err != nil {
			t.Fatal(err)
		}
		return f32s(y[0])
	}
	erf, ge, gt := run(stdlib["erf.v1"]), run(stdlib["gelu_erf.v1"]), run(stdlib["gelu_tanh.v1"])
	for i, v := range f32s(x) {
		if d := math.Abs(erf[i] - math.Erf(v)); d > 1.6e-7+16*0x1p-24 {
			t.Errorf("erf(%g) = %g, off by %g", v, erf[i], d)
		}
		exact := 0.5 * v * (1 + math.Erf(v/math.Sqrt2))
		if d := math.Abs(ge[i] - exact); d > 1e-6 {
			t.Errorf("gelu_erf(%g) = %g, want %g", v, ge[i], exact)
		}
		approx := 0.5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*v*v*v)))
		if d := math.Abs(gt[i] - approx); d > 1e-6 {
			t.Errorf("gelu_tanh(%g) = %g, want %g", v, gt[i], approx)
		}
	}
}

func TestLayerNormAndMean(t *testing.T) {
	x := f32t([]int{2, 2}, 1, 3, -2, 2)
	scale, bias := f32t([]int{2}, 2, 1), f32t([]int{2}, 0, 10)
	y, err := layerNormV1([]mim.Attr{iAttr("axis", 1), fAttr("epsilon", 0)}, []*ref.Tensor{x, scale, bias},
		[]mim.DType{mim.F32}, [][]int{{2, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if got := f32s(y[0]); !sameFloats(got, []float64{-2, 11, -2, 11}) {
		t.Errorf("layer_norm: %v", got)
	}
	m, err := meanV1([]mim.Attr{intsAttr("axes", 0)}, []*ref.Tensor{x}, []mim.DType{mim.F32}, [][]int{{1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if got := f32s(m[0]); !sameFloats(got, []float64{-0.5, 2.5}) {
		t.Errorf("mean: %v", got)
	}
	if _, err := layerNormV1(nil, []*ref.Tensor{x, f32t([]int{3}, 1, 1, 1)}, []mim.DType{mim.F32}, [][]int{{2, 2}}); err == nil {
		t.Error("a scale of the wrong shape was accepted")
	}
}

func sameFloats(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
