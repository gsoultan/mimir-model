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
	"math/rand"
	"testing"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/internal/kernels"
	"github.com/gsoultan/mimir-model/ref"
)

func intsAttr(k string, v ...int64) mim.Attr { return mim.Attr{Key: k, Type: mim.AttrInts, Ints: v} }
func intAttr(k string, v int64) mim.Attr     { return mim.Attr{Key: k, Type: mim.AttrInt, Int: v} }

// naiveConv is the definition of a grouped 2-D convolution, in float64.
func naiveConv(x, w, bias []float32, nb, c, h, wd, f, kh, kw, g, sh, sw, dh, dw, top, left, ho, wo int) []float64 {
	cg, fg := c/g, f/g
	y := make([]float64, nb*f*ho*wo)
	for n := 0; n < nb; n++ {
		for o := 0; o < f; o++ {
			for oy := 0; oy < ho; oy++ {
				for ox := 0; ox < wo; ox++ {
					s := 0.0
					if bias != nil {
						s = float64(bias[o])
					}
					for ci := 0; ci < cg; ci++ {
						for i := 0; i < kh; i++ {
							for j := 0; j < kw; j++ {
								iy, ix := oy*sh+i*dh-top, ox*sw+j*dw-left
								if iy < 0 || iy >= h || ix < 0 || ix >= wd {
									continue
								}
								s += float64(x[((n*c+o/fg*cg+ci)*h+iy)*wd+ix]) * float64(w[((o*cg+ci)*kh+i)*kw+j])
							}
						}
					}
					y[((n*f+o)*ho+oy)*wo+ox] = s
				}
			}
		}
	}
	return y
}

func TestConv2d(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	cases := []struct {
		nb, c, h, w, f, kh, kw, g, sh, sw, dh, dw, top, bottom, left, right int
		bias                                                                bool
	}{
		{1, 3, 6, 5, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, true},
		{2, 4, 7, 8, 6, 3, 2, 2, 2, 3, 1, 1, 0, 1, 2, 0, false},
		{1, 4, 9, 9, 4, 3, 3, 4, 1, 2, 2, 1, 2, 2, 1, 1, true},
		{1, 2, 5, 5, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false},
	}
	for ci, tc := range cases {
		x := randF32(rng, []int{tc.nb, tc.c, tc.h, tc.w}, 1)
		w := randF32(rng, []int{tc.f, tc.c / tc.g, tc.kh, tc.kw}, 1)
		in := []*ref.Tensor{x, w}
		var bias []float32
		if tc.bias {
			b := randF32(rng, []int{tc.f}, 1)
			in = append(in, b)
			bias = f32View(b)
		}
		attrs := []mim.Attr{
			intsAttr("strides", int64(tc.sh), int64(tc.sw)),
			intsAttr("dilations", int64(tc.dh), int64(tc.dw)),
			intsAttr("pads", int64(tc.top), int64(tc.bottom), int64(tc.left), int64(tc.right)),
			iAttr("groups", int64(tc.g)),
		}
		body := checkFast(t, "conv2d.v1", attrs, in, 1e-5)
		ho := (tc.h+tc.top+tc.bottom-tc.dh*(tc.kh-1)-1)/tc.sh + 1
		wo := (tc.w+tc.left+tc.right-tc.dw*(tc.kw-1)-1)/tc.sw + 1
		if !sameInts(body.Shape, []int{tc.nb, tc.f, ho, wo}) {
			t.Fatalf("case %d: shape %v", ci, body.Shape)
		}
		want := naiveConv(f32View(x), f32View(w), bias, tc.nb, tc.c, tc.h, tc.w, tc.f, tc.kh, tc.kw, tc.g,
			tc.sh, tc.sw, tc.dh, tc.dw, tc.top, tc.left, ho, wo)
		got := f32View(body)
		for i := range want {
			if d := float64(got[i]) - want[i]; d > 1e-5 || d < -1e-5 {
				t.Fatalf("case %d element %d: body %g, definition %g", ci, i, got[i], want[i])
			}
		}
	}
	if _, err := stdlib["conv2d.v1"]([]mim.Attr{iAttr("groups", 2)},
		[]*ref.Tensor{randF32(rng, []int{1, 3, 4, 4}, 1), randF32(rng, []int{2, 1, 1, 1}, 1)}, []mim.DType{mim.F32}, nil); err == nil {
		t.Fatal("3 channels in 2 groups were accepted")
	}
}

// TestConvGemm checks the matrix-product convolution against the
// definition, computed in float64 and rounded to float32, within the
// composite's tolerance, Q u sum |x w| + u |B|, plus 2 u |result| for the
// two roundings of a sum to float32 (after the bias is added, and the
// reference's own). The shapes have tiles with fewer than 12 filters and
// 32 output positions, receptive fields longer than a run of the kernel,
// groups, strides, dilations, padding, and 1x1 convolutions that use the
// input as it is.
func TestConvGemm(t *testing.T) {
	if !kernels.Sgemm() {
		t.Skip("no matrix-product kernel on this machine")
	}
	rng := rand.New(rand.NewSource(8))
	cases := []struct {
		nb, c, h, w, f, kh, kw, g, sh, sw, dh, dw, top, bottom, left, right int
		bias                                                                bool
	}{
		{1, 3, 6, 5, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, true},
		{2, 4, 7, 8, 6, 3, 2, 2, 2, 3, 1, 1, 0, 1, 2, 0, false},
		{1, 4, 9, 9, 4, 3, 3, 4, 1, 2, 2, 1, 2, 2, 1, 1, true},
		{1, 2, 5, 5, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false},
		{2, 40, 11, 13, 25, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, true},
		{1, 64, 9, 9, 30, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, true},
		{1, 32, 12, 12, 16, 1, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, false},
		{1, 3, 33, 35, 13, 7, 7, 1, 2, 2, 1, 1, 3, 3, 3, 3, false},
		{3, 8, 10, 10, 8, 3, 3, 8, 1, 1, 1, 1, 1, 1, 1, 1, true},
		{1, 6, 4, 4, 6, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, false},
	}
	const u = 1.0 / (1 << 24)
	p := &Program{isFixed: map[*ref.Tensor]bool{}}
	for ci, tc := range cases {
		x := randF32(rng, []int{tc.nb, tc.c, tc.h, tc.w}, 1)
		w := randF32(rng, []int{tc.f, tc.c / tc.g, tc.kh, tc.kw}, 1)
		in := []*ref.Tensor{x, w}
		var bias []float32
		if tc.bias {
			b := randF32(rng, []int{tc.f}, 1)
			in = append(in, b)
			bias = f32View(b)
		}
		attrs := []mim.Attr{
			intsAttr("strides", int64(tc.sh), int64(tc.sw)),
			intsAttr("dilations", int64(tc.dh), int64(tc.dw)),
			intsAttr("pads", int64(tc.top), int64(tc.bottom), int64(tc.left), int64(tc.right)),
			iAttr("groups", int64(tc.g)),
		}
		out, handled, err := p.convGemm(attrs, in, []mim.DType{mim.F32}, nil)
		if err != nil || !handled {
			t.Fatalf("case %d: handled %v, %v", ci, handled, err)
		}
		ho := (tc.h+tc.top+tc.bottom-tc.dh*(tc.kh-1)-1)/tc.sh + 1
		wo := (tc.w+tc.left+tc.right-tc.dw*(tc.kw-1)-1)/tc.sw + 1
		if !sameInts(out[0].Shape, []int{tc.nb, tc.f, ho, wo}) {
			t.Fatalf("case %d: shape %v", ci, out[0].Shape)
		}
		want := naiveConv(f32View(x), f32View(w), bias, tc.nb, tc.c, tc.h, tc.w, tc.f, tc.kh, tc.kw, tc.g,
			tc.sh, tc.sw, tc.dh, tc.dw, tc.top, tc.left, ho, wo)
		// The sums of |x w| over each receptive field.
		xa, wa := absF32(f32View(x)), absF32(f32View(w))
		abs := naiveConv(xa, wa, nil, tc.nb, tc.c, tc.h, tc.w, tc.f, tc.kh, tc.kw, tc.g,
			tc.sh, tc.sw, tc.dh, tc.dw, tc.top, tc.left, ho, wo)
		q := float64(tc.c / tc.g * tc.kh * tc.kw)
		got := f32View(out[0])
		for i := range want {
			r := float64(float32(want[i]))
			tol := q*u*abs[i] + 2*u*math.Abs(r)
			if bias != nil {
				tol += u * math.Abs(float64(bias[i/(ho*wo)%tc.f]))
			}
			if d := math.Abs(float64(got[i]) - r); d > tol {
				t.Fatalf("case %d element %d: %g, definition %g, tolerance %g", ci, i, got[i], r, tol)
			}
		}
	}
}

func absF32(v []float32) []float32 {
	a := make([]float32, len(v))
	for i, x := range v {
		a[i] = float32(math.Abs(float64(x)))
	}
	return a
}
