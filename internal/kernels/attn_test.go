// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mmf, created and owned by Gembit Soultan Shirazi.
// It is licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License in the LICENSE file at the root
// of this repository or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

package kernels

import (
	"math"
	"math/rand"
	"testing"
)

func TestScoresF32(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, d := range []int{8, 64, 72, 128, 5} {
		for _, n := range []int{1, 3, 4, 7, 33} {
			q := make([]float32, d)
			k := make([]float32, n*d)
			for i := range q {
				q[i] = float32(rng.NormFloat64())
			}
			for i := range k {
				k[i] = float32(rng.NormFloat64())
			}
			out := make([]float32, n)
			ScoresF32(q, k, 0.125, out)
			for j := range out {
				var s, mag float64
				for i := range q {
					p := float64(q[i]) * float64(k[j*d+i])
					s += p
					mag += math.Abs(p)
				}
				s *= 0.125
				if tol := float64(d+2) * 0x1p-24 * mag * 0.125; math.Abs(float64(out[j])-s) > tol {
					t.Errorf("d=%d n=%d: score %d is %g, want %g within %g", d, n, j, out[j], s, tol)
				}
			}
		}
	}
}

// ulps returns the distance between a and b in units in the last place of
// float32.
func ulps(a, b float32) int64 {
	d := int64(math.Float32bits(a)) - int64(math.Float32bits(b))
	return max(d, -d)
}

func TestExpSumF32(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	x := []float32{0, -1e-8, -0.5, -0.34657359, -0.34657362, -1, -10, -87, -87.3, -87.33, -87.34, -88, -100, -1e30,
		float32(math.Inf(-1))}
	for len(x) < 1000 {
		x = append(x, -float32(rng.ExpFloat64()*8))
	}
	for _, m := range []float32{0, 3.5, -2} {
		in := make([]float32, len(x))
		for i, v := range x {
			in[i] = v + m
		}
		got := append([]float32(nil), in...)
		sum := ExpSumF32(got, m)
		var want float64
		for i, v := range in {
			e := math.Exp(float64(v - m))
			want += e
			w := float32(e)
			// Arguments whose exp is not a normal float32 may give 0.
			if got[i] == 0 && float64(v-m) < -87.33 {
				continue
			}
			if ulps(got[i], w) > 2 {
				t.Errorf("m=%g: exp(%g) is %g, want %g", m, v-m, got[i], w)
			}
		}
		if math.Abs(float64(sum)-want) > 1e-5*want {
			t.Errorf("m=%g: sum is %g, want %g", m, sum, want)
		}
	}
}

func TestWeightedSumF32(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, d := range []int{8, 24, 64, 72, 160, 5} {
		for _, n := range []int{1, 2, 9} {
			p := make([]float32, n)
			v := make([]float32, n*d)
			acc := make([]float32, d)
			for i := range p {
				p[i] = rng.Float32()
			}
			for i := range v {
				v[i] = float32(rng.NormFloat64())
			}
			for i := range acc {
				acc[i] = float32(rng.NormFloat64())
			}
			want := append([]float32(nil), acc...)
			for j, e := range p {
				AxpyF32(e, v[j*d:(j+1)*d], want)
			}
			WeightedSumF32(p, v, acc)
			for i := range acc {
				if acc[i] != want[i] {
					t.Fatalf("d=%d n=%d: element %d is %g, want %g as AxpyF32 gives", d, n, i, acc[i], want[i])
				}
			}
		}
	}
}
