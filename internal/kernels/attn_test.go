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

	"github.com/gsoultan/mmf"
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

// TestF16Attention checks that the f16 kernels give what the float32 ones
// give on the converted keys and values, bit for bit.
func TestF16Attention(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for _, d := range []int{8, 64, 72, 160, 5} {
		for _, n := range []int{1, 3, 4, 9} {
			kb := randomRows(rng, mmf.F16, n, d)
			k := make([]float32, n*d)
			f16Row(kb, k)
			q := make([]float32, d)
			p := make([]float32, n)
			for i := range q {
				q[i] = float32(rng.NormFloat64())
			}
			for i := range p {
				p[i] = rng.Float32()
			}
			got, want := make([]float32, n), make([]float32, n)
			ScoresF16(q, kb, 0.125, got)
			ScoresF32(q, k, 0.125, want)
			for j := range got {
				if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
					t.Errorf("d=%d n=%d: score %d is %g, ScoresF32 gives %g", d, n, j, got[j], want[j])
				}
			}
			acc, wacc := make([]float32, d), make([]float32, d)
			WeightedSumF16(p, kb, acc)
			WeightedSumF32(p, k, wacc)
			for i := range acc {
				if math.Float32bits(acc[i]) != math.Float32bits(wacc[i]) {
					t.Fatalf("d=%d n=%d: element %d is %g, WeightedSumF32 gives %g", d, n, i, acc[i], wacc[i])
				}
			}
		}
	}
}

// TestSiluMulF32 checks the float32 kernel against silu computed in
// float64, within swiglu.v1's tolerance, rel (20 + 2|a|) u.
func TestSiluMulF32(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	n := 1003
	a, b, out := make([]float32, n), make([]float32, n), make([]float32, n)
	for i := range a {
		a[i] = float32(rng.NormFloat64() * 8)
		b[i] = float32(rng.NormFloat64())
	}
	special := []float32{0, float32(math.Copysign(0, -1)), 1e-30, -1e-30, 87, -87, 88.8, -88.8, 89, -89, 100, -100,
		1e30, -1e30, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}
	copy(a, special)
	SiluMulF32(a, b, out)
	const u = 0x1p-24
	for i := range a {
		x := float64(a[i])
		want := x / (1 + math.Exp(-x)) * float64(b[i])
		got := float64(out[i])
		if math.IsNaN(want) || math.IsNaN(got) {
			if math.IsNaN(want) != math.IsNaN(got) {
				t.Errorf("a=%g b=%g: got %g, want %g", a[i], b[i], got, want)
			}
			continue
		}
		if math.IsInf(want, 0) {
			if got != want {
				t.Errorf("a=%g b=%g: got %g, want %g", a[i], b[i], got, want)
			}
			continue
		}
		tol := (20 + 2*math.Abs(x)) * u * math.Abs(want)
		if math.Abs(got-want) > tol && math.Abs(got-want) >= 0x1p-126 { // results below the smallest normal float may round to zero
			t.Errorf("a=%g b=%g: got %g, want %g (diff %g, tolerance %g)", a[i], b[i], got, want, got-want, tol)
		}
	}
}

// BenchmarkAttention times one query of one head against 1024 cached keys
// and values in f16, head size 64: scores, softmax, weighted sum.
func BenchmarkAttention(b *testing.B) {
	rng := rand.New(rand.NewSource(5))
	const d, n = 64, 1024
	k, v := randomRows(rng, mmf.F16, n, d), randomRows(rng, mmf.F16, n, d)
	q, sc, acc := make([]float32, d), make([]float32, n), make([]float32, d)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	b.SetBytes(4 * n * d)
	for i := 0; i < b.N; i++ {
		ScoresF16(q, k, 0.125, sc)
		var mx float32
		for _, s := range sc {
			mx = max(mx, s)
		}
		ExpSumF32(sc, mx)
		clear(acc)
		WeightedSumF16(sc, v, acc)
	}
}

// BenchmarkSiluMul times SwiGLU's gate over the 5632 rows of TinyLlama's
// feed-forward layer.
func BenchmarkSiluMul(b *testing.B) {
	rng := rand.New(rand.NewSource(6))
	const n = 5632
	a, g, out := make([]float32, n), make([]float32, n), make([]float32, n)
	for i := range a {
		a[i], g[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	b.SetBytes(4 * n)
	for i := 0; i < b.N; i++ {
		SiluMulF32(a, g, out)
	}
}
