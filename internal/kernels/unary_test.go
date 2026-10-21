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

package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// TestUnaryF32 checks SigmoidF32 and TanhF32 against float64 within the
// composites' tolerances (spec/stdlib.md section 4): sigmoid rel
// (16 + 2|x|) u, tanh abs 16 u, with the special values exact.
func TestUnaryF32(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	n := 1003
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 8)
	}
	special := []float32{0, float32(math.Copysign(0, -1)), 1e-30, -1e-30, 1e-4, -1e-4, 0.5, -0.5, 44, -44, 87, -87,
		88.8, -88.8, 89, -89, 100, -100, 1e30, -1e30, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}
	copy(x, special)
	nan := float32(math.NaN())
	const u = 0x1p-24
	for _, tc := range []struct {
		name string
		run  func(y, x []float32)
		want func(x float64) float64
		tol  func(x, want float64) float64
	}{
		{"sigmoid", func(y, x []float32) { SigmoidF32(y, x, nan) }, func(x float64) float64 { return 1 / (1 + math.Exp(-x)) },
			func(x, want float64) float64 { return (16 + 2*math.Abs(x)) * u * math.Abs(want) }},
		{"tanh", func(y, x []float32) { TanhF32(y, x, nan) }, math.Tanh,
			func(x, want float64) float64 { return 16 * u }},
	} {
		y := make([]float32, n)
		tc.run(y, x)
		for i, v := range x {
			want := tc.want(float64(v))
			got := float64(y[i])
			if math.IsNaN(want) || math.IsNaN(got) {
				if math.IsNaN(want) != math.IsNaN(got) {
					t.Errorf("%s(%g) = %g, want %g", tc.name, v, got, want)
				}
				continue
			}
			if math.IsInf(float64(v), 0) && got != want {
				t.Errorf("%s(%g) = %g, want %g", tc.name, v, got, want)
			}
			if d := math.Abs(got - want); d > tc.tol(float64(v), want) && d >= 0x1p-126 {
				t.Errorf("%s(%g) = %g, want %g (off by %g)", tc.name, v, got, want, d)
			}
		}
		// In place.
		z := append([]float32(nil), x...)
		tc.run(z, z)
		for i := range z {
			if z[i] != y[i] && !(z[i] != z[i] && y[i] != y[i]) {
				t.Fatalf("%s in place differs at %d: %g vs %g", tc.name, i, z[i], y[i])
			}
		}
	}
}

func BenchmarkUnaryF32(b *testing.B) {
	x := make([]float32, 1024)
	for i := range x {
		x[i] = float32(i%7) / 3
	}
	y := make([]float32, len(x))
	b.Run("sigmoid", func(b *testing.B) {
		for range b.N {
			SigmoidF32(y, x, 0)
		}
	})
	b.Run("tanh", func(b *testing.B) {
		for range b.N {
			TanhF32(y, x, 0)
		}
	})
}
