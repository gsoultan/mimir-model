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

// refBinF32 is one float32 operation of BinaryF32, as the interpreter's
// scalar code does it.
func refBinF32(op int, a, b, nan float32) float32 {
	switch op {
	case BinAdd, BinMul:
		r := a + b
		if op == BinMul {
			r = a * b
		}
		if r != r {
			return nan
		}
		return r
	}
	if a != a || b != b {
		return nan
	}
	if a == 0 && b == 0 {
		if math.Signbit(float64(a)) && math.Signbit(float64(b)) {
			return a
		}
		return 0
	}
	return max(a, b)
}

// TestBinaryF32 checks the vector kernel against the scalar operations
// bit for bit, on every length up to a few vectors, with a scalar or a
// vector second operand, in place, and with NaNs, infinities, zeros of
// both signs, equal operands and subnormals.
func TestBinaryF32(t *testing.T) {
	if !Sgemm() {
		t.Skip("no AVX-512")
	}
	rng := rand.New(rand.NewSource(9))
	inf := float32(math.Inf(1))
	special := []float32{0, float32(math.Copysign(0, -1)), 1, -1, inf, -inf, float32(math.NaN()),
		math.Float32frombits(0xffc00001), 1e-45, -1e-45, math.MaxFloat32, 3}
	pick := func() float32 {
		if rng.Intn(3) == 0 {
			return special[rng.Intn(len(special))]
		}
		return float32(rng.NormFloat64())
	}
	nan := math.Float32frombits(0x7fc00000)
	for op := BinAdd; op <= BinMax; op++ {
		for n := 0; n <= 70; n++ {
			for _, scalar := range []bool{false, true} {
				for rep := 0; rep < 20; rep++ {
					x, z := make([]float32, n+1), make([]float32, n+1)
					for j := range x {
						x[j], z[j] = pick(), pick()
						if rng.Intn(5) == 0 {
							z[j] = x[j]
						}
					}
					want := make([]float32, n)
					for j := range want {
						b := z[j]
						if scalar {
							b = z[0]
						}
						want[j] = refBinF32(op, x[j], b, nan)
					}
					y := make([]float32, n+1)
					y[n] = 7
					inPlace := rep%2 == 1
					if inPlace {
						y = x
					}
					if !BinaryF32(op, y[:n], x, z, scalar, nan) {
						t.Fatal("no kernel")
					}
					for j := range want {
						if math.Float32bits(y[j]) != math.Float32bits(want[j]) {
							t.Fatalf("op %d n %d scalar %v element %d: %x op %x = %x, want %x", op, n, scalar, j,
								math.Float32bits(x[j]), math.Float32bits(z[j]), math.Float32bits(y[j]), math.Float32bits(want[j]))
						}
					}
					if !inPlace && y[n] != 7 {
						t.Fatalf("op %d n %d: wrote past the end", op, n)
					}
				}
			}
		}
	}
}
