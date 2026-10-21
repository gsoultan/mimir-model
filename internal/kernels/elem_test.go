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

// TestPoolF32 checks PoolF32 against BinaryF32 over the taps, for every
// width under 40, a few heights and tap counts, and both strides.
func TestPoolF32(t *testing.T) {
	if !useAVX512 {
		t.Skip("no AVX-512 on this machine")
	}
	rng := rand.New(rand.NewSource(7))
	nan := float32(math.NaN())
	vals := []float32{0, float32(math.Copysign(0, -1)), 1, -1, nan, 2.5, -3}
	pick := func() float32 {
		if rng.Intn(4) == 0 {
			return vals[rng.Intn(len(vals))]
		}
		return float32(rng.NormFloat64())
	}
	for w := 1; w < 40; w++ {
		for _, rows := range []int{1, 2, 3, 4, 5, 8, 9, 11, 17} {
			for _, stride := range []int{1, 2} {
				for _, taps := range []int{1, 2, 4, 9} {
					for op := BinAdd; op <= BinMax; op++ {
						xrs := (w-1)*stride + 1 + rng.Intn(3)
						yrs := w + rng.Intn(3)
						offs := make([]int, taps)
						for i := range offs {
							offs[i] = rng.Intn(5)
						}
						x := make([]float32, 4+(rows-1)*xrs+(w-1)*stride+1)
						for i := range x {
							x[i] = pick()
						}
						ny := (rows-1)*yrs + w
						want := make([]float32, ny+3)
						for i := range want {
							want[i] = float32(i) + 100 // the gaps between rows, kept
						}
						got := append([]float32(nil), want...)
						for ti, off := range offs {
							for r := range rows {
								for j := range w {
									v := x[off+r*xrs+j*stride]
									if ti == 0 {
										want[r*yrs+j] = v
									} else {
										var o [1]float32
										BinaryF32(op, o[:], want[r*yrs+j:r*yrs+j+1], []float32{v}, false, nan)
										want[r*yrs+j] = o[0]
									}
								}
							}
						}
						if !PoolF32(op, got[:ny], x, offs, w, rows, stride, xrs, yrs, nan) {
							t.Fatal("no kernel")
						}
						for j := range got {
							if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
								t.Fatalf("w %d rows %d stride %d yrs %d taps %d op %d: element %d is %g, want %g", w, rows, stride, yrs, taps, op, j, got[j], want[j])
							}
						}
					}
				}
			}
		}
	}
}

// TestPoolGatherF32 checks PoolGatherF32 against BinaryF32 on the
// gathered elements, for every length under 40.
func TestPoolGatherF32(t *testing.T) {
	if !useAVX512 {
		t.Skip("no AVX-512 on this machine")
	}
	rng := rand.New(rand.NewSource(8))
	nan := float32(math.NaN())
	vals := []float32{0, float32(math.Copysign(0, -1)), 1, -1, nan, 2.5, -3}
	pick := func() float32 {
		if rng.Intn(4) == 0 {
			return vals[rng.Intn(len(vals))]
		}
		return float32(rng.NormFloat64())
	}
	x := make([]float32, 100)
	for i := range x {
		x[i] = pick()
	}
	for n := range 40 {
		for _, taps := range []int{1, 2, 4, 9} {
			for op := BinAdd; op <= BinMax; op++ {
				np := (n + 15) &^ 15
				if rng.Intn(2) == 0 {
					np += 16
				}
				idx := make([]int32, taps*np)
				for i := range idx {
					idx[i] = int32(rng.Intn(len(x)))
				}
				want := make([]float32, n)
				for ti := range taps {
					z := make([]float32, n)
					for j := range z {
						z[j] = x[idx[ti*np+j]]
					}
					if ti == 0 {
						copy(want, z)
					} else {
						BinaryF32(op, want, want, z, false, nan)
					}
				}
				got := make([]float32, n+3)
				got[n], got[n+1], got[n+2] = 7, 8, 9
				if !PoolGatherF32(op, got[:n], x, idx, np, taps, nan) {
					t.Fatal("no kernel")
				}
				for j := range n {
					if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
						t.Fatalf("n %d taps %d op %d: element %d is %g, want %g", n, taps, op, j, got[j], want[j])
					}
				}
				if got[n] != 7 || got[n+1] != 8 || got[n+2] != 9 {
					t.Fatalf("n %d: wrote past the row", n)
				}
			}
		}
	}
}

// TestRowsF32 checks RowsF32 against BinaryF32 row by row, for every
// plane under 40.
func TestRowsF32(t *testing.T) {
	if !useAVX512 {
		t.Skip("no AVX-512 on this machine")
	}
	rng := rand.New(rand.NewSource(9))
	nan := float32(math.NaN())
	vals := []float32{0, float32(math.Copysign(0, -1)), 1, -1, nan, 2.5, -3}
	pick := func() float32 {
		if rng.Intn(4) == 0 {
			return vals[rng.Intn(len(vals))]
		}
		return float32(rng.NormFloat64())
	}
	for plane := 1; plane < 40; plane++ {
		for _, rows := range []int{1, 2, 5} {
			for op := BinAdd; op <= BinMax; op++ {
				y := make([]float32, rows*plane+3)
				for i := range y {
					y[i] = pick()
				}
				y[rows*plane], y[rows*plane+1], y[rows*plane+2] = 7, 8, 9
				v := make([]float32, rows)
				for i := range v {
					v[i] = pick()
				}
				want := append([]float32(nil), y...)
				for r := range rows {
					BinaryF32(op, want[r*plane:(r+1)*plane], y[r*plane:(r+1)*plane], v[r:], true, nan)
				}
				if !RowsF32(op, y[:rows*plane], v, rows, plane, nan) {
					t.Fatal("no kernel")
				}
				for j := range y {
					if math.Float32bits(y[j]) != math.Float32bits(want[j]) {
						t.Fatalf("plane %d rows %d op %d: element %d is %g, want %g", plane, rows, op, j, y[j], want[j])
					}
				}
			}
		}
	}
}

// TestEpilogueF32 checks EpilogueF32 against BinaryF32 step by step,
// for every plane under 40.
func TestEpilogueF32(t *testing.T) {
	if !useAVX512 {
		t.Skip("no AVX-512 on this machine")
	}
	rng := rand.New(rand.NewSource(10))
	nan := float32(math.NaN())
	vals := []float32{0, float32(math.Copysign(0, -1)), 1, -1, nan, 2.5, -3}
	pick := func() float32 {
		if rng.Intn(4) == 0 {
			return vals[rng.Intn(len(vals))]
		}
		return float32(rng.NormFloat64())
	}
	for plane := 1; plane < 40; plane++ {
		for _, rows := range []int{1, 3} {
			for _, steps := range []int{1, 2, 3, 5} {
				ops := make([]int, steps)
				for i := range ops {
					ops[i] = rng.Intn(3)
				}
				vs := rows + rng.Intn(2)
				v := make([]float32, steps*vs)
				for i := range v {
					v[i] = pick()
				}
				y := make([]float32, rows*plane+3)
				for i := range y {
					y[i] = pick()
				}
				y[rows*plane], y[rows*plane+1], y[rows*plane+2] = 7, 8, 9
				want := append([]float32(nil), y...)
				for r := range rows {
					for s, op := range ops {
						row := want[r*plane : (r+1)*plane]
						BinaryF32(op, row, row, v[s*vs+r:], true, nan)
					}
				}
				if !EpilogueF32(y[:rows*plane], rows, plane, ops, v, vs, nan) {
					t.Fatal("no kernel")
				}
				for j := range y {
					if math.Float32bits(y[j]) != math.Float32bits(want[j]) {
						t.Fatalf("plane %d rows %d ops %v: element %d is %g, want %g", plane, rows, ops, j, y[j], want[j])
					}
				}
			}
		}
	}
}

// TestPadPlaneF32 checks PadPlaneF32 against a plain loop, for padded
// widths under 40.
func TestPadPlaneF32(t *testing.T) {
	if !useAVX512 {
		t.Skip("no AVX-512 on this machine")
	}
	rng := rand.New(rand.NewSource(11))
	for wp := 1; wp < 40; wp++ {
		for _, hp := range []int{1, 2, 5} {
			for range 4 {
				h, w := rng.Intn(hp+1), rng.Intn(wp+1)
				top, left := rng.Intn(hp-h+1), rng.Intn(wp-w+1)
				v := float32(rng.NormFloat64())
				x := make([]float32, h*w)
				for i := range x {
					x[i] = float32(rng.NormFloat64())
				}
				want := make([]float32, hp*wp+3)
				for i := range want {
					want[i] = v
				}
				want[hp*wp], want[hp*wp+1], want[hp*wp+2] = 7, 8, 9
				got := append([]float32(nil), want...)
				for r := range h {
					copy(want[(top+r)*wp+left:], x[r*w:(r+1)*w])
				}
				if !PadPlaneF32(got[:hp*wp], x, h, w, hp, wp, top, left, v) {
					t.Fatal("no kernel")
				}
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("hp %d wp %d h %d w %d top %d left %d: element %d is %g, want %g", hp, wp, h, w, top, left, i, got[i], want[i])
					}
				}
			}
		}
	}
}
