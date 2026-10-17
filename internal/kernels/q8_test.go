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
	"slices"
	"testing"

	"github.com/gsoultan/mim"
)

func TestQuantizeQ8(t *testing.T) {
	if !useAVX2 {
		t.Skip("no 8-bit kernels on this machine")
	}
	rng := rand.New(rand.NewSource(6))
	x := randomVec(rng, 96)
	x[40] = 0 // a block with a zero
	for i := 64; i < 96; i++ {
		x[i] = 0 // an all-zero block
	}
	var a Q8
	if !a.Quantize(x) {
		t.Fatal("Quantize refused finite input")
	}
	for i, v := range x {
		d := a.d[i/32]
		if e := math.Abs(float64(v) - float64(a.q[i])*float64(d)); e > float64(d)/2*(1+1e-6) {
			t.Errorf("x[%d] = %g quantized to %d*%g", i, v, a.q[i], d)
		}
	}
	for r := 0; r < 3; r++ {
		var sum int16
		for i := 0; i < 32; i++ {
			sum += int16(a.q[32*r+i])
		}
		if a.xs[r] != sum {
			t.Errorf("xs[%d] = %d, want %d", r, a.xs[r], sum)
		}
		for i := 0; i < 16; i++ {
			if want := 32 * (int16(a.q[32*r+2*i]) + int16(a.q[32*r+2*i+1])); a.m32[16*r+i] != want {
				t.Errorf("m32[%d] = %d, want %d", 16*r+i, a.m32[16*r+i], want)
			}
		}
	}
	// Ties go to even: with max|x| = 254, x = 1 is 0.5 steps and x = 3 is 1.5.
	y := make([]float32, 32)
	y[0], y[1], y[2] = 254, 1, 3
	a.Quantize(y)
	if a.q[0] != 127 || a.q[1] != 0 || a.q[2] != 2 {
		t.Errorf("ties: got %d %d %d, want 127 0 2", a.q[0], a.q[1], a.q[2])
	}
	for _, bad := range []float32{float32(math.Inf(1)), float32(math.NaN()), 1e30} {
		y[5] = bad
		if a.Quantize(y) {
			t.Errorf("Quantize accepted %g", bad)
		}
	}
	// Blocks of 256 share one scale; each run of 32 repeats it.
	z := randomVec(rng, 512)
	z[300] = 40
	if !a.QuantizeBlocks(z, 256) {
		t.Fatal("QuantizeBlocks refused finite input")
	}
	for r := 8; r < 16; r++ {
		if a.d[r] != 40.0/127 {
			t.Errorf("d[%d] = %g, want %g", r, a.d[r], 40.0/127)
		}
	}
	if a.q[300] != 127 {
		t.Errorf("q[300] = %d, want 127", a.q[300])
	}
	if a.QuantizeBlocks(make([]float32, 288), 256) || a.QuantizeBlocks(make([]float32, 64), 64) {
		t.Error("QuantizeBlocks accepted a bad length or block")
	}
	if a.Quantize(make([]float32, 48)) {
		t.Error("Quantize accepted a length that is not a multiple of 32")
	}
}

// TestQuantizeQ8Rejects checks that QuantizeBlocks refuses values the
// 8-bit kernels cannot take, with blocks of 256.
func TestQuantizeQ8Rejects(t *testing.T) {
	if !q8K {
		t.Skip("no 8-bit kernels on this machine")
	}
	x := randomVec(rand.New(rand.NewSource(15)), 512)
	var a Q8
	if !a.QuantizeBlocks(x, 256) {
		t.Fatal("QuantizeBlocks refused finite input")
	}
	for _, bad := range []float32{float32(math.Inf(-1)), float32(math.NaN()), 1e30} {
		x[300] = bad
		if a.QuantizeBlocks(x, 256) {
			t.Errorf("QuantizeBlocks accepted %g", bad)
		}
	}
}

// TestQuantQ8 checks the rounding of QuantizeBlocks against the float64
// formula, on values near the half-integers where it matters, and the
// maximum and sums against the plain Go versions.
func TestQuantQ8(t *testing.T) {
	if !q8K {
		t.Skip("no 8-bit kernels on this machine")
	}
	rng := rand.New(rand.NewSource(13))
	x := make([]float32, 4096)
	for trial := 0; trial < 20; trial++ {
		amax := float32(rng.ExpFloat64() * 10)
		for i := range x {
			// Half-integer steps of amax/127, nudged by a few ulps.
			h := float64(rng.Intn(255)-127) / 2
			v := float32(h * float64(amax) / 127)
			for range rng.Intn(3) {
				v = math.Nextafter32(v, float32(rng.NormFloat64()))
			}
			x[i] = max(-amax, min(amax, v))
		}
		q := make([]int8, len(x))
		quantQ8(x, q, amax)
		for i, v := range x {
			if want := int8(math.RoundToEven(127 * float64(v) / float64(amax))); q[i] != want {
				t.Fatalf("x = %g, amax = %g: got %d, want %d", v, amax, q[i], want)
			}
		}
		quantQ8Go(x, q, amax)
		for i, v := range x {
			if want := int8(math.RoundToEven(127 * float64(v) / float64(amax))); q[i] != want {
				t.Fatalf("quantQ8Go: x = %g, amax = %g: got %d, want %d", v, amax, q[i], want)
			}
		}
		x[rng.Intn(len(x))] = -amax
		if m, ok := amaxQ8(x); !ok || m != amax {
			t.Fatalf("amaxQ8 = %g, %v, want %g", m, ok, amax)
		}
		if m, ok := amaxQ8Go(x); !ok || m != amax {
			t.Fatalf("amaxQ8Go = %g, %v, want %g", m, ok, amax)
		}
		m32, xs := make([]int16, len(q)/2), make([]int16, len(q)/32)
		wm32, wxs := make([]int16, len(q)/2), make([]int16, len(q)/32)
		sumsQ8(q, m32, xs)
		sumsQ8Go(q, wm32, wxs)
		if !slices.Equal(m32, wm32) || !slices.Equal(xs, wxs) {
			t.Fatal("sumsQ8 differs from sumsQ8Go")
		}
	}
}

// offsets returns, for each weight of a Q4_K row, the magnitude of the
// offset dmin*m its decode subtracts (zero for other dtypes): the decode of
// the row with every quant set to zero.
func offsets(dt mim.DType, row []byte, k int) []float64 {
	o := make([]float64, k)
	if dt != mim.Q4_K {
		return o
	}
	z := append([]byte{}, row...)
	for b := 0; b < len(z); b += 144 {
		clear(z[b+16 : b+144])
	}
	w := make([]float32, k)
	mim.Dequantize(dt, z, w)
	for i, v := range w {
		o[i] = math.Abs(float64(v))
	}
	return o
}

func TestRowDotQ8(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, c := range []struct {
		dt    mim.DType
		block int
	}{{mim.Q4_K, 32}, {mim.Q6_K, 32}, {mim.Q8_0, 32}, {mim.Q4_K, 256}, {mim.Q6_K, 256}} {
		dt := c.dt
		if !FusedQ8Blocks(dt, c.block) {
			continue
		}
		for _, k := range []int{256, 2048, 5632} {
			row := randomRows(rng, dt, 1, k)
			if dt == mim.Q8_0 {
				row[2] = 0x80 // a weight of -128
			}
			w := make([]float32, k)
			mim.Dequantize(dt, row, w)
			off := offsets(dt, row, k)
			r := NewRowDot(dt, k)
			r.LoadQ8(row)
			for trial := 0; trial < 3; trial++ {
				x := randomVec(rng, k)
				if trial == 1 {
					for i := range x {
						x[i] *= 1e-3 // small activations
					}
				}
				x[3] = 0
				var a Q8
				if !a.QuantizeBlocks(x, c.block) {
					t.Fatal("Quantize refused")
				}
				var want, bound float64
				for i := range x {
					xq := float64(a.q[i]) * float64(a.d[i/32])
					want += xq * float64(w[i])
					bound += math.Abs(xq) * (math.Abs(float64(w[i])) + 2*off[i])
				}
				g, ok := r.DotQ8(&a)
				if !ok {
					t.Fatalf("%s: DotQ8 refused finite scales", dt)
				}
				got := float64(g)
				if tol := float64(k+2) * 0x1p-24 * bound; math.Abs(got-want) > tol {
					t.Errorf("%s/%d K=%d: got %g, want %g (diff %g > %g)", dt, c.block, k, got, want, math.Abs(got-want), tol)
				}
			}
		}
	}
}

// TestDotQ8x4 checks that the four-row kernel returns what DotQ8 returns
// for each row, bit for bit, and refuses the rows DotQ8 refuses.
func TestDotQ8x4(t *testing.T) {
	rng := rand.New(rand.NewSource(10))
	for _, kb := range [][2]int{{256, 32}, {2048, 32}, {5632, 32}, {256, 256}, {2048, 256}, {5632, 256}} {
		k, block := kb[0], kb[1]
		if !FusedQ8Blocks(mim.Q4_K, block) || (block == 32 && !FusedQ8x4(mim.Q4_K)) {
			continue
		}
		const n = 6
		var b Q8Batch
		b.Reset(k, n)
		for i := 0; i < n; i++ {
			x := randomVec(rng, k)
			if i == 2 {
				for j := range x {
					x[j] *= 1e-3
				}
			}
			if !b.Row(i).QuantizeBlocks(x, block) {
				t.Fatal("Quantize refused")
			}
		}
		for trial := 0; trial < 6; trial++ {
			dt, bad := mim.Q4_K, 2 // dmin of the first block
			if trial >= 3 {
				dt, bad = mim.Q6_K, 208 // d of the first block
			}
			row := randomRows(rng, dt, 1, k)
			if trial%3 == 2 {
				row[bad], row[bad+1] = 0x00, 0x7C // +inf
			}
			r := NewRowDot(dt, k)
			r.LoadQ8(row)
			for i := 0; i+4 <= n; i++ {
				var out [4]float32
				ok := r.DotQ8x4(&b, i, &out)
				for j := 0; j < 4; j++ {
					want, wok := r.DotQ8(b.Row(i + j))
					if ok != wok {
						t.Fatalf("%s/%d K=%d trial %d: DotQ8x4 ok=%v, DotQ8 ok=%v", dt, block, k, trial, ok, wok)
					}
					if ok && math.Float32bits(out[j]) != math.Float32bits(want) {
						t.Errorf("%s/%d K=%d rows %d+%d: got %g, DotQ8 %g", dt, block, k, i, j, out[j], want)
					}
				}
			}
		}
	}
}

func TestDotQ8RejectsNonFiniteScales(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	for _, c := range []struct {
		dt  mim.DType
		off int
	}{{mim.Q4_K, 0}, {mim.Q4_K, 2}, {mim.Q6_K, 208}, {mim.Q8_0, 0}} {
		row := randomRows(rng, c.dt, 1, 512)
		size := int(c.dt.BlockBytes())
		row[size+c.off], row[size+c.off+1] = 0x00, 0x7C // +inf in the second block
		for _, block := range []int{32, 256} {
			if !FusedQ8Blocks(c.dt, block) {
				continue
			}
			var a Q8
			a.QuantizeBlocks(randomVec(rng, 512), block)
			r := NewRowDot(c.dt, 512)
			r.LoadQ8(row)
			if _, ok := r.DotQ8(&a); ok {
				t.Errorf("%s/%d: accepted an infinite scale at byte %d", c.dt, block, c.off)
			}
		}
	}
}

func BenchmarkRowDotQ8(b *testing.B) {
	if !q8K {
		b.Skip("no 8-bit kernels on this machine")
	}
	rng := rand.New(rand.NewSource(9))
	const k = 2048
	var a Q8
	a.Quantize(randomVec(rng, k))
	for _, dt := range []mim.DType{mim.Q4_K, mim.Q6_K, mim.Q8_0} {
		row := randomRows(rng, dt, 1, k)
		r := NewRowDot(dt, k)
		if FusedQ8(dt) {
			b.Run(dt.String(), func(b *testing.B) {
				b.SetBytes(int64(len(row)))
				for i := 0; i < b.N; i++ {
					r.LoadQ8(row)
					r.DotQ8(&a)
				}
			})
			b.Run(dt.String()+"/dot", func(b *testing.B) {
				r.LoadQ8(row)
				b.SetBytes(int64(len(row)))
				for i := 0; i < b.N; i++ {
					r.DotQ8(&a)
				}
			})
		}
		if FusedQ8x4(dt) {
			var batch Q8Batch
			batch.Reset(k, 4)
			for i := 0; i < 4; i++ {
				batch.Row(i).Quantize(randomVec(rng, k))
			}
			b.Run(dt.String()+"/dot4", func(b *testing.B) {
				r.LoadQ8(row)
				b.SetBytes(4 * int64(len(row)))
				var out [4]float32
				for i := 0; i < b.N; i++ {
					r.DotQ8x4(&batch, 0, &out)
				}
			})
		}
		if FusedQ8Blocks(dt, 256) {
			var a256 Q8
			a256.QuantizeBlocks(randomVec(rng, k), 256)
			b.Run(dt.String()+"/dot256", func(b *testing.B) {
				r.LoadQ8(row)
				b.SetBytes(int64(len(row)))
				for i := 0; i < b.N; i++ {
					r.DotQ8(&a256)
				}
			})
			var batch Q8Batch
			batch.Reset(k, 4)
			for i := 0; i < 4; i++ {
				batch.Row(i).QuantizeBlocks(randomVec(rng, k), 256)
			}
			b.Run(dt.String()+"/dot4x256", func(b *testing.B) {
				r.LoadQ8(row)
				b.SetBytes(4 * int64(len(row)))
				var out [4]float32
				for i := 0; i < b.N; i++ {
					r.DotQ8x4(&batch, 0, &out)
				}
			})
		}
	}
}
