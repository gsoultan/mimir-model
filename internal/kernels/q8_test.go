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
	if a.Quantize(make([]float32, 48)) {
		t.Error("Quantize accepted a length that is not a multiple of 32")
	}
}

// offsets returns, for each weight of a Q4_K row, the magnitude of the
// offset dmin*m its decode subtracts (zero for other dtypes): the decode of
// the row with every quant set to zero.
func offsets(dt mmf.DType, row []byte, k int) []float64 {
	o := make([]float64, k)
	if dt != mmf.Q4_K {
		return o
	}
	z := append([]byte{}, row...)
	for b := 0; b < len(z); b += 144 {
		clear(z[b+16 : b+144])
	}
	w := make([]float32, k)
	mmf.Dequantize(dt, z, w)
	for i, v := range w {
		o[i] = math.Abs(float64(v))
	}
	return o
}

func TestRowDotQ8(t *testing.T) {
	if !useAVX2 {
		t.Skip("no 8-bit kernels on this machine")
	}
	rng := rand.New(rand.NewSource(7))
	for _, dt := range []mmf.DType{mmf.Q4_K, mmf.Q6_K, mmf.Q8_0} {
		for _, k := range []int{256, 2048, 5632} {
			row := randomRows(rng, dt, 1, k)
			if dt == mmf.Q8_0 {
				row[2] = 0x80 // a weight of -128
			}
			w := make([]float32, k)
			mmf.Dequantize(dt, row, w)
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
				if !a.Quantize(x) {
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
					t.Errorf("%s K=%d: got %g, want %g (diff %g > %g)", dt, k, got, want, math.Abs(got-want), tol)
				}
			}
		}
	}
}

func TestDotQ8RejectsNonFiniteScales(t *testing.T) {
	if !useAVX2 {
		t.Skip("no 8-bit kernels on this machine")
	}
	rng := rand.New(rand.NewSource(8))
	for _, c := range []struct {
		dt  mmf.DType
		off int
	}{{mmf.Q4_K, 0}, {mmf.Q4_K, 2}, {mmf.Q6_K, 208}, {mmf.Q8_0, 0}} {
		row := randomRows(rng, c.dt, 1, 512)
		size := int(c.dt.BlockBytes())
		row[size+c.off], row[size+c.off+1] = 0x00, 0x7C // +inf in the second block
		var a Q8
		a.Quantize(randomVec(rng, 512))
		r := NewRowDot(c.dt, 512)
		r.LoadQ8(row)
		if _, ok := r.DotQ8(&a); ok {
			t.Errorf("%s: accepted an infinite scale at byte %d", c.dt, c.off)
		}
	}
}

func BenchmarkRowDotQ8(b *testing.B) {
	if !useAVX2 {
		b.Skip("no 8-bit kernels on this machine")
	}
	rng := rand.New(rand.NewSource(9))
	const k = 2048
	var a Q8
	a.Quantize(randomVec(rng, k))
	for _, dt := range []mmf.DType{mmf.Q4_K, mmf.Q6_K, mmf.Q8_0} {
		row := randomRows(rng, dt, 1, k)
		r := NewRowDot(dt, k)
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
}
