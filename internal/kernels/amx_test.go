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

	mim "github.com/gsoultan/mimir-model"
)

// TestTileDot checks that the AMX and VNNI kernels return what
// RowDot.DotQ8 returns, bit for bit, for every row of activations they
// are given.
func TestTileDot(t *testing.T) {
	if !TileQ8(mim.Q4_K) {
		t.Skip("no AMX or VNNI kernels on this machine")
	}
	testTileDot(t)
	if useAMX && useVNNI {
		t.Run("vnni", func(t *testing.T) {
			noAMX = true
			defer func() { noAMX = false }()
			testTileDot(t)
		})
	}
}

func testTileDot(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, dt := range []mim.DType{mim.Q4_K, mim.Q6_K} {
		for _, k := range []int{256, 2048, 5632} {
			for _, rows := range []int{1, 5, 16, 21, 37} {
				var b Q8Batch
				b.Reset(k, rows)
				use := make([]bool, rows)
				for i := range rows {
					x := randomVec(rng, k)
					switch i % 7 {
					case 3:
						for j := range x {
							x[j] *= 1e-3
						}
					case 5:
						for j := range x {
							x[j] = 2 // every quant 127
						}
					}
					use[i] = i%9 != 4 // some rows are left out
					if !b.Row(i).QuantizeBlocks(x, 256) {
						t.Fatal("QuantizeBlocks refused")
					}
				}
				var a TileActs
				a.Set(&b, use)
				w := randomRows(rng, dt, TileRows, k)
				if k == 2048 && dt == mim.Q4_K {
					// The largest integer sums: every scale, offset and quant at its
					// maximum in row 1.
					rb := int(uint64(k) / 256 * dt.BlockBytes())
					for o := rb; o < 2*rb; o += 144 {
						for j := o + 4; j < o+144; j++ {
							w[j] = 0xFF
						}
					}
				}
				d := NewTileDot(dt, k)
				out := make([]float32, rows*TileRows)
				if !d.Dot(w, &a, out, TileRows, nil) {
					t.Fatalf("%s K=%d: Dot refused finite weights", dt, k)
				}
				r := NewRowDot(dt, k)
				rb := len(w) / TileRows
				for n := range TileRows {
					r.LoadQ8(w[n*rb : (n+1)*rb])
					for i := range rows {
						if !use[i] {
							continue
						}
						want, _ := r.DotQ8(b.Row(i))
						if got := out[i*TileRows+n]; math.Float32bits(got) != math.Float32bits(want) {
							t.Errorf("%s K=%d rows=%d: weight row %d, activation row %d: got %g, DotQ8 %g", dt, k, rows, n, i, got, want)
						}
					}
				}
				// A scale that is not finite, in the last row's last block.
				bad := len(w) - int(dt.BlockBytes())
				if dt == mim.Q6_K {
					bad += 208
				}
				w[bad], w[bad+1] = 0x00, 0x7C
				if d.Dot(w, &a, out, TileRows, nil) {
					t.Errorf("%s K=%d: Dot accepted an infinite scale", dt, k)
				}
			}
		}
	}
}

func BenchmarkTileDot(b *testing.B) {
	if !TileQ8(mim.Q4_K) {
		b.Skip("no AMX or VNNI kernels on this machine")
	}
	rng := rand.New(rand.NewSource(12))
	const k = 2048
	for _, dt := range []mim.DType{mim.Q4_K, mim.Q6_K} {
		for _, rows := range []int{16, 32, 64} {
			var batch Q8Batch
			batch.Reset(k, rows)
			use := make([]bool, rows)
			for i := range rows {
				batch.Row(i).QuantizeBlocks(randomVec(rng, k), 256)
				use[i] = true
			}
			var a TileActs
			a.Set(&batch, use)
			w := randomRows(rng, dt, TileRows, k)
			d := NewTileDot(dt, k)
			out := make([]float32, TileRows*rows)
			b.Run(dt.String()+"/"+itoa(rows), func(b *testing.B) {
				b.SetBytes(int64(len(w)))
				for i := 0; i < b.N; i++ {
					d.Dot(w, &a, out, TileRows, nil)
				}
				b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(TileRows*rows*k)*1e3, "ps/mac")
			})
		}
	}
}

func itoa(n int) string {
	if n < 10 {
		return string(rune('0' + n))
	}
	return itoa(n/10) + string(rune('0'+n%10))
}

func BenchmarkTileActs(b *testing.B) {
	rng := rand.New(rand.NewSource(13))
	const k, rows = 2048, 38
	var batch Q8Batch
	batch.Reset(k, rows)
	use := make([]bool, rows)
	for i := range rows {
		batch.Row(i).QuantizeBlocks(randomVec(rng, k), 256)
		use[i] = true
	}
	var a TileActs
	b.SetBytes(k * rows)
	for i := 0; i < b.N; i++ {
		a.Set(&batch, use)
	}
}

// streamK is the row length of BenchmarkTileDotStream.
var streamK = 4096

// BenchmarkTileDotStream runs the tile kernels over a weight matrix larger
// than the caches, 16 rows at a time, as a prompt's matrix product does.
func BenchmarkTileDotStream(b *testing.B) {
	if !TileQ8(mim.Q4_K) {
		b.Skip("no AMX or VNNI kernels on this machine")
	}
	rng := rand.New(rand.NewSource(14))
	k, n := streamK, 8192*4096/streamK
	for _, dt := range []mim.DType{mim.Q4_K, mim.Q6_K} {
		w := randomRows(rng, dt, TileRows, k)
		rb := len(w) / TileRows
		big := make([]byte, n*rb)
		for o := 0; o < len(big); o += len(w) {
			copy(big[o:], w)
		}
		for _, rows := range []int{19, 64} {
			var batch Q8Batch
			batch.Reset(k, rows)
			use := make([]bool, rows)
			for i := range rows {
				batch.Row(i).QuantizeBlocks(randomVec(rng, k), 256)
				use[i] = true
			}
			var a TileActs
			a.Set(&batch, use)
			d := NewTileDot(dt, k)
			out := make([]float32, TileRows*rows)
			b.Run(dt.String()+"/"+itoa(rows), func(b *testing.B) {
				b.SetBytes(int64(len(big)))
				for i := 0; i < b.N; i++ {
					for j := 0; j < n; j += TileRows {
						d.Dot(big[j*rb:(j+TileRows)*rb], &a, out, TileRows, nil)
					}
				}
				b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(n*rows*k)*1e3, "ps/mac")
			})
		}
	}
}
