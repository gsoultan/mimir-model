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
	"math/bits"
	"math/rand"
	"testing"
)

// TestSgemmTile checks SgemmTile against the product computed in float64
// and rounded to float32, within K u sum |a b| + u |bias| for the chain of
// multiply-adds and the bias, plus 2 u |result| for the two roundings to
// float32 of a sum (after the bias is added, and the reference's own), for
// tiles with fewer rows and
// columns than a whole one and for k longer than a run of sgemmKC; and
// that the elements of C outside the tile are not written.
func TestSgemmTile(t *testing.T) {
	if !Sgemm() {
		t.Skip("no AVX-512 on this machine")
	}
	rng := rand.New(rand.NewSource(21))
	const u = 1.0 / (1 << 24)
	for _, k := range []int{1, 3, 64, 255, 256, 257, 700} {
		for _, m := range []int{1, 5, 12, 13, 30} {
			for _, cols := range []int{1, 7, 16, 17, 32} {
				ldb, ldc := cols+rng.Intn(9), cols+rng.Intn(9)
				a := make([]float32, m*k)
				for i := range a {
					a[i] = float32(rng.NormFloat64())
				}
				b := make([]float32, (k-1)*ldb+cols)
				for i := range b {
					b[i] = float32(rng.NormFloat64())
				}
				bias := make([]float32, m)
				for i := range bias {
					bias[i] = float32(rng.NormFloat64())
				}
				pa := PackSgemmA(a, m, k, k)
				c := make([]float32, (m-1)*ldc+cols+SgemmCols)
				const sentinel = 12345
				for i := range c {
					c[i] = sentinel
				}
				withBias := k%2 == 1
				for p := 0; p*SgemmRows < m; p++ {
					rows := min(SgemmRows, m-p*SgemmRows)
					var bs []float32
					if withBias {
						bs = bias[p*SgemmRows:]
					}
					SgemmTile(pa, k, p, rows, b, ldb, cols, c[p*SgemmRows*ldc:], ldc, bs)
				}
				for r := range m {
					for j := range ldc {
						got := c[r*ldc+j]
						if j >= cols {
							if got != sentinel {
								t.Fatalf("k %d m %d cols %d: C[%d][%d] outside the tile was written", k, m, cols, r, j)
							}
							continue
						}
						var exact, abs float64
						for kk := range k {
							p := float64(a[r*k+kk]) * float64(b[kk*ldb+j])
							exact += p
							abs += math.Abs(p)
						}
						tol := float64(k) * u * abs
						if withBias {
							exact += float64(bias[r])
							tol += u * math.Abs(float64(bias[r]))
						}
						ref := float64(float32(exact))
						tol += 2 * u * math.Abs(ref)
						if d := math.Abs(float64(got) - ref); d > tol {
							t.Fatalf("k %d m %d cols %d: C[%d][%d] = %g, reference %g, tolerance %g", k, m, cols, r, j, got, ref, tol)
						}
					}
				}
			}
		}
	}
}

// TestSgemmWindow checks SgemmWindow against SgemmTile, bit for bit, on
// rows of B placed at random in one array, with the columns of a random
// mask packed together.
func TestSgemmWindow(t *testing.T) {
	if !Sgemm() {
		t.Skip("no AVX-512 on this machine")
	}
	rng := rand.New(rand.NewSource(22))
	const cols = SgemmCols
	for _, k := range []int{1, 3, 64, 257, 700} {
		for _, m := range []int{1, 4, 5, 8, 9, 12, 13} {
			for _, mask := range []uint32{1, 1<<cols - 1, 0x0000ffff, 0xffff0000, 0x8001, 0x5a5a5a5a, 0x00f0_0300} {
				a := make([]float32, m*k)
				for i := range a {
					a[i] = float32(rng.NormFloat64())
				}
				b := make([]float32, k*cols)
				for i := range b {
					b[i] = float32(rng.NormFloat64())
				}
				big := make([]float32, 3*k*(cols+2)+8)
				for i := range big {
					big[i] = float32(rng.NormFloat64())
				}
				boff := make([]int, k)
				for kk, slot := range rng.Perm(3 * k)[:k] { // windows apart from each other
					boff[kk] = slot*(cols+2) + rng.Intn(2)
					copy(big[boff[kk]:boff[kk]+cols], b[kk*cols:(kk+1)*cols])
				}
				bias := make([]float32, m)
				for i := range bias {
					bias[i] = float32(rng.NormFloat64())
				}
				pa := PackSgemmA(a, m, k, k)
				full := make([]float32, m*cols)
				n := bits.OnesCount32(mask)
				ldc := n + 3
				got := make([]float32, m*ldc)
				const sentinel = 12345
				for i := range got {
					got[i] = sentinel
				}
				for p := 0; p*SgemmRows < m; p++ {
					rows := min(SgemmRows, m-p*SgemmRows)
					SgemmTile(pa, k, p, rows, b, cols, cols, full[p*SgemmRows*cols:], cols, bias[p*SgemmRows:])
					SgemmWindow(pa, k, p, rows, big, boff, mask, got[p*SgemmRows*ldc:], ldc, bias[p*SgemmRows:])
				}
				for r := range m {
					j := 0
					for col := range cols {
						if mask&(1<<col) == 0 {
							continue
						}
						if g, w := got[r*ldc+j], full[r*cols+col]; math.Float32bits(g) != math.Float32bits(w) {
							t.Fatalf("k %d m %d mask %#x: C[%d][%d] is %g, SgemmTile %g", k, m, mask, r, j, g, w)
						}
						j++
					}
					for ; j < ldc; j++ {
						if got[r*ldc+j] != sentinel {
							t.Fatalf("k %d m %d mask %#x: C[%d][%d] past the columns was written", k, m, mask, r, j)
						}
					}
				}
			}
		}
	}
}

func BenchmarkSgemmTile(b *testing.B) {
	if !Sgemm() {
		b.Skip("no AVX-512 on this machine")
	}
	const m, k, n = 12, 576, 32
	rng := rand.New(rand.NewSource(1))
	a := make([]float32, m*k)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	bm := make([]float32, k*n)
	for i := range bm {
		bm[i] = float32(rng.NormFloat64())
	}
	pa := PackSgemmA(a, m, k, k)
	c := make([]float32, m*n)
	for i := 0; i < b.N; i++ {
		SgemmTile(pa, k, 0, m, bm, n, n, c, n, nil)
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(m*k*n)*1e3, "ps/mac")
}

// TestConvGatherF32 checks ConvGatherF32 against SgemmTile on the same
// windows, bit for bit.
func TestConvGatherF32(t *testing.T) {
	if !Sgemm() {
		t.Skip("no AVX-512 on this machine")
	}
	rng := rand.New(rand.NewSource(23))
	x := make([]float32, 500)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	for _, k := range []int{1, 2, 9, 27, 100} {
		for rows := 1; rows <= ConvGatherRows; rows++ {
			for _, chunks := range []int{1, 2, 3} {
				for _, tail := range []int{1, 0x00ff, 0xffff, 0x8421} {
					ws := rows + rng.Intn(3)
					w := make([]float32, k*ws)
					for i := range w {
						w[i] = float32(rng.NormFloat64())
					}
					idx := make([]int32, chunks*k*16)
					masks := make([]uint16, chunks*k)
					for i := range idx {
						idx[i] = int32(rng.Intn(len(x)))
					}
					for i := range masks {
						masks[i] = uint16(rng.Uint32())
					}
					var bias []float32
					if rng.Intn(2) == 0 {
						bias = make([]float32, rows)
						for i := range bias {
							bias[i] = float32(rng.NormFloat64())
						}
					}
					plane := chunks*16 + 5
					got := make([]float32, rows*plane)
					const sentinel = 12345
					for i := range got {
						got[i] = sentinel
					}
					ConvGatherF32(x, idx, masks, chunks, k, w, ws, rows, bias, got, plane, tail)
					// The same as a GEMM: A the filters' weights, B the gathered
					// windows, a chunk's 16 positions as the columns.
					a := make([]float32, rows*k)
					for f := range rows {
						for tp := range k {
							a[f*k+tp] = w[tp*ws+f]
						}
					}
					pa := PackSgemmA(a, rows, k, k)
					for c := range chunks {
						b := make([]float32, k*16)
						for tp := range k {
							for j := range 16 {
								if masks[c*k+tp]&(1<<j) != 0 {
									b[tp*16+j] = x[idx[(c*k+tp)*16+j]]
								}
							}
						}
						want := make([]float32, rows*16)
						SgemmTile(pa, k, 0, rows, b, 16, 16, want, 16, bias)
						for f := range rows {
							for j := range 16 {
								g := got[f*plane+c*16+j]
								if c == chunks-1 && tail&(1<<j) == 0 {
									if g != sentinel {
										t.Fatalf("k %d rows %d chunks %d tail %#x: lane %d past the tail was written", k, rows, chunks, tail, j)
									}
									continue
								}
								if math.Float32bits(g) != math.Float32bits(want[f*16+j]) {
									t.Fatalf("k %d rows %d chunk %d: y[%d][%d] is %g, SgemmTile %g", k, rows, c, f, c*16+j, g, want[f*16+j])
								}
							}
						}
					}
					for f := range rows {
						for j := chunks * 16; j < plane; j++ {
							if got[f*plane+j] != sentinel {
								t.Fatalf("k %d rows %d: wrote past the chunks", k, rows)
							}
						}
					}
				}
			}
		}
	}
}
