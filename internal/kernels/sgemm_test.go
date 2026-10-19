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
