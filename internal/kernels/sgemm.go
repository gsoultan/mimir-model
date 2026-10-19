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

// A float32 matrix product C = A B (+ bias) on AVX-512, for convolutions
// and other float32 products: a tile of 12 rows of A by 32 columns of B
// at a time, with A packed once into panels of 12 rows. Each element of C
// is one chain of fused multiply-adds over k in order, then the bias:
// within K u sum |a b| of the exact product (u = 2^-24), plus u |bias|.

// SgemmRows and SgemmCols are the rows of A and columns of B of a tile.
const SgemmRows, SgemmCols = 12, 32

// sgemmKC is the length of the runs of k a tile works through at a time,
// so that its 32 columns of B and 12 rows of A stay in the L1 cache.
const sgemmKC = 256

// Sgemm reports whether SgemmTile has a vector kernel on this machine.
func Sgemm() bool { return useAVX512 }

// PackSgemmA packs the m by k row-major matrix a (rows lda apart) into
// panels of SgemmRows rows, each k groups of SgemmRows floats, the rows
// past m zero.
func PackSgemmA(a []float32, m, k, lda int) []float32 {
	np := (m + SgemmRows - 1) / SgemmRows
	p := make([]float32, np*k*SgemmRows)
	for r := range m {
		row := a[r*lda : r*lda+k]
		dst := p[r/SgemmRows*k*SgemmRows+r%SgemmRows:]
		for kk, v := range row {
			dst[kk*SgemmRows] = v
		}
	}
	return p
}

// SgemmTile computes rows panel*SgemmRows to panel*SgemmRows+rows-1 of C
// = A B for the first cols columns (at most SgemmCols) of b and c: row kk
// of B is b[kk*ldb:], row r of the tile is c[r*ldc:], and pa is A packed
// by PackSgemmA with k columns. bias, when not nil, holds the tile's rows'
// biases. Sgemm must be true.
func SgemmTile(pa []float32, k, panel, rows int, b []float32, ldb, cols int, c []float32, ldc int, bias []float32) {
	if !useAVX512 || rows < 1 || rows > SgemmRows || cols < 1 || cols > SgemmCols || k < 1 ||
		len(pa) < (panel+1)*k*SgemmRows || len(b) < (k-1)*ldb+cols || len(c) < (rows-1)*ldc+cols ||
		(bias != nil && len(bias) < rows) {
		panic("kernels: SgemmTile with operands of another shape")
	}
	mask := uint32(1)<<cols - 1
	if cols == 32 {
		mask = ^uint32(0)
	}
	a := pa[panel*k*SgemmRows:]
	for k0 := 0; k0 < k; k0 += sgemmKC {
		kc := min(sgemmKC, k-k0)
		var bp *float32
		if k0+kc == k && bias != nil {
			bp = &bias[0]
		}
		acc := 0
		if k0 > 0 {
			acc = 1
		}
		sgemm12x32(&a[k0*SgemmRows], &b[k0*ldb], ldb, &c[0], ldc, kc, rows, mask, acc, bp)
	}
}
