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

import "math/bits"

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

// SgemmWindow is SgemmTile for the products of convolutions: row kk of B
// is b[boff[kk]:] rather than b[kk*ldb:], so the rows are windows of b,
// such as the receptive fields in a padded image; and the columns are
// those of mask (bit j for column j), which row r of the tile holds
// packed together from c[r*ldc], so that the positions past the end of
// an output row are dropped as they are made. Every boff[kk] plus the
// last column's index must be below len(b), which the caller sees to
// (checking each row would cost as much as its multiply-adds); boff has
// at least k entries.
func SgemmWindow(pa []float32, k, panel, rows int, b []float32, boff []int, mask uint32, c []float32, ldc int, bias []float32) {
	cols := bits.OnesCount32(mask)
	if !useAVX512 || rows < 1 || rows > SgemmRows || cols < 1 || k < 1 ||
		len(pa) < (panel+1)*k*SgemmRows || len(boff) < k || len(c) < (rows-1)*ldc+cols ||
		(bias != nil && len(bias) < rows) {
		panic("kernels: SgemmWindow with operands of another shape")
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
		sgemm12x32w(&a[k0*SgemmRows], &b[0], &boff[k0], &c[0], ldc, kc, rows, mask, acc, bp)
	}
}

// ConvGatherRows is the most filters ConvGatherF32 computes in one call.
const ConvGatherRows = 8

// ConvGatherF32 computes a convolution's outputs for rows filters (at
// most ConvGatherRows) sixteen positions at a time, each tap's inputs
// gathered: for chunk c < chunks and filter f, y[f*plane + c*16 + j] =
// bias[f] + the sum over the taps t < k of w[t*ws+f] * x[idx[(c*k+t)*16
// + j]], for the lanes j set in masks[c*k+t] (the others read 0), the
// sum accumulated by fused multiply-adds in the taps' order as SgemmTile
// does, and bias (when not nil) added last. The last chunk stores only
// the lanes of tail. The indices of set lanes are all below len(x),
// which the caller sees to. Sgemm must be true.
func ConvGatherF32(x []float32, idx []int32, masks []uint16, chunks, k int, w []float32, ws, rows int, bias []float32, y []float32, plane, tail int) {
	if !useAVX512 || chunks < 1 || k < 1 || rows < 1 || rows > ConvGatherRows || ws < rows || plane < 1 || tail < 1 || tail > 0xffff ||
		len(x) == 0 || len(idx) < chunks*k*16 || len(masks) < chunks*k || len(w) < (k-1)*ws+rows ||
		(bias != nil && len(bias) < rows) || len(y) < (rows-1)*plane+(chunks-1)*16+bits.Len(uint(tail)) {
		panic("kernels: ConvGatherF32 with operands of another shape")
	}
	var bp *float32
	if bias != nil {
		bp = &bias[0]
	}
	convGatherF32(&x[0], &idx[0], &masks[0], chunks, k, &w[0], ws, rows, bp, &y[0], plane, tail)
}
