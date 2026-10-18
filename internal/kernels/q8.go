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

	mim "github.com/gsoultan/mimir-model"
)

// 8-bit activations, for dequant_matmul_a8.v1 and v2 (spec/stdlib.md
// section 3.4). A row of activations is split into blocks of 32 or 256
// along K; each block is stored as int8 values q and a scale
// d = max|x|/127, with q = round(x/d) to nearest, ties to even, so that
// |x - q*d| <= d/2. The weights are still decoded exactly; the dot product
// of a block is then an integer dot product of the 8-bit activations with
// the weights' integer quants, scaled once per block.

// Q8 is a row of activations quantized to 8-bit blocks.
type Q8 struct {
	k     int
	block int // 32 or 256
	q     []int8
	d     []float32 // scale of the block holding each run of 32
	dbs   []float32 // d times the sum of each run of 32 quants (Q4_K offsets)
	m32   []int16   // 32*(q[2i]+q[2i+1]) (Q6_K's bias of 32)
	xs    []int16   // the sum of each run of 32 quants
	ok    bool
}

// Quantize sets a to x quantized to 8-bit blocks of 32. It returns false,
// and the caller must use the float32 kernels, if x is not a multiple of
// 32 long, holds a value that is not finite or is very large, or the
// assembly kernels are not available.
func (a *Q8) Quantize(x []float32) bool { return a.QuantizeBlocks(x, 32) }

// QuantizeBlocks is Quantize with blocks of block elements, 32 or 256.
// x must be a multiple of block long.
func (a *Q8) QuantizeBlocks(x []float32, block int) bool {
	a.ok = false
	n := len(x)
	if !q8K || n == 0 || (block != 32 && block != 256) || n%block != 0 || (block == 32 && !(useAVX2 && hasF16C)) {
		return false
	}
	a.k, a.block = n, block
	if cap(a.q) < n {
		a.q = make([]int8, n)
		a.m32 = make([]int16, n/2)
		a.d = make([]float32, n/32)
		a.dbs = make([]float32, n/32)
		a.xs = make([]int16, n/32)
	}
	a.q, a.m32, a.d, a.dbs, a.xs = a.q[:n], a.m32[:n/2], a.d[:n/32], a.dbs[:n/32], a.xs[:n/32]
	for b := 0; b < n/block; b++ {
		xb := x[block*b : block*b+block]
		// amaxQ8 refuses NaN and any |x| above 2^60, beyond which the
		// scaled integer sums could overflow float32 where the exact
		// product would not.
		amax, ok := amaxQ8(xb)
		if !ok {
			return false
		}
		qb := a.q[block*b : block*b+block]
		if amax == 0 {
			clear(qb)
		} else {
			// 127*x is exact in float64, so the quotient is within
			// 2^-46 of the exact 127*x/amax, which for float32 operands
			// is either a half-integer or at least 2^-34 away from one:
			// the result is the exact quotient rounded to nearest even.
			// quantQ8 computes RoundToEven(127*float64(v)/float64(amax))
			// for each v.
			quantQ8(xb, qb, amax)
		}
		// For each run of 32: m32 = 32*(q[2i]+q[2i+1]) and xs = sum q,
		// |xs| <= 32*127.
		sumsQ8(qb, a.m32[block*b/2:block*(b+1)/2], a.xs[block*b/32:block*(b+1)/32])
		d := amax / 127
		for r := block * b / 32; r < block*(b+1)/32; r++ {
			a.d[r] = d
			a.dbs[r] = d * float32(a.xs[r])
		}
	}
	a.ok = true
	return true
}

// FusedQ8 reports whether rows of dt have an 8-bit activation kernel for
// blocks of 32.
func FusedQ8(dt mim.DType) bool {
	return useAVX2 && hasF16C && (dt == mim.Q4_K || dt == mim.Q6_K || dt == mim.Q8_0)
}

// FusedQ8Blocks reports whether rows of dt have an 8-bit activation kernel
// for blocks of block elements, one-row and four-row. Blocks of 256 have
// kernels for AVX2 on amd64 and for NEON with the dot product
// instructions on arm64; blocks of 32 only for AVX2.
func FusedQ8Blocks(dt mim.DType, block int) bool {
	switch block {
	case 32:
		return FusedQ8(dt)
	case 256:
		return q8K && (dt == mim.Q4_K || dt == mim.Q6_K)
	}
	return false
}

// LoadQ8 prepares row, the bytes of one row of a dtype for which FusedQ8
// is true, for DotQ8.
func (r *RowDot) LoadQ8(row []byte) {
	r.row = row[:r.rowBytes]
}

// DotQ8 returns the dot product of the row loaded with LoadQ8 and the
// quantized activations a. It returns false, and the caller must use Load
// and Dot instead, if a block scale of the row is not finite: the integer
// kernels would turn an infinite scale times a zero sum into infinity
// where the exact product is NaN.
func (r *RowDot) DotQ8(a *Q8) (float32, bool) {
	if !a.ok || a.k != r.k {
		panic("kernels: DotQ8 with activations of another length")
	}
	if a.block == 256 {
		switch r.dt {
		case mim.Q4_K:
			return dotQ4KQ8K(&r.row[0], &a.q[0], &a.d[0], &a.xs[0], r.k/256)
		case mim.Q6_K:
			return dotQ6KQ8K(&r.row[0], &a.q[0], &a.d[0], &a.m32[0], r.k/256)
		}
		panic("kernels: DotQ8 with blocks of 256 on a dtype without a kernel")
	}
	switch r.dt {
	case mim.Q4_K:
		return dotQ4KQ8AVX2(&r.row[0], &a.q[0], &a.d[0], &a.dbs[0], r.k/256)
	case mim.Q6_K:
		return dotQ6KQ8AVX2(&r.row[0], &a.q[0], &a.d[0], &a.m32[0], r.k/256)
	case mim.Q8_0:
		return dotQ8_0Q8AVX2(&r.row[0], &a.q[0], &a.d[0], r.k/32)
	}
	panic("kernels: DotQ8 on a dtype without a kernel")
}

// Q8Batch holds rows of activations quantized to 8-bit blocks, stored one
// after another so that a kernel can read several rows at fixed strides.
// The rows of one batch must all use one block length.
type Q8Batch struct {
	k    int
	rows []Q8
	q    []int8
	d    []float32
	dbs  []float32
	m32  []int16
	xs   []int16
}

// Reset prepares b for n rows of k activations. The rows are then set with
// b.Row(i).Quantize or QuantizeBlocks, which may run concurrently for
// different rows. It keeps b's buffers when they are large enough.
func (b *Q8Batch) Reset(k, n int) {
	b.k = k
	b.rows = grow(b.rows, n)
	clear(b.rows)
	if k%32 != 0 {
		return // Quantize refuses such rows
	}
	b.q = grow(b.q, n*k)
	b.m32 = grow(b.m32, n*k/2)
	b.d = grow(b.d, n*k/32)
	b.dbs = grow(b.dbs, n*k/32)
	b.xs = grow(b.xs, n*k/32)
	for i := range b.rows {
		r := &b.rows[i]
		r.q = b.q[i*k : (i+1)*k : (i+1)*k]
		r.m32 = b.m32[i*k/2 : (i+1)*k/2 : (i+1)*k/2]
		r.d = b.d[i*k/32 : (i+1)*k/32 : (i+1)*k/32]
		r.dbs = b.dbs[i*k/32 : (i+1)*k/32 : (i+1)*k/32]
		r.xs = b.xs[i*k/32 : (i+1)*k/32 : (i+1)*k/32]
	}
}

// Row returns row i of b.
func (b *Q8Batch) Row(i int) *Q8 { return &b.rows[i] }

// FusedQ8x4 reports whether rows of dt have a kernel that multiplies one
// row by four rows of activations in blocks of 32 at once
// (RowDot.DotQ8x4). Every dtype FusedQ8Blocks accepts for blocks of 256
// has one.
func FusedQ8x4(dt mim.DType) bool {
	return useAVX2 && hasF16C && (dt == mim.Q4_K || dt == mim.Q6_K)
}

// DotQ8x4 sets out to the dot products of the row loaded with LoadQ8 and
// rows i to i+3 of b, which must all have been quantized. The results are
// those DotQ8 returns for each row, bit for bit: the four products are
// computed side by side, in the same order, and share the decoding of the
// weights. It returns false where DotQ8 would.
func (r *RowDot) DotQ8x4(b *Q8Batch, i int, out *[4]float32) bool {
	if b.k != r.k || i < 0 || i+4 > len(b.rows) {
		panic("kernels: DotQ8x4 with activations of another shape")
	}
	block := b.rows[i].block
	for j := i; j < i+4; j++ {
		if !b.rows[j].ok || b.rows[j].block != block {
			panic("kernels: DotQ8x4 with a row that is not quantized, or quantized differently")
		}
	}
	k := r.k
	if block == 256 {
		switch r.dt {
		case mim.Q4_K:
			return dotQ4KQ8Kx4(&r.row[0], &b.q[i*k], &b.d[i*k/32], &b.xs[i*k/32], k, k/32*4, k/256, out)
		case mim.Q6_K:
			return dotQ6KQ8Kx4(&r.row[0], &b.q[i*k], &b.d[i*k/32], &b.m32[i*k/2], k, k/32*4, k/256, out)
		}
		panic("kernels: DotQ8x4 with blocks of 256 on a dtype without a kernel")
	}
	switch r.dt {
	case mim.Q4_K:
		return dotQ4KQ8x4AVX2(&r.row[0], &b.q[i*k], &b.d[i*k/32], &b.dbs[i*k/32], k, k/32*4, k/256, out)
	case mim.Q6_K:
		return dotQ6KQ8x4AVX2(&r.row[0], &b.q[i*k], &b.d[i*k/32], &b.m32[i*k/2], k, k/32*4, k/256, out)
	}
	panic("kernels: DotQ8x4 on a dtype without a kernel")
}

// amaxQ8Go returns max|x| for amaxQ8, and false if x holds NaN or a value
// above 2^60 in magnitude.
func amaxQ8Go(x []float32) (float32, bool) {
	var m float32
	for _, v := range x {
		a := math.Float32frombits(math.Float32bits(v) &^ (1 << 31))
		if !(a <= 0x1p60) {
			return 0, false
		}
		m = max(m, a)
	}
	return m, true
}

// quantQ8Go sets q[i] to x[i] quantized for the scale amax/127, rounded
// to nearest even, for quantQ8.
func quantQ8Go(x []float32, q []int8, amax float32) {
	x = x[:len(q)]
	for i, v := range x {
		q[i] = int8(math.RoundToEven(127 * float64(v) / float64(amax)))
	}
}

// sumsQ8Go sets, for sumsQ8, m32[i] = 32*(q[2i]+q[2i+1]) and xs[r] to the
// sum of the run of 32 quants r.
func sumsQ8Go(q []int8, m32, xs []int16) {
	for r := range xs {
		var s int16
		for _, v := range q[32*r : 32*r+32] {
			s += int16(v)
		}
		xs[r] = s
	}
	for i := range m32 {
		m32[i] = 32 * (int16(q[2*i]) + int16(q[2*i+1]))
	}
}
