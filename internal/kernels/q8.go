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

	"github.com/gsoultan/mmf"
)

// 8-bit activations, for dequant_matmul_a8.v1 (spec/stdlib.md section
// 3.4). A row of activations is split into blocks of 32 along K; each
// block is stored as 32 int8 values q and a scale d = max|x|/127, with
// q = round(x/d) to nearest, ties to even, so that |x - q*d| <= d/2. The
// weights are still decoded exactly; the dot product of a block is then an
// integer dot product of the 8-bit activations with the weights' integer
// quants, scaled once per block.

// Q8 is a row of activations quantized to 8-bit blocks.
type Q8 struct {
	k   int
	q   []int8
	d   []float32 // scale of each block of 32
	dbs []float32 // d times the sum of the block's quants (Q4_K offsets)
	m32 []int16   // 32*(q[2i]+q[2i+1]) (Q6_K's bias of 32)
	ok  bool
}

// maxQ8 bounds the activations that are quantized: beyond it the scaled
// integer sums could overflow float32 where the exact product would not.
const maxQ8 = 0x1p60

// Quantize sets a to x quantized to 8-bit blocks. It returns false, and
// the caller must use the float32 kernels, if x is not a multiple of 32
// long, holds a value that is not finite or is very large, or the
// assembly kernels are not available.
func (a *Q8) Quantize(x []float32) bool {
	a.ok = false
	n := len(x)
	if !useAVX2 || !hasF16C || n == 0 || n%32 != 0 {
		return false
	}
	a.k = n
	if cap(a.q) < n {
		a.q = make([]int8, n)
		a.m32 = make([]int16, n/2)
		a.d = make([]float32, n/32)
		a.dbs = make([]float32, n/32)
	}
	a.q, a.m32, a.d, a.dbs = a.q[:n], a.m32[:n/2], a.d[:n/32], a.dbs[:n/32]
	for b := 0; b < n/32; b++ {
		xb := x[32*b : 32*b+32]
		var amax float32
		for _, v := range xb {
			if v < 0 {
				v = -v
			}
			if !(v <= maxQ8) { // also NaN
				return false
			}
			amax = max(amax, v)
		}
		qb := a.q[32*b : 32*b+32]
		var sum int32
		if amax == 0 {
			clear(qb)
		} else {
			// 127*x is exact in float64, so the quotient is within
			// 2^-46 of the exact 127*x/amax, which for float32 operands
			// is either a half-integer or at least 2^-34 away from one:
			// the result is the exact quotient rounded to nearest even.
			for i, v := range xb {
				q := int8(math.RoundToEven(127 * float64(v) / float64(amax)))
				qb[i] = q
				sum += int32(q)
			}
		}
		for i := 0; i < 16; i++ {
			a.m32[16*b+i] = 32 * (int16(qb[2*i]) + int16(qb[2*i+1]))
		}
		a.d[b] = amax / 127
		a.dbs[b] = a.d[b] * float32(sum)
	}
	a.ok = true
	return true
}

// FusedQ8 reports whether rows of dt have an 8-bit activation kernel.
func FusedQ8(dt mmf.DType) bool {
	return useAVX2 && hasF16C && (dt == mmf.Q4_K || dt == mmf.Q6_K || dt == mmf.Q8_0)
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
	switch r.dt {
	case mmf.Q4_K:
		return dotQ4KQ8AVX2(&r.row[0], &a.q[0], &a.d[0], &a.dbs[0], r.k/256)
	case mmf.Q6_K:
		return dotQ6KQ8AVX2(&r.row[0], &a.q[0], &a.d[0], &a.m32[0], r.k/256)
	case mmf.Q8_0:
		return dotQ8_0Q8AVX2(&r.row[0], &a.q[0], &a.d[0], r.k/32)
	}
	panic("kernels: DotQ8 on a dtype without a kernel")
}
