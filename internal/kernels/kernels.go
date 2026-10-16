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

// Package kernels holds the executor's inner loops: float32 dot products
// and dot products against rows of a weight matrix stored in a block
// dtype, decoded inside the kernel.
//
// On amd64 with AVX2 and FMA the loops are assembly; elsewhere, or when
// built with the purego tag, they are plain Go. Both decode every weight
// exactly as mmf.Dequantize does, so the only difference between them is
// the order in which the products are summed, and both stay within the
// dequant_matmul.v1 tolerance (spec/stdlib.md section 3.3). Activations
// are never quantized.
package kernels

import (
	"encoding/binary"

	"github.com/gsoultan/mmf"
)

// SIMD reports whether the assembly kernels are in use.
func SIMD() bool { return useAVX2 }

// DotF32 returns the dot product of a and b[:len(a)].
func DotF32(a, b []float32) float32 {
	b = b[:len(a)]
	if useAVX2 && len(a) >= 8 {
		n := len(a) &^ 7
		s := dotF32AVX2(&a[0], &b[0], n)
		for i := n; i < len(a); i++ {
			s += a[i] * b[i]
		}
		return s
	}
	return dotF32Go(a, b)
}

func dotF32Go(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	n := len(a)
	b = b[:n]
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// AxpyF32 adds alpha*x to y[:len(x)].
func AxpyF32(alpha float32, x, y []float32) {
	y = y[:len(x)]
	i := 0
	if useAVX2 && len(x) >= 8 {
		i = len(x) &^ 7
		axpyF32AVX2(alpha, &x[0], &y[0], i)
	}
	for ; i < len(x); i++ {
		y[i] += alpha * x[i]
	}
}

// Fused reports whether rows of dt are decoded inside the dot product
// rather than into a buffer first.
func Fused(dt mmf.DType) bool {
	if !useAVX2 {
		return false
	}
	switch dt {
	case mmf.Q4_K, mmf.Q6_K, mmf.Q8_0, mmf.BF16:
		return true
	case mmf.F16:
		return hasF16C
	}
	return false
}

// DecodeOnce is the number of dot products with one row above which
// Load decodes the row into a buffer, even for a dtype with a fused
// kernel: decoding once and then running plain float32 dot products is
// cheaper than decoding inside each of many dot products.
const DecodeOnce = 8

// RowDot computes dot products of float32 vectors with a row of K
// elements of one dtype: Load a row, then call Dot for each vector. It
// holds scratch space, so each goroutine needs its own.
type RowDot struct {
	dt       mmf.DType
	k        int
	rowBytes int
	row      []byte    // the loaded row, for the fused kernels
	fused    bool      // whether the loaded row is decoded inside Dot
	buf      []float32 // the loaded row decoded, otherwise
	tab      []float32 // per-block scales for the fused kernels
}

// NewRowDot returns a RowDot for rows of k elements of dtype dt. For block
// dtypes k must be a multiple of the block size.
func NewRowDot(dt mmf.DType, k int) *RowDot {
	return &RowDot{dt: dt, k: k, rowBytes: int(uint64(k) / dt.BlockElems() * dt.BlockBytes())}
}

// Load prepares row, the bytes of one row, for uses calls to Dot.
func (r *RowDot) Load(row []byte, uses int) {
	r.fused = Fused(r.dt) && uses <= DecodeOnce && r.k > 0
	if !r.fused {
		if r.buf == nil {
			r.buf = make([]float32, r.k)
		}
		// Dequantize cannot fail: callers pass rows of the size K implies.
		_ = mmf.Dequantize(r.dt, row, r.buf)
		return
	}
	r.row = row
	switch r.dt {
	case mmf.Q4_K:
		nb := r.k / 256
		r.row = row[:nb*144]
		r.table(nb * 16)
		for b := 0; b < nb; b++ {
			q4kScales(row[144*b:144*b+144], r.tab[16*b:16*b+16])
		}
	case mmf.Q6_K:
		nb := r.k / 256
		r.row = row[:nb*210]
		r.table(nb * 16)
		for b := 0; b < nb; b++ {
			blk := row[210*b : 210*b+210]
			d := mmf.F16ToF32(binary.LittleEndian.Uint16(blk[208:]))
			t := r.tab[16*b : 16*b+16]
			for i, s := range blk[192:208] {
				t[i] = d * float32(int8(s))
			}
		}
	case mmf.Q8_0:
		nb := r.k / 32
		r.row = row[:nb*34]
		r.table(nb)
		for b := range r.tab {
			r.tab[b] = mmf.F16ToF32(binary.LittleEndian.Uint16(row[34*b:]))
		}
	case mmf.F16, mmf.BF16:
		r.row = row[:2*r.k]
	}
}

func (r *RowDot) table(n int) {
	if cap(r.tab) < n {
		r.tab = make([]float32, n)
	}
	r.tab = r.tab[:n]
}

// Dot returns the dot product of x (len K) with the loaded row.
func (r *RowDot) Dot(x []float32) float32 {
	x = x[:r.k]
	if !r.fused {
		return DotF32(x, r.buf)
	}
	switch r.dt {
	case mmf.Q4_K:
		return dotQ4KAVX2(&r.row[0], &x[0], &r.tab[0], r.k/256)
	case mmf.Q6_K:
		return dotQ6KAVX2(&r.row[0], &x[0], &r.tab[0], r.k/256)
	case mmf.Q8_0:
		return dotQ8_0AVX2(&r.row[0], &x[0], &r.tab[0], r.k/32)
	}
	n := r.k &^ 7
	var s float32
	if r.dt == mmf.F16 {
		if n > 0 {
			s = dotF16AVX2(&r.row[0], &x[0], n)
		}
		for i := n; i < r.k; i++ {
			s += x[i] * mmf.F16ToF32(binary.LittleEndian.Uint16(r.row[2*i:]))
		}
		return s
	}
	if n > 0 {
		s = dotBF16AVX2(&r.row[0], &x[0], n)
	}
	for i := n; i < r.k; i++ {
		s += x[i] * mmf.BF16ToF32(binary.LittleEndian.Uint16(r.row[2*i:]))
	}
	return s
}

// q4kScales writes the eight sub-block scales d*sc and then the eight
// offsets dmin*m of one Q4_K block, rounded to float32 as
// mmf.Dequantize computes them.
func q4kScales(blk []byte, t []float32) {
	blk = blk[:16]
	t = t[:16]
	d := mmf.F16ToF32(binary.LittleEndian.Uint16(blk[0:]))
	dmin := mmf.F16ToF32(binary.LittleEndian.Uint16(blk[2:]))
	q := blk[4:16]
	for j := 0; j < 4; j++ {
		t[j] = d * float32(q[j]&63)
		t[8+j] = dmin * float32(q[j+4]&63)
		t[4+j] = d * float32(q[j+8]&0xF|(q[j]>>6)<<4)
		t[12+j] = dmin * float32(q[j+8]>>4|(q[j+4]>>6)<<4)
	}
}
