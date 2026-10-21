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

// Element-wise float32 add, multiply and maximum on AVX-512, with the
// results of the scalar float32 operations bit for bit: add and multiply
// are single IEEE operations, a NaN result is replaced by a given NaN,
// and the maximum is NaN when either operand is, and -0 only when both
// operands are -0.

// The operations of BinaryF32.
const (
	BinAdd = iota
	BinMul
	BinMax
)

// BinaryF32 sets y[j] = x[j] op z[j], or x[j] op z[0] when scalar is
// true, for j < len(y), and reports whether it had a vector kernel to do
// so; when it returns false it has done nothing. A NaN result is nan.
// x and z have at least len(y) floats (z one when scalar); y may be x or
// z but must not overlap them otherwise.
func BinaryF32(op int, y, x, z []float32, scalar bool, nan float32) bool {
	if !useAVX512 || len(y) == 0 {
		return useAVX512
	}
	if op < BinAdd || op > BinMax || len(x) < len(y) || (scalar && len(z) < 1) || (!scalar && len(z) < len(y)) {
		panic("kernels: BinaryF32 with operands of another shape")
	}
	s := 0
	if scalar {
		s = 1
	}
	binF32(op, &y[0], &x[0], &z[0], len(y), s, nan)
	return true
}

// PoolF32 computes rows rows of w outputs each, y[r*yrs+j] = the op over
// the taps t, in order, of x[offs[t] + r*xrs + j*stride] as BinaryF32
// would apply it, and reports whether it had a vector kernel to do so
// (one for strides 1 and 2); when it returns false it has done nothing.
// y has at least (rows-1)*yrs + w floats and x at least offs[t] +
// (rows-1)*xrs + (w-1)*stride + 1 for every tap. With one tap, it is a
// strided copy.
func PoolF32(op int, y, x []float32, offs []int, w, rows, stride, xrs, yrs int, nan float32) bool {
	if !useAVX512 || stride < 1 || stride > 2 {
		return false
	}
	if rows <= 0 || w <= 0 {
		return true
	}
	if op < BinAdd || op > BinMax || xrs < 0 || yrs < w || len(y) < (rows-1)*yrs+w || len(offs) == 0 {
		panic("kernels: PoolF32 with operands of another shape")
	}
	for _, off := range offs {
		if off < 0 || len(x) < off+(rows-1)*xrs+(w-1)*stride+1 {
			panic("kernels: PoolF32 with operands of another shape")
		}
	}
	poolF32(op, &y[0], &x[0], &offs[0], len(offs), w, rows, stride, xrs, yrs, nan)
	return true
}

// PoolGatherF32 computes y[j] = the op over the taps t, in order, of
// x[idx[t*np+j]] for j < len(y) as BinaryF32 would apply it, and reports
// whether it had a vector kernel to do so; when it returns false it has
// done nothing. np is a multiple of 16 of at least len(y), and the
// indices read are all below len(x), which the caller sees to.
func PoolGatherF32(op int, y, x []float32, idx []int32, np, taps int, nan float32) bool {
	if !useAVX512 {
		return false
	}
	if len(y) == 0 {
		return true
	}
	if op < BinAdd || op > BinMax || taps <= 0 || np%16 != 0 || np < len(y) || len(idx) < (taps-1)*np+(len(y)+15)&^15 || len(x) == 0 {
		panic("kernels: PoolGatherF32 with operands of another shape")
	}
	poolGatherF32(op, &y[0], &x[0], &idx[0], np, taps, len(y), nan)
	return true
}

// RowsF32 sets y[r*plane+j] = y[r*plane+j] op v[r] for r < rows and j <
// plane as BinaryF32 would, and reports whether it had a vector kernel
// to do so; when it returns false it has done nothing. y has at least
// rows*plane floats and v rows.
func RowsF32(op int, y, v []float32, rows, plane int, nan float32) bool {
	if !useAVX512 {
		return false
	}
	if rows <= 0 || plane <= 0 {
		return true
	}
	if op < BinAdd || op > BinMax || len(y) < rows*plane || len(v) < rows {
		panic("kernels: RowsF32 with operands of another shape")
	}
	rowsF32(op, &y[0], &v[0], rows, plane, nan)
	return true
}

// EpilogueF32 applies the steps of an element-wise epilogue to y, rows
// rows of plane floats, in one pass: for each row r and step s in
// order, y[r*plane+j] = y[r*plane+j] ops[s] v[s*vs+r] as BinaryF32
// would compute it, and reports whether it had a vector kernel to do
// so; when it returns false it has done nothing. v has at least
// (len(ops)-1)*vs + rows floats.
func EpilogueF32(y []float32, rows, plane int, ops []int, v []float32, vs int, nan float32) bool {
	if !useAVX512 {
		return false
	}
	if rows <= 0 || plane <= 0 || len(ops) == 0 {
		return true
	}
	if len(y) < rows*plane || vs < rows || len(v) < (len(ops)-1)*vs+rows {
		panic("kernels: EpilogueF32 with operands of another shape")
	}
	for _, op := range ops {
		if op < BinAdd || op > BinMax {
			panic("kernels: EpilogueF32 with an op it has not")
		}
	}
	epilogueF32(&y[0], rows, plane, &ops[0], len(ops), &v[0], vs, nan)
	return true
}

// PadPlaneF32 sets y, hp rows of wp floats, to x, h rows of w, at row
// top, column left, and v everywhere else, and reports whether it had a
// vector kernel to do so; when it returns false it has done nothing.
func PadPlaneF32(y, x []float32, h, w, hp, wp, top, left int, v float32) bool {
	if !useAVX512 || left+w > 64 {
		return false
	}
	if hp <= 0 || wp <= 0 {
		return true
	}
	if h < 0 || w < 0 || top < 0 || left < 0 || top+h > hp || left+w > wp || len(y) < hp*wp || (h > 0 && w > 0 && len(x) < h*w) {
		panic("kernels: PadPlaneF32 with operands of another shape")
	}
	if h == 0 || w == 0 {
		x, w = []float32{0}, 0
	}
	padPlaneF32(&y[0], &x[0], h, w, hp, wp, top, left, v)
	return true
}
