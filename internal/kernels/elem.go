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
