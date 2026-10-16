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

//go:build !amd64 || purego

package kernels

// q8K reports whether the 8-bit kernels for activation blocks of 256 are
// available; there are none in plain Go.
const q8K = false

// The 8-bit kernels are never called when q8K is false.

func dotQ4KQ8K(q *byte, xq *int8, xd *float32, xs *int16, nb int) (float32, bool) {
	panic("no assembly")
}
func dotQ6KQ8K(q *byte, xq *int8, xd *float32, m32 *int16, nb int) (float32, bool) {
	panic("no assembly")
}
func dotQ4KQ8Kx4(q *byte, xq *int8, xd *float32, xs *int16, xstride, dstride, nb int, out *[4]float32) bool {
	panic("no assembly")
}
func dotQ6KQ8Kx4(q *byte, xq *int8, xd *float32, m32 *int16, xstride, dstride, nb int, out *[4]float32) bool {
	panic("no assembly")
}

func amaxQ8(x []float32) (float32, bool)          { return amaxQ8Go(x) }
func quantQ8(x []float32, q []int8, amax float32) { quantQ8Go(x, q, amax) }
func sumsQ8(q []int8, m32, xs []int16)            { sumsQ8Go(q, m32, xs) }
