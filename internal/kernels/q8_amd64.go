// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mlm, created and owned by Gembit Soultan Shirazi.
// It is licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License in the LICENSE file at the root
// of this repository or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

//go:build amd64 && !purego

package kernels

// q8K reports whether the 8-bit kernels for activation blocks of 256 are
// available: on amd64, the AVX2 ones.
var q8K = useAVX2 && hasF16C

func dotQ4KQ8K(q *byte, xq *int8, xd *float32, xs *int16, nb int) (float32, bool) {
	return dotQ4KQ8KAVX2(q, xq, xd, xs, nb)
}

func dotQ6KQ8K(q *byte, xq *int8, xd *float32, m32 *int16, nb int) (float32, bool) {
	return dotQ6KQ8KAVX2(q, xq, xd, m32, nb)
}

func dotQ4KQ8Kx4(q *byte, xq *int8, xd *float32, xs *int16, xstride, dstride, nb int, out *[4]float32) bool {
	return dotQ4KQ8Kx4AVX2(q, xq, xd, xs, xstride, dstride, nb, out)
}

func dotQ6KQ8Kx4(q *byte, xq *int8, xd *float32, m32 *int16, xstride, dstride, nb int, out *[4]float32) bool {
	return dotQ6KQ8Kx4AVX2(q, xq, xd, m32, xstride, dstride, nb, out)
}

func amaxQ8(x []float32) (float32, bool) { return amaxAVX2(&x[0], len(x)) }

func quantQ8(x []float32, q []int8, amax float32) { quantQ8AVX2(&x[0], &q[0], len(q), amax) }

func sumsQ8(q []int8, m32, xs []int16) { sumsQ8AVX2(&q[0], &m32[0], &xs[0], len(q)) }
