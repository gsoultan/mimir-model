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

//go:build amd64 && !purego

package kernels

// cpuid and xgetbv are implemented in kernels_amd64.s.
func cpuid(eax, ecx uint32) (a, b, c, d uint32)
func xgetbv() (eax, edx uint32)

var useAVX2, hasF16C = detect()

// detect reports whether the CPU and the operating system support AVX2
// with FMA, and F16C.
func detect() (avx2, f16c bool) {
	maxID, _, _, _ := cpuid(0, 0)
	if maxID < 7 {
		return false, false
	}
	_, _, c1, _ := cpuid(1, 0)
	const (
		fma     = 1 << 12
		osxsave = 1 << 27
		avx     = 1 << 28
		f16cBit = 1 << 29
	)
	if c1&(fma|osxsave|avx) != fma|osxsave|avx {
		return false, false
	}
	// The OS must save the XMM and YMM registers on context switches.
	if lo, _ := xgetbv(); lo&6 != 6 {
		return false, false
	}
	_, b7, _, _ := cpuid(7, 0)
	if b7&(1<<5) == 0 {
		return false, false
	}
	return true, c1&f16cBit != 0
}

//go:noescape
func dotF32AVX2(a, b *float32, n int) float32

//go:noescape
func axpyF32AVX2(alpha float32, x, y *float32, n int)

//go:noescape
func dotQ4KAVX2(q *byte, x, tab *float32, nb int) float32

//go:noescape
func dotQ6KAVX2(q *byte, x, tab *float32, nb int) float32

//go:noescape
func dotQ8_0AVX2(q *byte, x, tab *float32, nb int) float32

//go:noescape
func dotF16AVX2(w *byte, x *float32, n int) float32

//go:noescape
func dotBF16AVX2(w *byte, x *float32, n int) float32

//go:noescape
func dotQ4KQ8AVX2(q *byte, xq *int8, xd, xdbs *float32, nb int) (float32, bool)

//go:noescape
func dotQ6KQ8AVX2(q *byte, xq *int8, xd *float32, m32 *int16, nb int) (float32, bool)

//go:noescape
func dotQ8_0Q8AVX2(q *byte, xq *int8, xd *float32, nb int) (float32, bool)

//go:noescape
func dotQ4KQ8x4AVX2(q *byte, xq *int8, xd, xdbs *float32, xstride, dstride, nb int, out *[4]float32) bool

//go:noescape
func dotQ6KQ8x4AVX2(q *byte, xq *int8, xd *float32, m32 *int16, xstride, dstride, nb int, out *[4]float32) bool

//go:noescape
func dotQ4KQ8KAVX2(q *byte, xq *int8, xd *float32, xs *int16, nb int) (float32, bool)

//go:noescape
func dotQ6KQ8KAVX2(q *byte, xq *int8, xd *float32, m32 *int16, nb int) (float32, bool)

//go:noescape
func dotQ4KQ8Kx4AVX2(q *byte, xq *int8, xd *float32, xs *int16, xstride, dstride, nb int, out *[4]float32) bool

//go:noescape
func dotQ6KQ8Kx4AVX2(q *byte, xq *int8, xd *float32, m32 *int16, xstride, dstride, nb int, out *[4]float32) bool

//go:noescape
func quantQ8AVX2(x *float32, q *int8, n int, amax float32)

//go:noescape
func amaxAVX2(x *float32, n int) (amax float32, ok bool)

//go:noescape
func sumsQ8AVX2(q *int8, m32 *int16, xs *int16, n int)

//go:noescape
func scoresF32AVX2(q, k *float32, d, n int, scale float32, out *float32)

//go:noescape
func expSumF32AVX2(x *float32, n int, m float32) float32

//go:noescape
func wsumF32AVX2(p, v *float32, n, d int, acc *float32)

//go:noescape
func scoresF16AVX2(q *float32, k *byte, d, n int, scale float32, out *float32)

//go:noescape
func wsumF16AVX2(p *float32, v *byte, n, d int, acc *float32)
