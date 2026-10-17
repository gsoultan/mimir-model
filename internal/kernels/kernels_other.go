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

const useAVX2, hasF16C, useAMX, useVNNI = false, false, false, false

// The assembly kernels are never called when useAVX2 is false.

func dotF32AVX2(a, b *float32, n int) float32              { panic("no assembly") }
func axpyF32AVX2(alpha float32, x, y *float32, n int)      { panic("no assembly") }
func dotQ4KAVX2(q *byte, x, tab *float32, nb int) float32  { panic("no assembly") }
func dotQ6KAVX2(q *byte, x, tab *float32, nb int) float32  { panic("no assembly") }
func dotQ8_0AVX2(q *byte, x, tab *float32, nb int) float32 { panic("no assembly") }
func dotF16AVX2(w *byte, x *float32, n int) float32        { panic("no assembly") }
func dotBF16AVX2(w *byte, x *float32, n int) float32       { panic("no assembly") }

func scoresF32AVX2(q, k *float32, d, n int, scale float32, out *float32) { panic("no assembly") }
func expSumF32AVX2(x *float32, n int, m float32) float32                 { panic("no assembly") }
func siluMulF32AVX2(a, b, out *float32, n int)                           { panic("no assembly") }
func wsumF32AVX2(p, v *float32, n, d int, acc *float32)                  { panic("no assembly") }

func scoresF16AVX2(q *float32, k *byte, d, n int, scale float32, out *float32) { panic("no assembly") }
func wsumF16AVX2(p *float32, v *byte, n, d int, acc *float32)                  { panic("no assembly") }

func dotQ4KQ8AVX2(q *byte, xq *int8, xd, xdbs *float32, nb int) (float32, bool) { panic("no assembly") }
func dotQ6KQ8AVX2(q *byte, xq *int8, xd *float32, m32 *int16, nb int) (float32, bool) {
	panic("no assembly")
}
func dotQ8_0Q8AVX2(q *byte, xq *int8, xd *float32, nb int) (float32, bool) { panic("no assembly") }
func dotQ4KQ8x4AVX2(q *byte, xq *int8, xd, xdbs *float32, xstride, dstride, nb int, out *[4]float32) bool {
	panic("no assembly")
}
func dotQ6KQ8x4AVX2(q *byte, xq *int8, xd *float32, m32 *int16, xstride, dstride, nb int, out *[4]float32) bool {
	panic("no assembly")
}

func amxQ4K(a *tileArgs) bool { panic("no assembly") }
func amxQ6K(a *tileArgs) bool { panic("no assembly") }
func vnniQ4K(a *vnniArgs)     { panic("no assembly") }
func vnniQ6K(a *vnniArgs)     { panic("no assembly") }
