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

// vecF32 reports whether the float32 kernels of vec_*.go are in use, and
// vecF16 whether those over f16 rows are: on amd64, the AVX2 ones, which
// convert f16 with F16C.
var vecF32, vecF16 = useAVX2, useAVX2 && hasF16C

func dotF32Vec(a, b *float32, n int) float32         { return dotF32AVX2(a, b, n) }
func axpyF32Vec(alpha float32, x, y *float32, n int) { axpyF32AVX2(alpha, x, y, n) }

func scoresF32Vec(q, k *float32, d, n int, scale float32, out *float32) {
	scoresF32AVX2(q, k, d, n, scale, out)
}
func expSumF32Vec(x *float32, n int, m float32) float32 { return expSumF32AVX2(x, n, m) }
func siluMulF32Vec(a, b, out *float32, n int)           { siluMulF32AVX2(a, b, out, n) }
func wsumF32Vec(p, v *float32, n, d int, acc *float32)  { wsumF32AVX2(p, v, n, d, acc) }

func scoresF16Vec(q *float32, k *byte, d, n int, scale float32, out *float32) {
	scoresF16AVX2(q, k, d, n, scale, out)
}
func wsumF16Vec(p *float32, v *byte, n, d int, acc *float32) { wsumF16AVX2(p, v, n, d, acc) }
