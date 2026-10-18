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

//go:build arm64 && !purego

package kernels

// vecF32 reports whether the float32 kernels of vec_*.go are in use, and
// vecF16 whether those over f16 rows are: on arm64, the NEON ones in
// vec_arm64.s, which need nothing beyond the base architecture (FCVTL
// converts f16).
const vecF32, vecF16 = true, true

//go:noescape
func dotF32NEON(a, b *float32, n int) float32

//go:noescape
func axpyF32NEON(alpha float32, x, y *float32, n int)

//go:noescape
func scoresF32NEON(q, k *float32, d, n int, scale float32, out *float32)

//go:noescape
func expSumF32NEON(x *float32, n int, m float32) float32

//go:noescape
func siluMulF32NEON(a, b, out *float32, n int)

//go:noescape
func wsumF32NEON(p, v *float32, n, d int, acc *float32)

//go:noescape
func scoresF16NEON(q *float32, k *byte, d, n int, scale float32, out *float32)

//go:noescape
func wsumF16NEON(p *float32, v *byte, n, d int, acc *float32)

func dotF32Vec(a, b *float32, n int) float32         { return dotF32NEON(a, b, n) }
func axpyF32Vec(alpha float32, x, y *float32, n int) { axpyF32NEON(alpha, x, y, n) }

func scoresF32Vec(q, k *float32, d, n int, scale float32, out *float32) {
	scoresF32NEON(q, k, d, n, scale, out)
}
func expSumF32Vec(x *float32, n int, m float32) float32 { return expSumF32NEON(x, n, m) }
func siluMulF32Vec(a, b, out *float32, n int)           { siluMulF32NEON(a, b, out, n) }
func wsumF32Vec(p, v *float32, n, d int, acc *float32)  { wsumF32NEON(p, v, n, d, acc) }

func scoresF16Vec(q *float32, k *byte, d, n int, scale float32, out *float32) {
	scoresF16NEON(q, k, d, n, scale, out)
}
func wsumF16Vec(p *float32, v *byte, n, d int, acc *float32) { wsumF16NEON(p, v, n, d, acc) }
