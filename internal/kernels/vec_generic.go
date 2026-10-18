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

//go:build (!amd64 && !arm64) || purego

package kernels

const vecF32, vecF16 = false, false

// The vector kernels are never called when vecF32 is false.

func dotF32Vec(a, b *float32, n int) float32                            { panic("no assembly") }
func axpyF32Vec(alpha float32, x, y *float32, n int)                    { panic("no assembly") }
func scoresF32Vec(q, k *float32, d, n int, scale float32, out *float32) { panic("no assembly") }
func maxF32Vec(x *float32, n int) float32                               { panic("no assembly") }
func expSumF32Vec(x *float32, n int, m float32) float32                 { panic("no assembly") }
func siluMulF32Vec(a, b, out *float32, n int)                           { panic("no assembly") }
func wsumF32Vec(p, v *float32, n, d int, acc *float32)                  { panic("no assembly") }

func scoresF16Vec(q *float32, k *byte, d, n int, scale float32, out *float32) { panic("no assembly") }
func wsumF16Vec(p *float32, v *byte, n, d int, acc *float32)                  { panic("no assembly") }
