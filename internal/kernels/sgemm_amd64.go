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

//go:build amd64 && !purego

package kernels

//go:noescape
func sgemm12x32(a, b *float32, ldb int, c *float32, ldc, k, rows int, mask uint32, acc int, bias *float32)

// useAVX512 reports whether the CPU has AVX-512 F and the operating system
// saves its state.
var useAVX512 = detectAVX512()

func detectAVX512() bool {
	if !useAVX2 {
		return false
	}
	_, b7, _, _ := cpuid(7, 0)
	if b7&(1<<16) == 0 {
		return false
	}
	lo, _ := xgetbv()
	return lo&0xE6 == 0xE6
}
