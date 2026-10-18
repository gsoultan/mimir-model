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

//go:noescape
func amxQ4K(a *tileArgs) bool

//go:noescape
func amxQ6K(a *tileArgs) bool

var useAMX = detectAMX()

// detectAMX reports whether the CPU has AMX tiles with int8 products and
// AVX-512 (F, BW, VL, VBMI and VNNI, for the kernels' other instructions), the
// operating system saves their state, and it lets this process use the
// tiles (amxperm_*.go).
func detectAMX() bool {
	if !useAVX2 || !hasF16C {
		return false
	}
	_, b7, c7, d7 := cpuid(7, 0)
	const (
		avx512f    = 1 << 16
		avx512bw   = 1 << 30
		avx512vl   = 1 << 31
		avx512vbmi = 1 << 1 // in ECX
		avx512vnni = 1 << 11
		amxTile    = 1 << 24 // in EDX
		amxInt8    = 1 << 25
	)
	if b7&(avx512f|avx512bw|avx512vl) != avx512f|avx512bw|avx512vl || c7&(avx512vbmi|avx512vnni) != avx512vbmi|avx512vnni ||
		d7&(amxTile|amxInt8) != amxTile|amxInt8 {
		return false
	}
	// XCR0: opmask, the upper ZMM halves, ZMM16..31, tile config and data.
	const xcr0 = 0xE0 | 1<<17 | 1<<18
	if lo, _ := xgetbv(); lo&xcr0 != xcr0 {
		return false
	}
	return requestAMX()
}

//go:noescape
func vnniQ4K(a *vnniArgs)

//go:noescape
func vnniQ6K(a *vnniArgs)

var useVNNI = detectVNNI()

// detectVNNI reports whether the CPU has AVX-512 (F and BW) with VNNI and
// the operating system saves the AVX-512 state.
func detectVNNI() bool {
	if !useAVX2 || !hasF16C {
		return false
	}
	_, b7, c7, _ := cpuid(7, 0)
	const (
		avx512f    = 1 << 16
		avx512bw   = 1 << 30
		avx512vnni = 1 << 11 // in ECX
	)
	if b7&(avx512f|avx512bw) != avx512f|avx512bw || c7&avx512vnni == 0 {
		return false
	}
	// XCR0: opmask, the upper ZMM halves and ZMM16..31.
	if lo, _ := xgetbv(); lo&0xE0 != 0xE0 {
		return false
	}
	return true
}
