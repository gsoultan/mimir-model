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

import (
	"encoding/binary"
	"os"
	"runtime"
)

// q8K reports whether the 8-bit kernels for activation blocks of 256 are
// available: on arm64, the NEON ones in q8_arm64.s, which need the dot
// product instructions (FEAT_DotProd). Every Apple Silicon CPU has them,
// as do ARM servers since Neoverse N1 (AWS Graviton2).
var q8K = hasDotProd()

func hasDotProd() bool {
	switch runtime.GOOS {
	case "darwin", "ios":
		return true
	case "linux", "android":
		const atHWCAP, hwcapASIMDDP = 16, 1 << 20
		return auxv(atHWCAP)&hwcapASIMDDP != 0
	}
	return false
}

// auxv returns the value of the entry key in the process's auxiliary
// vector, or 0.
func auxv(key uint64) uint64 {
	b, err := os.ReadFile("/proc/self/auxv")
	if err != nil {
		return 0
	}
	for ; len(b) >= 16; b = b[16:] {
		if binary.LittleEndian.Uint64(b) == key {
			return binary.LittleEndian.Uint64(b[8:])
		}
	}
	return 0
}

//go:noescape
func dotQ4KQ8KNEON(q *byte, xq *int8, xd *float32, xs *int16, nb int) (float32, bool)

//go:noescape
func dotQ6KQ8KNEON(q *byte, xq *int8, xd *float32, nb int) (float32, bool)

//go:noescape
func dotQ4KQ8Kx4NEON(q *byte, xq *int8, xd *float32, xs *int16, xstride, dstride, nb int, out *[4]float32) bool

//go:noescape
func dotQ6KQ8Kx4NEON(q *byte, xq *int8, xd *float32, xstride, dstride, nb int, out *[4]float32) bool

func dotQ4KQ8K(q *byte, xq *int8, xd *float32, xs *int16, nb int) (float32, bool) {
	return dotQ4KQ8KNEON(q, xq, xd, xs, nb)
}

func dotQ6KQ8K(q *byte, xq *int8, xd *float32, _ *int16, nb int) (float32, bool) {
	return dotQ6KQ8KNEON(q, xq, xd, nb)
}

func dotQ4KQ8Kx4(q *byte, xq *int8, xd *float32, xs *int16, xstride, dstride, nb int, out *[4]float32) bool {
	return dotQ4KQ8Kx4NEON(q, xq, xd, xs, xstride, dstride, nb, out)
}

func dotQ6KQ8Kx4(q *byte, xq *int8, xd *float32, _ *int16, xstride, dstride, nb int, out *[4]float32) bool {
	return dotQ6KQ8Kx4NEON(q, xq, xd, xstride, dstride, nb, out)
}

func amaxQ8(x []float32) (float32, bool) { return amaxQ8Go(x) }

func quantQ8(x []float32, q []int8, amax float32) { quantQ8Go(x, q, amax) }

func sumsQ8(q []int8, m32, xs []int16) { sumsQ8Go(q, m32, xs) }
