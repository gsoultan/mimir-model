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

package kernels

import (
	"encoding/binary"
	"math"

	"github.com/gsoultan/mmf"
)

// Kernels for the attention fast path: the scores of one query against a
// run of keys, their softmax, and the weighted sum of the values.

// ScoresF32 sets out[j] to scale times the dot product of q with row j of
// k, whose rows are len(q) long, for each j of out.
func ScoresF32(q, k []float32, scale float32, out []float32) {
	d := len(q)
	k = k[:len(out)*d]
	if useAVX2 && d >= 8 && d%8 == 0 && len(out) > 0 {
		scoresF32AVX2(&q[0], &k[0], d, len(out), scale, &out[0])
		return
	}
	for j := range out {
		out[j] = DotF32(q, k[j*d:(j+1)*d]) * scale
	}
}

// ExpSumF32 sets each x[i] to exp(x[i]-m) and returns their sum. m must be
// finite and no less than any x[i]; an x[i] of -inf gives 0. With AVX2 the
// exponentials are computed in float32, within a few units in the last
// place.
func ExpSumF32(x []float32, m float32) float32 {
	var s float32
	i := 0
	if useAVX2 && len(x) >= 8 {
		i = len(x) &^ 7
		s = expSumF32AVX2(&x[0], i, m)
	}
	for ; i < len(x); i++ {
		e := float32(math.Exp(float64(x[i] - m)))
		x[i] = e
		s += e
	}
	return s
}

// SiluMulF32 sets out[i] to silu(a[i]) * b[i], silu(x) = x/(1+exp(-x)),
// for the elements of out; a and b must be at least as long. With AVX2 the
// exponentials are computed in float32 as in ExpSumF32, within a few units
// in the last place, and so is the rest; otherwise in float64.
func SiluMulF32(a, b, out []float32) {
	a, b = a[:len(out)], b[:len(out)]
	i := 0
	if useAVX2 && len(out) >= 8 {
		i = len(out) &^ 7
		siluMulF32AVX2(&a[0], &b[0], &out[0], i)
	}
	for ; i < len(out); i++ {
		out[i] = float32(float64(a[i]) / (1 + math.Exp(-float64(a[i]))) * float64(b[i]))
	}
}

// WeightedSumF32 adds p[j] times row j of v, whose rows are len(acc) long,
// to acc for each j of p, in order: the result is that of AxpyF32 called
// for each row.
func WeightedSumF32(p, v, acc []float32) {
	d := len(acc)
	v = v[:len(p)*d]
	if useAVX2 && d >= 8 && d%8 == 0 && len(p) > 0 {
		wsumF32AVX2(&p[0], &v[0], len(p), d, &acc[0])
		return
	}
	for j, e := range p {
		AxpyF32(e, v[j*d:(j+1)*d], acc)
	}
}

// ScoresF16 is ScoresF32 for keys stored as f16, len(q) elements (2*len(q)
// bytes) per row. The keys convert to float32 exactly, and the result is
// that of ScoresF32 on the converted keys, bit for bit.
func ScoresF16(q []float32, k []byte, scale float32, out []float32) {
	d := len(q)
	k = k[:2*len(out)*d]
	if useAVX2 && hasF16C && d >= 8 && d%8 == 0 && len(out) > 0 {
		scoresF16AVX2(&q[0], &k[0], d, len(out), scale, &out[0])
		return
	}
	row := make([]float32, d)
	for j := range out {
		f16Row(k[2*j*d:], row)
		ScoresF32(q, row, scale, out[j:j+1])
	}
}

// WeightedSumF16 is WeightedSumF32 for values stored as f16, len(acc)
// elements per row, with the same result on the converted values.
func WeightedSumF16(p []float32, v []byte, acc []float32) {
	d := len(acc)
	v = v[:2*len(p)*d]
	if useAVX2 && hasF16C && d >= 8 && d%8 == 0 && len(p) > 0 {
		wsumF16AVX2(&p[0], &v[0], len(p), d, &acc[0])
		return
	}
	row := make([]float32, d)
	for j, e := range p {
		f16Row(v[2*j*d:], row)
		AxpyF32(e, row, acc)
	}
}

// f16Row converts len(dst) f16 values from b.
func f16Row(b []byte, dst []float32) {
	for i := range dst {
		dst[i] = mmf.F16ToF32(binary.LittleEndian.Uint16(b[2*i:]))
	}
}
