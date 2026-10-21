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

package kernels

import "math"

// Element-wise sigmoid and tanh of float32, in float32 with a vector
// kernel where there is one (within a few units in the last place: see
// SigmoidF32), and in float64 otherwise.

// SigmoidF32 sets y[i] = 1/(1+exp(-x[i])) for i < len(y); x has at least
// len(y) floats, and may be y. A NaN gives nan. With a vector kernel the
// result is within a few units in the last place of the exact value
// relative to it; otherwise it is computed in float64 and rounded once.
func SigmoidF32(y, x []float32, nan float32) {
	x = x[:len(y)]
	i := 0
	if vecUnary > 0 && len(y) >= vecUnary {
		i = len(y) &^ (vecUnary - 1)
		sigmoidF32Vec(&x[0], &y[0], i, nan)
	}
	for ; i < len(y); i++ {
		r := float32(1 / (1 + math.Exp(-float64(x[i]))))
		if r != r {
			r = nan
		}
		y[i] = r
	}
}

// TanhF32 sets y[i] = tanh(x[i]) for i < len(y); x has at least len(y)
// floats, and may be y. A NaN gives nan. With a vector kernel the result
// is within a few units in the last place of 1 of the exact value;
// otherwise it is computed in float64 and rounded once.
func TanhF32(y, x []float32, nan float32) {
	x = x[:len(y)]
	i := 0
	if vecUnary > 0 && len(y) >= vecUnary {
		i = len(y) &^ (vecUnary - 1)
		tanhF32Vec(&x[0], &y[0], i, nan)
	}
	for ; i < len(y); i++ {
		r := float32(math.Tanh(float64(x[i])))
		if r != r {
			r = nan
		}
		y[i] = r
	}
}
