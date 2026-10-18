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

package ref

import (
	"math"

	"github.com/gsoultan/mim"
)

// isStorage reports whether dt is a storage float (spec/primitives.md 2.2).
func isStorage(dt mim.DType) bool { return dt == mim.F8E4M3 || dt == mim.F8E5M2 }

// castBits converts one element following spec/primitives.md section 5.8.
// Conversions to storage floats saturate finite overflow
// (spec/dtypes.md section 2.3).
func castBits(from, to mim.DType, b uint64) uint64 {
	if to == mim.Bool {
		if IsFloat(from) {
			return b2u(DecodeFloat(from, b) != 0 || math.IsNaN(DecodeFloat(from, b)))
		}
		return b2u(b != 0)
	}
	sat := isStorage(to)
	switch {
	case from == mim.Bool:
		v := b2u(b != 0)
		if IsFloat(to) {
			return roundInt(to, false, v, sat)
		}
		return v
	case IsFloat(from):
		x := DecodeFloat(from, b)
		if IsFloat(to) {
			return RoundFloat(to, x, sat)
		}
		return floatToInt(to, x)
	default: // integer source
		if IsFloat(to) {
			if isSigned(from) {
				v := sext(from, b)
				if v < 0 {
					return roundInt(to, true, uint64(-v), sat)
				}
				return roundInt(to, false, uint64(v), sat)
			}
			return roundInt(to, false, b, sat)
		}
		if isSigned(from) {
			return mask(to, uint64(sext(from, b)))
		}
		return mask(to, b)
	}
}

// floatToInt truncates toward zero and saturates; NaN becomes 0.
func floatToInt(to mim.DType, x float64) uint64 {
	if math.IsNaN(x) {
		return 0
	}
	x = math.Trunc(x)
	lo, hi, hiU := intRange(to)
	if !isSigned(to) {
		if x <= 0 {
			return 0
		}
		if x >= math.Ldexp(1, int(intWidth(to))) {
			return hiU
		}
		return uint64(x)
	}
	if x < float64(lo) {
		return mask(to, uint64(lo))
	}
	if x >= -float64(lo) {
		return mask(to, uint64(hi))
	}
	return mask(to, uint64(int64(x)))
}

func evalCast(x *Tensor, to mim.DType) (*Tensor, error) {
	y, err := New(to, x.Shape)
	if err != nil {
		return nil, err
	}
	for i := 0; i < x.NumElems(); i++ {
		y.SetBits(i, castBits(x.DType, to, x.Bits(i)))
	}
	return y, nil
}
