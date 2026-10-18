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

package ref

import (
	"math"

	"github.com/gsoultan/mlm"
)

// evalReduce folds x over axes. Float sums and products accumulate in
// float64 and are rounded once to the output dtype, which meets the bound of
// spec/primitives.md section 3.6 for every float dtype.
func evalReduce(x *Tensor, mode string, axes []int, outShape []int) (*Tensor, error) {
	dt := x.DType
	y, err := New(dt, outShape)
	if err != nil {
		return nil, err
	}
	reduced := make([]bool, len(x.Shape))
	for _, a := range axes {
		reduced[a] = true
	}
	n := y.NumElems()
	accF := make([]float64, n)
	accI := make([]uint64, n)
	for i := range accF {
		switch mode {
		case "add":
			accF[i], accI[i] = 0, 0
		case "mul":
			accF[i], accI[i] = 1, 1
		case "max":
			accF[i] = math.Inf(-1)
			lo, _, _ := intRange(dt)
			accI[i] = uint64(lo)
		}
	}
	ost := strides(outShape)
	idx := make([]int, len(x.Shape))
	for i := 0; i < x.NumElems(); i++ {
		unravel(i, x.Shape, idx)
		o := 0
		for k := range idx {
			if !reduced[k] {
				o += idx[k] * ost[k]
			}
		}
		b := x.Bits(i)
		if isArith(dt) {
			v := DecodeFloat(dt, b)
			switch mode {
			case "add":
				accF[o] += v
			case "mul":
				accF[o] *= v
			case "max":
				accF[o] = floatMax(accF[o], v)
			}
			continue
		}
		switch mode {
		case "add":
			accI[o] += b
		case "mul":
			accI[o] *= b
		case "max":
			accI[o], _ = binaryInt(mlm.OpMax, dt, accI[o], b)
		}
	}
	for i := 0; i < n; i++ {
		if isArith(dt) {
			y.SetBits(i, RoundFloat(dt, accF[i], false))
		} else {
			y.SetBits(i, mask(dt, accI[i]))
		}
	}
	return y, nil
}
