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

package ref

import (
	"fmt"
	"math"

	"github.com/gsoultan/mmf"
)

// broadcastShape applies spec/primitives.md section 3.1 to equal-rank shapes.
func broadcastShape(shapes ...[]int) ([]int, error) {
	r := len(shapes[0])
	out := make([]int, r)
	for _, s := range shapes {
		if len(s) != r {
			return nil, fmt.Errorf("broadcast: rank %d and %d differ", r, len(s))
		}
	}
	for i := 0; i < r; i++ {
		zero, m := false, 1
		for _, s := range shapes {
			d := s[i]
			if d == 0 {
				zero = true
			}
			if d != 1 && m != 1 && d != m {
				return nil, fmt.Errorf("broadcast: dimension %d: %d and %d", i, m, d)
			}
			if d != 1 {
				m = d
			}
		}
		if zero {
			m = 0
		}
		out[i] = m
	}
	return out, nil
}

// broadcastMap returns, for each flat index of out, the flat index of the
// broadcast source element of a tensor of shape in.
func broadcastMap(out, in []int) []int {
	n := NumElems(out)
	res := make([]int, n)
	st := strides(in)
	idx := make([]int, len(out))
	for i := 0; i < n; i++ {
		unravel(i, out, idx)
		j := 0
		for k := range idx {
			if in[k] != 1 {
				j += idx[k] * st[k]
			}
		}
		res[i] = j
	}
	return res
}

// unaryFloat maps an elementwise float primitive to its float64 function.
// Every float dtype is a subset of float64, and for f32, f16 and bf16 the
// float64 result rounded once to the output dtype is correctly rounded for
// the exact operations and within the primitive bounds for the others.
var unaryFloat = map[mmf.Opcode]func(float64) float64{
	mmf.OpRecip: func(x float64) float64 { return 1 / x },
	mmf.OpSqrt:  math.Sqrt,
	mmf.OpExp2:  math.Exp2,
	mmf.OpLog2:  math.Log2,
	mmf.OpSin:   math.Sin,
	mmf.OpTrunc: math.Trunc,
}

func evalUnary(op mmf.Opcode, x *Tensor) (*Tensor, error) {
	dt := x.DType
	y, err := New(dt, x.Shape)
	if err != nil {
		return nil, err
	}
	n := x.NumElems()
	switch {
	case isArith(dt) && op == mmf.OpNeg:
		sb := floatFormats[dt].signBit()
		for i := 0; i < n; i++ {
			y.SetBits(i, x.Bits(i)^sb)
		}
	case isArith(dt):
		fn, ok := unaryFloat[op]
		if !ok {
			return nil, fmt.Errorf("%s: no float kernel", op)
		}
		for i := 0; i < n; i++ {
			y.SetBits(i, RoundFloat(dt, fn(DecodeFloat(dt, x.Bits(i))), false))
		}
	case IsInt(dt) && op == mmf.OpNeg:
		for i := 0; i < n; i++ {
			y.SetBits(i, mask(dt, -x.Bits(i)))
		}
	case IsInt(dt) && op == mmf.OpTrunc:
		copy(y.Data, x.Data)
	default:
		return nil, fmt.Errorf("%s: unsupported dtype %s", op, dt)
	}
	return y, nil
}

func floatMax(a, b float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.NaN()
	}
	if a == 0 && b == 0 {
		if math.Signbit(a) && math.Signbit(b) {
			return a
		}
		return 0
	}
	if a > b {
		return a
	}
	return b
}

// floatPow is pow with the NaN rule of spec/primitives.md section 5.13: a
// NaN operand always gives NaN. Go's math.Pow follows the IEEE 754 special
// cases for the remaining values.
func floatPow(a, b float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.NaN()
	}
	return pow64(a, b)
}

var binaryFloat = map[mmf.Opcode]func(a, b float64) float64{
	mmf.OpAdd: func(a, b float64) float64 { return a + b },
	mmf.OpMul: func(a, b float64) float64 { return a * b },
	mmf.OpMax: floatMax,
	mmf.OpPow: floatPow,
}

// binaryInt computes an integer primitive on bit patterns of dtype dt. The
// result is masked by the caller. ok is false when the result is
// unspecified (division by zero), in which case 0 is stored.
func binaryInt(op mmf.Opcode, dt mmf.DType, a, b uint64) (r uint64, ok bool) {
	w := uint64(intWidth(dt))
	signed := isSigned(dt)
	switch op {
	case mmf.OpAdd:
		return a + b, true
	case mmf.OpMul:
		return a * b, true
	case mmf.OpMax:
		if signed {
			if sext(dt, a) > sext(dt, b) {
				return a, true
			}
			return b, true
		}
		if a > b {
			return a, true
		}
		return b, true
	case mmf.OpIdiv, mmf.OpImod:
		if b == 0 {
			return 0, false
		}
		if !signed {
			if op == mmf.OpIdiv {
				return a / b, true
			}
			return a % b, true
		}
		sa, sb := sext(dt, a), sext(dt, b)
		if sb == -1 { // avoids the MIN / -1 trap; wraps as the spec requires
			if op == mmf.OpIdiv {
				return uint64(-sa), true
			}
			return 0, true
		}
		if op == mmf.OpIdiv {
			return uint64(sa / sb), true
		}
		return uint64(sa % sb), true
	case mmf.OpShl:
		if b >= w {
			return 0, true
		}
		return a << b, true
	case mmf.OpShr:
		if signed {
			if b >= w {
				b = w - 1
			}
			return uint64(sext(dt, a) >> b), true
		}
		if b >= w {
			return 0, true
		}
		return a >> b, true
	case mmf.OpAnd:
		return a & b, true
	case mmf.OpOr:
		return a | b, true
	case mmf.OpXor:
		return a ^ b, true
	case mmf.OpCmplt:
		if signed {
			return b2u(sext(dt, a) < sext(dt, b)), true
		}
		return b2u(a < b), true
	case mmf.OpCmpeq:
		return b2u(a == b), true
	}
	return 0, false
}

func b2u(b bool) uint64 {
	if b {
		return 1
	}
	return 0
}

func evalBinary(op mmf.Opcode, a, b *Tensor, outDType mmf.DType, shape []int) (*Tensor, error) {
	dt := a.DType
	y, err := New(outDType, shape)
	if err != nil {
		return nil, err
	}
	ma, mb := broadcastMap(shape, a.Shape), broadcastMap(shape, b.Shape)
	for i := range ma {
		x, z := a.Bits(ma[i]), b.Bits(mb[i])
		switch {
		case isArith(dt) && (op == mmf.OpCmplt || op == mmf.OpCmpeq):
			u, v := DecodeFloat(dt, x), DecodeFloat(dt, z)
			if op == mmf.OpCmplt {
				y.SetBits(i, b2u(u < v))
			} else {
				y.SetBits(i, b2u(u == v))
			}
		case isArith(dt):
			r := binaryFloat[op](DecodeFloat(dt, x), DecodeFloat(dt, z))
			y.SetBits(i, RoundFloat(dt, r, false))
		case dt == mmf.Bool:
			u, v := b2u(x != 0), b2u(z != 0)
			switch op {
			case mmf.OpAnd:
				y.SetBits(i, u&v)
			case mmf.OpOr:
				y.SetBits(i, u|v)
			case mmf.OpXor:
				y.SetBits(i, u^v)
			case mmf.OpCmpeq:
				y.SetBits(i, b2u(u == v))
			}
		default:
			r, _ := binaryInt(op, dt, x, z)
			if outDType == mmf.Bool {
				y.SetBits(i, r)
			} else {
				y.SetBits(i, mask(dt, r))
			}
		}
	}
	return y, nil
}

func evalWhere(c, a, b *Tensor, shape []int) (*Tensor, error) {
	y, err := New(a.DType, shape)
	if err != nil {
		return nil, err
	}
	mc, ma, mb := broadcastMap(shape, c.Shape), broadcastMap(shape, a.Shape), broadcastMap(shape, b.Shape)
	for i := range mc {
		if c.Bits(mc[i]) != 0 {
			y.SetBits(i, a.Bits(ma[i]))
		} else {
			y.SetBits(i, b.Bits(mb[i]))
		}
	}
	return y, nil
}
