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
	"math"
	"math/bits"

	"github.com/gsoultan/mmf"
)

// floatFormat describes a binary floating-point encoding.
type floatFormat struct {
	p      int  // precision in bits, including the hidden bit
	emin   int  // exponent of the smallest normal number
	ebits  int  // exponent field width
	hasInf bool // false for f8_e4m3, whose top exponent holds finite values
}

var floatFormats = map[mmf.DType]floatFormat{
	mmf.F64:    {53, -1022, 11, true},
	mmf.F32:    {24, -126, 8, true},
	mmf.F16:    {11, -14, 5, true},
	mmf.BF16:   {8, -126, 8, true},
	mmf.F8E4M3: {4, -6, 4, false},
	mmf.F8E5M2: {3, -14, 5, true},
}

func (f floatFormat) width() int { return f.p + f.ebits }
func (f floatFormat) bias() int  { return 1 - f.emin }
func (f floatFormat) emax() int {
	if f.hasInf {
		return 1<<f.ebits - 2 - f.bias()
	}
	return 1<<f.ebits - 1 - f.bias()
}
func (f floatFormat) signBit() uint64 { return 1 << (f.width() - 1) }
func (f floatFormat) expMask() uint64 { return 1<<f.ebits - 1 }
func (f floatFormat) manMask() uint64 { return 1<<(f.p-1) - 1 }

// maxFiniteBits is the bit pattern of the largest finite value.
func (f floatFormat) maxFiniteBits() uint64 {
	if f.hasInf {
		return (f.expMask()-1)<<(f.p-1) | f.manMask()
	}
	return f.expMask()<<(f.p-1) | (f.manMask() - 1)
}

func (f floatFormat) nanBits() uint64 {
	if f.hasInf {
		return f.expMask()<<(f.p-1) | 1<<(f.p-2)
	}
	return f.signBit() - 1
}

func (f floatFormat) infBits() uint64 { return f.expMask() << (f.p - 1) }

// IsFloat reports whether dt is a float or storage float dtype.
func IsFloat(dt mmf.DType) bool { _, ok := floatFormats[dt]; return ok }

// isArith reports whether dt is in the float class of spec/primitives.md
// section 2.2 (f64, f32, f16, bf16).
func isArith(dt mmf.DType) bool {
	return dt == mmf.F64 || dt == mmf.F32 || dt == mmf.F16 || dt == mmf.BF16
}

// IsInt reports whether dt is an integer dtype.
func IsInt(dt mmf.DType) bool {
	switch dt {
	case mmf.I8, mmf.U8, mmf.I16, mmf.U16, mmf.I32, mmf.U32, mmf.I64, mmf.U64:
		return true
	}
	return false
}

// isSigned reports whether dt is a signed integer dtype.
func isSigned(dt mmf.DType) bool {
	return dt == mmf.I8 || dt == mmf.I16 || dt == mmf.I32 || dt == mmf.I64
}

func intWidth(dt mmf.DType) uint { return uint(8 * elemSize(dt)) }

// sext interprets the low intWidth(dt) bits of b as a signed integer.
func sext(dt mmf.DType, b uint64) int64 {
	s := 64 - intWidth(dt)
	return int64(b<<s) >> s
}

// mask keeps the low intWidth(dt) bits of b.
func mask(dt mmf.DType, b uint64) uint64 {
	w := intWidth(dt)
	if w == 64 {
		return b
	}
	return b & (1<<w - 1)
}

// intRange returns the smallest and largest value of an integer dtype. For
// unsigned dtypes lo is 0 and hi is returned in hiU.
func intRange(dt mmf.DType) (lo int64, hi int64, hiU uint64) {
	w := intWidth(dt)
	if isSigned(dt) {
		return -1 << (w - 1), 1<<(w-1) - 1, 1<<(w-1) - 1
	}
	if w == 64 {
		return 0, math.MaxInt64, math.MaxUint64
	}
	return 0, 1<<w - 1, 1<<w - 1
}

// DecodeFloat returns the exact value of a float bit pattern of dtype dt.
func DecodeFloat(dt mmf.DType, b uint64) float64 {
	f := floatFormats[dt]
	if dt == mmf.F64 {
		return math.Float64frombits(b)
	}
	sign := b&f.signBit() != 0
	e := (b >> (f.p - 1)) & f.expMask()
	m := b & f.manMask()
	var v float64
	switch {
	case f.hasInf && e == f.expMask():
		if m != 0 {
			return math.NaN()
		}
		v = math.Inf(1)
	case !f.hasInf && e == f.expMask() && m == f.manMask():
		return math.NaN()
	case e == 0:
		v = math.Ldexp(float64(m), f.emin-(f.p-1))
	default:
		v = math.Ldexp(float64(m|1<<(f.p-1)), int(e)-f.bias()-(f.p-1))
	}
	if sign {
		v = -v
	}
	return v
}

// roundExact rounds (-1)^neg * m * 2^e to dt, to nearest with ties to even,
// and returns the bit pattern. Finite values beyond the largest finite value
// become infinity, or the largest finite value when sat is set or dt has no
// infinity.
func roundExact(dt mmf.DType, neg bool, m uint64, e int, sat bool) uint64 {
	f := floatFormats[dt]
	var s uint64
	if neg {
		s = f.signBit()
	}
	if m == 0 {
		return s
	}
	l := bits.Len64(m)
	ex := e + l - 1 // floor(log2(value))
	if ex < f.emin {
		ex = f.emin
	}
	shift := ex - (f.p - 1) - e // quantum exponent minus e
	var n uint64
	switch {
	case shift <= 0:
		n = m << uint(-shift)
	case shift > 64:
		n = 0
	case shift == 64:
		if m > 1<<63 {
			n = 1
		}
	default:
		n = m >> uint(shift)
		rem := m & (1<<uint(shift) - 1)
		half := uint64(1) << uint(shift-1)
		if rem > half || rem == half && n&1 == 1 {
			n++
		}
	}
	if n == 0 {
		return s
	}
	if n == 1<<f.p {
		n >>= 1
		ex++
	}
	overflow := ex > f.emax() || !f.hasInf && ex == f.emax() && n == 1<<f.p-1
	if overflow {
		if f.hasInf && !sat {
			return s | f.infBits()
		}
		return s | f.maxFiniteBits()
	}
	if n < 1<<(f.p-1) {
		return s | n
	}
	return s | uint64(ex+f.bias())<<(f.p-1) | (n - 1<<(f.p-1))
}

// RoundFloat rounds x to dt (a float or storage float dtype), to nearest with
// ties to even, and returns the bit pattern. NaN becomes the canonical quiet
// NaN of dt. Infinities stay infinite where dt has them; f8_e4m3, which has
// none, saturates. With sat set, finite values that overflow saturate to the
// largest finite value of the same sign instead of becoming infinite.
func RoundFloat(dt mmf.DType, x float64, sat bool) uint64 {
	f := floatFormats[dt]
	if dt == mmf.F64 {
		return math.Float64bits(x)
	}
	if math.IsNaN(x) {
		return f.nanBits()
	}
	b := math.Float64bits(x)
	neg := b>>63 != 0
	var s uint64
	if neg {
		s = f.signBit()
	}
	if math.IsInf(x, 0) {
		if f.hasInf {
			return s | f.infBits()
		}
		return s | f.maxFiniteBits()
	}
	ef := int(b>>52) & 0x7FF
	m := b & (1<<52 - 1)
	e := -1074
	if ef != 0 {
		m |= 1 << 52
		e = ef - 1075
	}
	return roundExact(dt, neg, m, e, sat)
}

// roundInt rounds an integer, given as a sign and magnitude, to dt.
func roundInt(dt mmf.DType, neg bool, mag uint64, sat bool) uint64 {
	return roundExact(dt, neg, mag, 0, sat)
}

// ULP returns the spacing of dt's values at the magnitude of x: the unit in
// the last place used by the tolerance rules of spec/primitives.md
// section 3.5. x must be finite.
func ULP(dt mmf.DType, x float64) float64 {
	f := floatFormats[dt]
	e := f.emin
	if x != 0 {
		_, fe := math.Frexp(math.Abs(x))
		if fe-1 > e {
			e = fe - 1
		}
	}
	return math.Ldexp(1, e-(f.p-1))
}
