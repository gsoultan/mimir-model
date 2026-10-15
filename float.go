package mmf

import "math"

// This file holds pure-Go conversions between float32 and the narrow float
// encodings used by the plain and block dtypes. Every conversion is exact
// except the narrowing ones, which round to nearest, ties to even.

// F16ToF32 converts an IEEE 754 binary16 bit pattern to float32. The
// conversion is exact; subnormals, infinities and NaNs are preserved.
func F16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h & 0x3FF)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal: value = mant * 2^-24. Normalize into a float32.
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		return math.Float32frombits(sign | e<<23 | (mant&0x3FF)<<13)
	case 0x1F:
		return math.Float32frombits(sign | 0x7F800000 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
}

// F32ToF16 converts a float32 to an IEEE 754 binary16 bit pattern, rounding to
// nearest with ties to even. Values too large for binary16 become infinity,
// values too small become zero or a subnormal, and NaN stays NaN (quiet, with
// the top payload bits kept).
func F32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int32(b>>23) & 0xFF
	mant := b & 0x7FFFFF
	if exp == 0xFF {
		if mant == 0 {
			return sign | 0x7C00
		}
		return sign | 0x7E00 | uint16(mant>>13)
	}
	e := exp - 127 + 15
	if e >= 0x1F {
		return sign | 0x7C00
	}
	if e <= 0 {
		// Result is zero or subnormal in binary16: value = q * 2^-24.
		if e < -11 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - e)
		half := uint32(1) << (shift - 1)
		rem := mant & (half<<1 - 1)
		q := mant >> shift
		if rem > half || (rem == half && q&1 == 1) {
			q++ // may carry into exponent 1, which is correct
		}
		return sign | uint16(q)
	}
	q := uint32(e)<<10 | mant>>13
	rem := mant & 0x1FFF
	if rem > 0x1000 || (rem == 0x1000 && q&1 == 1) {
		q++ // may carry into the exponent, up to 0x7C00 (infinity), which is correct
	}
	return sign | uint16(q)
}

// BF16ToF32 converts a bfloat16 bit pattern to float32. The conversion is exact.
func BF16ToF32(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}

// F32ToBF16 converts a float32 to bfloat16, rounding to nearest with ties to
// even. NaN stays NaN (quiet).
func F32ToBF16(f float32) uint16 {
	b := math.Float32bits(f)
	if b&0x7F800000 == 0x7F800000 && b&0x7FFFFF != 0 {
		return uint16(b>>16) | 0x40
	}
	return uint16((b + 0x7FFF + (b>>16)&1) >> 16)
}

// F8E4M3ToF32 converts an OCP FP8 E4M3 (the "fn" variant) bit pattern to
// float32: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits, no
// infinities; 0x7F and 0xFF are NaN. Largest finite value is 448.
func F8E4M3ToF32(b uint8) float32 {
	sign := uint32(b&0x80) << 24
	exp := uint32(b>>3) & 0xF
	mant := uint32(b & 7)
	if exp == 0xF && mant == 7 {
		return math.Float32frombits(sign | 0x7FC00000)
	}
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal: value = mant * 2^-9.
		e := uint32(127 - 7 + 1)
		for mant&8 == 0 {
			mant <<= 1
			e--
		}
		return math.Float32frombits(sign | e<<23 | (mant&7)<<20)
	}
	return math.Float32frombits(sign | (exp+127-7)<<23 | mant<<20)
}

// F8E5M2ToF32 converts an OCP FP8 E5M2 bit pattern to float32: 1 sign bit,
// 5 exponent bits (bias 15), 2 mantissa bits, with IEEE-style infinities and
// NaNs. It is binary16 truncated to one byte.
func F8E5M2ToF32(b uint8) float32 {
	return F16ToF32(uint16(b) << 8)
}

// E8M0ToF32 converts an OCP MX E8M0 shared scale to float32: 2^(e-127) for
// e in 0..254 (e = 0 is the subnormal 2^-127) and NaN for 0xFF.
func E8M0ToF32(e uint8) float32 {
	switch e {
	case 0xFF:
		return math.Float32frombits(0x7FC00000)
	case 0:
		return math.Float32frombits(0x00400000) // 2^-127
	}
	return math.Float32frombits(uint32(e) << 23)
}

// e2m1Table holds the eight non-negative E2M1 values indexed by the low 3 bits.
var e2m1Table = [8]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}

// E2M1ToF32 converts an OCP MX FP4 E2M1 value held in the low 4 bits of b to
// float32: bit 3 is the sign, bits 2..1 the exponent (bias 1) and bit 0 the
// mantissa. The magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4 and 6. The high 4 bits
// of b are ignored.
func E2M1ToF32(b uint8) float32 {
	return math.Float32frombits(math.Float32bits(e2m1Table[b&7]) | uint32(b&8)<<28)
}
