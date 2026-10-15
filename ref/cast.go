package ref

import (
	"math"

	"github.com/gsoultan/mmf"
)

// isStorage reports whether dt is a storage float (spec/primitives.md 2.2).
func isStorage(dt mmf.DType) bool { return dt == mmf.F8E4M3 || dt == mmf.F8E5M2 }

// castBits converts one element following spec/primitives.md section 5.8.
// Conversions to storage floats saturate finite overflow
// (spec/dtypes.md section 2.3).
func castBits(from, to mmf.DType, b uint64) uint64 {
	if to == mmf.Bool {
		if IsFloat(from) {
			return b2u(DecodeFloat(from, b) != 0 || math.IsNaN(DecodeFloat(from, b)))
		}
		return b2u(b != 0)
	}
	sat := isStorage(to)
	switch {
	case from == mmf.Bool:
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
func floatToInt(to mmf.DType, x float64) uint64 {
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

func evalCast(x *Tensor, to mmf.DType) (*Tensor, error) {
	y, err := New(to, x.Shape)
	if err != nil {
		return nil, err
	}
	for i := 0; i < x.NumElems(); i++ {
		y.SetBits(i, castBits(x.DType, to, x.Bits(i)))
	}
	return y, nil
}
