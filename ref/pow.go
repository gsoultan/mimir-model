package ref

import "math"

// pow64 is pow for non-NaN operands.
func pow64(a, b float64) float64 { return math.Pow(a, b) }
