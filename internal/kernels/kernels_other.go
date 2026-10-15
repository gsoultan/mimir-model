//go:build !amd64 || purego

package kernels

const useAVX2, hasF16C = false, false

// The assembly kernels are never called when useAVX2 is false.

func dotF32AVX2(a, b *float32, n int) float32              { panic("no assembly") }
func axpyF32AVX2(alpha float32, x, y *float32, n int)      { panic("no assembly") }
func dotQ4KAVX2(q *byte, x, tab *float32, nb int) float32  { panic("no assembly") }
func dotQ6KAVX2(q *byte, x, tab *float32, nb int) float32  { panic("no assembly") }
func dotQ8_0AVX2(q *byte, x, tab *float32, nb int) float32 { panic("no assembly") }
func dotF16AVX2(w *byte, x *float32, n int) float32        { panic("no assembly") }
func dotBF16AVX2(w *byte, x *float32, n int) float32       { panic("no assembly") }

func dotQ4KQ8AVX2(q *byte, xq *int8, xd, xdbs *float32, nb int) (float32, bool) { panic("no assembly") }
func dotQ6KQ8AVX2(q *byte, xq *int8, xd *float32, m32 *int16, nb int) (float32, bool) {
	panic("no assembly")
}
func dotQ8_0Q8AVX2(q *byte, xq *int8, xd *float32, nb int) (float32, bool) { panic("no assembly") }
