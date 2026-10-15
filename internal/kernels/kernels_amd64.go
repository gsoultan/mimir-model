//go:build amd64 && !purego

package kernels

// cpuid and xgetbv are implemented in kernels_amd64.s.
func cpuid(eax, ecx uint32) (a, b, c, d uint32)
func xgetbv() (eax, edx uint32)

var useAVX2, hasF16C = detect()

// detect reports whether the CPU and the operating system support AVX2
// with FMA, and F16C.
func detect() (avx2, f16c bool) {
	maxID, _, _, _ := cpuid(0, 0)
	if maxID < 7 {
		return false, false
	}
	_, _, c1, _ := cpuid(1, 0)
	const (
		fma     = 1 << 12
		osxsave = 1 << 27
		avx     = 1 << 28
		f16cBit = 1 << 29
	)
	if c1&(fma|osxsave|avx) != fma|osxsave|avx {
		return false, false
	}
	// The OS must save the XMM and YMM registers on context switches.
	if lo, _ := xgetbv(); lo&6 != 6 {
		return false, false
	}
	_, b7, _, _ := cpuid(7, 0)
	if b7&(1<<5) == 0 {
		return false, false
	}
	return true, c1&f16cBit != 0
}

//go:noescape
func dotF32AVX2(a, b *float32, n int) float32

//go:noescape
func axpyF32AVX2(alpha float32, x, y *float32, n int)

//go:noescape
func dotQ4KAVX2(q *byte, x, tab *float32, nb int) float32

//go:noescape
func dotQ6KAVX2(q *byte, x, tab *float32, nb int) float32

//go:noescape
func dotQ8_0AVX2(q *byte, x, tab *float32, nb int) float32

//go:noescape
func dotF16AVX2(w *byte, x *float32, n int) float32

//go:noescape
func dotBF16AVX2(w *byte, x *float32, n int) float32

//go:noescape
func dotQ4KQ8AVX2(q *byte, xq *int8, xd, xdbs *float32, nb int) (float32, bool)

//go:noescape
func dotQ6KQ8AVX2(q *byte, xq *int8, xd *float32, m32 *int16, nb int) (float32, bool)

//go:noescape
func dotQ8_0Q8AVX2(q *byte, xq *int8, xd *float32, nb int) (float32, bool)
