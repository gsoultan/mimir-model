package kernels

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/gsoultan/mmf"
)

// randomRows returns n rows of k elements of dt with finite weights of
// moderate size.
func randomRows(rng *rand.Rand, dt mmf.DType, n, k int) []byte {
	rowBytes := int(uint64(k) / dt.BlockElems() * dt.BlockBytes())
	b := make([]byte, n*rowBytes)
	rng.Read(b)
	f16 := func(p []byte, v float32) { binary.LittleEndian.PutUint16(p, mmf.F32ToF16(v)) }
	switch dt {
	case mmf.Q4_K:
		for o := 0; o < len(b); o += 144 {
			f16(b[o:], rng.Float32()*0.01)
			f16(b[o+2:], rng.Float32()*0.01)
		}
	case mmf.Q6_K:
		for o := 0; o < len(b); o += 210 {
			f16(b[o+208:], rng.Float32()*0.01-0.005)
		}
	case mmf.Q8_0:
		for o := 0; o < len(b); o += 34 {
			f16(b[o:], rng.Float32()*0.01-0.005)
		}
	case mmf.Q4_0:
		for o := 0; o < len(b); o += 18 {
			f16(b[o:], rng.Float32()*0.01-0.005)
		}
	case mmf.F32:
		for o := 0; o < len(b); o += 4 {
			binary.LittleEndian.PutUint32(b[o:], math.Float32bits(float32(rng.NormFloat64())))
		}
	case mmf.F16:
		for o := 0; o < len(b); o += 2 {
			f16(b[o:], float32(rng.NormFloat64()))
		}
	case mmf.BF16:
		for o := 0; o < len(b); o += 2 {
			binary.LittleEndian.PutUint16(b[o:], uint16(math.Float32bits(float32(rng.NormFloat64()))>>16))
		}
	}
	return b
}

func randomVec(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

// check compares got with the float64 dot product of x and w, within the
// linear.v1 bound K u sum |x w| (spec/stdlib.md section 3.2).
func check(t *testing.T, what string, got float32, x, w []float32) {
	t.Helper()
	var want, mag float64
	for i := range x {
		p := float64(x[i]) * float64(w[i])
		want += p
		mag += math.Abs(p)
	}
	tol := float64(len(x)+1) * 0x1p-24 * mag
	if d := math.Abs(float64(got) - want); d > tol {
		t.Errorf("%s: got %g, want %g (diff %g > %g)", what, got, want, d, tol)
	}
}

var rowDTypes = []mmf.DType{mmf.Q4_K, mmf.Q6_K, mmf.Q8_0, mmf.F16, mmf.BF16, mmf.F32, mmf.Q4_0}

func TestRowDot(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, dt := range rowDTypes {
		for _, k := range []int{256, 512, 2048, 5632} {
			const n = 4
			rows := randomRows(rng, dt, n, k)
			rowBytes := len(rows) / n
			r := NewRowDot(dt, k)
			w := make([]float32, k)
			for i := 0; i < n; i++ {
				x := randomVec(rng, k)
				row := rows[i*rowBytes : (i+1)*rowBytes]
				if err := mmf.Dequantize(dt, row, w); err != nil {
					t.Fatal(err)
				}
				for _, uses := range []int{1, DecodeOnce + 1} {
					r.Load(row, uses)
					check(t, dt.String(), r.Dot(x), x, w)
				}
			}
		}
	}
}

// TestRowDotDecodesExactly multiplies rows by unit vectors, which picks out
// each decoded weight exactly: the kernels must decode bit for bit as
// mmf.Dequantize does.
func TestRowDotDecodesExactly(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, dt := range rowDTypes {
		const k = 512
		row := randomRows(rng, dt, 1, k)
		w := make([]float32, k)
		if err := mmf.Dequantize(dt, row, w); err != nil {
			t.Fatal(err)
		}
		r := NewRowDot(dt, k)
		r.Load(row, 1)
		x := make([]float32, k)
		for i := 0; i < k; i++ {
			x[i] = 1
			if got := r.Dot(x); math.Float32bits(got) != math.Float32bits(w[i]) && !(got == 0 && w[i] == 0) {
				t.Errorf("%s weight %d: got %g, want %g", dt, i, got, w[i])
			}
			x[i] = 0
		}
	}
}

func TestDotF32(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for n := 0; n < 300; n++ {
		a, b := randomVec(rng, n), randomVec(rng, n+3)
		check(t, "dot", DotF32(a, b), a, b[:n])
	}
}

func TestAxpyF32(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for n := 0; n < 100; n++ {
		x, y := randomVec(rng, n), randomVec(rng, n+2)
		want := append([]float32{}, y...)
		alpha := float32(rng.NormFloat64())
		for i := range x {
			want[i] = float32(float64(want[i]) + float64(alpha)*float64(x[i]))
		}
		AxpyF32(alpha, x, y)
		for i := range y {
			// One rounding (fused) or two (separate multiply and add).
			if d := math.Abs(float64(y[i] - want[i])); d > 2*0x1p-24*(math.Abs(float64(want[i]))+math.Abs(float64(alpha*x0(x, i)))) {
				t.Fatalf("n=%d: y[%d] = %g, want %g", n, i, y[i], want[i])
			}
		}
	}
}

func x0(x []float32, i int) float32 {
	if i < len(x) {
		return x[i]
	}
	return 0
}

func BenchmarkRowDot(b *testing.B) {
	rng := rand.New(rand.NewSource(5))
	for _, dt := range []mmf.DType{mmf.Q4_K, mmf.Q6_K, mmf.Q8_0, mmf.F16} {
		const k = 2048
		row := randomRows(rng, dt, 1, k)
		x := randomVec(rng, k)
		r := NewRowDot(dt, k)
		b.Run(dt.String(), func(b *testing.B) {
			b.SetBytes(int64(len(row)))
			for i := 0; i < b.N; i++ {
				r.Load(row, 1)
				r.Dot(x)
			}
		})
	}
}
