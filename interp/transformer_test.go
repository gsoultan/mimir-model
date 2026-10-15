package interp

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/ref"
)

func randF32(rng *rand.Rand, shape []int, scale float64) *ref.Tensor {
	v := make([]float32, ref.NumElems(shape))
	for i := range v {
		v[i] = float32(rng.NormFloat64() * scale)
	}
	return f32Tensor(shape, v)
}

func quantized(t *testing.T, dt mmf.DType, rng *rand.Rand, n, k int) *ref.Tensor {
	t.Helper()
	src := make([]float32, n*k)
	for i := range src {
		src[i] = float32(rng.NormFloat64())
	}
	size, err := ref.ByteSize(dt, []int{n, k})
	if err != nil {
		t.Fatal(err)
	}
	b := make([]byte, size)
	if err := mmf.Quantize(dt, src, b); err != nil {
		t.Fatal(err)
	}
	return &ref.Tensor{DType: dt, Shape: []int{n, k}, Data: b}
}

// closeTo compares a fast path's result with the canonical body's, element
// by element, with an absolute tolerance relative to the largest magnitude.
func closeTo(t *testing.T, name string, got, want *ref.Tensor, tol float64) {
	t.Helper()
	if got.DType != want.DType || !sameInts(got.Shape, want.Shape) {
		t.Fatalf("%s: got %s%v, want %s%v", name, got.DType, got.Shape, want.DType, want.Shape)
	}
	g, w := f32Values(got), f32Values(want)
	var big float64
	for _, x := range w {
		big = math.Max(big, math.Abs(float64(x)))
	}
	for i := range g {
		if d := math.Abs(float64(g[i] - w[i])); d > tol*math.Max(big, 1) || math.IsNaN(float64(g[i])) != math.IsNaN(float64(w[i])) {
			t.Fatalf("%s: element %d is %g, canonical body gives %g", name, i, g[i], w[i])
		}
	}
}

// checkFast runs a composite's canonical body and its fast path on the same
// inputs and requires them to agree.
func checkFast(t *testing.T, key string, attrs []mmf.Attr, in []*ref.Tensor, tol float64) *ref.Tensor {
	t.Helper()
	want, err := stdlib[key](attrs, in, []mmf.DType{mmf.F32}, nil)
	if err != nil {
		t.Fatalf("%s body: %v", key, err)
	}
	got, handled, err := fastPaths[key](attrs, in, []mmf.DType{mmf.F32}, nil)
	if err != nil || !handled {
		t.Fatalf("%s fast path: handled %v, %v", key, handled, err)
	}
	closeTo(t, key, got[0], want[0], tol)
	return want[0]
}

func TestFastPathsMatchBodies(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	t.Run("dequant_matmul", func(t *testing.T) {
		x := randF32(rng, []int{1, 3, 64}, 1)
		for _, dt := range []mmf.DType{mmf.Q4_0, mmf.Q8_0} {
			checkFast(t, "dequant_matmul.v1", nil, []*ref.Tensor{x, quantized(t, dt, rng, 5, 64)}, 1e-5)
		}
		w := quantized(t, mmf.Q8_0, rng, 4, 64)
		checkFast(t, "dequant_matmul.v1", nil, []*ref.Tensor{x, w, randF32(rng, []int{4}, 1)}, 1e-5)
	})
	t.Run("linear", func(t *testing.T) {
		checkFast(t, "linear.v1", nil, []*ref.Tensor{randF32(rng, []int{2, 7}, 1), randF32(rng, []int{3, 7}, 1)}, 1e-5)
	})
	t.Run("rms_norm", func(t *testing.T) {
		attrs := []mmf.Attr{fAttr("epsilon", 1e-5)}
		checkFast(t, "rms_norm.v1", attrs, []*ref.Tensor{randF32(rng, []int{1, 2, 16}, 3), randF32(rng, []int{16}, 1)}, 1e-5)
	})
	t.Run("swiglu", func(t *testing.T) {
		checkFast(t, "swiglu.v1", nil, []*ref.Tensor{randF32(rng, []int{2, 9}, 3), randF32(rng, []int{2, 9}, 1)}, 1e-5)
	})
	t.Run("rope", func(t *testing.T) {
		x := randF32(rng, []int{1, 2, 3, 8}, 1)
		pos := i64t([]int{1, 3}, 0, 5, 1000)
		for _, inter := range []int64{0, 1} {
			for _, rd := range []int64{8, 4} {
				attrs := []mmf.Attr{fAttr("theta", 10000), iAttr("interleaved", inter), iAttr("rotary_dims", rd)}
				checkFast(t, "rope.v1", attrs, []*ref.Tensor{x, pos}, 1e-4)
			}
		}
		yarn := []mmf.Attr{iAttr("yarn_original_context", 64), fAttr("yarn_factor", 4)}
		checkFast(t, "rope.v1", yarn, []*ref.Tensor{x, pos}, 1e-4)
	})
	t.Run("attention", func(t *testing.T) {
		q := randF32(rng, []int{1, 4, 3, 8}, 1)
		k := randF32(rng, []int{1, 2, 5, 8}, 1)
		v := randF32(rng, []int{1, 2, 5, 6}, 1)
		gqa := []mmf.Attr{iAttr("num_kv_heads", 2)}
		checkFast(t, "attention.v1", gqa, []*ref.Tensor{q, k, v}, 1e-5)
		checkFast(t, "attention.v1", append(gqa, iAttr("causal", 1)), []*ref.Tensor{q, k, v}, 1e-5)
		mask, _ := ref.New(mmf.Bool, []int{1, 1, 3, 5})
		for i := range mask.Data {
			mask.Data[i] = byte(i % 3 % 2)
			if i%5 == 0 {
				mask.Data[i] = 1
			}
		}
		checkFast(t, "attention.v1", gqa, []*ref.Tensor{q, k, v, mask}, 1e-5)
		checkFast(t, "attention.v1", gqa, []*ref.Tensor{q, k, v, randF32(rng, []int{1, 4, 3, 1}, 1)}, 1e-5)
	})
	t.Run("embedding", func(t *testing.T) {
		table := quantized(t, mmf.Q8_0, rng, 6, 32)
		got := checkFast(t, "embedding.v1", nil, []*ref.Tensor{table, i64t([]int{1, 3}, 5, 0, 5)}, 0)
		if !sameInts(got.Shape, []int{1, 3, 32}) {
			t.Fatalf("shape %v", got.Shape)
		}
	})
}

// TestRopeDefinition checks rope.v1 against the rotation it defines, on one
// pair, in both layouts.
func TestRopeDefinition(t *testing.T) {
	x := f32t([]int{1, 1, 1, 4}, 1, 2, 3, 4)
	pos := i64t([]int{1, 1}, 3)
	theta := 100.0
	f0, f1 := 3.0, 3*math.Pow(theta, -0.5)
	rot := func(a, b, ang float64) (float64, float64) {
		return a*math.Cos(ang) - b*math.Sin(ang), b*math.Cos(ang) + a*math.Sin(ang)
	}
	for _, c := range []struct {
		inter int64
		want  [4]float64
	}{
		{0, func() (w [4]float64) { w[0], w[2] = rot(1, 3, f0); w[1], w[3] = rot(2, 4, f1); return }()},
		{1, func() (w [4]float64) { w[0], w[1] = rot(1, 2, f0); w[2], w[3] = rot(3, 4, f1); return }()},
	} {
		out, err := ropeV1([]mmf.Attr{fAttr("theta", theta), iAttr("interleaved", c.inter)}, []*ref.Tensor{x, pos}, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		for i, g := range f32Values(out[0]) {
			if math.Abs(float64(g)-c.want[i]) > 1e-5 {
				t.Errorf("interleaved=%d element %d: %g, want %g", c.inter, i, g, c.want[i])
			}
		}
	}
}

// TestAttentionDefinition checks attention.v1 with grouped heads and a
// causal mask against a direct float64 evaluation.
func TestAttentionDefinition(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	q := randF32(rng, []int{1, 2, 2, 4}, 1)
	k := randF32(rng, []int{1, 1, 3, 4}, 1)
	v := randF32(rng, []int{1, 1, 3, 2}, 1)
	out, err := attentionV1([]mmf.Attr{iAttr("num_kv_heads", 1), iAttr("causal", 1)}, []*ref.Tensor{q, k, v}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	qv, kv, vv, ov := f32Values(q), f32Values(k), f32Values(v), f32Values(out[0])
	for h := 0; h < 2; h++ {
		for i := 0; i < 2; i++ {
			// Query i sees keys 0..i+1 (Tk - Tq = 1).
			var w []float64
			var sum float64
			for j := 0; j <= i+1; j++ {
				var s float64
				for d := 0; d < 4; d++ {
					s += float64(qv[(h*2+i)*4+d]) * float64(kv[j*4+d])
				}
				e := math.Exp(s / 2)
				w = append(w, e)
				sum += e
			}
			for d := 0; d < 2; d++ {
				var o float64
				for j, e := range w {
					o += e / sum * float64(vv[j*2+d])
				}
				if g := ov[(h*2+i)*2+d]; math.Abs(float64(g)-o) > 1e-5 {
					t.Errorf("head %d query %d dim %d: %g, want %g", h, i, d, g, o)
				}
			}
		}
	}
}
