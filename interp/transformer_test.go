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

package interp

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/gsoultan/mlm"
	"github.com/gsoultan/mlm/ref"
)

func randF32(rng *rand.Rand, shape []int, scale float64) *ref.Tensor {
	v := make([]float32, ref.NumElems(shape))
	for i := range v {
		v[i] = float32(rng.NormFloat64() * scale)
	}
	return f32Tensor(shape, v)
}

func quantized(t *testing.T, dt mlm.DType, rng *rand.Rand, n, k int) *ref.Tensor {
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
	if err := mlm.Quantize(dt, src, b); err != nil {
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
func checkFast(t *testing.T, key string, attrs []mlm.Attr, in []*ref.Tensor, tol float64) *ref.Tensor {
	t.Helper()
	want, err := stdlib[key](attrs, in, []mlm.DType{mlm.F32}, nil)
	if err != nil {
		t.Fatalf("%s body: %v", key, err)
	}
	got, handled, err := fastPaths[key](attrs, in, []mlm.DType{mlm.F32}, nil)
	if err != nil || !handled {
		t.Fatalf("%s fast path: handled %v, %v", key, handled, err)
	}
	closeTo(t, key, got[0], want[0], tol)
	return want[0]
}

// castTo casts x with the reference cast.
func castTo(t *testing.T, x *ref.Tensor, dt mlm.DType) *ref.Tensor {
	t.Helper()
	y, err := ref.Eval(mlm.OpCast, []mlm.Attr{{Key: "to", Type: mlm.AttrDType, DType: dt}}, []*ref.Tensor{x}, dt, x.Shape)
	if err != nil {
		t.Fatal(err)
	}
	return y
}

func TestFastPathsMatchBodies(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	t.Run("dequant_matmul", func(t *testing.T) {
		x := randF32(rng, []int{1, 3, 64}, 1)
		for _, dt := range []mlm.DType{mlm.Q4_0, mlm.Q8_0} {
			checkFast(t, "dequant_matmul.v1", nil, []*ref.Tensor{x, quantized(t, dt, rng, 5, 64)}, 1e-5)
		}
		w := quantized(t, mlm.Q8_0, rng, 4, 64)
		checkFast(t, "dequant_matmul.v1", nil, []*ref.Tensor{x, w, randF32(rng, []int{4}, 1)}, 1e-5)
	})
	t.Run("linear", func(t *testing.T) {
		checkFast(t, "linear.v1", nil, []*ref.Tensor{randF32(rng, []int{2, 7}, 1), randF32(rng, []int{3, 7}, 1)}, 1e-5)
	})
	t.Run("rms_norm", func(t *testing.T) {
		attrs := []mlm.Attr{fAttr("epsilon", 1e-5)}
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
				attrs := []mlm.Attr{fAttr("theta", 10000), iAttr("interleaved", inter), iAttr("rotary_dims", rd)}
				checkFast(t, "rope.v1", attrs, []*ref.Tensor{x, pos}, 1e-4)
			}
		}
		yarn := []mlm.Attr{iAttr("yarn_original_context", 64), fAttr("yarn_factor", 4)}
		checkFast(t, "rope.v1", yarn, []*ref.Tensor{x, pos}, 1e-4)
	})
	t.Run("attention", func(t *testing.T) {
		q := randF32(rng, []int{1, 4, 3, 8}, 1)
		k := randF32(rng, []int{1, 2, 5, 8}, 1)
		v := randF32(rng, []int{1, 2, 5, 6}, 1)
		gqa := []mlm.Attr{iAttr("num_kv_heads", 2)}
		checkFast(t, "attention.v1", gqa, []*ref.Tensor{q, k, v}, 1e-5)
		checkFast(t, "attention.v1", append(gqa, iAttr("causal", 1)), []*ref.Tensor{q, k, v}, 1e-5)
		mask, _ := ref.New(mlm.Bool, []int{1, 1, 3, 5})
		for i := range mask.Data {
			mask.Data[i] = byte(i % 3 % 2)
			if i%5 == 0 {
				mask.Data[i] = 1
			}
		}
		checkFast(t, "attention.v1", gqa, []*ref.Tensor{q, k, v, mask}, 1e-5)
		checkFast(t, "attention.v1", gqa, []*ref.Tensor{q, k, v, randF32(rng, []int{1, 4, 3, 1}, 1)}, 1e-5)

		// Rows long enough for the vector kernels, with a causal mask as
		// the llama importer writes it: a query sees the keys up to its
		// position, and the rest of the cache is masked.
		q = randF32(rng, []int{1, 4, 11, 64}, 1)
		k = randF32(rng, []int{1, 2, 40, 64}, 1)
		v = randF32(rng, []int{1, 2, 40, 64}, 1)
		causal, _ := ref.New(mlm.Bool, []int{1, 1, 11, 40})
		for i := range 11 {
			for j := range 40 {
				if j <= i+20 {
					causal.Data[i*40+j] = 1
				}
			}
		}
		checkFast(t, "attention.v1", gqa, []*ref.Tensor{q, k, v, causal}, 1e-5)
		checkFast(t, "attention.v1", append(gqa, iAttr("causal", 1)), []*ref.Tensor{q, k, v}, 1e-5)

		// attention.v2 with the keys and values in f16 or bf16, as a cache
		// in f16 holds them.
		for _, dt := range []mlm.DType{mlm.F16, mlm.BF16} {
			k2, v2 := castTo(t, k, dt), castTo(t, v, dt)
			checkFast(t, "attention.v2", gqa, []*ref.Tensor{q, k2, v2, causal}, 1e-5)
			checkFast(t, "attention.v2", append(gqa, iAttr("causal", 1)), []*ref.Tensor{q, k2, v2}, 1e-5)
			checkFast(t, "attention.v2", gqa, []*ref.Tensor{q, k2, v, randF32(rng, []int{1, 4, 11, 1}, 1)}, 1e-5)
		}
		// Converting the f16 rows to float32 before the kernels changes
		// no bit of the result.
		k2, v2 := castTo(t, k, mlm.F16), castTo(t, v, mlm.F16)
		for _, in := range [][]*ref.Tensor{{q, k2, v2, causal}, {q, k2, v2}} {
			at := gqa
			if len(in) == 3 {
				at = append(gqa, iAttr("causal", 1))
			}
			conv, _, err := fastAttentionV2(at, in, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			attnConvertRows = 1 << 40
			rows, _, err := fastAttentionV2(at, in, nil, nil)
			attnConvertRows = 8
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(conv[0].Data, rows[0].Data) {
				t.Errorf("%d inputs: converted rows give another result", len(in))
			}
		}
		if _, err := stdlib["attention.v1"](gqa, []*ref.Tensor{q, castTo(t, k, mlm.F16), v}, []mlm.DType{mlm.F32}, nil); err == nil {
			t.Error("attention.v1 accepted f16 keys with f32 queries")
		}
	})
	t.Run("embedding", func(t *testing.T) {
		table := quantized(t, mlm.Q8_0, rng, 6, 32)
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
		out, err := ropeV1([]mlm.Attr{fAttr("theta", theta), iAttr("interleaved", c.inter)}, []*ref.Tensor{x, pos}, nil, nil)
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
	out, err := attentionCanonical("attention.v1", []mlm.Attr{iAttr("num_kv_heads", 1), iAttr("causal", 1)}, []*ref.Tensor{q, k, v})
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

// kquant returns an [n, k] tensor of dtype dt with random quants and
// moderate scales, for the K-quant dtypes mlm.Quantize does not encode.
func kquant(rng *rand.Rand, dt mlm.DType, n, k int) *ref.Tensor {
	size, _ := ref.ByteSize(dt, []int{n, k})
	b := make([]byte, size)
	rng.Read(b)
	put := func(o int, v float32) { binary.LittleEndian.PutUint16(b[o:], mlm.F32ToF16(v)) }
	switch dt {
	case mlm.Q4_K:
		for o := 0; o < len(b); o += 144 {
			put(o, rng.Float32()*0.01)
			put(o+2, rng.Float32()*0.01)
		}
	case mlm.Q6_K:
		for o := 0; o < len(b); o += 210 {
			put(o+208, rng.Float32()*0.01-0.005)
		}
	}
	return &ref.Tensor{DType: dt, Shape: []int{n, k}, Data: b}
}

// TestDequantMatmulA8RowsTogether checks that each row of a
// dequant_matmul_a8.v2 call with activation blocks of 256 comes out the
// same, bit for bit, whether it is multiplied alone or with 20 others: on
// CPUs with AMX the 21 rows take the tile kernels, elsewhere the four-row
// kernels. The weights have 40 rows, so the last 8 are not a whole tile,
// and one weight row and one activation row are not finite.
func TestDequantMatmulA8RowsTogether(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	// 150 rows are several of linearA8's blocks of rows.
	const k, rows, n = 512, 150, 40
	attrs := []mlm.Attr{intAttr("block", 256)}
	for _, dt := range []mlm.DType{mlm.Q4_K, mlm.Q6_K} {
		x := randF32(rng, []int{rows, k}, 1)
		x.SetBits(9*k+3, uint64(math.Float32bits(float32(math.Inf(1)))))
		x.SetBits(97*k+300, uint64(math.Float32bits(float32(math.NaN()))))
		w := kquant(rng, dt, n, k)
		rb := len(w.Data) / n
		bad := 20*rb + rb/2 // a block of weight row 20
		bad -= bad % int(dt.BlockBytes())
		if dt == mlm.Q6_K {
			bad += 208
		}
		w.Data[bad], w.Data[bad+1] = 0x00, 0x7C // +inf
		in := []*ref.Tensor{x, w, randF32(rng, []int{n}, 1)}
		all, handled, err := fastPaths["dequant_matmul_a8.v2"](attrs, in, []mlm.DType{mlm.F32}, nil)
		if err != nil || !handled {
			t.Fatalf("%s: handled %v, %v", dt, handled, err)
		}
		got := f32Values(all[0])
		xv := f32Values(x)
		for i := range rows {
			xi := f32Tensor([]int{1, k}, xv[i*k:(i+1)*k])
			one, _, err := fastPaths["dequant_matmul_a8.v2"](attrs, []*ref.Tensor{xi, w, in[2]}, []mlm.DType{mlm.F32}, nil)
			if err != nil {
				t.Fatal(err)
			}
			for j, want := range f32Values(one[0]) {
				g := got[i*n+j]
				if math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) {
					// The float32 path of a row or weight that is not finite
					// may sum in another order; the result is not finite either way.
					if !math.IsNaN(float64(g)) && !math.IsInf(float64(g), 0) {
						t.Errorf("%s [%d,%d]: got %g, alone %g", dt, i, j, g, want)
					}
					continue
				}
				if math.Float32bits(g) != math.Float32bits(want) {
					t.Errorf("%s [%d,%d]: got %g, alone %g", dt, i, j, g, want)
				}
			}
		}
	}
}

// TestDequantMatmulA8 checks the 8-bit activation fast paths of
// dequant_matmul_a8.v1 and v2 against the exact canonical body with their
// tolerance: one quantization step of each activation block per weight,
// plus rounding. Five rows of x take both the four-row and the one-row
// kernels.
func TestDequantMatmulA8(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const k, rows = 512, 5
	x := randF32(rng, []int{1, rows, k}, 1)
	xv := f32Values(x)
	for _, c := range []struct {
		name  string
		attrs []mlm.Attr
		block int
	}{
		{"dequant_matmul_a8.v1", nil, 32},
		{"dequant_matmul_a8.v2", nil, 32},
		{"dequant_matmul_a8.v2", []mlm.Attr{intAttr("block", 32)}, 32},
		{"dequant_matmul_a8.v2", []mlm.Attr{intAttr("block", 256)}, 256},
	} {
		for _, w := range []*ref.Tensor{
			quantized(t, mlm.Q8_0, rng, 6, k), kquant(rng, mlm.Q4_K, 6, k), kquant(rng, mlm.Q6_K, 6, k), quantized(t, mlm.Q4_0, rng, 6, k),
		} {
			in := []*ref.Tensor{x, w, randF32(rng, []int{6}, 1)}
			want, err := stdlib[c.name](c.attrs, in, []mlm.DType{mlm.F32}, nil)
			if err != nil {
				t.Fatal(err)
			}
			got, handled, err := fastPaths[c.name](c.attrs, in, []mlm.DType{mlm.F32}, nil)
			if err != nil || !handled {
				t.Fatalf("%s %s: handled %v, %v", c.name, w.DType, handled, err)
			}
			wd := make([]float32, 6*k)
			mlm.Dequantize(w.DType, w.Data, wd)
			g, e := f32Values(got[0]), f32Values(want[0])
			for i := 0; i < rows; i++ {
				for j := 0; j < 6; j++ {
					var bound float64
					for b := 0; b < k; b += c.block {
						var amax float64
						for _, v := range xv[i*k+b : i*k+b+c.block] {
							amax = math.Max(amax, math.Abs(float64(v)))
						}
						for kk := b; kk < b+c.block; kk++ {
							aw := math.Abs(float64(wd[j*k+kk]))
							bound += amax/127*aw + 1e-4*(math.Abs(float64(xv[i*k+kk]))+amax/127)*aw
						}
					}
					if d := math.Abs(float64(g[i*6+j] - e[i*6+j])); d > bound {
						t.Errorf("%s/%d %s [%d,%d]: got %g, exact %g, diff %g > %g", c.name, c.block, w.DType, i, j, g[i*6+j], e[i*6+j], d, bound)
					}
				}
			}
		}
	}
	for _, bad := range [][]mlm.Attr{{intAttr("block", 64)}, {intAttr("other", 1)}} {
		in := []*ref.Tensor{x, kquant(rng, mlm.Q4_K, 6, k)}
		if _, err := stdlib["dequant_matmul_a8.v2"](bad, in, []mlm.DType{mlm.F32}, nil); err == nil {
			t.Errorf("body accepted %v", bad)
		}
		if _, _, err := fastPaths["dequant_matmul_a8.v2"](bad, in, []mlm.DType{mlm.F32}, nil); err == nil {
			t.Errorf("fast path accepted %v", bad)
		}
	}
	// Activations that are not finite take the float32 path, so infinities
	// and NaN come out as the exact product's.
	bad := randF32(rng, []int{1, 1, k}, 1)
	bad.SetBits(7, uint64(math.Float32bits(float32(math.Inf(1)))))
	checkFast(t, "dequant_matmul_a8.v1", nil, []*ref.Tensor{bad, kquant(rng, mlm.Q4_K, 4, k)}, 1e-5)
}
