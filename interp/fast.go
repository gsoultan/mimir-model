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

package interp

import (
	"encoding/binary"
	"math"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/internal/kernels"
	"github.com/gsoultan/mmf/ref"
)

// Fast paths (spec/stdlib.md section 1.2). Each one computes a composite
// directly in float32 instead of evaluating its canonical body element by
// element, and must stay within the composite's tolerance; the tests check
// each against its body. A fast path that does not handle its inputs (an
// unusual dtype or broadcast) returns handled = false and the executor
// falls back to the canonical body.
//
// They are parallelised over output rows. The inner loops are in package
// internal/kernels: AVX2 assembly on amd64, which decodes block weights
// inside the dot product, and plain Go elsewhere. Weights are always
// decoded exactly and activations are never quantized, so the
// dequant_matmul.v1 tolerance holds.

type fastFunc func(attrs []mmf.Attr, in []*ref.Tensor, dts []mmf.DType, shapes [][]int) (out []*ref.Tensor, handled bool, err error)

var fastPaths = map[string]fastFunc{
	"linear.v1":            fastLinear,
	"dequant_matmul.v1":    fastDequantMatmul,
	"dequant_matmul_a8.v1": fastDequantMatmulA8,
	"dequant_matmul_a8.v2": fastDequantMatmulA8V2,
	"rms_norm.v1":          fastRMSNorm,
	"swiglu.v1":            fastSwiglu,
	"rope.v1":              fastRope,
	"attention.v1":         fastAttention,
	"embedding.v1":         fastEmbedding,
	"conv2d.v1":            fastConv2d,
}

// matrixRows decodes rows of a [N, K] matrix of any decodable dtype.
type matrixRows struct {
	t        *ref.Tensor
	k        int
	rowBytes int
}

func newMatrixRows(t *ref.Tensor) (*matrixRows, bool) {
	if len(t.Shape) != 2 || (t.DType.IsBlock() && uint64(t.Shape[1])%t.DType.BlockElems() != 0) {
		return nil, false
	}
	n, err := ref.ByteSize(t.DType, []int{t.Shape[1]})
	if err != nil {
		return nil, false
	}
	return &matrixRows{t: t, k: t.Shape[1], rowBytes: n}, true
}

func (m *matrixRows) rowBytesOf(i int) []byte {
	return m.t.Data[i*m.rowBytes : (i+1)*m.rowBytes]
}

func (m *matrixRows) row(i int, dst []float32) {
	// The length checks of Dequantize cannot fail: the row size was
	// computed from the dtype and K.
	_ = mmf.Dequantize(m.t.DType, m.rowBytesOf(i), dst)
}

// linearF32 computes x [rows, K] times w^T for w [N, K], plus bias.
func linearF32(x []float32, rows int, w *matrixRows, bias []float32) []float32 {
	n, k := w.t.Shape[0], w.k
	y := make([]float32, rows*n)
	parallel(n, func(lo, hi int) {
		rd := kernels.NewRowDot(w.t.DType, k)
		for j := lo; j < hi; j++ {
			rd.Load(w.rowBytesOf(j), rows)
			for i := 0; i < rows; i++ {
				s := rd.Dot(x[i*k : (i+1)*k])
				if bias != nil {
					s += bias[j]
				}
				y[i*n+j] = s
			}
		}
	})
	return y
}

func linearFast(in []*ref.Tensor) ([]*ref.Tensor, bool) {
	x, w := in[0], in[1]
	if x.DType != mmf.F32 || len(x.Shape) < 2 {
		return nil, false
	}
	m, ok := newMatrixRows(w)
	if !ok || x.Shape[len(x.Shape)-1] != m.k {
		return nil, false
	}
	var bias []float32
	if len(in) == 3 {
		if in[2].DType != mmf.F32 || len(in[2].Shape) != 1 || in[2].Shape[0] != w.Shape[0] {
			return nil, false
		}
		bias = f32View(in[2])
	}
	rows := x.NumElems() / m.k
	y := linearF32(f32View(x), rows, m, bias)
	shape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), w.Shape[0])
	return []*ref.Tensor{f32Tensor(shape, y)}, true
}

func fastLinear(list []mmf.Attr, in []*ref.Tensor, _ []mmf.DType, _ [][]int) ([]*ref.Tensor, bool, error) {
	if len(list) != 0 || len(in) < 2 || len(in) > 3 || in[1].DType != mmf.F32 {
		return nil, false, nil
	}
	out, ok := linearFast(in)
	return out, ok, nil
}

func fastDequantMatmul(list []mmf.Attr, in []*ref.Tensor, _ []mmf.DType, _ [][]int) ([]*ref.Tensor, bool, error) {
	if len(list) != 0 || len(in) < 2 || len(in) > 3 {
		return nil, false, nil
	}
	if err := checkDequantMatmul(in); err != nil {
		return nil, false, err
	}
	out, ok := linearFast(in)
	return out, ok, nil
}

// fastDequantMatmulA8 runs dequant_matmul_a8.v1 with the activations
// quantized to 8-bit blocks where the kernels allow (spec/stdlib.md section
// 3.4), and as dequant_matmul.v1 otherwise.
func fastDequantMatmulA8(list []mmf.Attr, in []*ref.Tensor, _ []mmf.DType, _ [][]int) ([]*ref.Tensor, bool, error) {
	if len(list) != 0 {
		return nil, false, nil
	}
	return dequantMatmulA8(in, 32)
}

// fastDequantMatmulA8V2 runs dequant_matmul_a8.v2 as fastDequantMatmulA8
// does, with activation blocks of the call's block attribute.
func fastDequantMatmulA8V2(list []mmf.Attr, in []*ref.Tensor, _ []mmf.DType, _ [][]int) ([]*ref.Tensor, bool, error) {
	block, err := a8Block(list)
	if err != nil {
		return nil, false, err
	}
	return dequantMatmulA8(in, block)
}

func dequantMatmulA8(in []*ref.Tensor, block int) ([]*ref.Tensor, bool, error) {
	if len(in) < 2 || len(in) > 3 {
		return nil, false, nil
	}
	if err := checkDequantMatmul(in); err != nil {
		return nil, false, err
	}
	x, w := in[0], in[1]
	if !kernels.FusedQ8Blocks(w.DType, block) {
		out, ok := linearFast(in)
		return out, ok, nil
	}
	m, ok := newMatrixRows(w)
	if !ok || x.DType != mmf.F32 || x.Shape[len(x.Shape)-1] != m.k {
		return nil, false, nil
	}
	var bias []float32
	if len(in) == 3 {
		if in[2].DType != mmf.F32 {
			return nil, false, nil
		}
		bias = f32View(in[2])
	}
	rows := x.NumElems() / m.k
	y := linearA8(f32View(x), rows, m, bias, block)
	shape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), w.Shape[0])
	return []*ref.Tensor{f32Tensor(shape, y)}, true, nil
}

// linearA8 is linearF32 with each row of x quantized to 8-bit blocks. Rows
// of x or w that the 8-bit kernels refuse (values that are not finite or
// very large) use the float32 kernels. Where the weights' dtype has a
// four-row kernel, rows of x go through it four at a time; its results are
// those of the one-row kernel, bit for bit, so the output does not depend
// on how many rows are multiplied together. The activation blocks are block
// long, and kernels.FusedQ8Blocks must accept the weights' dtype for it.
func linearA8(x []float32, rows int, w *matrixRows, bias []float32, block int) []float32 {
	n, k := w.t.Shape[0], w.k
	var acts kernels.Q8Batch
	acts.Reset(k, rows)
	quantized := make([]bool, rows)
	parallel(rows, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			quantized[i] = acts.Row(i).QuantizeBlocks(x[i*k:(i+1)*k], block)
		}
	})
	y := make([]float32, rows*n)
	// tiled[j/16] reports whether the AMX kernels computed weight rows j to
	// j+15 for the quantized rows of x.
	var tiled []bool
	if block == 256 && rows >= tileMinRows && kernels.TileQ8(w.t.DType) {
		tiled = linearA8Tiles(&acts, quantized, w, bias, y)
	}
	// four[i] reports whether rows i to i+3 are all quantized.
	x4 := kernels.FusedQ8x4(w.t.DType) || block == 256
	four := make([]bool, rows)
	for i := 0; x4 && i+4 <= rows; i++ {
		four[i] = quantized[i] && quantized[i+1] && quantized[i+2] && quantized[i+3]
	}
	parallel(n, func(lo, hi int) {
		rd := kernels.NewRowDot(w.t.DType, k)
		var out [4]float32
		for j := lo; j < hi; j++ {
			row := w.rowBytesOf(j)
			rd.LoadQ8(row)
			q8, loaded := true, false
			done := j/kernels.TileRows < len(tiled) && tiled[j/kernels.TileRows]
			for i := 0; i < rows; {
				if done && quantized[i] {
					i++
					continue
				}
				if q8 && four[i] {
					if q8 = rd.DotQ8x4(&acts, i, &out); q8 {
						for r, s := range out {
							if bias != nil {
								s += bias[j]
							}
							y[(i+r)*n+j] = s
						}
						i += 4
						continue
					}
				}
				var s float32
				if q8 && quantized[i] {
					s, q8 = rd.DotQ8(acts.Row(i))
				}
				if !q8 || !quantized[i] {
					if !loaded {
						rd.Load(row, rows)
						loaded = true
					}
					s = rd.Dot(x[i*k : (i+1)*k])
				}
				if bias != nil {
					s += bias[j]
				}
				y[i*n+j] = s
				i++
			}
		}
	})
	return y
}

// tileMinRows is the number of rows of activations from which linearA8
// multiplies with the AMX kernels: with fewer, the four-row kernels are as
// fast, since a tile product costs the same for 1 row as for 16.
const tileMinRows = 8

// linearA8Tiles computes y for the quantized rows of x with the AMX
// kernels, 16 weight rows at a time, and reports for each run of 16 weight
// rows whether it did. Its results are those of the other 8-bit kernels,
// bit for bit; a run with a block scale that is not finite, and the weight
// rows after the last whole run, are left to the caller.
func linearA8Tiles(acts *kernels.Q8Batch, quantized []bool, w *matrixRows, bias, y []float32) []bool {
	n, k := w.t.Shape[0], w.k
	var ta kernels.TileActs
	ta.Set(acts, quantized)
	tiled := make([]bool, n/kernels.TileRows)
	parallel(len(tiled), func(lo, hi int) {
		td := kernels.NewTileDot(w.t.DType, k)
		for t := lo; t < hi; t++ {
			j0 := t * kernels.TileRows
			wt := w.t.Data[j0*w.rowBytes : (j0+kernels.TileRows)*w.rowBytes]
			var b []float32
			if bias != nil {
				b = bias[j0 : j0+kernels.TileRows]
			}
			tiled[t] = td.Dot(wt, &ta, y[j0:], n, b)
		}
	})
	return tiled
}

func fastRMSNorm(list []mmf.Attr, in []*ref.Tensor, _ []mmf.DType, _ [][]int) ([]*ref.Tensor, bool, error) {
	a, err := readAttrs(list, "axis", "epsilon")
	if err != nil {
		return nil, false, err
	}
	if len(in) < 1 || len(in) > 2 || in[0].DType != mmf.F32 || len(in[0].Shape) == 0 {
		return nil, false, nil
	}
	x := in[0]
	r := len(x.Shape)
	ax, err := a.int("axis", -1)
	if err != nil {
		return nil, false, err
	}
	if ax != -1 && ax != int64(r-1) {
		return nil, false, nil
	}
	eps, err := a.float("epsilon", 1e-6)
	if err != nil {
		return nil, false, err
	}
	n := x.Shape[r-1]
	var scale []float32
	if len(in) == 2 {
		if in[1].DType != mmf.F32 || len(in[1].Shape) != 1 || in[1].Shape[0] != n {
			return nil, false, nil
		}
		scale = f32View(in[1])
	}
	v := f32Values(x)
	if n == 0 {
		return []*ref.Tensor{x}, true, nil
	}
	for i := 0; i < len(v); i += n {
		row := v[i : i+n]
		var ss float64
		for _, e := range row {
			ss += float64(e) * float64(e)
		}
		inv := float32(1 / math.Sqrt(float64(float32(ss/float64(n)))+float64(float32(eps))))
		for j := range row {
			row[j] *= inv
			if scale != nil {
				row[j] *= scale[j]
			}
		}
	}
	return []*ref.Tensor{f32Tensor(x.Shape, v)}, true, nil
}

func fastSwiglu(list []mmf.Attr, in []*ref.Tensor, _ []mmf.DType, _ [][]int) ([]*ref.Tensor, bool, error) {
	if len(list) != 0 || len(in) != 2 || in[0].DType != mmf.F32 || in[1].DType != mmf.F32 ||
		!sameInts(in[0].Shape, in[1].Shape) {
		return nil, false, nil
	}
	a, b := f32Values(in[0]), f32View(in[1])
	parallel(len(a), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			x := a[i]
			a[i] = float32(float64(x) / (1 + math.Exp(-float64(x))) * float64(b[i]))
		}
	})
	return []*ref.Tensor{f32Tensor(in[0].Shape, a)}, true, nil
}

func position(t *ref.Tensor, i int) float64 {
	if t.DType == mmf.I32 {
		return float64(int32(binary.LittleEndian.Uint32(t.Data[4*i:])))
	}
	return float64(int64(binary.LittleEndian.Uint64(t.Data[8*i:])))
}

func fastRope(list []mmf.Attr, in []*ref.Tensor, _ []mmf.DType, _ [][]int) ([]*ref.Tensor, bool, error) {
	if err := checkRope(in); err != nil {
		return nil, false, err
	}
	x, pos := in[0], in[1]
	if x.DType != mmf.F32 {
		return nil, false, nil
	}
	B, H, T, D := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	p, err := readRope(list, D)
	if err != nil {
		return nil, false, err
	}
	h := p.rotary / 2
	v := f32Values(x)
	cs := make([]float32, 2*h)
	for b := 0; b < B; b++ {
		pb := 0
		if pos.Shape[0] != 1 {
			pb = b
		}
		for t := 0; t < T; t++ {
			pp := position(pos, pb*T+t)
			for i := 0; i < h; i++ {
				ang := pp * p.invFreq[i]
				cs[2*i] = float32(math.Cos(ang) * p.ms)
				cs[2*i+1] = float32(math.Sin(ang) * p.ms)
			}
			for hd := 0; hd < H; hd++ {
				row := v[((b*H+hd)*T+t)*D:][:D]
				for i := 0; i < h; i++ {
					i1, i2 := i, i+h
					if p.interleaved {
						i1, i2 = 2*i, 2*i+1
					}
					x1, x2 := row[i1], row[i2]
					c, s := cs[2*i], cs[2*i+1]
					row[i1] = x1*c - x2*s
					row[i2] = x2*c + x1*s
				}
			}
		}
	}
	return []*ref.Tensor{f32Tensor(x.Shape, v)}, true, nil
}

func fastAttention(list []mmf.Attr, in []*ref.Tensor, _ []mmf.DType, _ [][]int) ([]*ref.Tensor, bool, error) {
	p, err := readAttention(list, in)
	if err != nil {
		return nil, false, err
	}
	q, k, v := in[0], in[1], in[2]
	if q.DType != mmf.F32 || p.softcap > 0 {
		return nil, false, nil
	}
	var mask *ref.Tensor
	var mst []int
	if len(in) == 4 {
		mask = in[3]
		if mask.DType != mmf.Bool && mask.DType != mmf.F32 {
			return nil, false, nil
		}
		// Broadcast strides: 0 on size-1 axes.
		mst = make([]int, 4)
		acc := 1
		for i := 3; i >= 0; i-- {
			if mask.Shape[i] != 1 {
				mst[i] = acc
			}
			acc *= mask.Shape[i]
		}
	}
	B, Hq, Tq, D := q.Shape[0], q.Shape[1], q.Shape[2], q.Shape[3]
	Hkv, Tk, Dv := k.Shape[1], k.Shape[2], v.Shape[3]
	G := Hq / Hkv
	qv, kv, vv := f32View(q), f32View(k), f32View(v)
	out := make([]float32, B*Hq*Tq*Dv)
	scale := float32(p.scale)
	// ends holds, for each query row, the end of its visible keys: keys
	// past the last one a boolean mask allows, or past the causal limit,
	// get no weight, so the loops below stop there. With a cache sized for
	// the whole context, most keys are past the end.
	var ends []int
	if mask != nil && mask.DType == mmf.Bool && mst[3] == 1 {
		ends = make([]int, B*Hq*Tq)
		for b := 0; b < B; b++ {
			for h := 0; h < Hq; h++ {
				for i := 0; i < Tq; i++ {
					at := (b*Hq+h)*Tq + i
					if h > 0 && mst[1] == 0 {
						ends[at] = ends[at-h*Tq]
						continue
					}
					row := mask.Data[b*mst[0]+h*mst[1]+i*mst[2]:][:Tk]
					e := len(row)
					for e > 0 && row[e-1] == 0 {
						e--
					}
					ends[at] = e
				}
			}
		}
	}
	parallel(B*Hq, func(lo, hi int) {
		sc := make([]float32, Tk)
		acc := make([]float32, Dv)
		for bh := lo; bh < hi; bh++ {
			b, h := bh/Hq, bh%Hq
			kvh := b*Hkv + h/G
			kb := kv[kvh*Tk*D:]
			vb := vv[kvh*Tk*Dv:]
			for i := 0; i < Tq; i++ {
				qi := qv[(bh*Tq+i)*D:][:D]
				end := Tk
				if ends != nil {
					end = ends[bh*Tq+i]
				}
				if p.causal {
					end = max(0, min(end, i+(Tk-Tq)+1))
				}
				sc := sc[:end]
				mx := float32(math.Inf(-1))
				for j := 0; j < end; j++ {
					s := float32(math.Inf(-1))
					allowed := !p.causal || j <= i+(Tk-Tq)
					var add float32
					if mask != nil {
						mi := b*mst[0] + h*mst[1] + i*mst[2] + j*mst[3]
						if mask.DType == mmf.Bool {
							allowed = allowed && mask.Data[mi] != 0
						} else {
							add = math.Float32frombits(binary.LittleEndian.Uint32(mask.Data[4*mi:]))
						}
					}
					if allowed {
						s = kernels.DotF32(qi, kb[j*D:(j+1)*D])*scale + add
					}
					sc[j] = s
					if s > mx || s != s {
						mx = s
					}
				}
				var sum float32
				for j := range sc {
					if math.IsInf(float64(sc[j]), -1) && !math.IsInf(float64(mx), -1) {
						sc[j] = 0 // exp(-inf) without the call; most of a cache is masked
						continue
					}
					e := float32(math.Exp(float64(sc[j] - mx)))
					sc[j] = e
					sum += e
				}
				clear(acc)
				for j, e := range sc {
					if e == 0 {
						continue
					}
					kernels.AxpyF32(e, vb[j*Dv:][:Dv], acc)
				}
				o := out[(bh*Tq+i)*Dv:][:Dv]
				for d := range o {
					o[d] = acc[d] / sum
				}
			}
		}
	})
	return []*ref.Tensor{f32Tensor([]int{B, Hq, Tq, Dv}, out)}, true, nil
}

func fastEmbedding(list []mmf.Attr, in []*ref.Tensor, dts []mmf.DType, _ [][]int) ([]*ref.Tensor, bool, error) {
	if len(list) != 0 || len(in) != 2 || len(dts) != 1 || dts[0] != mmf.F32 {
		return nil, false, nil
	}
	table, ids := in[0], in[1]
	if ids.DType != mmf.I32 && ids.DType != mmf.I64 {
		return nil, false, nil
	}
	m, ok := newMatrixRows(table)
	if !ok {
		return nil, false, nil
	}
	n := ids.NumElems()
	y := make([]float32, n*m.k)
	for i := 0; i < n; i++ {
		id := int64(position(ids, i))
		// Out-of-range rows are unspecified (spec/primitives.md 5.30); they
		// are left zero, as the reference gather does.
		if id >= 0 && id < int64(table.Shape[0]) {
			m.row(int(id), y[i*m.k:(i+1)*m.k])
		}
	}
	shape := append(append([]int{}, ids.Shape...), m.k)
	return []*ref.Tensor{f32Tensor(shape, y)}, true, nil
}
