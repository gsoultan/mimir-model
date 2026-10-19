// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mim, created and owned by Gembit Soultan Shirazi.
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
	"fmt"
	"sync"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/internal/kernels"
	"github.com/gsoultan/mimir-model/ref"
)

// convParams are the attributes of conv2d.v1 and the shapes they imply.
type convParams struct {
	nb, c, h, w, f, kh, kw, g int
	sh, sw, dh, dw            int
	top, bottom, left, right  int
	ho, wo                    int
}

func readConv(list []mim.Attr, in []*ref.Tensor, dts []mim.DType) (*convParams, error) {
	a, err := readAttrs(list, "strides", "dilations", "pads", "groups")
	if err != nil {
		return nil, err
	}
	if len(in) < 2 || len(in) > 3 || len(dts) != 1 {
		return nil, fmt.Errorf("conv2d.v1 takes X, Wt and an optional bias, and has one output")
	}
	x, w := in[0], in[1]
	if !isFloat(x.DType) || w.DType != x.DType || len(x.Shape) != 4 || len(w.Shape) != 4 {
		return nil, fmt.Errorf("conv2d.v1: X %s%v and Wt %s%v", x.DType, x.Shape, w.DType, w.Shape)
	}
	ints := func(key string, n int, def ...int) ([]int, error) {
		v := a[key]
		if v == nil {
			return def, nil
		}
		if v.Type != mim.AttrInts || len(v.Ints) != n {
			return nil, fmt.Errorf("conv2d.v1: attribute %q must be %d ints", key, n)
		}
		out := make([]int, n)
		for i, x := range v.Ints {
			if x < 0 || x > 1<<20 {
				return nil, fmt.Errorf("conv2d.v1: attribute %q is %v", key, v.Ints)
			}
			out[i] = int(x)
		}
		return out, nil
	}
	s, err := ints("strides", 2, 1, 1)
	if err != nil {
		return nil, err
	}
	d, err := ints("dilations", 2, 1, 1)
	if err != nil {
		return nil, err
	}
	pads, err := ints("pads", 4, 0, 0, 0, 0)
	if err != nil {
		return nil, err
	}
	g, err := a.int("groups", 1)
	if err != nil {
		return nil, err
	}
	p := &convParams{
		nb: x.Shape[0], c: x.Shape[1], h: x.Shape[2], w: x.Shape[3],
		f: w.Shape[0], kh: w.Shape[2], kw: w.Shape[3], g: int(g),
		sh: s[0], sw: s[1], dh: d[0], dw: d[1],
		top: pads[0], bottom: pads[1], left: pads[2], right: pads[3],
	}
	if g <= 0 || g > 1<<20 || p.c%p.g != 0 || p.f%p.g != 0 || w.Shape[1] != p.c/p.g {
		return nil, fmt.Errorf("conv2d.v1: %d groups for X %v and Wt %v", g, x.Shape, w.Shape)
	}
	if min(p.sh, p.sw, p.dh, p.dw, p.kh, p.kw) <= 0 {
		return nil, fmt.Errorf("conv2d.v1: strides %v, dilations %v, kernel %dx%d", s, d, p.kh, p.kw)
	}
	eh, ew := p.dh*(p.kh-1)+1, p.dw*(p.kw-1)+1
	if p.h+p.top+p.bottom < eh || p.w+p.left+p.right < ew {
		return nil, fmt.Errorf("conv2d.v1: the kernel is larger than the padded input")
	}
	p.ho = (p.h+p.top+p.bottom-eh)/p.sh + 1
	p.wo = (p.w+p.left+p.right-ew)/p.sw + 1
	if len(in) == 3 {
		if b := in[2]; b.DType != x.DType || len(b.Shape) != 1 || b.Shape[0] != p.f {
			return nil, fmt.Errorf("conv2d.v1: bias %s%v for %d filters", b.DType, b.Shape, p.f)
		}
	}
	return p, nil
}

func (b *body) shrink(x *ref.Tensor, ax, begin, end int) *ref.Tensor {
	if x == nil {
		return nil
	}
	bs, es := make([]int64, len(x.Shape)), make([]int64, len(x.Shape))
	s := append([]int{}, x.Shape...)
	for i := range es {
		es[i] = -1
	}
	bs[ax], es[ax], s[ax] = int64(begin), int64(end), end-begin
	return b.eval(mim.OpShrink, []mim.Attr{
		{Key: "begin", Type: mim.AttrInts, Ints: bs},
		{Key: "end", Type: mim.AttrInts, Ints: es},
	}, x.DType, s, x)
}

// takeStrided is the body macro of spec/stdlib.md section 2.2.
func (b *body) takeStrided(x *ref.Tensor, ax, start, n, step int) *ref.Tensor {
	y := b.shrink(x, ax, start, start+n*step)
	if y == nil {
		return nil
	}
	s := y.Shape
	split := append(append(append([]int{}, s[:ax]...), n, step), s[ax+1:]...)
	y = b.shrink(b.reshape(y, split), ax+1, 0, 1)
	out := append(append(append([]int{}, s[:ax]...), n), s[ax+1:]...)
	return b.reshape(y, out)
}

// conv2dV1 evaluates the canonical body of spec/stdlib.md section 3.5.
func conv2dV1(list []mim.Attr, in []*ref.Tensor, dts []mim.DType, _ [][]int) ([]*ref.Tensor, error) {
	p, err := readConv(list, in, dts)
	if err != nil {
		return nil, err
	}
	dt := in[0].DType
	cd := computeDType(dt)
	b := &body{}
	x, w := b.cast(in[0], cd), b.cast(in[1], cd)
	cg, fg := p.c/p.g, p.f/p.g
	hp, wp := p.h+p.top+p.bottom+p.sh-1, p.w+p.left+p.right+p.sw-1
	xp := b.eval(mim.OpPad, []mim.Attr{{Key: "pads", Type: mim.AttrInts,
		Ints: ints([]int{0, 0, 0, 0, p.top, p.bottom + p.sh - 1, p.left, p.right + p.sw - 1})}},
		cd, []int{p.nb, p.c, hp, wp}, x)
	xg := b.reshape(xp, []int{p.nb, p.g, cg, 1, hp, wp})
	wg := b.reshape(w, []int{p.g, fg, cg, p.kh, p.kw})
	wg = b.eval(mim.OpPermute, []mim.Attr{{Key: "perm", Type: mim.AttrInts, Ints: []int64{0, 2, 1, 3, 4}}},
		cd, []int{p.g, cg, fg, p.kh, p.kw}, wg)
	wg = b.reshape(wg, []int{1, p.g, cg, fg, p.kh, p.kw})
	var acc *ref.Tensor
	for i := 0; i < p.kh; i++ {
		for j := 0; j < p.kw; j++ {
			xs := b.takeStrided(xg, 4, i*p.dh, p.ho, p.sh)
			xs = b.takeStrided(xs, 5, j*p.dw, p.wo, p.sw)
			ws := b.shrink(b.shrink(wg, 4, i, i+1), 5, j, j+1)
			t := b.reduce("add", b.bin(mim.OpMul, xs, ws), []int{2})
			if acc == nil {
				acc = t
			} else {
				acc = b.bin(mim.OpAdd, acc, t)
			}
			if b.err != nil {
				return nil, b.err
			}
		}
	}
	y := b.reshape(acc, []int{p.nb, p.f, p.ho, p.wo})
	if len(in) == 3 {
		y = b.bin(mim.OpAdd, y, b.reshape(b.cast(in[2], cd), []int{1, p.f, 1, 1}))
	}
	return []*ref.Tensor{b.cast(y, dt)}, b.err
}

// fastConv2d computes a float32 convolution directly, summing each output
// element's receptive field in float64 and rounding once, which is well
// within the composite's tolerance.
func fastConv2d(list []mim.Attr, in []*ref.Tensor, dts []mim.DType, _ [][]int) ([]*ref.Tensor, bool, error) {
	if len(in) < 2 || in[0].DType != mim.F32 {
		return nil, false, nil
	}
	p, err := readConv(list, in, dts)
	if err != nil {
		return nil, false, err
	}
	x, w := f32View(in[0]), f32View(in[1])
	var bias []float32
	if len(in) == 3 {
		bias = f32View(in[2])
	}
	cg, fg := p.c/p.g, p.f/p.g
	plane := p.ho * p.wo
	y := make([]float32, p.nb*p.f*plane)
	parallel(p.nb*p.f, func(lo, hi int) {
		for nf := lo; nf < hi; nf++ {
			n, f := nf/p.f, nf%p.f
			c0 := f / fg * cg
			out := y[nf*plane : (nf+1)*plane]
			for oy := 0; oy < p.ho; oy++ {
				for ox := 0; ox < p.wo; ox++ {
					var s float64
					for c := 0; c < cg; c++ {
						xc := x[(n*p.c+c0+c)*p.h*p.w:]
						wc := w[((f*cg+c)*p.kh)*p.kw:]
						for i := 0; i < p.kh; i++ {
							iy := oy*p.sh + i*p.dh - p.top
							if iy < 0 || iy >= p.h {
								continue
							}
							for j := 0; j < p.kw; j++ {
								ix := ox*p.sw + j*p.dw - p.left
								if ix < 0 || ix >= p.w {
									continue
								}
								s += float64(xc[iy*p.w+ix]) * float64(wc[i*p.kw+j])
							}
						}
					}
					r := float32(s)
					if bias != nil {
						r += bias[f]
					}
					out[oy*p.wo+ox] = r
				}
			}
		}
	})
	return []*ref.Tensor{f32Tensor([]int{p.nb, p.f, p.ho, p.wo}, y)}, true, nil
}

// convGemm computes a float32 convolution as a matrix product per image
// and group, on the AVX-512 kernel: the weights of a group, Fg rows of
// K = Cg*KH*KW, times the K by Ho*Wo matrix of the input's receptive
// fields (the input itself for a 1x1 convolution with no stride or
// padding), plus the bias. Each output element is one chain of fused
// multiply-adds over its receptive field, then the bias: within the
// composite's tolerance, Q u sum |x w| + u |B|. The weights of a graph's
// own tensors are laid out for the kernel once and kept.
func (p *Program) convGemm(list []mim.Attr, in []*ref.Tensor, dts []mim.DType) ([]*ref.Tensor, bool, error) {
	if !kernels.Sgemm() || len(in) < 2 || in[0].DType != mim.F32 {
		return nil, false, nil
	}
	c, err := readConv(list, in, dts)
	if err != nil {
		return nil, false, err
	}
	x, w := f32View(in[0]), f32View(in[1])
	var bias []float32
	if len(in) == 3 {
		bias = f32View(in[2])
	}
	cg, fg := c.c/c.g, c.f/c.g
	k := cg * c.kh * c.kw
	plane, hw := c.ho*c.wo, c.h*c.w
	y := make([]float32, c.nb*c.f*plane)
	if len(y) == 0 {
		return []*ref.Tensor{f32Tensor([]int{c.nb, c.f, c.ho, c.wo}, y)}, true, nil
	}
	pa := p.packConv(in[1], w, c.g, fg, k)
	panels := (fg + kernels.SgemmRows - 1) / kernels.SgemmRows
	strips := (plane + kernels.SgemmCols - 1) / kernels.SgemmCols
	// The receptive fields, packed by runs of SgemmCols output positions:
	// strip s holds k rows of SgemmCols floats, so that the kernel reads
	// each row of a strip from the next 128 bytes.
	col := colBufs.get(strips * k * kernels.SgemmCols)
	defer colBufs.put(col)
	for n := range c.nb {
		for g := range c.g {
			im2col(c, x[(n*c.c+g*cg)*hw:(n*c.c+(g+1)*cg)*hw], col)
			yg := y[(n*c.f+g*fg)*plane : (n*c.f+(g+1)*fg)*plane]
			var bg []float32
			if bias != nil {
				bg = bias[g*fg : (g+1)*fg]
			}
			parallel(panels*strips, func(lo, hi int) {
				for t := lo; t < hi; t++ {
					s, pn := t/panels, t%panels
					rows := min(kernels.SgemmRows, fg-pn*kernels.SgemmRows)
					j0 := s * kernels.SgemmCols
					var bp []float32
					if bg != nil {
						bp = bg[pn*kernels.SgemmRows:]
					}
					kernels.SgemmTile(pa[g], k, pn, rows, col[s*k*kernels.SgemmCols:], kernels.SgemmCols,
						min(kernels.SgemmCols, plane-j0), yg[pn*kernels.SgemmRows*plane+j0:], plane, bp)
				}
			})
		}
	}
	return []*ref.Tensor{f32Tensor([]int{c.nb, c.f, c.ho, c.wo}, y)}, true, nil
}

// packConv returns the weights of each group packed for kernels.SgemmTile,
// kept with the program when wt is one of its own tensors.
func (p *Program) packConv(wt *ref.Tensor, w []float32, g, fg, k int) [][]float32 {
	fixed := p.isFixed[wt]
	if fixed {
		if v, ok := p.packed.Load(wt); ok {
			return v.([][]float32)
		}
	}
	pa := make([][]float32, g)
	for i := range pa {
		pa[i] = kernels.PackSgemmA(w[i*fg*k:(i+1)*fg*k], fg, k, k)
	}
	if fixed {
		p.packed.Store(wt, pa)
	}
	return pa
}

// im2col writes the receptive fields of the output positions of x, one
// group's Cg channels of one image, into col by strips of SgemmCols
// output positions: row r = (ci*KH+i)*KW+j of strip s holds, for output
// positions s*SgemmCols to s*SgemmCols+SgemmCols-1, the input element
// under kernel element (i, j) of channel ci, 0 in the padding and past
// the last position.
func im2col(c *convParams, x, col []float32) {
	cg := c.c / c.g
	k := cg * c.kh * c.kw
	const sc = kernels.SgemmCols
	strip := k * sc
	plane := c.ho * c.wo
	strips := (plane + sc - 1) / sc
	parallel(k, func(lo, hi int) {
		row := make([]float32, strips*sc) // one row, before it is cut into strips
		for r := lo; r < hi; r++ {
			ci, i, j := r/(c.kh*c.kw), r/c.kw%c.kh, r%c.kw
			xc := x[ci*c.h*c.w : (ci+1)*c.h*c.w]
			// The output columns whose input column is inside the image.
			ox0, ox1 := 0, c.wo
			for ox0 < c.wo && ox0*c.sw+j*c.dw-c.left < 0 {
				ox0++
			}
			for ox1 > ox0 && (ox1-1)*c.sw+j*c.dw-c.left >= c.w {
				ox1--
			}
			for oy := range c.ho {
				out := row[oy*c.wo : (oy+1)*c.wo]
				iy := oy*c.sh + i*c.dh - c.top
				if iy < 0 || iy >= c.h || ox0 >= ox1 {
					clear(out)
					continue
				}
				clear(out[:ox0])
				clear(out[ox1:])
				src := xc[iy*c.w:]
				ix := ox0*c.sw + j*c.dw - c.left
				if c.sw == 1 {
					copy(out[ox0:ox1], src[ix:ix+ox1-ox0])
					continue
				}
				for ox := ox0; ox < ox1; ox++ {
					out[ox] = src[ix]
					ix += c.sw
				}
			}
			clear(row[plane:])
			for st := range strips {
				copy(col[st*strip+r*sc:st*strip+(r+1)*sc], row[st*sc:(st+1)*sc])
			}
		}
	})
}

// colBuf keeps one buffer between calls, for the receptive fields of
// convGemm.
type colBuf struct {
	mu  sync.Mutex
	buf []float32
}

var colBufs colBuf

// get returns a buffer of n floats, the kept one when it is free and
// large enough.
func (b *colBuf) get(n int) []float32 {
	b.mu.Lock()
	buf := b.buf
	b.buf = nil
	b.mu.Unlock()
	if cap(buf) < n {
		return make([]float32, n)
	}
	return buf[:n]
}

func (b *colBuf) put(buf []float32) {
	b.mu.Lock()
	if cap(buf) > cap(b.buf) {
		b.buf = buf
	}
	b.mu.Unlock()
}
