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

	once sync.Once
	lay  *convLayout // see layout
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
	parallelWork(p.nb*p.f, len(y)*cg*p.kh*p.kw*2, func(lo, hi int) {
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

// convGemm computes a float32 convolution as matrix products on the
// AVX-512 kernel, per image and group: the weights of a group, Fg rows of
// K = Cg*KH*KW, times the K by Ho*Wo matrix of the input's receptive
// fields, plus the bias. Each output element is one chain of fused
// multiply-adds over its receptive field, then the bias: within the
// composite's tolerance, Q u sum |x w| + u |B|. The weights of a graph's
// own tensors are laid out for the kernel once and kept.
//
// With epi not nil, its ops run on each tile of the output as it is made.
func (p *Program) convGemm(a *arena, list []mim.Attr, in []*ref.Tensor, dts []mim.DType, epi *epilogue) ([]*ref.Tensor, bool, error) {
	if !kernels.Sgemm() || len(in) < 2 || in[0].DType != mim.F32 {
		return nil, false, nil
	}
	c, err := readConv(list, in, dts)
	if err != nil {
		return nil, false, err
	}
	y := a.f32(c.nb*c.f*c.ho*c.wo, false)
	p.convInto(c, in, y, epi)
	return []*ref.Tensor{f32Tensor([]int{c.nb, c.f, c.ho, c.wo}, y)}, true, nil
}

// convInto is convGemm writing the convolution of c, with operands in, into
// y, which holds nb*f*ho*wo floats.
//
// The receptive fields are not gathered: the products run over windows
// of the padded input (kernels.SgemmWindow). The padded image is laid out
// in SH*SW phases, phase (py, px) holding the rows and columns that are
// py and px modulo the strides, each of HP2 = ceil(HP/SH) rows of WP2 =
// ceil(WP/SW); then output position (oy, ox) under kernel element (i, j)
// of channel ci reads row oy + (i*DH)/SH, column ox + (j*DW)/SW of phase
// ((i*DH) mod SH, (j*DW) mod SW), at a fixed offset from q = oy*WP2 + ox.
// So the products run over the Ho*WP2 positions q in a row, and the
// kernel drops those with ox >= Wo as it stores, packing each row of y.
// (With unit strides there is one phase, the padded image itself.)
func (p *Program) convInto(c *convParams, in []*ref.Tensor, y []float32, epi *epilogue) {
	r := p.newConvRun(c, in[1])
	var bias []float32
	if len(in) == 3 {
		bias = f32View(in[2])
	}
	r.run(f32View(in[0]), bias, y, epi)
}

// newConvRun readies a convRun of c with the weights w, to run over and
// over.
func (p *Program) newConvRun(c *convParams, w *ref.Tensor) *convRun {
	l := c.layout()
	r := &convRun{c: c, l: l}
	r.pa, r.gw = p.packConv(w, f32View(w), c.g, c.f/c.g, l.k, l.gidx != nil)
	r.images = func(lo, hi int) {
		xp := getScratch(l.xpLen)
		for ng := lo; ng < hi; ng++ {
			r.image(ng/c.g, ng%c.g, xp.b, nil)
		}
		putScratch(xp)
	}
	r.split = func(f func(lo, hi int)) { parallelWork(l.panels*l.strips, l.work, f) }
	return r
}

// run computes the convolution of x (with bias, which may be nil) into
// y, then epi.
func (r *convRun) run(x, bias, y []float32, epi *epilogue) {
	if len(y) == 0 {
		return
	}
	c, l := r.c, r.l
	r.x, r.bias, r.y, r.epi = x, bias, y, epi
	if epi != nil {
		epi.prepare(c.f)
	}
	defer func() { r.x, r.bias, r.y, r.epi = nil, nil, nil, nil }()
	if c.nb*c.g > 1 && l.work < minParallelWork {
		// Images too small to split go whole to one worker each.
		parallelWork(c.nb*c.g, c.nb*c.g*l.work, r.images)
		return
	}
	xp := getScratch(l.xpLen)
	for n := range c.nb {
		for g := range c.g {
			r.image(n, g, xp.b, r.split)
		}
	}
	putScratch(xp)
}

// A convLayout is how convInto lays out the padded images and products
// of a convolution of some shape.
type convLayout struct {
	k, plane       int
	hp2, wp2       int // the rows and columns of a phase
	phase          int // the elements of a phase
	qn             int // the positions of the products, those past wo on each row included
	panels, strips int
	offs           []int    // each tap's window: its offset from the output position
	xpLen          int      // the padded image, and the last window of the last position
	masks          []uint32 // each strip's output positions
	dsts           []int    // where each strip's first output position goes in its row
	work           int      // the multiply-adds of one image and group, and its padding
	ix0, x20       []int    // the first input column of each phase, and its column in the phase
	// With gidx set, the convolution is small enough that gathering each
	// tap's inputs costs less than padding and the GEMM: for each chunk
	// of 16 output positions and tap, the input index of each position
	// (within the image and group) and the lanes of the positions inside
	// the input; gtail is the lanes of the last chunk.
	gidx    []int32
	gmasks  []uint16
	gchunks int
	gtail   int
}

// gatherTable is the most entries of a layout's gidx.
const gatherTable = 64 << 10

// layout returns c's layout, made once.
func (c *convParams) layout() *convLayout {
	c.once.Do(func() {
		cg, fg := c.c/c.g, c.f/c.g
		l := &convLayout{k: cg * c.kh * c.kw, plane: c.ho * c.wo}
		hp, wp := c.h+c.top+c.bottom, c.w+c.left+c.right
		l.hp2, l.wp2 = (hp+c.sh-1)/c.sh, (wp+c.sw-1)/c.sw
		l.phase = l.hp2 * l.wp2
		l.qn = c.ho * l.wp2
		l.panels = (fg + kernels.SgemmRows - 1) / kernels.SgemmRows
		l.strips = (l.qn + kernels.SgemmCols - 1) / kernels.SgemmCols
		l.offs = make([]int, l.k)
		maxOff := 0
		t := 0
		for ci := range cg {
			for i, py, y2 := 0, 0, 0; i < c.kh; i++ {
				for j, px, x2 := 0, 0, 0; j < c.kw; j++ {
					o := ((ci*c.sh+py)*c.sw+px)*l.phase + y2*l.wp2 + x2
					l.offs[t] = o
					maxOff = max(maxOff, o)
					t++
					// Kernel element j+1 is dw columns on.
					px, x2 = (px+c.dw)%c.sw, x2+(px+c.dw)/c.sw
				}
				py, y2 = (py+c.dh)%c.sh, y2+(py+c.dh)/c.sh
			}
		}
		l.xpLen = max(cg*c.sh*c.sw*l.phase, maxOff+l.qn)
		l.masks, l.dsts = make([]uint32, l.strips), make([]int, l.strips)
		for q, ox, v := 0, 0, 0; q < l.qn; q++ {
			if ox < c.wo {
				s := q / kernels.SgemmCols
				if l.masks[s] == 0 {
					l.dsts[s] = v
				}
				l.masks[s] |= 1 << (q % kernels.SgemmCols)
				v++
			}
			if ox++; ox == l.wp2 {
				ox = 0
			}
		}
		l.work = fg*l.qn*l.k/16 + cg*hp*wp
		// Gather when it costs less: a gather per tap and chunk of 16
		// positions, plus a multiply-add per filter, against the GEMM's
		// tiles, the padding's copies, and their calls, in cycles.
		chunks := (l.plane + 15) / 16
		if gather, gemm := chunks*l.k*(12+fg), l.k*l.strips*l.panels*8+100*cg*c.sh*c.sw+200*l.strips*l.panels+l.xpLen/16; chunks*l.k*16 <= gatherTable && gather < gemm {
			l.gchunks = chunks
			l.gidx, l.gmasks = make([]int32, chunks*l.k*16), make([]uint16, chunks*l.k)
			for q := range l.plane {
				oy, ox := q/c.wo, q%c.wo
				ch, j := q/16, q%16
				t := 0
				for ci := range cg {
					for i := range c.kh {
						for jj := range c.kw {
							iy, ix := oy*c.sh-c.top+i*c.dh, ox*c.sw-c.left+jj*c.dw
							if iy >= 0 && iy < c.h && ix >= 0 && ix < c.w {
								l.gidx[(ch*l.k+t)*16+j] = int32((ci*c.h+iy)*c.w + ix)
								l.gmasks[ch*l.k+t] |= 1 << j
							}
							t++
						}
					}
				}
			}
			l.gtail = 1<<(l.plane-(chunks-1)*16) - 1
		}
		// Where the input's columns of each phase start: column ix0 of
		// the input is column x20 of phase px, then every sw-th column.
		l.ix0, l.x20 = make([]int, c.sw), make([]int, c.sw)
		for ix := range min(c.sw, c.w) {
			px := (ix + c.left) % c.sw
			l.ix0[px], l.x20[px] = ix, (ix+c.left)/c.sw
		}
		c.lay = l
	})
	return c.lay
}

// A convRun computes convolutions of one shape with one set of weights:
// see newConvRun and run.
type convRun struct {
	c  *convParams
	l  *convLayout
	pa [][]float32 // each group's weights packed for the GEMM, or
	gw [][]float32 // laid out by tap for gathering

	x, bias, y []float32 // the operands of the run going on
	epi        *epilogue

	images func(lo, hi int)         // computes images lo to hi-1, one worker's share
	split  func(f func(lo, hi int)) // runs f over the tiles of one image on the workers
}

// image computes image n, group g: padded into xp, its products stored
// in y, then the epilogue. With split not nil, the tiles of the products
// are computed by the function it is given, over the workers.
func (r *convRun) image(n, g int, xp []float32, split func(func(lo, hi int))) {
	c, l := r.c, r.l
	fg := c.f / c.g
	if l.gidx != nil {
		r.gather(n, g)
		return
	}
	r.pad(n, g, xp)
	y := r.y[(n*c.f+g*fg)*l.plane : (n*c.f+(g+1)*fg)*l.plane]
	if split != nil {
		split(func(lo, hi int) { r.tiles(g, xp, y, lo, hi) })
	} else {
		r.tiles(g, xp, y, 0, l.panels*l.strips)
	}
	if r.epi != nil {
		r.epi.applyImage(y, n, g*fg, fg, c.f, l.plane)
	}
}

// gather computes image n, group g by gathering, then the epilogue.
func (r *convRun) gather(n, g int) {
	c, l := r.c, r.l
	cg, fg := c.c/c.g, c.f/c.g
	x := r.x[(n*c.c+g*cg)*c.h*c.w : (n*c.c+(g+1)*cg)*c.h*c.w]
	y := r.y[(n*c.f+g*fg)*l.plane : (n*c.f+(g+1)*fg)*l.plane]
	for f0 := 0; f0 < fg; f0 += kernels.ConvGatherRows {
		rows := min(kernels.ConvGatherRows, fg-f0)
		var bias []float32
		if r.bias != nil {
			bias = r.bias[g*fg+f0:]
		}
		kernels.ConvGatherF32(x, l.gidx, l.gmasks, l.gchunks, l.k, r.gw[g][f0:], fg, rows, bias, y[f0*l.plane:], l.plane, l.gtail)
	}
	if r.epi != nil {
		r.epi.applyImage(y, n, g*fg, fg, c.f, l.plane)
	}
}

// tiles computes tiles lo to hi-1 of group g's products from xp into y,
// the group's output planes.
func (r *convRun) tiles(g int, xp, y []float32, lo, hi int) {
	l := r.l
	fg := r.c.f / r.c.g
	for t := lo; t < hi; t++ {
		s, pn := t/l.panels, t%l.panels
		if l.masks[s] == 0 {
			continue
		}
		rows := min(kernels.SgemmRows, fg-pn*kernels.SgemmRows)
		var bp []float32
		if r.bias != nil {
			bp = r.bias[g*fg+pn*kernels.SgemmRows:]
		}
		kernels.SgemmWindow(r.pa[g], l.k, pn, rows, xp[s*kernels.SgemmCols:], l.offs, l.masks[s],
			y[pn*kernels.SgemmRows*l.plane+l.dsts[s]:], l.plane, bp)
	}
}

// pad lays image n, group g out in xp, in phases.
func (r *convRun) pad(n, g int, xp []float32) {
	c, l := r.c, r.l
	cg := c.c / c.g
	clear(xp)
	var one [1]int
	for ci := range cg {
		src := r.x[(n*c.c+g*cg+ci)*c.h*c.w:]
		for py := range c.sh {
			// Input row iy0 is the first of phase py, then every sh-th.
			iy0 := (py - c.top%c.sh + c.sh) % c.sh
			if iy0 >= c.h {
				continue
			}
			rows, y20 := (c.h-iy0+c.sh-1)/c.sh, (iy0+c.top)/c.sh
			for px := range min(c.sw, c.w) {
				ix0, x20 := l.ix0[px], l.x20[px]
				cols := (c.w - ix0 + c.sw - 1) / c.sw
				dst := xp[((ci*c.sh+py)*c.sw+px)*l.phase+y20*l.wp2+x20:]
				from := src[iy0*c.w+ix0:]
				if kernels.PoolF32(kernels.BinAdd, dst, from, one[:], cols, rows, c.sw, c.sh*c.w, l.wp2, 0) {
					continue
				}
				for r := range rows {
					row, drow := from[r*c.sh*c.w:], dst[r*l.wp2:]
					for j := range cols {
						drow[j] = row[j*c.sw]
					}
				}
			}
		}
	}
}

// scratch keeps float32 buffers between calls, for the padded images and
// products of convInto.
var scratch sync.Pool

type scratchBuf struct{ b []float32 }

func getScratch(n int) *scratchBuf {
	if v := scratch.Get(); v != nil {
		if s := v.(*scratchBuf); cap(s.b) >= n {
			s.b = s.b[:n]
			return s
		}
	}
	return &scratchBuf{make([]float32, n)}
}

func putScratch(s *scratchBuf) { scratch.Put(s) }

// packConv returns the weights of each group packed for kernels.SgemmTile
// or, with gather set, laid out tap by tap for kernels.ConvGatherF32,
// kept with the program when wt is one of its own tensors.
func (p *Program) packConv(wt *ref.Tensor, w []float32, g, fg, k int, gather bool) (pa, gw [][]float32) {
	fixed := p.isFixed[wt]
	if fixed {
		if v, ok := p.packed.Load(wt); ok {
			pk := v.(*packedConv)
			return pk.pa, pk.gw
		}
	}
	if gather {
		gw = make([][]float32, g)
		for i := range gw {
			gw[i] = make([]float32, k*fg)
			for f := range fg {
				for t := range k {
					gw[i][t*fg+f] = w[(i*fg+f)*k+t]
				}
			}
		}
	} else {
		pa = make([][]float32, g)
		for i := range pa {
			pa[i] = kernels.PackSgemmA(w[i*fg*k:(i+1)*fg*k], fg, k, k)
		}
	}
	if fixed {
		p.packed.Store(wt, &packedConv{pa, gw})
	}
	return pa, gw
}

// packedConv is a convolution's weights as packConv keeps them.
type packedConv struct{ pa, gw [][]float32 }
