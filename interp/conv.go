package interp

import (
	"fmt"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/ref"
)

// convParams are the attributes of conv2d.v1 and the shapes they imply.
type convParams struct {
	nb, c, h, w, f, kh, kw, g int
	sh, sw, dh, dw            int
	top, bottom, left, right  int
	ho, wo                    int
}

func readConv(list []mmf.Attr, in []*ref.Tensor, dts []mmf.DType) (*convParams, error) {
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
		if v.Type != mmf.AttrInts || len(v.Ints) != n {
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
	return b.eval(mmf.OpShrink, []mmf.Attr{
		{Key: "begin", Type: mmf.AttrInts, Ints: bs},
		{Key: "end", Type: mmf.AttrInts, Ints: es},
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
func conv2dV1(list []mmf.Attr, in []*ref.Tensor, dts []mmf.DType, _ [][]int) ([]*ref.Tensor, error) {
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
	xp := b.eval(mmf.OpPad, []mmf.Attr{{Key: "pads", Type: mmf.AttrInts,
		Ints: ints([]int{0, 0, 0, 0, p.top, p.bottom + p.sh - 1, p.left, p.right + p.sw - 1})}},
		cd, []int{p.nb, p.c, hp, wp}, x)
	xg := b.reshape(xp, []int{p.nb, p.g, cg, 1, hp, wp})
	wg := b.reshape(w, []int{p.g, fg, cg, p.kh, p.kw})
	wg = b.eval(mmf.OpPermute, []mmf.Attr{{Key: "perm", Type: mmf.AttrInts, Ints: []int64{0, 2, 1, 3, 4}}},
		cd, []int{p.g, cg, fg, p.kh, p.kw}, wg)
	wg = b.reshape(wg, []int{1, p.g, cg, fg, p.kh, p.kw})
	var acc *ref.Tensor
	for i := 0; i < p.kh; i++ {
		for j := 0; j < p.kw; j++ {
			xs := b.takeStrided(xg, 4, i*p.dh, p.ho, p.sh)
			xs = b.takeStrided(xs, 5, j*p.dw, p.wo, p.sw)
			ws := b.shrink(b.shrink(wg, 4, i, i+1), 5, j, j+1)
			t := b.reduce("add", b.bin(mmf.OpMul, xs, ws), []int{2})
			if acc == nil {
				acc = t
			} else {
				acc = b.bin(mmf.OpAdd, acc, t)
			}
			if b.err != nil {
				return nil, b.err
			}
		}
	}
	y := b.reshape(acc, []int{p.nb, p.f, p.ho, p.wo})
	if len(in) == 3 {
		y = b.bin(mmf.OpAdd, y, b.reshape(b.cast(in[2], cd), []int{1, p.f, 1, 1}))
	}
	return []*ref.Tensor{b.cast(y, dt)}, b.err
}

// fastConv2d computes a float32 convolution directly, summing each output
// element's receptive field in float64 and rounding once, which is well
// within the composite's tolerance.
func fastConv2d(list []mmf.Attr, in []*ref.Tensor, dts []mmf.DType, _ [][]int) ([]*ref.Tensor, bool, error) {
	if len(in) < 2 || in[0].DType != mmf.F32 {
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
