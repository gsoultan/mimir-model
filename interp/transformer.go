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
	"math"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/ref"
)

// This file holds the canonical bodies of the composites a decoder-only
// transformer needs beyond the basic ones in stdlib.go: dequant_matmul.v1
// and dequant_matmul_a8.v1 and v2, swiglu.v1, cos.v1, tanh.v1, rope.v1 and attention.v1 and v2 (spec/stdlib.md
// sections 3.3, 4 and 6). Like the rest of the executor's standard library
// they evaluate the body on the primitive kernels, so their results are the
// body's results.

func init() {
	stdlib["dequant_matmul.v1"] = dequantMatmulV1
	// dequant_matmul_a8.v1 has the same body; only its tolerance is looser.
	stdlib["dequant_matmul_a8.v1"] = dequantMatmulV1
	stdlib["dequant_matmul_a8.v2"] = dequantMatmulA8V2
	stdlib["swiglu.v1"] = swigluV1
	stdlib["cos.v1"] = unaryComposite(cosBody)
	stdlib["tanh.v1"] = unaryComposite(tanhBody)
	stdlib["rope.v1"] = ropeV1
	stdlib["attention.v1"] = attentionBody("attention.v1")
	stdlib["attention.v2"] = attentionBody("attention.v2")
}

func cosBody(b *body, x *ref.Tensor) *ref.Tensor {
	return b.un(mim.OpSin, b.sub(b.k(math.Pi/2, x.DType, len(x.Shape)), x))
}

func tanhBody(b *body, x *ref.Tensor) *ref.Tensor {
	r := len(x.Shape)
	s := sigmoidBody(b, b.bin(mim.OpMul, b.k(2, x.DType, r), x))
	return b.sub(b.bin(mim.OpMul, b.k(2, x.DType, r), s), b.k(1, x.DType, r))
}

// boolAttr reads a bool attribute, encoded as an int 0 or 1.
func (a attrs) bool(key string, def bool) (bool, error) {
	v := a[key]
	if v == nil {
		return def, nil
	}
	if v.Type != mim.AttrInt || (v.Int != 0 && v.Int != 1) {
		return false, fmt.Errorf("attribute %q must be an int 0 or 1", key)
	}
	return v.Int == 1, nil
}

// tensorConst is tensor_const of spec/stdlib.md section 2.1: values
// computed in float64 and rounded once to dt.
func tensorConst(dt mim.DType, shape []int, vals []float64) *ref.Tensor {
	t, _ := ref.New(dt, shape)
	for i, v := range vals {
		t.SetBits(i, ref.RoundFloat(dt, v, false))
	}
	return t
}

func (b *body) permute(x *ref.Tensor, perm []int) *ref.Tensor {
	if x == nil {
		return nil
	}
	s := make([]int, len(perm))
	for i, p := range perm {
		s[i] = x.Shape[p]
	}
	return b.eval(mim.OpPermute, []mim.Attr{{Key: "perm", Type: mim.AttrInts, Ints: ints(perm)}}, x.DType, s, x)
}

// slice is the body macro slice(x, ax, lo, hi).
func (b *body) slice(x *ref.Tensor, ax, lo, hi int) *ref.Tensor {
	if x == nil {
		return nil
	}
	begin := make([]int, len(x.Shape))
	end := append([]int{}, x.Shape...)
	begin[ax], end[ax] = lo, hi
	s := append([]int{}, x.Shape...)
	s[ax] = hi - lo
	return b.eval(mim.OpShrink, []mim.Attr{
		{Key: "begin", Type: mim.AttrInts, Ints: ints(begin)},
		{Key: "end", Type: mim.AttrInts, Ints: ints(end)},
	}, x.DType, s, x)
}

// padAxis pads x with zeros on one axis.
func (b *body) padAxis(x *ref.Tensor, ax, before, after int) *ref.Tensor {
	if x == nil {
		return nil
	}
	pads := make([]int, 2*len(x.Shape))
	pads[2*ax], pads[2*ax+1] = before, after
	s := append([]int{}, x.Shape...)
	s[ax] += before + after
	return b.eval(mim.OpPad, []mim.Attr{{Key: "pads", Type: mim.AttrInts, Ints: ints(pads)}}, x.DType, s, x)
}

// concat is the body macro concat(a, c, ax).
func (b *body) concat(a, c *ref.Tensor, ax int) *ref.Tensor {
	if a == nil || c == nil {
		return nil
	}
	return b.bin(mim.OpAdd, b.padAxis(a, ax, 0, c.Shape[ax]), b.padAxis(c, ax, a.Shape[ax], 0))
}

// insertAxis is unsqueeze(x, ax).
func (b *body) insertAxis(x *ref.Tensor, ax int) *ref.Tensor {
	if x == nil {
		return nil
	}
	s := append(append(append([]int{}, x.Shape[:ax]...), 1), x.Shape[ax:]...)
	return b.reshape(x, s)
}

// dropAxis is squeeze(x, ax).
func (b *body) dropAxis(x *ref.Tensor, ax int) *ref.Tensor {
	if x == nil {
		return nil
	}
	s := append(append([]int{}, x.Shape[:ax]...), x.Shape[ax+1:]...)
	return b.reshape(x, s)
}

func (b *body) where(c, x, y *ref.Tensor) *ref.Tensor {
	if b.err != nil || c == nil || x == nil || y == nil {
		return nil
	}
	s := make([]int, len(c.Shape))
	for i := range s {
		s[i] = max(c.Shape[i], x.Shape[i], y.Shape[i])
		if c.Shape[i] == 0 || x.Shape[i] == 0 || y.Shape[i] == 0 {
			s[i] = 0
		}
	}
	return b.eval(mim.OpWhere, nil, x.DType, s, c, x, y)
}

// linear is the body of linear.v1 on x and w already in the computation
// dtype; bias may be nil.
func (b *body) linear(x, w, bias *ref.Tensor) *ref.Tensor {
	if b.err != nil || x == nil || w == nil {
		return nil
	}
	r := len(x.Shape)
	n, k := w.Shape[0], w.Shape[1]
	wt := b.reshape(b.permute(w, []int{1, 0}), append(ones(r-2), k, n))
	y := b.matmul(x, wt)
	if bias != nil {
		y = b.bin(mim.OpAdd, y, b.reshape(bias, append(ones(r-1), n)))
	}
	return y
}

func dequantMatmulV1(list []mim.Attr, in []*ref.Tensor, dts []mim.DType, _ [][]int) ([]*ref.Tensor, error) {
	if len(list) != 0 || len(in) < 2 || len(in) > 3 || len(dts) != 1 {
		return nil, fmt.Errorf("dequant_matmul takes x, w and an optional bias, and no attributes")
	}
	x, w := in[0], in[1]
	if err := checkDequantMatmul(in); err != nil {
		return nil, err
	}
	cd := computeDType(x.DType)
	b := &body{}
	wd := b.eval(mim.OpDequantize, nil, mim.F32, w.Shape, w)
	var bias *ref.Tensor
	if len(in) == 3 {
		bias = b.cast(in[2], cd)
	}
	y := b.linear(b.cast(x, cd), b.cast(wd, cd), bias)
	return []*ref.Tensor{b.cast(y, x.DType)}, b.err
}

// dequantMatmulA8V2 checks the block attribute and evaluates the body,
// which is that of dequant_matmul.v1.
func dequantMatmulA8V2(list []mim.Attr, in []*ref.Tensor, dts []mim.DType, shapes [][]int) ([]*ref.Tensor, error) {
	if _, err := a8Block(list); err != nil {
		return nil, err
	}
	return dequantMatmulV1(nil, in, dts, shapes)
}

// a8Block returns the block attribute of dequant_matmul_a8.v2: 32 or 256,
// 32 by default.
func a8Block(list []mim.Attr) (int, error) {
	a, err := readAttrs(list, "block")
	if err != nil {
		return 0, fmt.Errorf("dequant_matmul_a8: %v", err)
	}
	b, err := a.int("block", 32)
	if err != nil {
		return 0, fmt.Errorf("dequant_matmul_a8: %v", err)
	}
	if b != 32 && b != 256 {
		return 0, fmt.Errorf("dequant_matmul_a8: block must be 32 or 256, got %d", b)
	}
	return int(b), nil
}

func checkDequantMatmul(in []*ref.Tensor) error {
	x, w := in[0], in[1]
	if len(x.Shape) < 2 || len(w.Shape) != 2 || x.Shape[len(x.Shape)-1] != w.Shape[1] {
		return fmt.Errorf("dequant_matmul: x %s%v and w %s%v", x.DType, x.Shape, w.DType, w.Shape)
	}
	if x.DType != mim.F32 && x.DType != mim.F16 && x.DType != mim.BF16 {
		return fmt.Errorf("dequant_matmul: x must be f32, f16 or bf16, got %s", x.DType)
	}
	switch {
	case w.DType.IsBlock(), w.DType == mim.F8E4M3, w.DType == mim.F8E5M2, w.DType == mim.F16, w.DType == mim.BF16:
	default:
		return fmt.Errorf("dequant_matmul: w must be a block, f8, f16 or bf16 dtype, got %s", w.DType)
	}
	if len(in) == 3 {
		bias := in[2]
		if bias.DType != x.DType || len(bias.Shape) != 1 || bias.Shape[0] != w.Shape[0] {
			return fmt.Errorf("dequant_matmul: bias %s%v", bias.DType, bias.Shape)
		}
	}
	return nil
}

func swigluV1(list []mim.Attr, in []*ref.Tensor, dts []mim.DType, _ [][]int) ([]*ref.Tensor, error) {
	if len(list) != 0 || len(in) != 2 || len(dts) != 1 || !isFloat(in[0].DType) || in[0].DType != in[1].DType {
		return nil, fmt.Errorf("swiglu.v1 takes two float inputs of one dtype and no attributes")
	}
	cd := computeDType(in[0].DType)
	b := &body{}
	a := b.cast(in[0], cd)
	y := b.bin(mim.OpMul, siluBody(b, a), b.cast(in[1], cd))
	return []*ref.Tensor{b.cast(y, in[0].DType)}, b.err
}

// ropeParams are the attributes of rope.v1 and the frequencies they give.
type ropeParams struct {
	rotary      int
	interleaved bool
	invFreq     []float64 // h = rotary/2 entries
	ms          float64
}

func readRope(list []mim.Attr, d int) (*ropeParams, error) {
	a, err := readAttrs(list, "theta", "rotary_dims", "interleaved", "yarn_factor",
		"yarn_original_context", "yarn_beta_fast", "yarn_beta_slow", "yarn_attn_factor")
	if err != nil {
		return nil, err
	}
	theta, err := a.float("theta", 10000)
	if err != nil {
		return nil, err
	}
	rd, err := a.int("rotary_dims", int64(d))
	if err != nil {
		return nil, err
	}
	if rd <= 0 || rd%2 != 0 || rd > int64(d) {
		return nil, fmt.Errorf("rope.v1: rotary_dims %d for D = %d", rd, d)
	}
	p := &ropeParams{rotary: int(rd), ms: 1}
	if p.interleaved, err = a.bool("interleaved", false); err != nil {
		return nil, err
	}
	factor, err := a.float("yarn_factor", 1)
	if err != nil {
		return nil, err
	}
	orig, err := a.int("yarn_original_context", 0)
	if err != nil {
		return nil, err
	}
	fast, err := a.float("yarn_beta_fast", 32)
	if err != nil {
		return nil, err
	}
	slow, err := a.float("yarn_beta_slow", 1)
	if err != nil {
		return nil, err
	}
	attn, err := a.float("yarn_attn_factor", 0.1*math.Log(factor)+1)
	if err != nil {
		return nil, err
	}
	if factor != 1 && orig <= 0 {
		return nil, fmt.Errorf("rope.v1: yarn_original_context is required when yarn_factor is %g", factor)
	}
	r := float64(p.rotary)
	h := p.rotary / 2
	p.invFreq = make([]float64, h)
	var lo, hi float64
	if factor != 1 {
		corr := func(n float64) float64 {
			return r * math.Log(float64(orig)/(n*2*math.Pi)) / (2 * math.Log(theta))
		}
		lo = math.Max(math.Floor(corr(fast)), 0)
		hi = math.Min(math.Ceil(corr(slow)), r-1)
		if lo == hi {
			hi += 0.001
		}
		p.ms = attn
	}
	for i := range p.invFreq {
		f := math.Pow(theta, -2*float64(i)/r)
		if factor == 1 {
			p.invFreq[i] = f
			continue
		}
		ramp := math.Min(math.Max((float64(i)-lo)/(hi-lo), 0), 1)
		p.invFreq[i] = (f/factor)*ramp + f*(1-ramp)
	}
	return p, nil
}

func checkRope(in []*ref.Tensor) error {
	if len(in) != 2 {
		return fmt.Errorf("rope.v1 takes x and positions")
	}
	x, pos := in[0], in[1]
	if len(x.Shape) != 4 || !isFloat(x.DType) {
		return fmt.Errorf("rope.v1: x must be a float [B, H, T, D], got %s%v", x.DType, x.Shape)
	}
	if (pos.DType != mim.I32 && pos.DType != mim.I64) || len(pos.Shape) != 2 || pos.Shape[1] != x.Shape[2] ||
		(pos.Shape[0] != 1 && pos.Shape[0] != x.Shape[0]) {
		return fmt.Errorf("rope.v1: positions must be i32 or i64 [B, T] or [1, T], got %s%v for x %v", pos.DType, pos.Shape, x.Shape)
	}
	return nil
}

func ropeV1(list []mim.Attr, in []*ref.Tensor, dts []mim.DType, _ [][]int) ([]*ref.Tensor, error) {
	if err := checkRope(in); err != nil {
		return nil, err
	}
	x0, pos := in[0], in[1]
	B, H, T, D := x0.Shape[0], x0.Shape[1], x0.Shape[2], x0.Shape[3]
	p, err := readRope(list, D)
	if err != nil {
		return nil, err
	}
	R, h := p.rotary, p.rotary/2
	cd := computeDType(x0.DType)
	b := &body{}
	x := b.cast(x0, cd)
	F := tensorConst(cd, []int{1, 1, 1, h}, p.invFreq)
	P := b.reshape(b.cast(pos, cd), []int{pos.Shape[0], 1, T, 1})
	ang := b.bin(mim.OpMul, P, F)
	c := b.bin(mim.OpMul, cosBody(b, ang), b.k(p.ms, cd, 4))
	s := b.bin(mim.OpMul, b.un(mim.OpSin, ang), b.k(p.ms, cd, 4))
	xr := x
	if R < D {
		xr = b.slice(x, 3, 0, R)
	}
	var x1, x2 *ref.Tensor
	if p.interleaved {
		xi := b.reshape(xr, []int{B, H, T, h, 2})
		x1 = b.dropAxis(b.slice(xi, 4, 0, 1), 4)
		x2 = b.dropAxis(b.slice(xi, 4, 1, 2), 4)
	} else {
		x1 = b.slice(xr, 3, 0, h)
		x2 = b.slice(xr, 3, h, R)
	}
	y1 := b.sub(b.bin(mim.OpMul, x1, c), b.bin(mim.OpMul, x2, s))
	y2 := b.bin(mim.OpAdd, b.bin(mim.OpMul, x2, c), b.bin(mim.OpMul, x1, s))
	var yr *ref.Tensor
	if p.interleaved {
		yr = b.reshape(b.concat(b.insertAxis(y1, 4), b.insertAxis(y2, 4), 4), []int{B, H, T, R})
	} else {
		yr = b.concat(y1, y2, 3)
	}
	y := yr
	if R < D {
		y = b.concat(yr, b.slice(x, 3, R, D), 3)
	}
	return []*ref.Tensor{b.cast(y, x0.DType)}, b.err
}

// attnParams are the attributes of attention.v1 and v2.
type attnParams struct {
	scale   float64
	causal  bool
	kvHeads int
	softcap float64
}

// readAttention checks the inputs and reads the attributes of name,
// attention.v1 or attention.v2. v2 takes keys and values of any float
// dtype, each independently of q.
func readAttention(name string, list []mim.Attr, in []*ref.Tensor) (*attnParams, error) {
	a, err := readAttrs(list, "scale", "causal", "num_kv_heads", "softcap")
	if err != nil {
		return nil, err
	}
	if len(in) < 3 || len(in) > 4 {
		return nil, fmt.Errorf("%s takes q, k, v and an optional mask (sinks are not supported by this executor)", name)
	}
	q, k, v := in[0], in[1], in[2]
	kvOK := k.DType == q.DType && v.DType == q.DType
	if name == "attention.v2" {
		kvOK = isFloat(k.DType) && isFloat(v.DType)
	}
	if len(q.Shape) != 4 || len(k.Shape) != 4 || len(v.Shape) != 4 || !isFloat(q.DType) || !kvOK {
		return nil, fmt.Errorf("%s: q %s%v, k %s%v, v %s%v", name, q.DType, q.Shape, k.DType, k.Shape, v.DType, v.Shape)
	}
	B, Hq, D := q.Shape[0], q.Shape[1], q.Shape[3]
	p := &attnParams{}
	if p.scale, err = a.float("scale", 1/math.Sqrt(float64(D))); err != nil {
		return nil, err
	}
	if p.causal, err = a.bool("causal", false); err != nil {
		return nil, err
	}
	kv, err := a.int("num_kv_heads", int64(Hq))
	if err != nil {
		return nil, err
	}
	if p.softcap, err = a.float("softcap", 0); err != nil {
		return nil, err
	}
	p.kvHeads = int(kv)
	if kv <= 0 || int64(Hq)%kv != 0 || k.Shape[1] != p.kvHeads || v.Shape[1] != p.kvHeads ||
		k.Shape[0] != B || v.Shape[0] != B || k.Shape[3] != D || v.Shape[2] != k.Shape[2] {
		return nil, fmt.Errorf("%s: q %v, k %v, v %v with num_kv_heads %d", name, q.Shape, k.Shape, v.Shape, kv)
	}
	if len(in) == 4 {
		m := in[3]
		want := []int{B, Hq, q.Shape[2], k.Shape[2]}
		if len(m.Shape) != 4 || (m.DType != mim.Bool && !isFloat(m.DType)) {
			return nil, fmt.Errorf("%s: mask %s%v", name, m.DType, m.Shape)
		}
		for i := range want {
			if m.Shape[i] != want[i] && m.Shape[i] != 1 {
				return nil, fmt.Errorf("%s: mask %v does not broadcast to %v", name, m.Shape, want)
			}
		}
	}
	return p, nil
}

// attentionBody returns the canonical body of name, attention.v1 or v2:
// the same body, which casts the keys and values to the compute dtype.
func attentionBody(name string) compositeFunc {
	return func(list []mim.Attr, in []*ref.Tensor, dts []mim.DType, shapes [][]int) ([]*ref.Tensor, error) {
		return attentionCanonical(name, list, in)
	}
}

func attentionCanonical(name string, list []mim.Attr, in []*ref.Tensor) ([]*ref.Tensor, error) {
	p, err := readAttention(name, list, in)
	if err != nil {
		return nil, err
	}
	q0 := in[0]
	B, Hq, Tq := q0.Shape[0], q0.Shape[1], q0.Shape[2]
	Hkv, Tk, D, Dv := in[1].Shape[1], in[1].Shape[2], in[1].Shape[3], in[2].Shape[3]
	G := Hq / Hkv
	cd := computeDType(q0.DType)
	b := &body{}
	q, k, v := b.cast(q0, cd), b.cast(in[1], cd), b.cast(in[2], cd)
	rep := func(t *ref.Tensor, d int) *ref.Tensor {
		return b.reshape(b.expand(b.insertAxis(t, 2), []int{B, Hkv, G, Tk, d}), []int{B, Hq, Tk, d})
	}
	kf, vf := rep(k, D), rep(v, Dv)
	s := b.bin(mim.OpMul, b.matmul(q, b.permute(kf, []int{0, 1, 3, 2})), b.k(p.scale, cd, 4))
	if p.softcap > 0 {
		sc := b.k(p.softcap, cd, 4)
		s = b.bin(mim.OpMul, tanhBody(b, b.div(s, sc)), sc)
	}
	if len(in) == 4 {
		m := in[3]
		if m.DType == mim.Bool {
			s = b.bin(mim.OpAdd, s, b.where(m, b.k(0, cd, 4), b.k(math.Inf(-1), cd, 4)))
		} else {
			s = b.bin(mim.OpAdd, s, b.cast(m, cd))
		}
	}
	if p.causal {
		// riota(s, 2) and riota(s, 3): their values are exact, so they are
		// built directly rather than through the cumsum.v1 loop.
		rq := riota([]int{1, 1, Tq, 1}, Tq)
		rk := riota([]int{1, 1, 1, Tk}, Tk)
		t := b.eval(mim.OpConst, []mim.Attr{{Key: "value", Type: mim.AttrInt, Int: 1}}, mim.Bool, ones(4))
		ok := b.eval(mim.OpXor, nil, mim.Bool, []int{1, 1, Tq, Tk},
			b.eval(mim.OpCmplt, nil, mim.Bool, []int{1, 1, Tq, Tk}, rk, rq), t)
		s = b.bin(mim.OpAdd, s, b.where(ok, b.k(0, cd, 4), b.k(math.Inf(-1), cd, 4)))
	}
	m := b.reduce("max", s, []int{3})
	e := expBody(b, b.sub(s, m))
	pr := b.div(e, b.reduce("add", e, []int{3}))
	o := b.matmul(pr, vf)
	return []*ref.Tensor{b.cast(o, q0.DType)}, b.err
}

// riota returns an i64 tensor of the given shape holding n-1, n-2, ..., 0
// along its one non-unit axis.
func riota(shape []int, n int) *ref.Tensor {
	t, _ := ref.New(mim.I64, shape)
	for i := 0; i < n; i++ {
		t.SetBits(i, uint64(n-1-i))
	}
	return t
}
