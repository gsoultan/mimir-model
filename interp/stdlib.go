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
	"fmt"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/ref"
)

// compositeFunc evaluates a standard-library composite. attrs excludes name
// and version; dts and shapes are the declared types of the outputs.
type compositeFunc func(attrs []mmf.Attr, in []*ref.Tensor, dts []mmf.DType, shapes [][]int) ([]*ref.Tensor, error)

// stdlib holds the composites this executor supports. Each one evaluates
// the canonical body of spec/stdlib.md on the primitive kernels, so its
// results are the body's results, not an approximation of them.
var stdlib = map[string]compositeFunc{
	"matmul.v1":    matmulV1,
	"linear.v1":    linearV1,
	"softmax.v1":   softmaxV1,
	"rms_norm.v1":  rmsNormV1,
	"exp.v1":       unaryComposite(expBody),
	"sigmoid.v1":   unaryComposite(sigmoidBody),
	"conv2d.v1":    conv2dV1,
	"silu.v1":      unaryComposite(siluBody),
	"embedding.v1": embeddingV1,
}

// Supported lists the standard-library composites the executor runs.
func Supported() []string {
	s := make([]string, 0, len(stdlib))
	for k := range stdlib {
		s = append(s, k)
	}
	return s
}

const log2e = 1.4426950408889634

// body evaluates primitives one at a time, keeping the first error.
type body struct{ err error }

func (b *body) eval(op mmf.Opcode, attrs []mmf.Attr, dt mmf.DType, shape []int, in ...*ref.Tensor) *ref.Tensor {
	if b.err != nil {
		return nil
	}
	for _, t := range in {
		if t == nil {
			return nil
		}
	}
	t, err := ref.Eval(op, attrs, in, dt, shape)
	if err != nil {
		b.err = err
	}
	return t
}

func ints(v []int) []int64 {
	r := make([]int64, len(v))
	for i, x := range v {
		r[i] = int64(x)
	}
	return r
}

func ones(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = 1
	}
	return s
}

func (b *body) un(op mmf.Opcode, x *ref.Tensor) *ref.Tensor {
	if x == nil {
		return nil
	}
	return b.eval(op, nil, x.DType, x.Shape, x)
}

func (b *body) bin(op mmf.Opcode, x, y *ref.Tensor) *ref.Tensor {
	if b.err != nil || x == nil || y == nil {
		return nil
	}
	if len(x.Shape) != len(y.Shape) {
		b.err = fmt.Errorf("%s: ranks %d and %d differ", op, len(x.Shape), len(y.Shape))
		return nil
	}
	s := make([]int, len(x.Shape))
	for i := range s {
		s[i] = max(x.Shape[i], y.Shape[i])
		if x.Shape[i] == 0 || y.Shape[i] == 0 {
			s[i] = 0
		}
	}
	return b.eval(op, nil, x.DType, s, x, y)
}

// k is the body macro k(v): a constant of the computation dtype with every
// dimension 1.
func (b *body) k(v float64, dt mmf.DType, rank int) *ref.Tensor {
	return b.eval(mmf.OpConst, []mmf.Attr{{Key: "value", Type: mmf.AttrFloat, Float: v}}, dt, ones(rank))
}

func (b *body) reduce(mode string, x *ref.Tensor, axes []int) *ref.Tensor {
	if x == nil {
		return nil
	}
	s := append([]int{}, x.Shape...)
	for _, a := range axes {
		s[a] = 1
	}
	return b.eval(mmf.OpReduce, []mmf.Attr{
		{Key: "mode", Type: mmf.AttrString, String: mode},
		{Key: "axes", Type: mmf.AttrInts, Ints: ints(axes)},
	}, x.DType, s, x)
}

func (b *body) reshape(x *ref.Tensor, shape []int) *ref.Tensor {
	if x == nil {
		return nil
	}
	return b.eval(mmf.OpReshape, []mmf.Attr{{Key: "shape", Type: mmf.AttrInts, Ints: ints(shape)}}, x.DType, shape, x)
}

func (b *body) expand(x *ref.Tensor, shape []int) *ref.Tensor {
	if x == nil {
		return nil
	}
	return b.eval(mmf.OpExpand, nil, x.DType, shape, x)
}

func (b *body) cast(x *ref.Tensor, dt mmf.DType) *ref.Tensor {
	if x == nil || x.DType == dt {
		return x
	}
	return b.eval(mmf.OpCast, []mmf.Attr{{Key: "to", Type: mmf.AttrDType, DType: dt}}, dt, x.Shape, x)
}

func (b *body) sub(x, y *ref.Tensor) *ref.Tensor { return b.bin(mmf.OpAdd, x, b.un(mmf.OpNeg, y)) }
func (b *body) div(x, y *ref.Tensor) *ref.Tensor { return b.bin(mmf.OpMul, x, b.un(mmf.OpRecip, y)) }

// computeDType is the computation dtype of spec/stdlib.md section 1.4.
func computeDType(dt mmf.DType) mmf.DType {
	if dt == mmf.F16 || dt == mmf.BF16 {
		return mmf.F32
	}
	return dt
}

// attrs reads composite attributes, rejecting unknown keys.
type attrs map[string]*mmf.Attr

func readAttrs(list []mmf.Attr, known ...string) (attrs, error) {
	a := attrs{}
	for i := range list {
		ok := false
		for _, k := range known {
			ok = ok || list[i].Key == k
		}
		if !ok {
			return nil, fmt.Errorf("unknown attribute %q", list[i].Key)
		}
		a[list[i].Key] = &list[i]
	}
	return a, nil
}

func (a attrs) int(key string, def int64) (int64, error) {
	v := a[key]
	if v == nil {
		return def, nil
	}
	if v.Type != mmf.AttrInt {
		return 0, fmt.Errorf("attribute %q must be an int", key)
	}
	return v.Int, nil
}

func (a attrs) float(key string, def float64) (float64, error) {
	v := a[key]
	if v == nil {
		return def, nil
	}
	if v.Type != mmf.AttrFloat {
		return 0, fmt.Errorf("attribute %q must be a float", key)
	}
	return v.Float, nil
}

// axis normalises a possibly negative axis for rank r.
func axis(ax int64, r int) (int, error) {
	if ax < 0 {
		ax += int64(r)
	}
	if ax < 0 || ax >= int64(r) {
		return 0, fmt.Errorf("axis %d is out of range for rank %d", ax, r)
	}
	return int(ax), nil
}

func isFloat(dt mmf.DType) bool {
	return dt == mmf.F64 || dt == mmf.F32 || dt == mmf.F16 || dt == mmf.BF16
}

func expBody(b *body, x *ref.Tensor) *ref.Tensor {
	return b.un(mmf.OpExp2, b.bin(mmf.OpMul, x, b.k(log2e, x.DType, len(x.Shape))))
}

func sigmoidBody(b *body, x *ref.Tensor) *ref.Tensor {
	r := len(x.Shape)
	e := b.un(mmf.OpExp2, b.bin(mmf.OpMul, b.un(mmf.OpNeg, x), b.k(log2e, x.DType, r)))
	return b.un(mmf.OpRecip, b.bin(mmf.OpAdd, b.k(1, x.DType, r), e))
}

func siluBody(b *body, x *ref.Tensor) *ref.Tensor {
	return b.bin(mmf.OpMul, x, sigmoidBody(b, x))
}

// unaryComposite wraps a float elementwise body with the entry and exit
// casts of spec/stdlib.md section 1.4.
func unaryComposite(f func(*body, *ref.Tensor) *ref.Tensor) compositeFunc {
	return func(list []mmf.Attr, in []*ref.Tensor, dts []mmf.DType, _ [][]int) ([]*ref.Tensor, error) {
		if len(list) != 0 || len(in) != 1 || len(dts) != 1 || !isFloat(in[0].DType) {
			return nil, fmt.Errorf("takes one float input, one output and no attributes")
		}
		b := &body{}
		x := b.cast(in[0], computeDType(in[0].DType))
		y := b.cast(f(b, x), in[0].DType)
		return []*ref.Tensor{y}, b.err
	}
}

// matmul is the body of matmul.v1 on tensors already in the computation
// dtype. batch is the broadcast batch shape.
func (b *body) matmul(a, bm *ref.Tensor) *ref.Tensor {
	if b.err != nil {
		return nil
	}
	r := len(a.Shape)
	if r < 2 || len(bm.Shape) != r || a.DType != bm.DType {
		b.err = fmt.Errorf("matmul: operands %s%v and %s%v", a.DType, a.Shape, bm.DType, bm.Shape)
		return nil
	}
	m, k, k2, n := a.Shape[r-2], a.Shape[r-1], bm.Shape[r-2], bm.Shape[r-1]
	if k != k2 {
		b.err = fmt.Errorf("matmul: contracted lengths %d and %d differ", k, k2)
		return nil
	}
	batch := make([]int, r-2)
	for i := range batch {
		x, y := a.Shape[i], bm.Shape[i]
		if x != y && x != 1 && y != 1 {
			b.err = fmt.Errorf("matmul: batch dimension %d: %d and %d", i, x, y)
			return nil
		}
		batch[i] = max(x, y)
		if x == 0 || y == 0 {
			batch[i] = 0
		}
	}
	as := append(append([]int{}, a.Shape[:r-1]...), k, 1)             // [..., M, K, 1]
	bs := append(append(append([]int{}, bm.Shape[:r-2]...), 1), k, n) // [..., 1, K, N]
	ps := append(append([]int{}, batch...), m, k, n)                  // [..., M, K, N]
	p := b.bin(mmf.OpMul, b.expand(b.reshape(a, as), ps), b.expand(b.reshape(bm, bs), ps))
	s := b.reduce("add", p, []int{r - 1}) // [..., M, 1, N]
	return b.reshape(s, append(append([]int{}, batch...), m, n))
}

func matmulV1(list []mmf.Attr, in []*ref.Tensor, dts []mmf.DType, _ [][]int) ([]*ref.Tensor, error) {
	if len(list) != 0 || len(in) != 2 || len(dts) != 1 {
		return nil, fmt.Errorf("matmul.v1 takes two inputs, one output and no attributes")
	}
	dt := in[0].DType
	b := &body{}
	y := b.cast(b.matmul(b.cast(in[0], computeDType(dt)), b.cast(in[1], computeDType(dt))), dt)
	return []*ref.Tensor{y}, b.err
}

func linearV1(list []mmf.Attr, in []*ref.Tensor, dts []mmf.DType, _ [][]int) ([]*ref.Tensor, error) {
	if len(list) != 0 || len(in) < 2 || len(in) > 3 || len(dts) != 1 {
		return nil, fmt.Errorf("linear.v1 takes x, w and an optional bias, and no attributes")
	}
	x, w := in[0], in[1]
	r := len(x.Shape)
	if r < 2 || len(w.Shape) != 2 || x.DType != w.DType {
		return nil, fmt.Errorf("linear.v1: x %s%v and w %s%v", x.DType, x.Shape, w.DType, w.Shape)
	}
	n, k := w.Shape[0], w.Shape[1]
	cd := computeDType(x.DType)
	b := &body{}
	wt := b.eval(mmf.OpPermute, []mmf.Attr{{Key: "perm", Type: mmf.AttrInts, Ints: []int64{1, 0}}}, w.DType, []int{k, n}, w)
	wt = b.reshape(wt, append(ones(r-2), k, n))
	y := b.matmul(b.cast(x, cd), b.cast(wt, cd))
	if len(in) == 3 {
		bias := in[2]
		if bias.DType != x.DType || len(bias.Shape) != 1 || bias.Shape[0] != n {
			return nil, fmt.Errorf("linear.v1: bias %s%v", bias.DType, bias.Shape)
		}
		y = b.bin(mmf.OpAdd, y, b.reshape(b.cast(bias, cd), append(ones(r-1), n)))
	}
	return []*ref.Tensor{b.cast(y, x.DType)}, b.err
}

func softmaxV1(list []mmf.Attr, in []*ref.Tensor, dts []mmf.DType, _ [][]int) ([]*ref.Tensor, error) {
	a, err := readAttrs(list, "axis")
	if err != nil {
		return nil, err
	}
	if len(in) != 1 || len(dts) != 1 || !isFloat(in[0].DType) {
		return nil, fmt.Errorf("softmax.v1 takes one float input")
	}
	ax64, err := a.int("axis", -1)
	if err != nil {
		return nil, err
	}
	ax, err := axis(ax64, len(in[0].Shape))
	if err != nil {
		return nil, err
	}
	b := &body{}
	x := b.cast(in[0], computeDType(in[0].DType))
	m := b.reduce("max", x, []int{ax})
	e := expBody(b, b.sub(x, m))
	s := b.reduce("add", e, []int{ax})
	return []*ref.Tensor{b.cast(b.div(e, s), in[0].DType)}, b.err
}

func rmsNormV1(list []mmf.Attr, in []*ref.Tensor, dts []mmf.DType, _ [][]int) ([]*ref.Tensor, error) {
	a, err := readAttrs(list, "axis", "epsilon")
	if err != nil {
		return nil, err
	}
	if len(in) < 1 || len(in) > 2 || len(dts) != 1 || !isFloat(in[0].DType) {
		return nil, fmt.Errorf("rms_norm.v1 takes x and an optional scale")
	}
	x0 := in[0]
	r := len(x0.Shape)
	ax64, err := a.int("axis", -1)
	if err != nil {
		return nil, err
	}
	ax, err := axis(ax64, r)
	if err != nil {
		return nil, err
	}
	eps, err := a.float("epsilon", 1e-6)
	if err != nil {
		return nil, err
	}
	axes := make([]int, 0, r-ax)
	count := 1
	for i := ax; i < r; i++ {
		axes = append(axes, i)
		count *= x0.Shape[i]
	}
	cd := computeDType(x0.DType)
	b := &body{}
	x := b.cast(x0, cd)
	// count(x, axes) is a reduction of ones; its value is exactly count.
	ms := b.div(b.reduce("add", b.bin(mmf.OpMul, x, x), axes), b.k(float64(count), cd, r))
	y := b.bin(mmf.OpMul, x, b.un(mmf.OpRecip, b.un(mmf.OpSqrt, b.bin(mmf.OpAdd, ms, b.k(eps, cd, r)))))
	if len(in) == 2 {
		sc := in[1]
		if sc.DType != x0.DType || !sameInts(sc.Shape, x0.Shape[ax:]) {
			return nil, fmt.Errorf("rms_norm.v1: scale %s%v, want %s%v", sc.DType, sc.Shape, x0.DType, x0.Shape[ax:])
		}
		y = b.bin(mmf.OpMul, y, b.reshape(b.cast(sc, cd), append(ones(ax), sc.Shape...)))
	}
	return []*ref.Tensor{b.cast(y, x0.DType)}, b.err
}

func embeddingV1(list []mmf.Attr, in []*ref.Tensor, dts []mmf.DType, _ [][]int) ([]*ref.Tensor, error) {
	if len(list) != 0 || len(in) != 2 || len(dts) != 1 {
		return nil, fmt.Errorf("embedding.v1 takes a table and ids, and no attributes")
	}
	table, ids := in[0], in[1]
	if len(table.Shape) != 2 {
		return nil, fmt.Errorf("embedding.v1: table must be [V, E], got %v", table.Shape)
	}
	b := &body{}
	t := table
	if table.DType.IsBlock() || table.DType == mmf.F8E4M3 || table.DType == mmf.F8E5M2 {
		t = b.eval(mmf.OpDequantize, nil, dts[0], table.Shape, table)
	}
	shape := append(append([]int{}, ids.Shape...), table.Shape[1])
	y := b.eval(mmf.OpGather, []mmf.Attr{{Key: "axis", Type: mmf.AttrInt, Int: 0}}, dts[0], shape, t, ids)
	return []*ref.Tensor{y}, b.err
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
