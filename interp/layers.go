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
	"encoding/binary"
	"fmt"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/ref"
)

// This file holds the canonical bodies of the composites that models
// exported from PyTorch use beyond the ones in stdlib.go and
// transformer.go: layer_norm.v1, mean.v1, erf.v1, gelu_erf.v1, gelu_tanh.v1
// and cumsum.v1 (spec/stdlib.md sections 4 and 5). Like the rest of the
// executor's standard library they evaluate the body on the primitive
// kernels.

func init() {
	stdlib["layer_norm.v1"] = layerNormV1
	stdlib["mean.v1"] = meanV1
	stdlib["erf.v1"] = unaryComposite(erfBody)
	stdlib["gelu_erf.v1"] = unaryComposite(geluErfBody)
	stdlib["gelu_tanh.v1"] = unaryComposite(geluTanhBody)
	stdlib["cumsum.v1"] = cumsumV1
}

func (b *body) abs(x *ref.Tensor) *ref.Tensor { return b.bin(mim.OpMax, x, b.un(mim.OpNeg, x)) }

// erfBody is Abramowitz and Stegun formula 7.1.26, as in spec/stdlib.md
// section 4.3.
func erfBody(b *body, x *ref.Tensor) *ref.Tensor {
	r, dt := len(x.Shape), x.DType
	k := func(v float64) *ref.Tensor { return b.k(v, dt, r) }
	s := b.where(b.eval(mim.OpCmplt, nil, mim.Bool, x.Shape, x, k(0)), k(-1), k(1))
	ax := b.abs(x)
	t := b.un(mim.OpRecip, b.bin(mim.OpAdd, k(1), b.bin(mim.OpMul, k(0.3275911), ax)))
	q := b.bin(mim.OpMul, k(1.061405429), t)
	for _, a := range []float64{-1.453152027, 1.421413741, -0.284496736, 0.254829592} {
		q = b.bin(mim.OpMul, b.bin(mim.OpAdd, q, k(a)), t)
	}
	if b.err != nil {
		return nil
	}
	e := expBody(b, b.un(mim.OpNeg, b.bin(mim.OpMul, ax, ax)))
	return b.bin(mim.OpMul, s, b.sub(k(1), b.bin(mim.OpMul, q, e)))
}

func geluErfBody(b *body, x *ref.Tensor) *ref.Tensor {
	r, dt := len(x.Shape), x.DType
	e := erfBody(b, b.bin(mim.OpMul, x, b.k(0.7071067811865476, dt, r)))
	return b.bin(mim.OpMul, b.bin(mim.OpMul, b.k(0.5, dt, r), x), b.bin(mim.OpAdd, b.k(1, dt, r), e))
}

func geluTanhBody(b *body, x *ref.Tensor) *ref.Tensor {
	r, dt := len(x.Shape), x.DType
	cube := b.bin(mim.OpMul, x, b.bin(mim.OpMul, x, x))
	inner := b.bin(mim.OpMul, b.k(0.7978845608028654, dt, r), b.bin(mim.OpAdd, x, b.bin(mim.OpMul, b.k(0.044715, dt, r), cube)))
	return b.bin(mim.OpMul, b.bin(mim.OpMul, b.k(0.5, dt, r), x), b.bin(mim.OpAdd, b.k(1, dt, r), tanhBody(b, inner)))
}

// trailingAxes returns the axes ax..r-1 and the number of elements they hold.
func trailingAxes(shape []int, ax int) ([]int, int) {
	axes := make([]int, 0, len(shape)-ax)
	n := 1
	for i := ax; i < len(shape); i++ {
		axes = append(axes, i)
		n *= shape[i]
	}
	return axes, n
}

func layerNormV1(list []mim.Attr, in []*ref.Tensor, dts []mim.DType, _ [][]int) ([]*ref.Tensor, error) {
	a, err := readAttrs(list, "axis", "epsilon")
	if err != nil {
		return nil, err
	}
	if len(in) < 2 || len(in) > 3 || len(dts) != 1 || !isFloat(in[0].DType) {
		return nil, fmt.Errorf("layer_norm.v1 takes X, scale and an optional bias")
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
	eps, err := a.float("epsilon", 1e-5)
	if err != nil {
		return nil, err
	}
	for _, p := range in[1:] {
		if p.DType != x0.DType || !sameInts(p.Shape, x0.Shape[ax:]) {
			return nil, fmt.Errorf("layer_norm.v1: scale or bias %s%v, want %s%v", p.DType, p.Shape, x0.DType, x0.Shape[ax:])
		}
	}
	axes, count := trailingAxes(x0.Shape, ax)
	cd := computeDType(x0.DType)
	b := &body{}
	x := b.cast(x0, cd)
	n := b.k(float64(count), cd, r) // count(x, axes) is exactly count
	mu := b.div(b.reduce("add", x, axes), n)
	d := b.sub(x, mu)
	v := b.div(b.reduce("add", b.bin(mim.OpMul, d, d), axes), n)
	inv := b.un(mim.OpRecip, b.un(mim.OpSqrt, b.bin(mim.OpAdd, v, b.k(eps, cd, r))))
	y := b.cast(b.bin(mim.OpMul, d, inv), x0.DType)
	y = b.bin(mim.OpMul, y, b.reshape(in[1], append(ones(ax), in[1].Shape...)))
	if len(in) == 3 {
		y = b.bin(mim.OpAdd, y, b.reshape(in[2], append(ones(ax), in[2].Shape...)))
	}
	return []*ref.Tensor{y}, b.err
}

// readAxes reads the axes attribute of a reduction composite.
func readAxes(a attrs, r int) ([]int, error) {
	v := a["axes"]
	if v == nil || v.Type != mim.AttrInts {
		return nil, fmt.Errorf("attribute \"axes\" must be ints")
	}
	seen := make([]bool, r)
	axes := make([]int, len(v.Ints))
	for i, x := range v.Ints {
		ax, err := axis(x, r)
		if err != nil {
			return nil, err
		}
		if seen[ax] {
			return nil, fmt.Errorf("axis %d given twice", ax)
		}
		seen[ax] = true
		axes[i] = ax
	}
	return axes, nil
}

func meanV1(list []mim.Attr, in []*ref.Tensor, dts []mim.DType, _ [][]int) ([]*ref.Tensor, error) {
	a, err := readAttrs(list, "axes")
	if err != nil {
		return nil, err
	}
	if len(in) != 1 || len(dts) != 1 || !isFloat(in[0].DType) {
		return nil, fmt.Errorf("mean.v1 takes one float input")
	}
	x0 := in[0]
	axes, err := readAxes(a, len(x0.Shape))
	if err != nil {
		return nil, err
	}
	count := 1
	for _, ax := range axes {
		count *= x0.Shape[ax]
	}
	cd := computeDType(x0.DType)
	b := &body{}
	x := b.cast(x0, cd)
	y := b.div(b.reduce("add", x, axes), b.k(float64(count), cd, len(x.Shape)))
	return []*ref.Tensor{b.cast(y, x0.DType)}, b.err
}

// cumsumV1 runs the while loop of spec/stdlib.md section 5.7 directly: one
// gather, add and scatter per position along the axis.
func cumsumV1(list []mim.Attr, in []*ref.Tensor, dts []mim.DType, _ [][]int) ([]*ref.Tensor, error) {
	a, err := readAttrs(list, "axis", "exclusive", "reverse")
	if err != nil {
		return nil, err
	}
	if len(in) != 1 || len(dts) != 1 {
		return nil, fmt.Errorf("cumsum.v1 takes one input")
	}
	x0 := in[0]
	r := len(x0.Shape)
	ax64, err := a.int("axis", 0)
	if err != nil {
		return nil, err
	}
	if a["axis"] == nil {
		return nil, fmt.Errorf("cumsum.v1: attribute \"axis\" is required")
	}
	ax, err := axis(ax64, r)
	if err != nil {
		return nil, err
	}
	exclusive, err := a.bool("exclusive", false)
	if err != nil {
		return nil, err
	}
	reverse, err := a.bool("reverse", false)
	if err != nil {
		return nil, err
	}
	dt := x0.DType
	cd := dt
	if isFloat(dt) {
		cd = computeDType(dt)
	}
	b := &body{}
	x := b.cast(x0, cd)
	rowShape := append([]int{}, x0.Shape...)
	rowShape[ax] = 1
	zero := []mim.Attr{{Key: "value", Type: mim.AttrInt, Int: 0}}
	if isFloat(cd) {
		zero = []mim.Attr{{Key: "value", Type: mim.AttrFloat, Float: 0}}
	}
	acc := b.eval(mim.OpConst, zero, cd, rowShape)
	y := b.eval(mim.OpConst, zero, cd, x0.Shape)
	gatherAt := []mim.Attr{{Key: "axis", Type: mim.AttrInt, Int: int64(ax)}}
	n := x0.Shape[ax]
	for i := 0; i < n && b.err == nil; i++ {
		pos := i
		if reverse {
			pos = n - 1 - i
		}
		j, _ := ref.New(mim.I64, []int{1})
		binary.LittleEndian.PutUint64(j.Data, uint64(pos))
		xj := b.eval(mim.OpGather, gatherAt, cd, rowShape, x, j)
		if exclusive {
			y = b.eval(mim.OpScatter, gatherAt, cd, x0.Shape, y, j, acc)
			acc = b.bin(mim.OpAdd, acc, xj)
		} else {
			acc = b.bin(mim.OpAdd, acc, xj)
			y = b.eval(mim.OpScatter, gatherAt, cd, x0.Shape, y, j, acc)
		}
	}
	return []*ref.Tensor{b.cast(y, dt)}, b.err
}
