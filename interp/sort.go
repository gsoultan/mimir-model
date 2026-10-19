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
	"slices"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/ref"
)

// top_k.v1 and argsort.v1 have no canonical body; spec/stdlib.md section 7
// defines them directly, and so are they here.

func init() {
	stdlib["top_k.v1"] = topKV1
	stdlib["argsort.v1"] = argsortV1
}

// sortKeys returns a comparison of the elements of x, by index, in the
// order of spec/stdlib.md section 7: integers in their natural order;
// floats with -0 = +0 and every NaN equal and above +inf.
func sortKeys(x *ref.Tensor) (func(i, j int) int, error) {
	dt := x.DType
	switch dt {
	case mim.I8, mim.I16, mim.I32, mim.I64:
		bits := 8 * int(dt.BlockBytes())
		v := func(i int) int64 { return int64(x.Bits(i)<<(64-bits)) >> (64 - bits) }
		return func(i, j int) int { return cmpInt(v(i), v(j)) }, nil
	case mim.U8, mim.U16, mim.U32, mim.U64:
		return func(i, j int) int {
			a, b := x.Bits(i), x.Bits(j)
			switch {
			case a < b:
				return -1
			case a > b:
				return 1
			}
			return 0
		}, nil
	}
	if dt.IsBlock() || !isSortFloat(dt) {
		return nil, fmt.Errorf("sorting %s is not defined", dt)
	}
	f := x
	if dt != mim.F64 {
		b := &body{}
		if f = b.cast(x, mim.F64); b.err != nil {
			return nil, b.err
		}
	}
	v := make([]float64, f.NumElems())
	for i := range v {
		v[i] = math.Float64frombits(f.Bits(i))
	}
	return func(i, j int) int {
		a, b := v[i], v[j]
		an, bn := math.IsNaN(a), math.IsNaN(b)
		switch {
		case an && bn:
			return 0
		case an:
			return 1
		case bn:
			return -1
		case a < b:
			return -1
		case a > b:
			return 1
		}
		return 0 // includes -0 == +0
	}, nil
}

func isSortFloat(dt mim.DType) bool {
	return isFloat(dt) || dt == mim.F8E4M3 || dt == mim.F8E5M2
}

func cmpInt(a, b int64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

// sortRows calls f with the element indices of each row of x along axis
// ax, in the order of the stable sort, descending if desc, with equal
// elements in ascending index order either way.
func sortRows(x *ref.Tensor, ax int, desc bool, f func(row []int, order []int)) error {
	cmp, err := sortKeys(x)
	if err != nil {
		return err
	}
	shape := x.Shape
	n := shape[ax]
	inner := 1
	for _, d := range shape[ax+1:] {
		inner *= d
	}
	outer := 1
	for _, d := range shape[:ax] {
		outer *= d
	}
	row := make([]int, n)
	order := make([]int, n)
	for o := 0; o < outer; o++ {
		for in := 0; in < inner; in++ {
			for j := range row {
				row[j] = (o*n+j)*inner + in
				order[j] = j
			}
			slices.SortStableFunc(order, func(a, b int) int {
				c := cmp(row[a], row[b])
				if desc {
					c = -c
				}
				return c
			})
			f(row, order)
		}
	}
	return nil
}

func topKV1(list []mim.Attr, in []*ref.Tensor, dts []mim.DType, shapes [][]int) ([]*ref.Tensor, error) {
	a, err := readAttrs(list, "k", "axis", "largest")
	if err != nil {
		return nil, err
	}
	if len(in) != 1 || len(dts) != 2 {
		return nil, fmt.Errorf("top_k.v1 takes x and gives values and indices")
	}
	x := in[0]
	ax64, err := a.int("axis", -1)
	if err != nil {
		return nil, err
	}
	ax, err := axis(ax64, len(x.Shape))
	if err != nil {
		return nil, err
	}
	k64, err := a.int("k", -1)
	if err != nil {
		return nil, err
	}
	if a["k"] == nil || k64 < 0 || k64 > int64(x.Shape[ax]) {
		return nil, fmt.Errorf("top_k.v1: k = %d for an axis of length %d", k64, x.Shape[ax])
	}
	largest, err := a.bool("largest", true)
	if err != nil {
		return nil, err
	}
	k := int(k64)
	want := slices.Clone(x.Shape)
	want[ax] = k
	if dts[0] != x.DType || dts[1] != mim.I64 || !sameInts(shapes[0], want) || !sameInts(shapes[1], want) {
		return nil, fmt.Errorf("top_k.v1: outputs %s%v and %s%v, want %s%v and i64%v", dts[0], shapes[0], dts[1], shapes[1], x.DType, want, want)
	}
	vals, err := ref.New(x.DType, want)
	if err != nil {
		return nil, err
	}
	idx, err := ref.New(mim.I64, want)
	if err != nil {
		return nil, err
	}
	inner := 1
	for _, d := range x.Shape[ax+1:] {
		inner *= d
	}
	n := x.Shape[ax]
	err = sortRows(x, ax, largest, func(row, order []int) {
		// row[0] is the first element of the row: recover its outer and
		// inner position to place the k results.
		o, i := row[0]/(n*inner), row[0]%inner
		for t := 0; t < k; t++ {
			dst := (o*k+t)*inner + i
			vals.SetBits(dst, x.Bits(row[order[t]]))
			idx.SetBits(dst, uint64(order[t]))
		}
	})
	if err != nil {
		return nil, fmt.Errorf("top_k.v1: %w", err)
	}
	return []*ref.Tensor{vals, idx}, nil
}

func argsortV1(list []mim.Attr, in []*ref.Tensor, dts []mim.DType, shapes [][]int) ([]*ref.Tensor, error) {
	a, err := readAttrs(list, "axis", "descending")
	if err != nil {
		return nil, err
	}
	if len(in) != 1 || len(dts) != 1 {
		return nil, fmt.Errorf("argsort.v1 takes x and gives indices")
	}
	x := in[0]
	ax64, err := a.int("axis", -1)
	if err != nil {
		return nil, err
	}
	ax, err := axis(ax64, len(x.Shape))
	if err != nil {
		return nil, err
	}
	desc, err := a.bool("descending", false)
	if err != nil {
		return nil, err
	}
	if dts[0] != mim.I64 || !sameInts(shapes[0], x.Shape) {
		return nil, fmt.Errorf("argsort.v1: output %s%v, want i64%v", dts[0], shapes[0], x.Shape)
	}
	idx, err := ref.New(mim.I64, x.Shape)
	if err != nil {
		return nil, err
	}
	n := x.Shape[ax]
	err = sortRows(x, ax, desc, func(row, order []int) {
		for t := range n {
			idx.SetBits(row[t], uint64(order[t]))
		}
	})
	if err != nil {
		return nil, fmt.Errorf("argsort.v1: %w", err)
	}
	return []*ref.Tensor{idx}, nil
}
