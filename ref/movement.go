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

package ref

import (
	"fmt"
	"math"

	mim "github.com/gsoultan/mimir-model"
)

// copyElem copies element j of src to element i of dst; both have dtype dt.
func copyElem(dt mim.DType, dst []byte, i int, src []byte, j int) {
	s := elemSize(dt)
	copy(dst[i*s:(i+1)*s], src[j*s:(j+1)*s])
}

func evalExpand(x *Tensor, shape []int) (*Tensor, error) {
	y, err := New(x.DType, shape)
	if err != nil {
		return nil, err
	}
	for i, j := range broadcastMap(shape, x.Shape) {
		copyElem(x.DType, y.Data, i, x.Data, j)
	}
	return y, nil
}

func evalPermute(x *Tensor, perm []int, shape []int) (*Tensor, error) {
	y, err := New(x.DType, shape)
	if err != nil {
		return nil, err
	}
	st := strides(x.Shape)
	idx := make([]int, len(shape))
	for i := 0; i < y.NumElems(); i++ {
		unravel(i, shape, idx)
		j := 0
		for k, p := range perm {
			j += idx[k] * st[p]
		}
		copyElem(x.DType, y.Data, i, x.Data, j)
	}
	return y, nil
}

func evalPad(x *Tensor, pads []int, value uint64, shape []int) (*Tensor, error) {
	y, err := New(x.DType, shape)
	if err != nil {
		return nil, err
	}
	st := strides(x.Shape)
	idx := make([]int, len(shape))
	for i := 0; i < y.NumElems(); i++ {
		unravel(i, shape, idx)
		j, inside := 0, true
		for k := range idx {
			s := idx[k] - pads[2*k]
			if s < 0 || s >= x.Shape[k] {
				inside = false
				break
			}
			j += s * st[k]
		}
		if inside {
			copyElem(x.DType, y.Data, i, x.Data, j)
		} else {
			y.SetBits(i, value)
		}
	}
	return y, nil
}

func evalShrink(x *Tensor, begin []int, shape []int) (*Tensor, error) {
	y, err := New(x.DType, shape)
	if err != nil {
		return nil, err
	}
	st := strides(x.Shape)
	idx := make([]int, len(shape))
	for i := 0; i < y.NumElems(); i++ {
		unravel(i, shape, idx)
		j := 0
		for k := range idx {
			j += (idx[k] + begin[k]) * st[k]
		}
		copyElem(x.DType, y.Data, i, x.Data, j)
	}
	return y, nil
}

// index returns element i of an i32 or i64 index tensor.
func index(t *Tensor, i int) int64 {
	if t.DType == mim.I32 {
		return sext(mim.I32, t.Bits(i))
	}
	return int64(t.Bits(i))
}

// gatherShape is data_shape[:axis] ++ indices_shape ++ data_shape[axis+1:].
func gatherShape(data, indices []int, axis int) []int {
	s := append([]int{}, data[:axis]...)
	s = append(s, indices...)
	return append(s, data[axis+1:]...)
}

// evalGather leaves output elements selected by an out-of-range index zero:
// their value is unspecified (spec/primitives.md section 5.30).
func evalGather(data, ind *Tensor, axis int, shape []int) (*Tensor, error) {
	y, err := New(data.DType, shape)
	if err != nil {
		return nil, err
	}
	q := len(ind.Shape)
	dst := strides(data.Shape)
	ist := strides(ind.Shape)
	idx := make([]int, len(shape))
	for i := 0; i < y.NumElems(); i++ {
		unravel(i, shape, idx)
		jf := 0
		for k := 0; k < q; k++ {
			jf += idx[axis+k] * ist[k]
		}
		k := index(ind, jf)
		if k < 0 || k >= int64(data.Shape[axis]) {
			continue
		}
		off := int(k) * dst[axis]
		for a := 0; a < axis; a++ {
			off += idx[a] * dst[a]
		}
		for b := axis + 1; b < len(data.Shape); b++ {
			off += idx[b+q-1] * dst[b]
		}
		copyElem(data.DType, y.Data, i, data.Data, off)
	}
	return y, nil
}

// evalScatter applies updates in row-major order of the update tensor, so
// for duplicate indices the later index position wins. Updates with an
// out-of-range index are skipped.
func evalScatter(data, ind, upd *Tensor, axis int) (*Tensor, error) {
	y := &Tensor{DType: data.DType, Shape: append([]int{}, data.Shape...), Data: append([]byte{}, data.Data...)}
	q := len(ind.Shape)
	dst := strides(data.Shape)
	ist := strides(ind.Shape)
	idx := make([]int, len(upd.Shape))
	// Row-major order over the update tensor visits index positions in
	// row-major order for each fixed (a, b), which is what decides the
	// winner among duplicates.
	for i := 0; i < upd.NumElems(); i++ {
		unravel(i, upd.Shape, idx)
		jf := 0
		for k := 0; k < q; k++ {
			jf += idx[axis+k] * ist[k]
		}
		k := index(ind, jf)
		if k < 0 || k >= int64(data.Shape[axis]) {
			continue
		}
		off := int(k) * dst[axis]
		for a := 0; a < axis; a++ {
			off += idx[a] * dst[a]
		}
		for b := axis + 1; b < len(data.Shape); b++ {
			off += idx[b+q-1] * dst[b]
		}
		copyElem(data.DType, y.Data, off, upd.Data, i)
	}
	return y, nil
}

// scalarBits converts the `value` attribute of const and pad to one element
// of dtype dt (spec/primitives.md section 5.32).
func scalarBits(dt mim.DType, a *mim.Attr) (uint64, error) {
	isInt := a.Type == mim.AttrInt
	switch {
	case IsFloat(dt):
		if isInt {
			v := a.Int
			if v < 0 {
				return roundInt(dt, true, uint64(-v), isStorage(dt)), nil
			}
			return roundInt(dt, false, uint64(v), isStorage(dt)), nil
		}
		return RoundFloat(dt, a.Float, isStorage(dt)), nil
	case dt == mim.Bool:
		v := float64(a.Int)
		if !isInt {
			v = a.Float
		}
		if v != 0 && v != 1 {
			return 0, fmt.Errorf("bool value must be 0 or 1, got %v", v)
		}
		return b2u(v == 1), nil
	case IsInt(dt):
		lo, hi, _ := intRange(dt)
		if isInt {
			if a.Int < lo || a.Int > hi {
				return 0, fmt.Errorf("value %d is not representable in %s", a.Int, dt)
			}
			return mask(dt, uint64(a.Int)), nil
		}
		f := a.Float
		_, _, hiU := intRange(dt)
		if f != math.Trunc(f) || f < float64(lo) || f > float64(hiU) || math.IsInf(f, 0) {
			return 0, fmt.Errorf("value %v is not representable in %s", f, dt)
		}
		if !isSigned(dt) {
			if f >= math.Ldexp(1, 64) {
				return 0, fmt.Errorf("value %v is not representable in %s", f, dt)
			}
			return mask(dt, uint64(f)), nil
		}
		if f >= math.Ldexp(1, 63) {
			return 0, fmt.Errorf("value %v is not representable in %s", f, dt)
		}
		return mask(dt, uint64(int64(f))), nil
	}
	return 0, fmt.Errorf("no scalar value for dtype %s", dt)
}

func evalConst(dt mim.DType, shape []int, value uint64) (*Tensor, error) {
	y, err := New(dt, shape)
	if err != nil {
		return nil, err
	}
	for i := 0; i < y.NumElems(); i++ {
		y.SetBits(i, value)
	}
	return y, nil
}

func evalDequantize(x *Tensor, to mim.DType) (*Tensor, error) {
	n := x.NumElems()
	f := make([]float32, n)
	if err := mim.Dequantize(x.DType, x.Data, f); err != nil {
		return nil, err
	}
	y, err := New(to, x.Shape)
	if err != nil {
		return nil, err
	}
	for i, v := range f {
		if to == mim.BF16 {
			y.SetBits(i, uint64(mim.F32ToBF16(v)))
		} else {
			y.SetBits(i, uint64(math.Float32bits(v)))
		}
	}
	return y, nil
}
