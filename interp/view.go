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
	"math"
	"unsafe"

	"github.com/gsoultan/mim"
	"github.com/gsoultan/mim/ref"
)

// mim stores float32 little-endian. On a little-endian machine an f32
// tensor's bytes are already a []float32, so the fast paths read inputs
// and return outputs in place instead of converting element by element.
// This matters for the key-value cache, which attention reads in full at
// every step.
var littleEndian = func() bool {
	x := uint16(1)
	return *(*byte)(unsafe.Pointer(&x)) == 1
}()

// f32View returns the elements of an f32 tensor without copying where it
// can. The result aliases t and must not be modified.
func f32View(t *ref.Tensor) []float32 {
	n := t.NumElems()
	if littleEndian && n > 0 && len(t.Data) >= 4*n && uintptr(unsafe.Pointer(&t.Data[0]))%4 == 0 {
		return unsafe.Slice((*float32)(unsafe.Pointer(&t.Data[0])), n)
	}
	return f32Values(t)
}

// f32Values returns a copy of the elements of an f32 tensor, which the
// caller may modify.
func f32Values(t *ref.Tensor) []float32 {
	v := make([]float32, t.NumElems())
	if littleEndian && len(v) > 0 {
		copy(unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), 4*len(v)), t.Data)
		return v
	}
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.Data[4*i:]))
	}
	return v
}

// f32Tensor returns an f32 tensor holding v. It takes ownership of v: on a
// little-endian machine the tensor's bytes are v's memory.
func f32Tensor(shape []int, v []float32) *ref.Tensor {
	t := &ref.Tensor{DType: mim.F32, Shape: append([]int{}, shape...)}
	if littleEndian {
		if len(v) > 0 {
			t.Data = unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), 4*len(v))
		} else {
			t.Data = []byte{}
		}
		return t
	}
	t.Data = make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(t.Data[4*i:], math.Float32bits(x))
	}
	return t
}
