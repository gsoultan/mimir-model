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

// Package ref holds the reference kernels of the mim primitives
// (spec/primitives.md). The kernels favour clarity over speed: every element
// is decoded, computed and re-encoded one at a time, and every result is
// rounded once to the output dtype. They are checked against the
// conformance corpus in testdata/conformance, which is generated from an
// independent float64 oracle (tools/gen_conformance.py).
//
// The kernels run in IEEE mode for every float dtype: subnormal inputs and
// outputs are never flushed (spec/primitives.md section 3.4).
package ref

import (
	"encoding/binary"
	"fmt"

	mim "github.com/gsoultan/mimir-model"
)

// Tensor is a dense tensor: a dtype, a shape and the packed little-endian
// element bytes in row-major order, laid out as in a mim file.
type Tensor struct {
	DType mim.DType
	Shape []int
	Data  []byte
}

// NumElems returns the number of elements of shape.
func NumElems(shape []int) int {
	n := 1
	for _, s := range shape {
		n *= s
	}
	return n
}

// ByteSize returns the number of bytes a tensor of dtype dt and the given
// shape occupies, or an error when the shape is invalid for dt.
func ByteSize(dt mim.DType, shape []int) (int, error) {
	var buf [8]uint64 // no allocation for the usual ranks
	us := buf[:0]
	if len(shape) > len(buf) {
		us = make([]uint64, 0, len(shape))
	}
	us = us[:len(shape)]
	for i, s := range shape {
		if s < 0 {
			return 0, fmt.Errorf("negative dimension %d", s)
		}
		us[i] = uint64(s)
	}
	n, err := mim.ByteSize(dt, us)
	if err != nil {
		return 0, err
	}
	return int(n), nil
}

// New returns a zero-filled tensor.
func New(dt mim.DType, shape []int) (*Tensor, error) {
	n, err := ByteSize(dt, shape)
	if err != nil {
		return nil, err
	}
	return &Tensor{DType: dt, Shape: append([]int{}, shape...), Data: make([]byte, n)}, nil
}

// Validate checks that the tensor's data length matches its dtype and shape.
func (t *Tensor) Validate() error {
	n, err := ByteSize(t.DType, t.Shape)
	if err != nil {
		return err
	}
	if n != len(t.Data) {
		return fmt.Errorf("%s%v needs %d bytes, has %d", t.DType, t.Shape, n, len(t.Data))
	}
	return nil
}

// NumElems returns the number of logical elements of t.
func (t *Tensor) NumElems() int { return NumElems(t.Shape) }

// elemSize is the byte width of one element of a plain dtype.
func elemSize(dt mim.DType) int { return int(dt.BlockBytes()) }

// Bits returns element i as a zero-extended bit pattern.
func (t *Tensor) Bits(i int) uint64 { return loadBits(t.DType, t.Data, i) }

// SetBits stores the low bits of b as element i.
func (t *Tensor) SetBits(i int, b uint64) { storeBits(t.DType, t.Data, i, b) }

func loadBits(dt mim.DType, data []byte, i int) uint64 {
	switch elemSize(dt) {
	case 1:
		return uint64(data[i])
	case 2:
		return uint64(binary.LittleEndian.Uint16(data[2*i:]))
	case 4:
		return uint64(binary.LittleEndian.Uint32(data[4*i:]))
	default:
		return binary.LittleEndian.Uint64(data[8*i:])
	}
}

func storeBits(dt mim.DType, data []byte, i int, b uint64) {
	switch elemSize(dt) {
	case 1:
		data[i] = byte(b)
	case 2:
		binary.LittleEndian.PutUint16(data[2*i:], uint16(b))
	case 4:
		binary.LittleEndian.PutUint32(data[4*i:], uint32(b))
	default:
		binary.LittleEndian.PutUint64(data[8*i:], b)
	}
}

// strides returns the row-major element strides of shape.
func strides(shape []int) []int {
	st := make([]int, len(shape))
	acc := 1
	for i := len(shape) - 1; i >= 0; i-- {
		st[i] = acc
		acc *= shape[i]
	}
	return st
}

// unravel writes the multi-index of flat index i into idx.
func unravel(i int, shape, idx []int) {
	for k := len(shape) - 1; k >= 0; k-- {
		if shape[k] == 0 {
			idx[k] = 0
			continue
		}
		idx[k] = i % shape[k]
		i /= shape[k]
	}
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
