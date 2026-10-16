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

package importer

import (
	"encoding/binary"
	"fmt"
	"math"
	"unicode/utf8"

	"github.com/gsoultan/mmf"
)

// ggmlTypes maps ggml tensor types to mmf dtypes. The block types have the
// same byte layout in both formats (spec/dtypes.md section 3).
var ggmlTypes = map[uint32]mmf.DType{
	0:  mmf.F32,
	1:  mmf.F16,
	2:  mmf.Q4_0,
	8:  mmf.Q8_0,
	12: mmf.Q4_K,
	14: mmf.Q6_K,
	24: mmf.I8,
	25: mmf.I16,
	26: mmf.I32,
	27: mmf.I64,
	28: mmf.F64,
	30: mmf.BF16,
	39: mmf.MXFP4,
}

// ggmlNames names the ggml types mmf does not (yet) define, for errors.
var ggmlNames = map[uint32]string{
	3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
	13: "Q5_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
	19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
	29: "IQ1_M", 34: "TQ1_0", 35: "TQ2_0",
}

// GGUF value types.
const (
	gUint8 = iota
	gInt8
	gUint16
	gInt16
	gUint32
	gInt32
	gFloat32
	gBool
	gString
	gArray
	gUint64
	gInt64
	gFloat64
)

const (
	maxGGUFCount  = 1 << 24 // tensors or metadata keys
	maxGGUFString = 1 << 26
)

type ggufReader struct {
	b   []byte
	off uint64
	err error
}

func (r *ggufReader) fail(format string, args ...any) {
	if r.err == nil {
		r.err = fmt.Errorf("gguf: "+format, args...)
	}
}

func (r *ggufReader) take(n uint64) []byte {
	if r.err != nil {
		return nil
	}
	if n > uint64(len(r.b))-r.off {
		r.fail("truncated at offset %d (need %d bytes)", r.off, n)
		return nil
	}
	p := r.b[r.off : r.off+n]
	r.off += n
	return p
}

func (r *ggufReader) u8() uint8 {
	if p := r.take(1); p != nil {
		return p[0]
	}
	return 0
}

func (r *ggufReader) u16() uint16 {
	if p := r.take(2); p != nil {
		return binary.LittleEndian.Uint16(p)
	}
	return 0
}

func (r *ggufReader) u32() uint32 {
	if p := r.take(4); p != nil {
		return binary.LittleEndian.Uint32(p)
	}
	return 0
}

func (r *ggufReader) u64() uint64 {
	if p := r.take(8); p != nil {
		return binary.LittleEndian.Uint64(p)
	}
	return 0
}

func (r *ggufReader) str() string {
	n := r.u64()
	if n > maxGGUFString {
		r.fail("string of %d bytes at offset %d", n, r.off)
		return ""
	}
	p := r.take(n)
	if r.err == nil && !utf8.Valid(p) {
		r.fail("string at offset %d is not UTF-8", r.off-n)
	}
	return string(p)
}

// scalar reads one value of a scalar GGUF type, converted to the mmf
// metadata type it maps to: int64, uint64, float64, bool or string.
func (r *ggufReader) scalar(t uint32) any {
	switch t {
	case gUint8:
		return uint64(r.u8())
	case gInt8:
		return int64(int8(r.u8()))
	case gUint16:
		return uint64(r.u16())
	case gInt16:
		return int64(int16(r.u16()))
	case gUint32:
		return uint64(r.u32())
	case gInt32:
		return int64(int32(r.u32()))
	case gUint64:
		return r.u64()
	case gInt64:
		return int64(r.u64())
	case gFloat32:
		return float64(math.Float32frombits(r.u32()))
	case gFloat64:
		return math.Float64frombits(r.u64())
	case gBool:
		b := r.u8()
		if b > 1 {
			r.fail("bool value %d", b)
		}
		return b == 1
	case gString:
		return r.str()
	}
	r.fail("unknown value type %d", t)
	return nil
}

// scalarSize is the encoded size of a fixed-size scalar type, or 0.
func scalarSize(t uint32) uint64 {
	switch t {
	case gUint8, gInt8, gBool:
		return 1
	case gUint16, gInt16:
		return 2
	case gUint32, gInt32, gFloat32:
		return 4
	case gUint64, gInt64, gFloat64:
		return 8
	}
	return 0
}

// value reads a metadata value. ok is false for values mmf metadata cannot
// hold (nested arrays, or u64 arrays beyond the int64 range); they are
// skipped, not errors.
func (r *ggufReader) value(t uint32) (v any, ok bool) {
	if t != gArray {
		return r.scalar(t), true
	}
	et := r.u32()
	n := r.u64()
	if r.err != nil {
		return nil, false
	}
	// Every element takes at least one byte (eight for a string length), so
	// a count larger than the remaining input is malformed.
	min := scalarSize(et)
	if et == gString || et == gArray {
		min = 8
	}
	if min == 0 {
		r.fail("unknown array element type %d", et)
		return nil, false
	}
	if n > (uint64(len(r.b))-r.off)/min {
		r.fail("array of %d elements at offset %d exceeds the file", n, r.off)
		return nil, false
	}
	switch et {
	case gArray:
		for i := uint64(0); i < n && r.err == nil; i++ {
			r.value(gArray)
		}
		return nil, false
	case gString:
		s := make([]string, n)
		for i := range s {
			s[i] = r.str()
		}
		return s, true
	case gFloat32, gFloat64:
		f := make([]float64, n)
		for i := range f {
			f[i] = r.scalar(et).(float64)
		}
		return f, true
	case gUint8:
		return append([]byte(nil), r.take(n)...), true
	default: // integers and bools
		ok = true
		ints := make([]int64, n)
		for i := range ints {
			switch x := r.scalar(et).(type) {
			case int64:
				ints[i] = x
			case uint64:
				if x > math.MaxInt64 {
					ok = false
				}
				ints[i] = int64(x)
			case bool:
				if x {
					ints[i] = 1
				}
			}
		}
		return ints, ok
	}
}

// ParseGGUF parses a GGUF v2 or v3 file. The returned tensors alias b.
// Dimensions are reversed: GGUF lists the innermost dimension first, mmf
// the outermost.
func ParseGGUF(b []byte) (*Source, error) {
	r := &ggufReader{b: b}
	if string(r.take(4)) != "GGUF" {
		return nil, fmt.Errorf("gguf: bad magic")
	}
	version := r.u32()
	if r.err != nil {
		return nil, r.err
	}
	if version != 2 && version != 3 {
		if version&0xFFFF == 0 {
			return nil, fmt.Errorf("gguf: big-endian files are not supported")
		}
		return nil, fmt.Errorf("gguf: unsupported version %d", version)
	}
	nt, nkv := r.u64(), r.u64()
	// A metadata entry takes at least 13 bytes and a tensor entry at least
	// 24, so larger counts cannot fit and must not drive allocations.
	rest := uint64(len(b)) - r.off
	if nt > maxGGUFCount || nkv > maxGGUFCount || nkv > rest/13 || nt > rest/24 {
		return nil, fmt.Errorf("gguf: %d tensors and %d keys do not fit in the file", nt, nkv)
	}
	md := mmf.Metadata{}
	var skipped []string
	alignment := uint64(32)
	for i := uint64(0); i < nkv && r.err == nil; i++ {
		key := r.str()
		t := r.u32()
		v, ok := r.value(t)
		if r.err != nil {
			break
		}
		if key == "" || len(key) > mmf.MaxMetadataKeyLen {
			return nil, fmt.Errorf("gguf: invalid metadata key %q", key)
		}
		if _, dup := md[key]; dup {
			return nil, fmt.Errorf("gguf: duplicate metadata key %q", key)
		}
		if key == "general.alignment" {
			a, isU := v.(uint64)
			if !isU || a == 0 || a&(a-1) != 0 || a > 1<<20 {
				return nil, fmt.Errorf("gguf: invalid general.alignment %v", v)
			}
			alignment = a
		}
		if !ok {
			skipped = append(skipped, key)
			continue
		}
		md[key] = v
	}
	if r.err != nil {
		return nil, r.err
	}
	if len(skipped) > 0 {
		md["import.skipped_keys"] = skipped
	}
	md["import.gguf_version"] = int64(version)

	type info struct {
		t      Tensor
		offset uint64
		size   uint64
	}
	infos := make([]info, 0, nt)
	names := map[string]bool{}
	for i := uint64(0); i < nt && r.err == nil; i++ {
		name := r.str()
		nd := r.u32()
		if r.err != nil {
			break
		}
		if nd > mmf.MaxRank {
			return nil, fmt.Errorf("gguf: tensor %q has %d dimensions", name, nd)
		}
		shape := make([]uint64, nd)
		for k := int(nd) - 1; k >= 0; k-- {
			shape[k] = r.u64()
		}
		gt := r.u32()
		off := r.u64()
		if r.err != nil {
			break
		}
		if names[name] {
			return nil, fmt.Errorf("gguf: duplicate tensor %q", name)
		}
		names[name] = true
		dt, known := ggmlTypes[gt]
		if !known {
			if n, ok := ggmlNames[gt]; ok {
				return nil, fmt.Errorf("gguf: tensor %q has type %s, which mmf does not define", name, n)
			}
			return nil, fmt.Errorf("gguf: tensor %q has unknown type %d", name, gt)
		}
		size, err := mmf.ByteSize(dt, shape)
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor %q: %w", name, err)
		}
		if off%alignment != 0 {
			return nil, fmt.Errorf("gguf: tensor %q offset %d is not aligned to %d", name, off, alignment)
		}
		infos = append(infos, info{Tensor{Name: name, DType: dt, Shape: shape}, off, size})
	}
	if r.err != nil {
		return nil, r.err
	}
	start := (r.off + alignment - 1) / alignment * alignment
	if start > uint64(len(b)) {
		return nil, fmt.Errorf("gguf: tensor data starts past the end of the file")
	}
	data := b[start:]
	src := &Source{Format: "gguf", Metadata: md}
	for _, in := range infos {
		if in.offset > uint64(len(data)) || in.size > uint64(len(data))-in.offset {
			return nil, fmt.Errorf("gguf: tensor %q data [%d, +%d) is outside the file", in.t.Name, in.offset, in.size)
		}
		in.t.Data = data[in.offset : in.offset+in.size]
		if err := checkTensor(in.t); err != nil {
			return nil, fmt.Errorf("gguf: %w", err)
		}
		src.Tensors = append(src.Tensors, in.t)
	}
	return src, nil
}
