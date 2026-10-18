// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mlm, created and owned by Gembit Soultan Shirazi.
// It is licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License in the LICENSE file at the root
// of this repository or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

package mlm

import (
	"encoding/binary"
	"math"
)

// This file decodes every dtype to float32 and holds reference encoders for a
// few block dtypes. The block layouts are ggml's (little-endian, packed) and
// the decoders reproduce ggml's dequantize_row_* bit for bit, including the
// order in which the float32 operations are applied. Explicit float32
// conversions in the arithmetic below stop the compiler from fusing a
// multiply and an add into an FMA, which would change the low bit.

// Dequantize decodes len(dst) elements of dtype dt from src into dst.
//
// src must hold exactly the bytes those elements occupy (for block dtypes
// len(dst) must be a multiple of the block size); otherwise CodeSizeMismatch
// is returned. Dtypes this function cannot decode return CodeUnsupportedDType
// and unknown dtypes CodeBadDType. Integer dtypes are converted to float32
// with Go's conversion rules (64-bit integers and F64 may lose precision).
func Dequantize(dt DType, src []byte, dst []float32) error {
	info, ok := dtypes[dt]
	if !ok {
		return errf(CodeBadDType, "unknown dtype %d", uint32(dt))
	}
	n := uint64(len(dst))
	if n%info.blockElems != 0 {
		return errf(CodeSizeMismatch, "%s: %d elements is not a multiple of the block size %d", info.name, n, info.blockElems)
	}
	if want := n / info.blockElems * info.blockBytes; uint64(len(src)) != want {
		return errf(CodeSizeMismatch, "%s: %d elements need %d bytes, got %d", info.name, n, want, len(src))
	}
	switch dt {
	case F32:
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[4*i:]))
		}
	case F16:
		for i := range dst {
			dst[i] = F16ToF32(binary.LittleEndian.Uint16(src[2*i:]))
		}
	case BF16:
		for i := range dst {
			dst[i] = BF16ToF32(binary.LittleEndian.Uint16(src[2*i:]))
		}
	case F64:
		for i := range dst {
			dst[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(src[8*i:])))
		}
	case I8:
		for i := range dst {
			dst[i] = float32(int8(src[i]))
		}
	case U8:
		for i := range dst {
			dst[i] = float32(src[i])
		}
	case I16:
		for i := range dst {
			dst[i] = float32(int16(binary.LittleEndian.Uint16(src[2*i:])))
		}
	case U16:
		for i := range dst {
			dst[i] = float32(binary.LittleEndian.Uint16(src[2*i:]))
		}
	case I32:
		for i := range dst {
			dst[i] = float32(int32(binary.LittleEndian.Uint32(src[4*i:])))
		}
	case U32:
		for i := range dst {
			dst[i] = float32(binary.LittleEndian.Uint32(src[4*i:]))
		}
	case I64:
		for i := range dst {
			dst[i] = float32(int64(binary.LittleEndian.Uint64(src[8*i:])))
		}
	case U64:
		for i := range dst {
			dst[i] = float32(binary.LittleEndian.Uint64(src[8*i:]))
		}
	case Bool:
		for i := range dst {
			if src[i] != 0 {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
	case F8E4M3:
		for i := range dst {
			dst[i] = F8E4M3ToF32(src[i])
		}
	case F8E5M2:
		for i := range dst {
			dst[i] = F8E5M2ToF32(src[i])
		}
	case Q8_0:
		dequantizeQ8_0(src, dst)
	case Q4_0:
		dequantizeQ4_0(src, dst)
	case Q4_K:
		dequantizeQ4_K(src, dst)
	case Q6_K:
		dequantizeQ6_K(src, dst)
	case MXFP4:
		dequantizeMXFP4(src, dst)
	case Affine4:
		dequantizeAffine4(src, dst)
	default:
		return errf(CodeUnsupportedDType, "cannot decode %s", info.name)
	}
	return nil
}

// Quantize encodes src into dst as dtype dt. Only Q8_0, Q4_0 and Affine4 are
// supported; they use the reference (non-optimal) quantizers: ggml's
// quantize_row_q8_0_ref and quantize_row_q4_0_ref, and for Affine4 the MLX
// affine min/max fit. Other dtypes return CodeUnsupportedDType. len(src) must
// be a multiple of the block size and len(dst) the matching byte size;
// otherwise CodeSizeMismatch is returned.
func Quantize(dt DType, src []float32, dst []byte) error {
	info, ok := dtypes[dt]
	if !ok {
		return errf(CodeBadDType, "unknown dtype %d", uint32(dt))
	}
	n := uint64(len(src))
	if n%info.blockElems != 0 {
		return errf(CodeSizeMismatch, "%s: %d elements is not a multiple of the block size %d", info.name, n, info.blockElems)
	}
	if want := n / info.blockElems * info.blockBytes; uint64(len(dst)) != want {
		return errf(CodeSizeMismatch, "%s: %d elements need %d bytes, got %d", info.name, n, want, len(dst))
	}
	switch dt {
	case Q8_0:
		quantizeQ8_0(src, dst)
	case Q4_0:
		quantizeQ4_0(src, dst)
	case Affine4:
		quantizeAffine4(src, dst)
	default:
		return errf(CodeUnsupportedDType, "cannot encode %s", info.name)
	}
	return nil
}

// Q8_0: f16 d; i8 qs[32]. y = d*q.
func dequantizeQ8_0(src []byte, dst []float32) {
	for b := 0; b < len(dst)/32; b++ {
		blk := src[34*b : 34*b+34]
		d := F16ToF32(binary.LittleEndian.Uint16(blk))
		y := dst[32*b : 32*b+32]
		for j := range y {
			y[j] = d * float32(int8(blk[2+j]))
		}
	}
}

func quantizeQ8_0(src []float32, dst []byte) {
	for b := 0; b < len(src)/32; b++ {
		x := src[32*b : 32*b+32]
		var amax float32
		for _, v := range x {
			if a := absf32(v); a > amax {
				amax = a
			}
		}
		d := amax / 127
		var id float32
		if d != 0 {
			id = 1 / d
		}
		blk := dst[34*b : 34*b+34]
		binary.LittleEndian.PutUint16(blk, F32ToF16(d))
		for j, v := range x {
			blk[2+j] = byte(int8(roundf32(v * id)))
		}
	}
}

// Q4_0: f16 d; u8 qs[16]. Element j is the low nibble of qs[j], element j+16
// the high nibble. y = d*(q-8).
func dequantizeQ4_0(src []byte, dst []float32) {
	for b := 0; b < len(dst)/32; b++ {
		blk := src[18*b : 18*b+18]
		d := F16ToF32(binary.LittleEndian.Uint16(blk))
		y := dst[32*b : 32*b+32]
		qs := blk[2:18]
		for j := 0; j < 16; j++ {
			y[j] = d * float32(int8(qs[j]&0xF)-8)
			y[j+16] = d * float32(int8(qs[j]>>4)-8)
		}
	}
}

func quantizeQ4_0(src []float32, dst []byte) {
	for b := 0; b < len(src)/32; b++ {
		x := src[32*b : 32*b+32]
		var amax, max float32
		for _, v := range x {
			if a := absf32(v); a > amax {
				amax, max = a, v
			}
		}
		d := max / -8
		var id float32
		if d != 0 {
			id = 1 / d
		}
		blk := dst[18*b : 18*b+18]
		binary.LittleEndian.PutUint16(blk, F32ToF16(d))
		qs := blk[2:18]
		for j := 0; j < 16; j++ {
			x0 := x[j] * id
			x1 := x[j+16] * id
			xi0 := min(uint8(int8(x0+8.5)), 15)
			xi1 := min(uint8(int8(x1+8.5)), 15)
			qs[j] = xi0 | xi1<<4
		}
	}
}

// getScaleMinK4 unpacks the 6-bit scale and min of sub-block j (0..7) from
// the 12 packed scale bytes of a Q4_K block (ggml's get_scale_min_k4).
func getScaleMinK4(j int, q []byte) (sc, m uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	sc = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
	m = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
	return sc, m
}

// Q4_K: f16 d; f16 dmin; u8 scales[12]; u8 qs[128]. 256 elements in eight
// sub-blocks of 32 with 6-bit scales and mins. For each 64-element step
// (sub-block pair 2i, 2i+1 over qs[32i:32i+32]) the low nibbles form
// sub-block 2i and the high nibbles sub-block 2i+1; y = d*sc*q - dmin*m.
func dequantizeQ4_K(src []byte, dst []float32) {
	for b := 0; b < len(dst)/256; b++ {
		blk := src[144*b : 144*b+144]
		d := F16ToF32(binary.LittleEndian.Uint16(blk[0:]))
		dmin := F16ToF32(binary.LittleEndian.Uint16(blk[2:]))
		scales := blk[4:16]
		q := blk[16:144]
		y := dst[256*b : 256*b+256]
		is := 0
		for j := 0; j < 256; j += 64 {
			sc, m := getScaleMinK4(is, scales)
			d1 := d * float32(sc)
			m1 := dmin * float32(m)
			sc, m = getScaleMinK4(is+1, scales)
			d2 := d * float32(sc)
			m2 := dmin * float32(m)
			for l := 0; l < 32; l++ {
				y[j+l] = float32(d1*float32(q[l]&0xF)) - m1
			}
			for l := 0; l < 32; l++ {
				y[j+32+l] = float32(d2*float32(q[l]>>4)) - m2
			}
			q = q[32:]
			is += 2
		}
	}
}

// Q6_K: u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d. 256 elements with
// 6-bit quants (4 low bits in ql, 2 high bits in qh) and 8-bit scales per 16
// elements. y = d*sc*(q-32), following ggml's dequantize_row_q6_K.
func dequantizeQ6_K(src []byte, dst []float32) {
	for b := 0; b < len(dst)/256; b++ {
		blk := src[210*b : 210*b+210]
		ql := blk[0:128]
		qh := blk[128:192]
		sc := blk[192:208]
		d := F16ToF32(binary.LittleEndian.Uint16(blk[208:]))
		y := dst[256*b : 256*b+256]
		for n := 0; n < 256; n += 128 {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int8((ql[l]&0xF)|((qh[l]>>0)&3)<<4) - 32
				q2 := int8((ql[l+32]&0xF)|((qh[l]>>2)&3)<<4) - 32
				q3 := int8((ql[l]>>4)|((qh[l]>>4)&3)<<4) - 32
				q4 := int8((ql[l+32]>>4)|((qh[l]>>6)&3)<<4) - 32
				y[l+0] = float32(d*float32(int8(sc[is+0]))) * float32(q1)
				y[l+32] = float32(d*float32(int8(sc[is+2]))) * float32(q2)
				y[l+64] = float32(d*float32(int8(sc[is+4]))) * float32(q3)
				y[l+96] = float32(d*float32(int8(sc[is+6]))) * float32(q4)
			}
			y = y[128:]
			ql = ql[64:]
			qh = qh[32:]
			sc = sc[8:]
		}
	}
}

// kvaluesMXFP4 is ggml's E2M1 table with every value doubled; it is used with
// the half scale so that the decoder is bit-identical to ggml's.
var kvaluesMXFP4 = [16]int8{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

// e8m0ToF32Half returns 2^(e-127)/2 the way ggml_e8m0_to_fp32_half does,
// including its treatment of 0xFF as a finite scale (2^127 here, i.e. a
// full scale of 2^128) rather than NaN.
func e8m0ToF32Half(e uint8) float32 {
	if e < 2 {
		return math.Float32frombits(0x00200000 << e)
	}
	return math.Float32frombits(uint32(e-1) << 23)
}

// MXFP4: u8 e (E8M0 shared scale); u8 qs[16]. Element j is the low nibble of
// qs[j], element j+16 the high nibble, each an E2M1 value. y = 2^(e-127)*E2M1(q),
// computed as ggml does (half scale times the doubled table).
func dequantizeMXFP4(src []byte, dst []float32) {
	for b := 0; b < len(dst)/32; b++ {
		blk := src[17*b : 17*b+17]
		d := e8m0ToF32Half(blk[0])
		qs := blk[1:17]
		y := dst[32*b : 32*b+32]
		for j := 0; j < 16; j++ {
			y[j] = d * float32(kvaluesMXFP4[qs[j]&0xF])
			y[j+16] = d * float32(kvaluesMXFP4[qs[j]>>4])
		}
	}
}

// Affine4 (affine4_g64): f16 scale; f16 bias; u8 qs[32]. 64 elements; element
// j is the low nibble of qs[j], element j+32 the high nibble, q unsigned.
// y = scale*q + bias. This is MLX's affine 4-bit, group size 64, repacked.
func dequantizeAffine4(src []byte, dst []float32) {
	for b := 0; b < len(dst)/64; b++ {
		blk := src[36*b : 36*b+36]
		scale := F16ToF32(binary.LittleEndian.Uint16(blk[0:]))
		bias := F16ToF32(binary.LittleEndian.Uint16(blk[2:]))
		qs := blk[4:36]
		y := dst[64*b : 64*b+64]
		for j := 0; j < 32; j++ {
			y[j] = float32(scale*float32(qs[j]&0xF)) + bias
			y[j+32] = float32(scale*float32(qs[j]>>4)) + bias
		}
	}
}

func quantizeAffine4(src []float32, dst []byte) {
	for b := 0; b < len(src)/64; b++ {
		x := src[64*b : 64*b+64]
		lo, hi := x[0], x[0]
		for _, v := range x[1:] {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		// Round the parameters through f16 first so the quants are chosen
		// against the values the decoder will actually use.
		scale := F16ToF32(F32ToF16((hi - lo) / 15))
		bias := F16ToF32(F32ToF16(lo))
		var inv float32
		if scale != 0 {
			inv = 1 / scale
		}
		blk := dst[36*b : 36*b+36]
		binary.LittleEndian.PutUint16(blk[0:], F32ToF16(scale))
		binary.LittleEndian.PutUint16(blk[2:], F32ToF16(bias))
		qs := blk[4:36]
		for j := 0; j < 32; j++ {
			qs[j] = affineQ(x[j], bias, inv) | affineQ(x[j+32], bias, inv)<<4
		}
	}
}

func affineQ(x, bias, inv float32) uint8 {
	q := roundf32((x - bias) * inv)
	if q < 0 {
		return 0
	}
	if q > 15 {
		return 15
	}
	return uint8(q)
}

func absf32(x float32) float32 {
	return math.Float32frombits(math.Float32bits(x) &^ (1 << 31))
}

// roundf32 rounds half away from zero, like C's roundf.
func roundf32(x float32) float32 {
	return float32(math.Round(float64(x)))
}
