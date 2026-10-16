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

package kernels

import (
	"encoding/binary"
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/gsoultan/mmf"
)

// fma32 returns a*b+c rounded once to float32.
func fma32(a, b, c float32) float32 {
	var p, s big.Float
	p.SetPrec(200).Mul(big.NewFloat(float64(a)).SetPrec(200), big.NewFloat(float64(b)))
	s.SetPrec(200).Add(&p, big.NewFloat(float64(c)))
	f, _ := s.Float32()
	return f
}

// dotQ8KRef is what the kernels for activation blocks of 256 compute, in
// plain Go: the exact integer sums of each block, then one fused
// multiply-add per block (see kernels_amd64.s and q8_arm64.s).
func dotQ8KRef(dt mmf.DType, row []byte, a *Q8) (float32, bool) {
	var tot, off float32
	f16 := func(b []byte) (float32, bool) {
		h := binary.LittleEndian.Uint16(b)
		return mmf.F16ToF32(h), h&0x7C00 != 0x7C00
	}
	for b := 0; b < a.k/256; b++ {
		y := a.q[256*b : 256*b+256]
		dx := a.d[8*b]
		if dt == mmf.Q4_K {
			blk := row[144*b : 144*b+144]
			d, ok1 := f16(blk[0:])
			dmin, ok2 := f16(blk[2:])
			if !ok1 || !ok2 {
				return 0, false
			}
			var sc, m [8]int32
			q := blk[4:16]
			for j := 0; j < 4; j++ {
				sc[j], m[j] = int32(q[j]&63), int32(q[j+4]&63)
				sc[4+j] = int32(q[j+8]&0xF | (q[j]>>6)<<4)
				m[4+j] = int32(q[j+8]>>4 | (q[j+4]>>6)<<4)
			}
			var s, mm int32
			for c := 0; c < 4; c++ {
				var lo, hi int32
				for i := 0; i < 32; i++ {
					v := blk[16+32*c+i]
					lo += int32(v&15) * int32(y[64*c+i])
					hi += int32(v>>4) * int32(y[64*c+32+i])
				}
				s += sc[2*c]*lo + sc[2*c+1]*hi
			}
			for j := 0; j < 8; j++ {
				mm += m[j] * int32(a.xs[8*b+j])
			}
			tot = fma32(float32(s), d*dx, tot)
			off = fma32(float32(mm), dmin*dx, off)
			continue
		}
		blk := row[210*b : 210*b+210]
		d, ok := f16(blk[208:])
		if !ok {
			return 0, false
		}
		var s int32
		for h := 0; h < 2; h++ {
			ql, qh, sc := blk[64*h:], blk[128+32*h:], blk[192+8*h:]
			yh := y[128*h:]
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int32(ql[l]&0xF|(qh[l]&3)<<4) - 32
				q2 := int32(ql[l+32]&0xF|(qh[l]>>2&3)<<4) - 32
				q3 := int32(ql[l]>>4|(qh[l]>>4&3)<<4) - 32
				q4 := int32(ql[l+32]>>4|(qh[l]>>6)<<4) - 32
				s += int32(int8(sc[is]))*q1*int32(yh[l]) +
					int32(int8(sc[is+2]))*q2*int32(yh[l+32]) +
					int32(int8(sc[is+4]))*q3*int32(yh[l+64]) +
					int32(int8(sc[is+6]))*q4*int32(yh[l+96])
			}
		}
		tot = fma32(float32(s), d*dx, tot)
	}
	return tot - off, true
}

// TestDotQ8KReference checks the kernels for activation blocks of 256
// against dotQ8KRef bit for bit, so that they return the same on every
// architecture that has them.
func TestDotQ8KReference(t *testing.T) {
	rng := rand.New(rand.NewSource(14))
	for _, dt := range []mmf.DType{mmf.Q4_K, mmf.Q6_K} {
		if !FusedQ8Blocks(dt, 256) {
			t.Skip("no 8-bit kernels for blocks of 256 on this machine")
		}
		for _, k := range []int{256, 2048, 5632} {
			const n = 5
			var b Q8Batch
			b.Reset(k, n)
			for i := 0; i < n; i++ {
				x := randomVec(rng, k)
				switch i {
				case 1:
					for j := range x {
						x[j] *= 1e-30 // tiny scales
					}
				case 2:
					for j := range x {
						x[j] *= 1e15
					}
				case 3:
					clear(x[:256]) // a zero block
				}
				if !b.Row(i).QuantizeBlocks(x, 256) {
					t.Fatal("QuantizeBlocks refused")
				}
			}
			for trial := 0; trial < 4; trial++ {
				row := randomRows(rng, dt, 1, k)
				if trial == 1 {
					// The largest finite scales, and all-ones quants.
					size := int(dt.BlockBytes())
					for o := 0; o < len(row); o += size {
						if dt == mmf.Q4_K {
							row[o], row[o+1], row[o+2], row[o+3] = 0xFF, 0x7B, 0xFF, 0x7B
							for i := o + 4; i < o+144; i++ {
								row[i] = 0xFF
							}
						} else {
							row[o+208], row[o+209] = 0xFF, 0x7B
							for i := o; i < o+192; i++ {
								row[i] = 0xFF
							}
							for i := o + 192; i < o+208; i++ {
								row[i] = 0x80 // -128
							}
						}
					}
				}
				if trial == 2 {
					// Subnormal f16 scales.
					if dt == mmf.Q4_K {
						row[0], row[1], row[2], row[3] = 0x01, 0x00, 0xFF, 0x03
					} else {
						row[208], row[209] = 0x01, 0x00
					}
				}
				r := NewRowDot(dt, k)
				r.LoadQ8(row)
				for i := 0; i < n; i++ {
					want, wok := dotQ8KRef(dt, row, b.Row(i))
					got, ok := r.DotQ8(b.Row(i))
					if ok != wok || ok && math.Float32bits(got) != math.Float32bits(want) {
						t.Errorf("%s K=%d trial %d row %d: got %g (%v), want %g (%v)", dt, k, trial, i, got, ok, want, wok)
					}
					if i+4 <= n {
						var out [4]float32
						ok4 := r.DotQ8x4(&b, i, &out)
						for j := 0; j < 4; j++ {
							w, wok := dotQ8KRef(dt, row, b.Row(i+j))
							if ok4 != wok || ok4 && math.Float32bits(out[j]) != math.Float32bits(w) {
								t.Errorf("%s K=%d trial %d rows %d+%d: DotQ8x4 %g (%v), want %g (%v)", dt, k, trial, i, j, out[j], ok4, w, wok)
							}
						}
					}
				}
			}
		}
	}
}
