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
	"unsafe"

	"github.com/gsoultan/mmf"
)

// AVX-512 VNNI kernels for activations in blocks of 256, for CPUs that
// have VNNI but not AMX (AMD Zen 4, Intel Xeon since Cascade Lake). They
// take the same operands as the AMX kernels and give the same results:
// VPDPBUSD multiplies four unsigned weight quants, broadcast to every
// lane, by four quants of each of the 16 activation rows of a group (one
// row of TileActs.q) and adds the four products into each lane's int32
// sum. Each sub-block's sums are scaled in int32, so a block ends with the
// integer sums of the one-row kernels, and the same float operations
// follow.

// vnniArgs are the arguments of the VNNI kernels in vnni_amd64.s, which
// read the fields at fixed offsets: keep the order.
type vnniArgs struct {
	w       *byte    // the first weight row
	wstride int      // bytes from one weight row to the next
	nb      int      // blocks of 256 in a row
	groups  int      // groups of 16 activation rows
	act     *int8    // TileActs.q
	gstride int      // bytes from one group's quants to the next
	xs      *byte    // Q4_K: TileActs.xs; Q6_K: TileActs.x32
	xsg     int      // bytes from one group's sums to the next
	dx      *float32 // TileActs.dx
	dxg     int      // bytes from one group's scales to the next
	tot     *float32 // the totals, [group][weight row][activation row]
	off     *float32 // the Q4_K offsets, laid out as tot
	sc      *int32   // the sub-block scales, [block][weight row][sub-block]
	mp      *int16   // the Q4_K sub-block offsets, [block][weight row][8]
	dd      *float32 // d and, for Q4_K, dmin, [block][weight row][2]
	scr     *byte    // scratch space for the quants of four weight rows' blocks, 1 KiB
}

// vnniPrep unpacks the scales of 16 rows of weights for the VNNI kernels.
// It reports false if a block scale is not finite.
func (d *TileDot) vnniPrep(w []byte) bool {
	nb := d.k / 256
	bb := int(d.dt.BlockBytes())
	for b := range nb {
		for r := range TileRows {
			blk := w[r*d.rowBytes+b*bb:][:bb]
			at := b*TileRows + r
			if d.dt == mmf.Q4_K {
				dv := mmf.F16ToF32(binary.LittleEndian.Uint16(blk[0:]))
				dmin := mmf.F16ToF32(binary.LittleEndian.Uint16(blk[2:]))
				if !finite(dv) || !finite(dmin) {
					return false
				}
				d.dd[2*at], d.dd[2*at+1] = dv, dmin
				q, sc, mp := blk[4:16], d.sc[8*at:8*at+8], d.mp[8*at:8*at+8]
				for j := range 4 {
					sc[j] = int32(q[j] & 63)
					mp[j] = int16(q[j+4] & 63)
					sc[4+j] = int32(q[j+8]&0xF | (q[j]>>6)<<4)
					mp[4+j] = int16(q[j+8]>>4 | (q[j+4]>>6)<<4)
				}
				continue
			}
			dv := mmf.F16ToF32(binary.LittleEndian.Uint16(blk[208:]))
			if !finite(dv) {
				return false
			}
			d.dd[2*at] = dv
			sc := d.sc[16*at : 16*at+16]
			for j := range 16 {
				sc[j] = int32(int8(blk[192+j]))
			}
		}
	}
	return true
}

func finite(f float32) bool { return !math.IsInf(float64(f), 0) && !math.IsNaN(float64(f)) }

// vnniDot runs the VNNI kernel of d's dtype over the 16 weight rows w,
// adding to tot and off as the AMX kernels do.
func (d *TileDot) vnniDot(w []byte, a *TileActs, tot, off []float32) bool {
	nb := d.k / 256
	if len(d.sc) < nb*TileRows*16 {
		d.sc = make([]int32, nb*TileRows*16)
		d.mp = make([]int16, nb*TileRows*8)
		d.dd = make([]float32, nb*TileRows*2)
	}
	if !d.vnniPrep(w) {
		return false
	}
	args := vnniArgs{
		w: &w[0], wstride: d.rowBytes, nb: nb, groups: a.groups,
		act: &a.q[0], gstride: TileRows * d.k,
		dx: &a.dx[0], dxg: nb * TileRows * 4,
		tot: &tot[0], off: &off[0],
		sc: &d.sc[0], mp: &d.mp[0], dd: &d.dd[0], scr: &d.scr[0],
	}
	if d.dt == mmf.Q4_K {
		args.xs, args.xsg = (*byte)(unsafe.Pointer(&a.xs[0])), nb*4*TileRows*2*2
		vnniQ4K(&args)
	} else {
		args.xs, args.xsg = (*byte)(unsafe.Pointer(&a.x32[0])), d.k/16*TileRows*4
		vnniQ6K(&args)
	}
	return true
}
