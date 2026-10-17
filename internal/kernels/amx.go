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
	"unsafe"

	"github.com/gsoultan/mmf"
)

// AMX kernels for activations in blocks of 256. On CPUs with Intel AMX
// (Xeon since Sapphire Rapids), a tile instruction multiplies a 16-row
// matrix of bytes by another and adds the int32 products into a 16 by 16
// tile. The kernels multiply 16 rows of weights by groups of 16 rows of
// activations at a time, one tile product per weight sub-block, and scale
// the sub-blocks' integer sums in int32. Each weight block then ends with
// the integer sums of the one-row kernel (dotQ4KQ8KAVX2, dotQ6KQ8KAVX2),
// and the same float operations follow, so the results are those of
// RowDot.DotQ8, bit for bit.

// TileRows is the number of weight rows, and of activation rows in a
// group, that the AMX kernels multiply at once.
const TileRows = 16

// TileQ8 reports whether rows of dt have AMX or VNNI kernels for
// activations in blocks of 256 (TileDot).
func TileQ8(dt mmf.DType) bool {
	return (useAMX || useVNNI) && (dt == mmf.Q4_K || dt == mmf.Q6_K)
}

// TileMinRows is the number of rows of activations from which the tile
// kernels are faster than the four-row kernels: an AMX tile product costs
// the same for 1 row as for 16, and a VNNI instruction works on 16 rows
// whether or not they are all used.
func TileMinRows() int {
	if useAMX && !noAMX {
		return 8
	}
	return 12
}

// noAMX makes TileDot use the VNNI kernels on a CPU with AMX, for tests.
var noAMX = false

// tileArgs are the arguments of the AMX kernels in amx_amd64.s, which read
// the fields at fixed offsets: keep the order.
type tileArgs struct {
	w       *byte    // the first weight row
	wstride int      // bytes from one weight row to the next
	nb      int      // blocks of 256 in a row
	groups  int      // groups of 16 activation rows
	act     *int8    // TileActs.q
	gstride int      // bytes from one group's quants to the next
	xs      *int16   // TileActs.xs
	xsg     int      // bytes from one group's sums to the next
	dx      *float32 // TileActs.dx
	dxg     int      // bytes from one group's scales to the next
	tot     *float32 // the totals, [group][weight row][activation row]
	off     *float32 // the Q4_K offsets, laid out as tot
	scr     *byte    // scratch space, tileScratch bytes
}

// TileActs holds rows of activations quantized in blocks of 256, in groups
// of 16 rows laid out for the tile products: each group's quants as rows
// of 64 bytes holding four quants of each of the 16 rows.
type TileActs struct {
	k, rows, groups int
	q               []int8    // [group][k/4][16][4]
	xs              []int16   // [group][block][pair][16][2], sums of each run of 32
	dx              []float32 // [group][block][16], the block scales
	x32             []int32   // [group][k/16][16], 32 times the sum of each run of 16 (VNNI, Q6_K)
	b               *Q8Batch  // the rows, between Reset and SetGroups
	use             []bool
}

// Set lays out the rows i of b for which use[i] is true, which must have
// been quantized in blocks of 256; the other rows are zeros, and the
// caller computes them some other way.
func (t *TileActs) Set(b *Q8Batch, use []bool) {
	t.Reset(b, use)
	t.SetGroups(0, t.groups)
}

// Reset prepares t for the rows of b, which Set or SetGroups then lays
// out. It keeps t's buffers when they are large enough, so a TileActs can
// be reused for calls of any size.
func (t *TileActs) Reset(b *Q8Batch, use []bool) {
	k, rows := b.k, len(b.rows)
	if k%256 != 0 || len(use) != rows {
		panic("kernels: TileActs.Set with activations of another shape")
	}
	nb := k / 256
	t.b, t.use = b, use
	t.k, t.rows, t.groups = k, rows, (rows+TileRows-1)/TileRows
	t.q = grow(t.q, t.groups*TileRows*k)
	t.xs = grow(t.xs, t.groups*nb*4*TileRows*2)
	t.dx = grow(t.dx, t.groups*nb*TileRows)
	// Only the VNNI kernels for Q6_K read the sums of 16.
	t.x32 = t.x32[:0]
	if !useAMX || noAMX {
		t.x32 = grow(t.x32, t.groups*k/16*TileRows)
	}
}

func grow[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n)
	}
	return s[:n]
}

// Groups returns the number of groups of 16 rows, after Reset.
func (t *TileActs) Groups() int { return t.groups }

// SetGroups lays out the rows of groups lo to hi-1 as Set does, after
// Reset. Calls for groups that do not overlap may run concurrently; each
// group's layout interleaves its 16 rows, so splitting a group between
// goroutines would have them write the same cache lines.
func (t *TileActs) SetGroups(lo, hi int) {
	k, nb := t.k, t.k/256
	for g := lo; g < hi; g++ {
		q := unsafe.Slice((*uint32)(unsafe.Pointer(&t.q[g*TileRows*k])), TileRows*k/4)
		for m := range TileRows {
			i := g*TileRows + m
			if i >= t.rows || !t.use[i] {
				for k4 := m; k4 < len(q); k4 += TileRows {
					q[k4] = 0
				}
				for gb := g * nb; gb < (g+1)*nb; gb++ {
					for p := range 4 {
						t.xs[(gb*4+p)*32+2*m] = 0
						t.xs[(gb*4+p)*32+2*m+1] = 0
					}
					t.dx[gb*TileRows+m] = 0
				}
				if len(t.x32) > 0 {
					for s := range k / 16 {
						t.x32[(g*k/16+s)*TileRows+m] = 0
					}
				}
				continue
			}
			r := &t.b.rows[i]
			if !r.ok || r.block != 256 {
				panic("kernels: TileActs.Set with a row that is not quantized in blocks of 256")
			}
			rq := unsafe.Slice((*uint32)(unsafe.Pointer(&r.q[0])), k/4)
			qm := q[m:]
			for k4, v := range rq {
				qm[k4*TileRows] = v
			}
			for blk := range nb {
				gb := g*nb + blk
				for p := range 4 {
					t.xs[(gb*4+p)*32+2*m] = r.xs[8*blk+2*p]
					t.xs[(gb*4+p)*32+2*m+1] = r.xs[8*blk+2*p+1]
				}
				t.dx[gb*TileRows+m] = r.d[8*blk]
			}
			if len(t.x32) > 0 {
				x32 := t.x32[g*k/16*TileRows:]
				for s := range k / 16 {
					var sum int32
					for _, v := range r.q[16*s : 16*s+16 : 16*s+16] {
						sum += int32(v)
					}
					x32[s*TileRows+m] = 32 * sum
				}
			}
		}
	}
}

// tileScratch is the size of the AMX kernels' scratch space, laid out as
// amx_amd64.s says, and tileConfig the offset of the tile configuration in
// it.
const tileScratch, tileConfig = 12800, 12736

// TileDot multiplies 16 rows of weights by the rows of a TileActs with
// the AMX kernels, or the VNNI ones (vnni.go) on CPUs without AMX. It
// holds scratch space, so each goroutine needs its own.
type TileDot struct {
	dt       mmf.DType
	k        int
	rowBytes int
	mem      []byte
	scr      []byte
	tot, off []float32
	sc       []int32   // VNNI: the sub-block scales
	mp       []int16   // VNNI: the Q4_K sub-block offsets
	dd       []float32 // VNNI: d and dmin
}

// NewTileDot returns a TileDot for rows of k elements of dt, for which
// TileQ8 must be true.
func NewTileDot(dt mmf.DType, k int) *TileDot {
	d := &TileDot{}
	d.Reset(dt, k)
	return d
}

// Reset makes d a TileDot for rows of k elements of dt, as NewTileDot
// does, keeping its buffers.
func (d *TileDot) Reset(dt mmf.DType, k int) {
	if !TileQ8(dt) || k%256 != 0 {
		panic("kernels: NewTileDot without a tile kernel")
	}
	d.dt, d.k, d.rowBytes = dt, k, k/256*int(dt.BlockBytes())
	if d.mem != nil {
		return
	}
	d.mem = make([]byte, tileScratch+64)
	o := int(-uintptr(unsafe.Pointer(&d.mem[0])) & 63)
	d.scr = d.mem[o : o+tileScratch]
	// Palette 1, every tile 16 rows of 64 bytes.
	cfg := d.scr[tileConfig:]
	cfg[0] = 1
	for t := range 8 {
		binary.LittleEndian.PutUint16(cfg[16+2*t:], 64)
		cfg[48+t] = 16
	}
}

// Dot sets y[i*stride+r] to the dot product of weight row r of w, which
// holds 16 rows, with row i of a, plus bias[r] if bias is not nil, for
// each row i < a's row count. The results for the rows a was given are
// those of RowDot.DotQ8 (plus the bias), bit for bit; the others are
// meaningless. It returns false, and the caller must compute the 16 rows
// some other way, if a block scale of one of them is not finite.
func (d *TileDot) Dot(w []byte, a *TileActs, y []float32, stride int, bias []float32) bool {
	if a.k != d.k || len(w) < TileRows*d.rowBytes || (a.rows > 0 && len(y) < (a.rows-1)*stride+TileRows) ||
		(bias != nil && len(bias) < TileRows) {
		panic("kernels: TileDot.Dot with operands of another shape")
	}
	nb := d.k / 256
	n := a.groups * TileRows * TileRows
	if cap(d.tot) < n {
		d.tot, d.off = make([]float32, n), make([]float32, n)
	}
	tot, off := d.tot[:n], d.off[:n]
	clear(tot)
	clear(off)
	args := tileArgs{
		w: &w[0], wstride: d.rowBytes, nb: nb, groups: a.groups,
		act: &a.q[0], gstride: TileRows * d.k,
		xs: &a.xs[0], xsg: nb * 4 * TileRows * 2 * 2,
		dx: &a.dx[0], dxg: nb * TileRows * 4,
		tot: &tot[0], off: &off[0], scr: &d.scr[0],
	}
	var ok bool
	switch {
	case !useAMX || noAMX:
		ok = d.vnniDot(w, a, tot, off)
	case d.dt == mmf.Q4_K:
		ok = amxQ4K(&args)
	default:
		ok = amxQ6K(&args)
	}
	if !ok {
		return false
	}
	// tot and off hold [group][weight row][activation row]: each weight
	// row's results are read in order and written down a column of y.
	for g := range a.groups {
		ms := min(TileRows, a.rows-g*TileRows)
		yg := y[g*TileRows*stride:]
		for r := range TileRows {
			t := tot[(g*TileRows+r)*TileRows:][:ms]
			f := off[(g*TileRows+r)*TileRows:][:ms]
			if bias == nil {
				for m, v := range t {
					yg[m*stride+r] = v - f[m]
				}
				continue
			}
			b := bias[r]
			for m, v := range t {
				yg[m*stride+r] = v - f[m] + b
			}
		}
	}
	return true
}
