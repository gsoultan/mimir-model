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
	"unsafe"

	"github.com/gsoultan/mimir-model/ref"
)

// An arena hands out the buffers of the values a run of the main graph
// computes on the fast paths, and takes back the buffer of each value the
// run drops (lastUses), so that a later value of the same size, in this
// run or the next, reuses its memory. A small model allocates and clears
// most of its time otherwise.
//
// A buffer goes back only once no value of the run refers to it: the
// arena counts the values whose data lie in each buffer it handed out,
// views such as a reshape included. Outputs, variable updates and values
// a subgraph reads are never dropped, so their buffers stay with them.
// Ops in subgraphs allocate as usual.
//
// An arena serves one run at a time; a program keeps a few in a pool, for
// runs of different sessions at once.
type arena struct {
	// free holds the buffers given back, by length; prev, those given
	// back in the previous run, which a buffer is taken from first. Those
	// left in prev when a run ends are let go, so the arena keeps no more
	// than one run's worth of memory when shapes change from run to run.
	free, prev map[int][][]byte
	held       []heldBuf // buffers handed out in this run and not given back
}

// heldBuf is a buffer handed out and the number of values that refer to it.
// A run holds a few dozen at most, so a slice searched in order is quicker
// than a map.
type heldBuf struct {
	start, end uintptr // the buffer's addresses
	buf        []byte
	refs       int
}

// minArenaBytes is the smallest buffer an arena keeps track of; smaller
// ones cost less to allocate than to track.
const minArenaBytes = 256

func newArena() *arena {
	return &arena{free: map[int][][]byte{}, prev: map[int][][]byte{}}
}

// begin readies a for a run.
func (a *arena) begin() {
	a.prev, a.free = a.free, a.prev
	clear(a.free)
	a.forget()
}

// forget drops the buffers handed out, which no longer come back.
func (a *arena) forget() {
	clear(a.held) // let go of the buffers
	a.held = a.held[:0]
}

// end lets go of what a run left: the buffers it handed out that still
// back values, which are now the caller's, and those it did not reuse.
func (a *arena) end() {
	a.forget()
	clear(a.prev)
}

// bytes returns a buffer of n bytes, all zero when zero is set. With a
// nil arena it allocates.
func (a *arena) bytes(n int, zero bool) []byte {
	if a == nil || n < minArenaBytes {
		return make([]byte, n)
	}
	b := a.take(a.free, n)
	if b == nil {
		b = a.take(a.prev, n)
	}
	if b == nil {
		b = make([]byte, n)
	} else if zero {
		clear(b)
	}
	start := uintptr(unsafe.Pointer(&b[0]))
	a.held = append(a.held, heldBuf{start: start, end: start + uintptr(n), buf: b})
	return b
}

func (a *arena) take(free map[int][][]byte, n int) []byte {
	l := free[n]
	if len(l) == 0 {
		return nil
	}
	b := l[len(l)-1]
	free[n] = l[:len(l)-1]
	return b
}

// f32 is bytes for n float32s.
func (a *arena) f32(n int, zero bool) []float32 {
	b := a.bytes(4*n, zero)
	if len(b) == 0 {
		return []float32{}
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), n)
}

// find returns the index in held of the buffer that t's data lie in, or
// -1.
func (a *arena) find(t *ref.Tensor) int {
	if a == nil || len(t.Data) == 0 {
		return -1
	}
	at := uintptr(unsafe.Pointer(&t.Data[0]))
	for i := range a.held {
		if h := &a.held[i]; at >= h.start && at < h.end {
			return i
		}
	}
	return -1
}

// retain counts a value of the run whose tensor is t.
func (a *arena) retain(t *ref.Tensor) {
	if i := a.find(t); i >= 0 {
		a.held[i].refs++
	}
}

// release uncounts a value the run drops, giving its buffer back when no
// other value refers to it.
func (a *arena) release(t *ref.Tensor) {
	i := a.find(t)
	if i < 0 {
		return
	}
	h := &a.held[i]
	if h.refs--; h.refs > 0 {
		return
	}
	a.free[len(h.buf)] = append(a.free[len(h.buf)], h.buf)
	last := len(a.held) - 1
	a.held[i] = a.held[last]
	a.held[last] = heldBuf{}
	a.held = a.held[:last]
}
