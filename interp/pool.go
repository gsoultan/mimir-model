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

package interp

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// parallel runs f(lo, hi) over [0, n) on the available CPUs. The range is
// cut into many more pieces than there are CPUs and the workers take them
// in turn, so a slow or busy CPU does not hold the others up.
//
// A forward pass calls parallel a few hundred times per token, each for a
// fraction of a millisecond, so the workers are kept between calls: they
// spin for a short while after each call, waiting for the next one, and
// then sleep. The calling goroutine takes pieces too. When the pool is
// already in use (two sessions running at once, or parallel called from
// inside f) the call starts goroutines of its own instead.
func parallel(n int, f func(lo, hi int)) {
	w := min(runtime.GOMAXPROCS(0), n)
	if w <= 1 {
		f(0, n)
		return
	}
	j := &job{f: f, n: n, chunk: max(1, n/(w*parallelPieces))}
	if !workers.mu.TryLock() {
		j.spawn(w)
		return
	}
	defer workers.mu.Unlock()
	workers.start(w - 1)
	workers.post(j)
	j.run()
	for j.done.Load() < int64(n) {
	}
	workers.cur.Store(nil) // f may hold large buffers
}

// parallelPieces is the number of pieces per CPU.
const parallelPieces = 8

// spinFor is how long an idle worker waits for the next call before it
// sleeps. Between two calls of one forward pass the caller runs only
// small serial steps, well under this.
const spinFor = 200_000 // nanoseconds

// A job is one call of parallel.
type job struct {
	f     func(lo, hi int)
	n     int
	chunk int
	next  atomic.Int64 // start of the next piece to take
	done  atomic.Int64 // elements finished
}

// run takes pieces of j until there are none left. A worker that comes
// late finds none and returns at once.
func (j *job) run() {
	c := int64(j.chunk)
	for {
		lo := j.next.Add(c) - c
		if lo >= int64(j.n) {
			return
		}
		hi := min(lo+c, int64(j.n))
		j.f(int(lo), int(hi))
		j.done.Add(hi - lo)
	}
}

// spawn runs j on w goroutines started for it.
func (j *job) spawn(w int) {
	var wg sync.WaitGroup
	for range w - 1 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			j.run()
		}()
	}
	j.run()
	wg.Wait()
}

// pool is the set of workers kept between calls of parallel. Calls are
// serialised by mu; the workers read the current job without it.
type pool struct {
	mu     sync.Mutex
	n      int                 // workers started
	seq    atomic.Uint64       // incremented for each job
	cur    atomic.Pointer[job] // the job for seq
	sleep  sync.Mutex
	wake   *sync.Cond
	asleep atomic.Int32
}

var workers pool

// start makes sure at least n workers are running. Called with mu held.
func (p *pool) start(n int) {
	if p.wake == nil {
		p.wake = sync.NewCond(&p.sleep)
	}
	for ; p.n < n; p.n++ {
		go p.work(p.seq.Load())
	}
}

// post hands j to the workers. Called with mu held.
func (p *pool) post(j *job) {
	p.cur.Store(j)
	p.seq.Add(1)
	if p.asleep.Load() > 0 {
		p.sleep.Lock()
		p.wake.Broadcast()
		p.sleep.Unlock()
	}
}

func (p *pool) work(seen uint64) {
	for {
		seen = p.wait(seen)
		if j := p.cur.Load(); j != nil {
			j.run()
		}
	}
}

// wait returns the sequence number of the next job after seen, spinning
// for spinFor and then sleeping until one is posted.
func (p *pool) wait(seen uint64) uint64 {
	var t0 int64
	for spins := 0; ; spins++ {
		if s := p.seq.Load(); s != seen {
			return s
		}
		if spins%1024 != 1023 {
			continue
		}
		now := nanotime()
		if t0 == 0 {
			t0 = now
		} else if now-t0 > spinFor {
			break
		}
	}
	p.asleep.Add(1)
	p.sleep.Lock()
	for p.seq.Load() == seen {
		p.wake.Wait()
	}
	p.sleep.Unlock()
	p.asleep.Add(-1)
	return p.seq.Load()
}

var epoch = time.Now()

// nanotime is a monotonic clock.
func nanotime() int64 { return int64(time.Since(epoch)) }
