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
	"sync"
	"sync/atomic"
	"testing"
)

// TestParallel checks that every element is visited exactly once, from
// one caller, from several callers at once (which cannot all use the
// pool), from inside f, and after the workers have gone to sleep.
func TestParallel(t *testing.T) {
	check := func(n int) {
		seen := make([]atomic.Int32, n)
		parallel(n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				seen[i].Add(1)
			}
		})
		for i := range seen {
			if c := seen[i].Load(); c != 1 {
				t.Fatalf("n=%d: element %d visited %d times", n, i, c)
			}
		}
	}
	for _, n := range []int{0, 1, 2, 3, 7, 64, 1000, 4097} {
		check(n)
	}

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for n := 1; n < 300; n += 7 {
				check(n)
			}
		}()
	}
	wg.Wait()

	var total atomic.Int64
	parallel(16, func(lo, hi int) {
		for range hi - lo {
			parallel(100, func(lo, hi int) { total.Add(int64(hi - lo)) })
		}
	})
	if total.Load() != 1600 {
		t.Fatalf("nested: %d elements, want 1600", total.Load())
	}

	workers.mu.Lock()
	for workers.asleep.Load() < int32(workers.n) {
	}
	workers.mu.Unlock()
	check(513)
}
