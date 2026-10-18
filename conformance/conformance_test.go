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

package conformance

import (
	"path/filepath"
	"testing"

	"github.com/gsoultan/mim"
	"github.com/gsoultan/mim/ref"
)

var corpusDir = filepath.Join("..", "testdata", "conformance", "primitives")

func loadCorpus(t *testing.T) []Case {
	t.Helper()
	cases, err := LoadDir(corpusDir)
	if err != nil {
		t.Fatal(err)
	}
	return cases
}

// TestCorpusCoversEveryPrimitive keeps the corpus honest: every primitive
// opcode has cases.
func TestCorpusCoversEveryPrimitive(t *testing.T) {
	cases := loadCorpus(t)
	seen := map[mim.Opcode]int{}
	ids := map[string]bool{}
	for _, c := range cases {
		seen[c.Opcode]++
		if ids[c.ID()] {
			t.Errorf("duplicate case %s", c.ID())
		}
		ids[c.ID()] = true
	}
	for op := mim.OpNeg; op <= mim.OpDequantize; op++ {
		if seen[op] == 0 {
			t.Errorf("no cases for %s", op)
		}
	}
}

// TestReferenceKernels runs the reference kernels against the corpus in
// IEEE mode, the mode they implement.
func TestReferenceKernels(t *testing.T) {
	cases := loadCorpus(t)
	r := Run(cases, ref.Eval, IEEE)
	for _, f := range r.Failures {
		t.Error(f)
	}
	t.Logf("%d cases, %d failures", r.Cases, len(r.Failures))
}

// TestCheckerDetectsErrors makes sure each check kind rejects a wrong result,
// so a broken checker cannot make every runtime pass.
func TestCheckerDetectsErrors(t *testing.T) {
	cases := loadCorpus(t)
	kinds := map[string]bool{}
	for i := range cases {
		c := &cases[i]
		w := c.Output.Tensor
		if kinds[c.Output.Check.Kind] || w.NumElems() < 2 || len(c.DontCare) > 0 {
			continue
		}
		bad := &ref.Tensor{DType: w.DType, Shape: w.Shape, Data: append([]byte{}, w.Data...)}
		changed := false
		for j := 0; j < w.NumElems() && !changed; j++ {
			b := w.Bits(j)
			var nb uint64
			switch {
			case w.DType == mim.Bool:
				nb = b ^ 1
			case ref.IsFloat(w.DType):
				v := ref.DecodeFloat(w.DType, b)
				if v != v || v == 0 || v > 1e300 || v < -1e300 {
					continue
				}
				nb = ref.RoundFloat(w.DType, v*1.5+1, false)
			default:
				nb = b ^ 0x5
			}
			bad.SetBits(j, nb)
			changed = nb != b
		}
		if !changed {
			continue
		}
		if err := c.Verify(bad, IEEE); err == nil {
			t.Errorf("%s: corrupted result passed the %s check", c.ID(), c.Output.Check.Kind)
		}
		kinds[c.Output.Check.Kind] = true
	}
	for _, k := range []string{"bits", "exact", "ulp", "tol"} {
		if !kinds[k] {
			t.Errorf("no case exercised check kind %q", k)
		}
	}
}

// TestRunReportsPanics checks that a crashing runtime fails instead of
// taking the test binary down.
func TestRunReportsPanics(t *testing.T) {
	cases := loadCorpus(t)[:3]
	r := Run(cases, func(mim.Opcode, []mim.Attr, []*ref.Tensor, mim.DType, []int) (*ref.Tensor, error) {
		panic("boom")
	}, IEEE)
	if len(r.Failures) != len(cases) {
		t.Fatalf("got %d failures, want %d", len(r.Failures), len(cases))
	}
}
