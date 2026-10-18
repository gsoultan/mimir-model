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

package conformance

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/gsoultan/mlm"
	"github.com/gsoultan/mlm/ref"
)

// maxReported is how many mismatching elements a failure lists.
const maxReported = 5

// Verify compares a runtime's result for c with the expected output for
// mode. It returns nil when the result conforms.
func (c *Case) Verify(got *ref.Tensor, m Mode) error {
	want := c.Expected(m)
	w := want.Tensor
	if got == nil {
		return fmt.Errorf("no result")
	}
	if got.DType != w.DType || !equalShape(got.Shape, w.Shape) {
		return fmt.Errorf("result is %s%v, want %s%v", got.DType, got.Shape, w.DType, w.Shape)
	}
	if err := got.Validate(); err != nil {
		return err
	}
	skip := make(map[int]bool, len(c.DontCare))
	for _, i := range c.DontCare {
		skip[i] = true
	}
	var bad []string
	nbad := 0
	for i := 0; i < w.NumElems(); i++ {
		if skip[i] {
			continue
		}
		if msg := checkElem(want.Check, w.DType, i, got.Bits(i), w.Bits(i)); msg != "" {
			if nbad < maxReported {
				bad = append(bad, fmt.Sprintf("[%d] %s", i, msg))
			}
			nbad++
		}
	}
	if nbad == 0 {
		return nil
	}
	if nbad > maxReported {
		bad = append(bad, fmt.Sprintf("... %d mismatches in total", nbad))
	}
	return fmt.Errorf("%s check failed: %s", want.Check.Kind, strings.Join(bad, "; "))
}

func equalShape(a, b []int) bool {
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

// checkElem returns "" when element i conforms, otherwise a description.
func checkElem(c Check, dt mlm.DType, i int, got, want uint64) string {
	isF := ref.IsFloat(dt)
	switch c.Kind {
	case "bits":
		if got != want {
			return fmt.Sprintf("bits %#x, want %#x", got, want)
		}
	case "exact":
		if isF {
			g, w := ref.DecodeFloat(dt, got), ref.DecodeFloat(dt, want)
			if math.IsNaN(w) && math.IsNaN(g) {
				return ""
			}
			if got != want {
				return fmt.Sprintf("%v (%#x), want %v (%#x)", g, got, w, want)
			}
			return ""
		}
		if got != want {
			return fmt.Sprintf("%#x, want %#x", got, want)
		}
	case "ulp":
		g, w := ref.DecodeFloat(dt, got), ref.DecodeFloat(dt, want)
		return within(g, w, c.ULP*ulpOrZero(dt, w), 0)
	case "tol":
		return within(ref.DecodeFloat(dt, got), c.Ref[i], c.Bound[i], c.MaxAbs)
	}
	return ""
}

func ulpOrZero(dt mlm.DType, w float64) float64 {
	if math.IsNaN(w) || math.IsInf(w, 0) {
		return 0
	}
	return ref.ULP(dt, w)
}

// within checks |g - w| <= bound; NaN and infinities in w must be matched
// exactly (any NaN matches any NaN).
func within(g, w, bound, maxAbs float64) string {
	switch {
	case math.IsNaN(w):
		if !math.IsNaN(g) {
			return fmt.Sprintf("%v, want NaN", g)
		}
		return ""
	case math.IsInf(w, 0):
		if g != w {
			return fmt.Sprintf("%v, want %v", g, w)
		}
		return ""
	case math.IsNaN(g) || math.IsInf(g, 0):
		return fmt.Sprintf("%v, want %v", g, w)
	}
	if maxAbs != 0 && math.Abs(g) > maxAbs {
		return fmt.Sprintf("%v exceeds %v in magnitude", g, maxAbs)
	}
	if d := math.Abs(g - w); d > bound {
		return fmt.Sprintf("%v, want %v within %g (off by %g)", g, w, bound, d)
	}
	return ""
}

// EvalFunc runs one primitive the way a runtime would. It receives the
// opcode, attributes, inputs and the declared output dtype and shape.
type EvalFunc func(op mlm.Opcode, attrs []mlm.Attr, in []*ref.Tensor, outDType mlm.DType, outShape []int) (*ref.Tensor, error)

// Failure is one case a runtime did not pass.
type Failure struct {
	Case *Case
	Err  error
}

func (f Failure) String() string { return f.Case.File + ": " + f.Case.ID() + ": " + f.Err.Error() }

// Report summarises a conformance run.
type Report struct {
	Mode     Mode
	Cases    int
	PerOp    map[string]int // cases run per op
	Failures []Failure
}

// Run evaluates every case with eval and checks the results for mode. A
// panic in eval is reported as a failure of that case: no input may crash a
// runtime (spec/primitives.md section 3.8).
func Run(cases []Case, eval EvalFunc, m Mode) Report {
	r := Report{Mode: m, PerOp: map[string]int{}}
	for i := range cases {
		c := &cases[i]
		r.Cases++
		r.PerOp[c.Op]++
		w := c.Expected(m).Tensor
		got, err := safeEval(eval, c, w.DType, w.Shape)
		if err == nil {
			err = c.Verify(got, m)
		}
		if err != nil {
			r.Failures = append(r.Failures, Failure{c, err})
		}
	}
	sort.SliceStable(r.Failures, func(i, j int) bool { return r.Failures[i].Case.ID() < r.Failures[j].Case.ID() })
	return r
}

func safeEval(eval EvalFunc, c *Case, dt mlm.DType, shape []int) (t *ref.Tensor, err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("panic: %v", p)
		}
	}()
	in := make([]*ref.Tensor, len(c.Inputs))
	for i, x := range c.Inputs { // runtimes must not modify their inputs; give each a copy
		in[i] = &ref.Tensor{DType: x.DType, Shape: append([]int{}, x.Shape...), Data: append([]byte{}, x.Data...)}
	}
	return eval(c.Opcode, c.Attrs, in, dt, append([]int{}, shape...))
}
