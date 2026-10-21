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
	"fmt"
	"path/filepath"
	"testing"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/ref"
)

// BenchmarkTinyCNN runs testdata/onnx/tiny-cnn.mim, about 170 ops on
// tensors of a few thousand elements, where the interpreter's cost per op
// is most of the time.
func BenchmarkTinyCNN(b *testing.B) {
	f, err := mim.Open(filepath.Join("..", "testdata", "onnx", "tiny-cnn.mim"))
	if err != nil {
		b.Fatal(err)
	}
	p, err := Load(f)
	if err != nil {
		b.Fatal(err)
	}
	p.FastPaths = true
	for _, batch := range []int{1, 8} {
		b.Run(fmt.Sprintf("batch%d", batch), func(b *testing.B) {
			x := make([]float32, batch*3*16*16)
			for i := range x {
				x[i] = float32(i%13)/13 - 0.5
			}
			in := map[string]*ref.Tensor{"x": f32t([]int{batch, 3, 16, 16}, x...)}
			s := p.NewSession()
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := s.Run(in); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestRunsReuseBuffers runs tiny-cnn again and again, as a server does,
// and checks that each run's outputs are its own: reusing the buffers of
// dropped values must neither change a result the caller holds nor give
// a later run a different answer for the same input.
func TestRunsReuseBuffers(t *testing.T) {
	f, err := mim.Open(filepath.Join("..", "testdata", "onnx", "tiny-cnn.mim"))
	if err != nil {
		t.Fatal(err)
	}
	p, err := Load(f)
	if err != nil {
		t.Fatal(err)
	}
	p.FastPaths = true
	input := func(batch int, seed float32) map[string]*ref.Tensor {
		x := make([]float32, batch*3*16*16)
		for i := range x {
			x[i] = float32(i%11)/11*seed - 0.5
		}
		return map[string]*ref.Tensor{"x": f32t([]int{batch, 3, 16, 16}, x...)}
	}
	s := p.NewSession()
	var held []map[string]*ref.Tensor // every run's outputs, and their bytes when made
	var copies []map[string][]byte
	for i := range 6 {
		out, err := s.Run(input(1+i%3, float32(i+1)))
		if err != nil {
			t.Fatal(err)
		}
		c := map[string][]byte{}
		for k, v := range out {
			c[k] = append([]byte{}, v.Data...)
		}
		held, copies = append(held, out), append(copies, c)
	}
	for i := range held {
		for k, v := range held[i] {
			if string(v.Data) != string(copies[i][k]) {
				t.Fatalf("run %d output %s changed under a later run", i, k)
			}
		}
		again, err := p.NewSession().Run(input(1+i%3, float32(i+1)))
		if err != nil {
			t.Fatal(err)
		}
		for k, v := range again {
			if string(v.Data) != string(copies[i][k]) {
				t.Fatalf("run %d output %s differs when run again", i, k)
			}
		}
	}
}
