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
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func f32bytes(v ...float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

func sampleWriter(t testing.TB) *Writer {
	w := NewWriter()
	must(t, w.AddTensor("layer.1.w", F32, []uint64{2, 3}, f32bytes(1, 2, 3, 4, 5, 6)))
	must(t, w.AddTensor("bias", F32, []uint64{3}, f32bytes(7, 8, 9)))
	must(t, w.AddTensor("scalar", I32, nil, []byte{1, 0, 0, 0}))
	must(t, w.AddTensor("empty", F16, []uint64{0, 4}, nil))
	must(t, w.AddTensor("big", U8, []uint64{LargeTensorThreshold}, make([]byte, LargeTensorThreshold)))
	must(t, w.AddTensor("q", Q8_0, []uint64{2, 32}, make([]byte, 68)))
	return w
}

func must(t testing.TB, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func encode(t testing.TB, w *Writer) []byte {
	var buf bytes.Buffer
	_, err := w.WriteTo(&buf)
	must(t, err)
	return buf.Bytes()
}

func TestRoundTrip(t *testing.T) {
	b := encode(t, sampleWriter(t))
	f, err := Parse(b)
	must(t, err)
	must(t, f.VerifyHashes())
	if f.NumTensors() != 6 {
		t.Fatalf("got %d tensors", f.NumTensors())
	}
	p := f.Preamble()
	if p.DataOffset%DefaultDataAlignment != 0 {
		t.Fatalf("data offset %d", p.DataOffset)
	}
	ti, ok := f.Tensor("layer.1.w")
	if !ok || ti.DType != F32 || len(ti.Shape) != 2 || ti.Shape[1] != 3 {
		t.Fatalf("lookup: %+v %v", ti, ok)
	}
	if !bytes.Equal(f.TensorData(ti), f32bytes(1, 2, 3, 4, 5, 6)) {
		t.Fatal("data mismatch")
	}
	big, _ := f.Tensor("big")
	if big.Offset%LargeTensorAlignment != 0 {
		t.Fatalf("big tensor at %d", big.Offset)
	}
	if _, ok := f.Tensor("missing"); ok {
		t.Fatal("found missing tensor")
	}
	for i := 1; i < f.NumTensors(); i++ {
		if f.TensorAt(i-1).Name >= f.TensorAt(i).Name {
			t.Fatal("index not sorted")
		}
	}
}

func TestEmptyFile(t *testing.T) {
	f, err := Parse(encode(t, NewWriter()))
	must(t, err)
	if f.NumTensors() != 0 || len(f.Weights()) != 0 {
		t.Fatal("expected empty")
	}
}

func TestOpenFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.mim")
	must(t, sampleWriter(t).WriteFile(path))
	f, err := Open(path)
	must(t, err)
	defer f.Close()
	must(t, f.VerifyHashes())
	ti, _ := f.Tensor("bias")
	if !bytes.Equal(f.TensorData(ti), f32bytes(7, 8, 9)) {
		t.Fatal("data mismatch")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
}

func TestWriterRejects(t *testing.T) {
	w := NewWriter()
	if CodeOf(w.AddTensor("a", F32, []uint64{2}, make([]byte, 4))) != CodeSizeMismatch {
		t.Fatal("size mismatch not caught")
	}
	if CodeOf(w.AddTensor("q", Q4_0, []uint64{31}, make([]byte, 18))) != CodeBadShape {
		t.Fatal("block divisibility not caught")
	}
	must(t, w.AddTensor("a", F32, nil, make([]byte, 4)))
	if CodeOf(w.AddTensor("a", F32, nil, make([]byte, 4))) != CodeDuplicate {
		t.Fatal("duplicate not caught")
	}
	if CodeOf(w.AddTensor("x", F32, []uint64{1 << 62, 8}, nil)) != CodeOverflow {
		t.Fatal("overflow not caught")
	}
}
