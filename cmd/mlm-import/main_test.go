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

package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gsoultan/mlm"
)

func copyFixture(t *testing.T, dir, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "testdata", "import", name))
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestRun(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"tiny.gguf", "tiny.safetensors"} {
		in := copyFixture(t, dir, name)
		var out bytes.Buffer
		if err := run(&out, []string{in}, options{bench: true, hash: true}); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(out.String(), "open+read") {
			t.Errorf("no benchmark table in:\n%s", out.String())
		}
		f, err := mlm.Open(defaultOutput(in))
		if err != nil {
			t.Fatal(err)
		}
		if err := f.VerifyHashes(); err != nil {
			t.Error(err)
		}
		if f.NumTensors() == 0 {
			t.Error("no tensors")
		}
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	}
}

func TestRunRejects(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "x.bin")
	os.WriteFile(bad, []byte("not a model"), 0o644)
	if err := run(&bytes.Buffer{}, []string{bad}, options{out: filepath.Join(dir, "o.mim"), hash: true}); err == nil {
		t.Error("accepted a non-model file")
	}
}

func TestRunONNX(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join("..", "..", "testdata", "onnx")
	b, err := os.ReadFile(filepath.Join(src, "tiny-cnn.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	in := filepath.Join(dir, "tiny-cnn.onnx")
	if err := os.WriteFile(in, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := run(&out, []string{in}, options{hash: true}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "from onnx") {
		t.Errorf("output %q", out.String())
	}
	got, err := os.ReadFile(defaultOutput(in))
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(filepath.Join(src, "tiny-cnn.mim"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Error("the conversion differs from testdata/onnx/tiny-cnn.mim")
	}
	if err := run(&bytes.Buffer{}, []string{in}, options{out: filepath.Join(dir, "b.mim"), bench: true}); err == nil {
		t.Error("-bench accepted an ONNX input")
	}
}
