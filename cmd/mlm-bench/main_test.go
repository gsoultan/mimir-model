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
	"github.com/gsoultan/mlm/importer"
)

func TestBench(t *testing.T) {
	dir := t.TempDir()
	src, err := os.ReadFile(filepath.Join("..", "..", "testdata", "llama", "tiny-q4km.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	gguf := filepath.Join(dir, "tiny.gguf")
	if err := os.WriteFile(gguf, src, 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := importer.Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := importer.BuildLlama(s, importer.LlamaOptions{Context: 16}); err != nil {
		t.Fatal(err)
	}
	w := mlm.NewWriter()
	if err := s.AddTo(w); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "tiny.mim")
	if err := w.WriteFile(out); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := run(&buf, []string{gguf, out}, 3, false, measure); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"| tiny.gguf | gguf |", "| tiny.mim | mim + graph |"} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("no %q row in:\n%s", want, buf.String())
		}
	}
}

func TestBenchRejects(t *testing.T) {
	bad := filepath.Join(t.TempDir(), "x.mim")
	os.WriteFile(bad, []byte("not a model"), 0o644)
	if err := run(&bytes.Buffer{}, []string{bad}, 1, false, measure); err == nil {
		t.Error("accepted a non-model file")
	}
}
