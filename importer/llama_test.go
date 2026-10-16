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

package importer

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gsoultan/mmf"
)

func llamaFixture(t *testing.T) *Source {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "testdata", "llama", "tiny-q4km.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	s, err := ParseGGUF(b)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestBuildLlama(t *testing.T) {
	s := llamaFixture(t)
	n := len(s.Tensors)
	c, err := BuildLlama(s, LlamaOptions{Context: 32})
	if err != nil {
		t.Fatal(err)
	}
	want := LlamaConfig{Layers: 2, Embed: 256, FFN: 256, Heads: 4, KVHeads: 2, HeadDim: 64, RopeDims: 64,
		RopeTheta: 10000, NormEps: float64(float32(1e-5)), Vocab: 441, Context: 32, MaxContext: 64}
	if *c != want {
		t.Fatalf("config %+v, want %+v", *c, want)
	}
	// The key position table; the caches start as zeros and are not stored.
	if len(s.Tensors) != n+1 {
		t.Fatalf("%d tensors added", len(s.Tensors)-n)
	}
	if _, err := BuildLlama(s, LlamaOptions{Context: 32}); err == nil {
		t.Fatal("built a second graph")
	}

	// The written file holds the graph, with refs pointing at the right
	// tensors after AddTo sorts them.
	w := mmf.NewWriter()
	if err := s.AddTo(w); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := w.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	f, err := mmf.Parse(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	g, err := f.Graph()
	if err != nil {
		t.Fatal(err)
	}
	vars := 0
	for _, v := range g.Values {
		switch v.Kind {
		case mmf.KindWeight, mmf.KindConst, mmf.KindVariable:
			if ti := f.TensorAt(int(v.Ref)); !v.Zero && ti.Name != v.Name {
				t.Errorf("value %q refers to tensor %q", v.Name, ti.Name)
			}
		}
		if v.Kind == mmf.KindVariable && v.Zero {
			vars++
		}
	}
	if vars != 2*c.Layers || len(g.Updates) != vars {
		t.Errorf("%d variables and %d updates", vars, len(g.Updates))
	}
}

func TestBuildLlamaRejects(t *testing.T) {
	for _, c := range []struct {
		name  string
		edit  func(*Source)
		error string
	}{
		{"architecture", func(s *Source) { s.Metadata["general.architecture"] = "gpt2" }, "architecture"},
		{"missing tensor", func(s *Source) {
			for i, t := range s.Tensors {
				if t.Name == "blk.1.ffn_up.weight" {
					s.Tensors = append(s.Tensors[:i], s.Tensors[i+1:]...)
					break
				}
			}
		}, "missing"},
		{"heads", func(s *Source) { s.Metadata["llama.attention.head_count_kv"] = uint64(3) }, "divide"},
		{"rope scaling", func(s *Source) { s.Metadata["llama.rope.scaling.type"] = "linear" }, "rope scaling"},
		{"epsilon", func(s *Source) { delete(s.Metadata, "llama.attention.layer_norm_rms_epsilon") }, "epsilon"},
	} {
		s := llamaFixture(t)
		c.edit(s)
		if _, err := BuildLlama(s, LlamaOptions{}); err == nil || !strings.Contains(err.Error(), c.error) {
			t.Errorf("%s: error %v, want one mentioning %q", c.name, err, c.error)
		}
	}
	if _, err := BuildLlama(llamaFixture(t), LlamaOptions{Context: maxLlamaContext + 1}); err == nil {
		t.Error("accepted an enormous context")
	}
}
