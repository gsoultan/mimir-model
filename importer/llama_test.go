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

package importer

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	mim "github.com/gsoultan/mimir-model"
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
	want := LlamaConfig{Arch: "llama", Layers: 2, Embed: 256, FFN: 256, Heads: 4, KVHeads: 2, HeadDim: 64, RopeDims: 64,
		RopeTheta: 10000, NormEps: float64(float32(1e-5)), Vocab: 441, Context: 32, MaxContext: 64, Slots: 1}
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
	w := mim.NewWriter()
	if err := s.AddTo(w); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := w.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	f, err := mim.Parse(buf.Bytes())
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
		case mim.KindWeight, mim.KindConst, mim.KindVariable:
			if ti := f.TensorAt(int(v.Ref)); !v.Zero && ti.Name != v.Name {
				t.Errorf("value %q refers to tensor %q", v.Name, ti.Name)
			}
		}
		if v.Kind == mim.KindVariable && v.Zero {
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
		{"rope scaling", func(s *Source) { s.Metadata["llama.rope.scaling.type"] = "yarn" }, "rope scaling"},
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

func TestBuildQwen2(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "testdata", "qwen2", "tiny-q4km.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	s, err := ParseGGUF(b)
	if err != nil {
		t.Fatal(err)
	}
	c, err := BuildLlama(s, LlamaOptions{Context: 32})
	if err != nil {
		t.Fatal(err)
	}
	want := LlamaConfig{Arch: "qwen2", Layers: 2, Embed: 256, FFN: 256, Heads: 4, KVHeads: 2, HeadDim: 64, RopeDims: 64,
		RopeTheta: 1e6, NormEps: float64(float32(1e-6)), Vocab: 406, Context: 32, MaxContext: 64, Slots: 1}
	if *c != want {
		t.Fatalf("config %+v, want %+v", *c, want)
	}
	biases := 0
	for _, v := range s.Graph.Values {
		if v.Kind == mim.KindWeight && strings.HasSuffix(v.Name, ".bias") {
			biases++
		}
	}
	if biases != 3*c.Layers {
		t.Errorf("%d bias weights in the graph, want %d", biases, 3*c.Layers)
	}

	s, _ = ParseGGUF(b)
	for i, t := range s.Tensors {
		if t.Name == "blk.0.attn_k.bias" {
			s.Tensors = append(s.Tensors[:i], s.Tensors[i+1:]...)
			break
		}
	}
	if _, err := BuildLlama(s, LlamaOptions{}); err == nil || !strings.Contains(err.Error(), "attn_k.bias") {
		t.Errorf("without a bias: error %v", err)
	}
}

func familyFixture(t *testing.T, arch string) *Source {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "testdata", arch, "tiny-q4km.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	s, err := ParseGGUF(b)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// TestBuildFamilies checks the hyperparameters read for each family the
// graph builder knows beyond llama and qwen2, and the parts of the graph
// that set the family apart. cmd/mim-run's tests run the graphs against
// float64 oracles that llama.cpp agrees with.
func TestBuildFamilies(t *testing.T) {
	for _, c := range []struct {
		arch  string
		want  LlamaConfig
		calls map[string]int // composite calls the graph must make, by name.version
	}{
		{"qwen3", LlamaConfig{Arch: "qwen3", Layers: 2, Embed: 256, FFN: 256, Heads: 4, KVHeads: 2, HeadDim: 96, RopeDims: 96,
			RopeTheta: 1e6, NormEps: float64(float32(1e-6)), Vocab: 406, Context: 32, MaxContext: 64, Slots: 1},
			map[string]int{"rms_norm.v1": 4*2 + 1, "rope.v1": 4, "swiglu.v1": 2}},
		{"gemma2", LlamaConfig{Arch: "gemma2", Layers: 2, Embed: 256, FFN: 256, Heads: 4, KVHeads: 2, HeadDim: 128, RopeDims: 128,
			RopeTheta: 10000, NormEps: float64(float32(1e-6)), Vocab: 444, Context: 32, MaxContext: 64, Slots: 1,
			Window: 8, LocalEvery: 2, LocalTheta: 10000, AttnScale: 1 / math.Sqrt(128), AttnSoftcap: 5, FinalSoftcap: 10},
			map[string]int{"rms_norm.v1": 4*2 + 1, "geglu.v1": 2, "tanh.v1": 1, "rope.v1": 4}},
		{"gemma3", LlamaConfig{Arch: "gemma3", Layers: 6, Embed: 256, FFN: 256, Heads: 4, KVHeads: 1, HeadDim: 128, RopeDims: 128,
			RopeTheta: 1e6, NormEps: float64(float32(1e-6)), Vocab: 444, Context: 32, MaxContext: 64, Slots: 1,
			RopeScale: 8, Window: 8, LocalEvery: 6, LocalTheta: 10000, AttnScale: 1 / math.Sqrt(128)},
			map[string]int{"rms_norm.v1": 6*6 + 1, "geglu.v1": 6, "rope.v1": 10, "rope.v2": 2}},
		{"phi3", LlamaConfig{Arch: "phi3", Layers: 2, Embed: 256, FFN: 256, Heads: 4, KVHeads: 2, HeadDim: 64, RopeDims: 48,
			RopeTheta: 10000, NormEps: float64(float32(1e-5)), Vocab: 406, Context: 32, MaxContext: 64, Slots: 1,
			RopeAttnFactor: float64(float32(1.19))},
			map[string]int{"rms_norm.v1": 2*2 + 1, "swiglu.v1": 2, "rope.v2": 4}},
	} {
		t.Run(c.arch, func(t *testing.T) {
			s := familyFixture(t, c.arch)
			got, err := BuildLlama(s, LlamaOptions{Context: 32})
			if err != nil {
				t.Fatal(err)
			}
			if *got != c.want {
				t.Fatalf("config %+v, want %+v", *got, c.want)
			}
			calls := map[string]int{}
			for _, op := range s.Graph.Subgraphs[0].Ops {
				if op.Opcode != mim.OpCall {
					continue
				}
				var name string
				var v int64
				for _, a := range op.Attrs {
					switch a.Key {
					case "name":
						name = a.String
					case "version":
						v = a.Int
					}
				}
				calls[fmt.Sprintf("%s.v%d", name, v)]++
			}
			for k, n := range c.calls {
				if calls[k] != n {
					t.Errorf("%d calls of %s, want %d (all: %v)", calls[k], k, n, calls)
				}
			}
		})
	}
}

// TestPhi3RopeFactors checks that Phi 3 takes its short rope factors for a
// cache no longer than its original context, and its long ones past it.
func TestPhi3RopeFactors(t *testing.T) {
	for _, c := range []struct {
		context int
		want    string
	}{{32, "rope_factors_short.weight"}, {64, "rope_factors_long.weight"}} {
		s := familyFixture(t, "phi3")
		if _, err := BuildLlama(s, LlamaOptions{Context: c.context}); err != nil {
			t.Fatal(err)
		}
		var used []string
		for _, v := range s.Graph.Values {
			if strings.HasPrefix(v.Name, "rope_factors") {
				used = append(used, v.Name)
			}
		}
		if len(used) != 1 || used[0] != c.want {
			t.Errorf("context %d: graph reads %v, want %s", c.context, used, c.want)
		}
	}
}

// TestBuildFamiliesReject checks what a family's graph cannot be built
// without.
func TestBuildFamiliesReject(t *testing.T) {
	for _, c := range []struct {
		arch, drop, error string
	}{
		{"qwen3", "blk.1.attn_k_norm.weight", "attn_k_norm"},
		{"gemma2", "blk.0.post_ffw_norm.weight", "post_ffw_norm"},
		{"gemma3", "blk.5.attn_q_norm.weight", "attn_q_norm"},
		{"phi3", "blk.0.attn_qkv.weight", "attn_q.weight"},
	} {
		s := familyFixture(t, c.arch)
		for i, tt := range s.Tensors {
			if tt.Name == c.drop {
				s.Tensors = append(s.Tensors[:i], s.Tensors[i+1:]...)
				break
			}
		}
		if _, err := BuildLlama(s, LlamaOptions{}); err == nil || !strings.Contains(err.Error(), c.error) {
			t.Errorf("%s without %s: error %v, want one mentioning %q", c.arch, c.drop, err, c.error)
		}
	}
	s := familyFixture(t, "gemma3")
	s.Metadata["gemma3.rope.scaling.type"] = "yarn"
	if _, err := BuildLlama(s, LlamaOptions{}); err == nil || !strings.Contains(err.Error(), "yarn") {
		t.Errorf("gemma3 with yarn scaling: error %v", err)
	}
}
