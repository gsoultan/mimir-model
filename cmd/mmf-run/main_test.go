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

package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/importer"
	"github.com/gsoultan/mmf/interp"
	"github.com/gsoultan/mmf/ref"
)

// The fixtures in testdata/llama and testdata/qwen2 are two-layer models
// with random weights, quantized to Q4_K_M by llama.cpp: a Llama with a
// SentencePiece vocabulary and a Qwen 2 with a byte-level BPE one. Each
// expected.json holds llama.cpp's token ids for a few strings and the
// float64 logits of an independent NumPy forward pass
// (tools/gen_llama_testdata.py, tools/gen_qwen2_testdata.py).

var fixtures = []string{"llama", "qwen2"}

type expected struct {
	dir      string
	Model    string `json:"model"`
	Vocab    int    `json:"vocab"`
	Tokenize []struct {
		Text string  `json:"text"`
		IDs  []int32 `json:"ids"`
	} `json:"tokenize"`
	Prompt        string   `json:"prompt"`
	PromptIDs     []int32  `json:"prompt_ids"`
	PromptLogits  string   `json:"prompt_logits"`
	Greedy        []int32  `json:"greedy"`
	GreedyLogits  []string `json:"greedy_logits"`
	LlamaCppError float64  `json:"llama_cpp_max_diff"`
}

func loadExpected(t *testing.T, fixture string) *expected {
	t.Helper()
	dir := filepath.Join("..", "..", "testdata", fixture)
	b, err := os.ReadFile(filepath.Join(dir, "expected.json"))
	if err != nil {
		t.Fatal(err)
	}
	e := expected{dir: dir}
	if err := json.Unmarshal(b, &e); err != nil {
		t.Fatal(err)
	}
	return &e
}

// eachFixture runs f as a subtest for every fixture.
func eachFixture(t *testing.T, f func(t *testing.T, e *expected)) {
	for _, name := range fixtures {
		t.Run(name, func(t *testing.T) { f(t, loadExpected(t, name)) })
	}
}

func f32b64(t *testing.T, s string) []float32 {
	t.Helper()
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		t.Fatal(err)
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

// importFixture converts the GGUF fixture with a graph, as
// `mmf-import -graph -ctx 64 -a8=false` does, and returns the output path.
func importFixture(t *testing.T, e *expected) string {
	return importFixtureWith(t, e, importer.LlamaOptions{Context: 64})
}

func importFixtureWith(t *testing.T, e *expected, o importer.LlamaOptions) string {
	t.Helper()
	in := filepath.Join(e.dir, e.Model)
	b, err := os.ReadFile(in)
	if err != nil {
		t.Fatal(err)
	}
	src, err := importer.ParseGGUF(b)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := importer.BuildLlama(src, o); err != nil {
		t.Fatal(err)
	}
	w := mmf.NewWriter()
	if err := src.AddTo(w); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "tiny.mmf")
	if err := w.WriteFile(out); err != nil {
		t.Fatal(err)
	}
	return out
}

// near requires got to match the oracle within tol times its largest
// magnitude.
func near(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d logits, want %d", what, len(got), len(want))
	}
	var big, worst float64
	for _, w := range want {
		big = math.Max(big, math.Abs(float64(w)))
	}
	for i := range got {
		worst = math.Max(worst, math.Abs(float64(got[i]-want[i])))
	}
	if worst > tol*big || math.IsNaN(worst) {
		t.Fatalf("%s: max |logit difference| %g, allowed %g", what, worst, tol*big)
	}
}

func TestTokenizerMatchesLlamaCpp(t *testing.T) { eachFixture(t, testTokenizer) }

func testTokenizer(t *testing.T, e *expected) {
	m, err := load(importFixture(t, e), false)
	if err != nil {
		t.Fatal(err)
	}
	defer m.f.Close()
	if m.tok.VocabSize() != e.Vocab {
		t.Fatalf("vocabulary %d, want %d", m.tok.VocabSize(), e.Vocab)
	}
	for _, c := range e.Tokenize {
		got := m.tok.Encode(c.Text, true)
		if !equalIDs(got, c.IDs) {
			t.Errorf("Encode(%q) = %v, llama.cpp gives %v", c.Text, got, c.IDs)
		}
		if dec := m.tok.Decode(got); dec != c.Text {
			t.Errorf("Decode(Encode(%q)) = %q", c.Text, dec)
		}
	}
}

func equalIDs(a, b []int32) bool {
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

// TestGenerateMatchesOracle runs the prompt and a greedy continuation with
// the fast paths, step by step through the key-value cache, and compares
// every logit with the float64 oracle.
func TestGenerateMatchesOracle(t *testing.T) { eachFixture(t, testGenerate) }

func testGenerate(t *testing.T, e *expected) {
	m, err := load(importFixture(t, e), false)
	if err != nil {
		t.Fatal(err)
	}
	defer m.f.Close()
	if ids := m.tok.Encode(e.Prompt, true); !equalIDs(ids, e.PromptIDs) {
		t.Fatalf("prompt ids %v, want %v", ids, e.PromptIDs)
	}
	s := m.prog.NewSession()
	all, err := stepAll(s, e.PromptIDs, 0)
	if err != nil {
		t.Fatal(err)
	}
	near(t, "prompt", all, f32b64(t, e.PromptLogits), 2e-5)

	logits := all[len(all)-e.Vocab:]
	pos := len(e.PromptIDs)
	for i, want := range e.Greedy {
		near(t, "step logits", logits, f32b64(t, e.GreedyLogits[i]), 2e-5)
		if got := argmax(logits); got != want {
			t.Fatalf("step %d: picked %d, oracle picks %d", i, got, want)
		}
		if logits, err = step(s, []int32{want}, pos); err != nil {
			t.Fatal(err)
		}
		pos++
	}

	// generate drives the same loop.
	var got []int32
	if _, _, _, err := m.generate(e.Prompt, len(e.Greedy), 0, 1, func(id int32) { got = append(got, id) }); err != nil {
		t.Fatal(err)
	}
	if !equalIDs(got, e.Greedy) {
		t.Fatalf("generate gave %v, want %v", got, e.Greedy)
	}
}

// TestGenerateA8 runs the prompt and the oracle's continuation through a
// graph whose matrix products allow 8-bit activations, as llama.cpp
// computes them, with the key-value cache in f32 and in f16. The logits
// must stay as close to the float64 oracle as llama.cpp's own, within 10%
// with the cache in f16.
func TestGenerateA8(t *testing.T) { eachFixture(t, testGenerateA8) }

func testGenerateA8(t *testing.T, e *expected) {
	for _, cache := range []mmf.DType{mmf.F32, mmf.F16} {
		t.Run(cache.String(), func(t *testing.T) {
			m, err := load(importFixtureWith(t, e, importer.LlamaOptions{Context: 64, A8: true, Cache: cache}), false)
			if err != nil {
				t.Fatal(err)
			}
			defer m.f.Close()
			s := m.prog.NewSession()
			all, err := stepAll(s, e.PromptIDs, 0)
			if err != nil {
				t.Fatal(err)
			}
			worst := maxDiff(all, f32b64(t, e.PromptLogits))
			logits := all[len(all)-e.Vocab:]
			pos := len(e.PromptIDs)
			for i, next := range e.Greedy {
				worst = math.Max(worst, maxDiff(logits, f32b64(t, e.GreedyLogits[i])))
				if logits, err = step(s, []int32{next}, pos); err != nil {
					t.Fatal(err)
				}
				pos++
			}
			t.Logf("max |logit difference| from the oracle: %.4f (llama.cpp: %.4f)", worst, e.LlamaCppError)
			// With the cache in f16, as llama.cpp's is, the keys and values
			// are rounded to 11 bits as well: on this model that moves the
			// largest difference from 0.41 to 0.55, against llama.cpp's
			// 0.53, which rounds them too and also the queries.
			bound := e.LlamaCppError
			if cache == mmf.F16 {
				bound *= 1.1
			}
			if worst > bound || math.IsNaN(worst) {
				t.Fatalf("max |logit difference| %g, more than %g (llama.cpp's is %g)", worst, bound, e.LlamaCppError)
			}
		})
	}
}

// TestCacheF16 runs the prompt through a graph with the key-value cache
// in f16 with the fast paths and with every composite on its canonical
// body: the two must agree as closely as they do with a cache in f32.
func TestCacheF16(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	e := loadExpected(t, "llama")
	path := importFixtureWith(t, e, importer.LlamaOptions{Context: 64, Cache: mmf.F16})
	ids := e.PromptIDs[:3]
	var out [2][]float32
	for i, useRef := range []bool{false, true} {
		m, err := load(path, useRef)
		if err != nil {
			t.Fatal(err)
		}
		if out[i], err = stepAll(m.prog.NewSession(), ids, 0); err != nil {
			t.Fatal(err)
		}
		m.f.Close()
	}
	near(t, "f16 cache, fast against reference", out[0], out[1], 2e-5)
}

func maxDiff(got, want []float32) float64 {
	var worst float64
	for i := range got {
		worst = math.Max(worst, math.Abs(float64(got[i]-want[i])))
	}
	return worst
}

// TestReferencePath runs the prompt with every composite inlined to its
// canonical body on the reference kernels: the spec's own definition of
// the model, with no fast path involved.
func TestReferencePath(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	e := loadExpected(t, "llama")
	m, err := load(importFixture(t, e), true)
	if err != nil {
		t.Fatal(err)
	}
	defer m.f.Close()
	ids := e.PromptIDs[:3]
	all, err := stepAll(m.prog.NewSession(), ids, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := f32b64(t, e.PromptLogits)[:len(all)]
	near(t, "reference prompt", all, want, 2e-5)
}

func TestRun(t *testing.T) {
	e := loadExpected(t, "llama")
	path := importFixture(t, e)
	var out, log bytes.Buffer
	if err := run(&out, &log, path, config{prompt: e.Prompt, tokens: 4}); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out.String(), e.Prompt) || !strings.Contains(log.String(), "generated 4 tokens") {
		t.Fatalf("output %q, log %q", out.String(), log.String())
	}
	if err := run(&out, &log, path, config{prompt: e.Prompt, tokens: 3, temp: 0.8, seed: 3}); err != nil {
		t.Fatal(err)
	}
}

// TestSlots imports the fixtures with a cache of three sequences, as
// `mmf-import -graph -slots 3` does, and runs the prompt in slots 2 and 0
// as one block, then the greedy continuation of both, one token each per
// block: every logit must match the oracle, so no sequence sees another's
// keys.
func TestSlots(t *testing.T) { eachFixture(t, testSlots) }

func testSlots(t *testing.T, e *expected) {
	f, err := mmf.Open(importFixtureWith(t, e, importer.LlamaOptions{Context: 64, Slots: 3}))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	prog, err := interp.Load(f)
	if err != nil {
		t.Fatal(err)
	}
	prog.FastPaths = true
	s := prog.NewSession()
	run := func(ids []int32, positions, slots []int64) []float32 {
		t.Helper()
		n := len(ids)
		tokens, _ := ref.New(mmf.I32, []int{1, n})
		pos, _ := ref.New(mmf.I64, []int{1, n})
		sl, _ := ref.New(mmf.I64, []int{1, n})
		for i := range ids {
			tokens.SetBits(i, uint64(uint32(ids[i])))
			pos.SetBits(i, uint64(positions[i]))
			sl.SetBits(i, uint64(slots[i]))
		}
		out, err := s.Run(map[string]*ref.Tensor{"tokens": tokens, "positions": pos, "slots": sl})
		if err != nil {
			t.Fatal(err)
		}
		for _, v := range out {
			return f32s(v, 0, v.NumElems())
		}
		t.Fatal("no output")
		return nil
	}

	p := len(e.PromptIDs)
	var ids []int32
	var positions, slots []int64
	for _, slot := range []int64{2, 0} {
		for i, id := range e.PromptIDs {
			ids = append(ids, id)
			positions = append(positions, int64(i))
			slots = append(slots, slot)
		}
	}
	all := run(ids, positions, slots)
	want := f32b64(t, e.PromptLogits)
	near(t, "prompt in slot 2", all[:p*e.Vocab], want, 2e-5)
	near(t, "prompt in slot 0", all[p*e.Vocab:], want, 2e-5)

	for i, next := range e.Greedy[:len(e.Greedy)-1] {
		all = run([]int32{next, next}, []int64{int64(p + i), int64(p + i)}, []int64{0, 2})
		want := f32b64(t, e.GreedyLogits[i+1])
		near(t, "step in slot 0", all[:e.Vocab], want, 2e-5)
		near(t, "step in slot 2", all[e.Vocab:], want, 2e-5)
	}
}
