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
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/importer"
)

// The fixtures are the tiny models of cmd/mmf-run's tests, whose
// expected.json holds the oracle's greedy continuation of a prompt.

type expected struct {
	Model  string  `json:"model"`
	Prompt string  `json:"prompt"`
	Greedy []int32 `json:"greedy"`
}

// engineFor imports a fixture with a cache of the given slots and opens
// an engine on it.
func engineFor(t *testing.T, fixture string, slots int) (*engine, *expected) {
	t.Helper()
	dir := filepath.Join("..", "..", "testdata", fixture)
	b, err := os.ReadFile(filepath.Join(dir, "expected.json"))
	if err != nil {
		t.Fatal(err)
	}
	var e expected
	if err := json.Unmarshal(b, &e); err != nil {
		t.Fatal(err)
	}
	if b, err = os.ReadFile(filepath.Join(dir, e.Model)); err != nil {
		t.Fatal(err)
	}
	src, err := importer.ParseGGUF(b)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := importer.BuildLlama(src, importer.LlamaOptions{Context: 64, Slots: slots}); err != nil {
		t.Fatal(err)
	}
	w := mmf.NewWriter()
	if err := src.AddTo(w); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "model.mmf")
	if err := w.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	en, err := open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { en.f.Close() })
	go en.loop()
	return en, &e
}

// TestConcurrent runs more requests than slots at once, with prompts of
// different lengths: each must get what it gets alone, the oracle's
// greedy continuation for the fixture's prompt.
func TestConcurrent(t *testing.T) {
	for _, fixture := range []string{"llama", "qwen2"} {
		t.Run(fixture, func(t *testing.T) {
			e, x := engineFor(t, fixture, 3)
			alone, _, err := e.generate(x.Prompt, len(x.Greedy), 0, 1)
			if err != nil {
				t.Fatal(err)
			}
			if !equal(alone, x.Greedy) {
				t.Fatalf("alone: %v, oracle %v", alone, x.Greedy)
			}
			prompts := []string{x.Prompt, "Once", x.Prompt, "the little girl wore a red riding cloak", x.Prompt, "Once"}
			solo := make([][]int32, len(prompts))
			for i, p := range prompts {
				if solo[i], _, err = e.generate(p, len(x.Greedy), 0, 1); err != nil {
					t.Fatal(err)
				}
			}
			got := make([][]int32, len(prompts))
			errs := make([]error, len(prompts))
			var wg sync.WaitGroup
			for i, p := range prompts {
				wg.Add(1)
				go func() {
					defer wg.Done()
					got[i], _, errs[i] = e.generate(p, len(x.Greedy)-i%3, 0, 1)
				}()
			}
			wg.Wait()
			for i := range prompts {
				if errs[i] != nil {
					t.Fatal(errs[i])
				}
				if want := solo[i][:len(x.Greedy)-i%3]; !equal(got[i], want) {
					t.Errorf("request %d (%q): %v, alone %v", i, prompts[i], got[i], want)
				}
			}
		})
	}
}

// TestLongPrompt splits a prompt longer than a step's block over several
// steps while another request generates.
func TestLongPrompt(t *testing.T) {
	e, x := engineFor(t, "llama", 2)
	e.maxBlock = 4
	var wg sync.WaitGroup
	var a, b []int32
	var errA, errB error
	wg.Add(2)
	go func() { defer wg.Done(); a, _, errA = e.generate(x.Prompt, len(x.Greedy), 0, 1) }()
	go func() { defer wg.Done(); b, _, errB = e.generate("Once", 20, 0, 1) }()
	wg.Wait()
	if errA != nil || errB != nil {
		t.Fatal(errA, errB)
	}
	if !equal(a, x.Greedy) {
		t.Fatalf("%v, oracle %v", a, x.Greedy)
	}
	if len(b) == 0 {
		t.Fatal("the other request generated nothing")
	}
}

func TestHTTP(t *testing.T) {
	e, x := engineFor(t, "llama", 2)
	srv := httptest.NewServer(e.handler())
	defer srv.Close()
	body, _ := json.Marshal(map[string]any{"prompt": x.Prompt, "max_tokens": len(x.Greedy)})
	resp, err := http.Post(srv.URL+"/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out generateResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	// JSON replaces the invalid UTF-8 of the random model's text.
	var text string
	js, _ := json.Marshal(e.tok.Decode(x.Greedy))
	json.Unmarshal(js, &text)
	if !equal(out.Tokens, x.Greedy) || out.Text != text || out.PromptTokens == 0 {
		t.Fatalf("response %+v", out)
	}
	for _, bad := range []string{`{"prompt": "x", "max_tokens": -1}`, `{"prompt": "x", "temperature": -1}`, `not json`} {
		resp, err := http.Post(srv.URL+"/generate", "application/json", strings.NewReader(bad))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("%s: status %d", bad, resp.StatusCode)
		}
	}
}

func TestBench(t *testing.T) {
	e, x := engineFor(t, "llama", 2)
	var out bytes.Buffer
	if err := runBench(&out, e, 2, x.Prompt, 4); err != nil {
		t.Fatal(err)
	}
	if strings.Count(out.String(), "tok/s") != 2 {
		t.Fatalf("output %q", out.String())
	}
}

func TestNeedsSlots(t *testing.T) {
	dir := t.TempDir()
	b, err := os.ReadFile(filepath.Join("..", "..", "testdata", "llama", "tiny-q4km.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	src, _ := importer.ParseGGUF(b)
	if _, err := importer.BuildLlama(src, importer.LlamaOptions{Context: 64}); err != nil {
		t.Fatal(err)
	}
	w := mmf.NewWriter()
	src.AddTo(w)
	path := filepath.Join(dir, "one.mmf")
	if err := w.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	if _, err := open(path); err == nil || !strings.Contains(err.Error(), "-slots") {
		t.Fatalf("error %v", err)
	}
}

func equal(a, b []int32) bool {
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
