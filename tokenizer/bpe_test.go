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

package tokenizer

import (
	"reflect"
	"testing"

	"github.com/gsoultan/mmf"
)

// The expected words are those of the pre-tokenizers' regular expressions
// as Python's regex module splits them.
func TestSplit(t *testing.T) {
	for _, c := range []struct {
		text  string
		pre   int
		words []string
	}{
		{"Hello world", preLlama3, []string{"Hello", " world"}},
		{"I'm here, they'RE", preLlama3, []string{"I", "'m", " here", ",", " they", "'RE"}},
		{"1234567 x", preLlama3, []string{"123", "456", "7", " x"}},
		{"1234567 x", preQwen2, []string{"1", "2", "3", "4", "5", "6", "7", " x"}},
		{"a  b", preLlama3, []string{"a", " ", " b"}},
		{"a   ", preLlama3, []string{"a", "   "}},
		{"x\n\n  y", preLlama3, []string{"x", "\n\n", " ", " y"}},
		{"f(x) += 1;\n", preLlama3, []string{"f", "(x", ")", " +=", " ", "1", ";\n"}},
		{"café 日本", preLlama3, []string{"café", " 日本"}},
		{" 😀!", preLlama3, []string{" 😀!"}},
	} {
		if got := split(c.text, c.pre); !reflect.DeepEqual(got, c.words) {
			t.Errorf("split(%q, %d) = %q, want %q", c.text, c.pre, got, c.words)
		}
	}
}

// bpeVocab builds a byte-level vocabulary: the 256 byte characters, the
// merged pieces in merge order, and an end-of-text control token.
func bpeVocab(pre string, merges ...string) mmf.Metadata {
	var tokens []string
	for b := 0; b < 256; b++ {
		tokens = append(tokens, string(byteRune[b]))
	}
	types := make([]int64, len(tokens))
	for i := range types {
		types[i] = typeNormal
	}
	for _, m := range merges {
		for i := 0; i < len(m); i++ {
			if m[i] == ' ' && i > 0 {
				tokens = append(tokens, m[:i]+m[i+1:])
				types = append(types, typeNormal)
				break
			}
		}
	}
	tokens = append(tokens, "<|endoftext|>")
	types = append(types, typeControl)
	eot := int64(len(tokens) - 1)
	return mmf.Metadata{
		"tokenizer.ggml.model":        "gpt2",
		"tokenizer.ggml.pre":          pre,
		"tokenizer.ggml.tokens":       tokens,
		"tokenizer.ggml.token_type":   types,
		"tokenizer.ggml.merges":       merges,
		"tokenizer.ggml.bos_token_id": eot,
		"tokenizer.ggml.eos_token_id": eot,
	}
}

func TestBPEEncode(t *testing.T) {
	// Ġ is the byte character of the space. Tokens 256.. in order: ab,
	// Ġa, bc, Ġab; abc is not a token.
	md := bpeVocab("qwen2", "a b", "Ġ a", "b c", "Ġa b")
	tok, err := New(md)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		text string
		want []int32
	}{
		{"", nil},
		// ab merges first (rank 0), so bc never forms.
		{"abc", []int32{256, 'c'}},
		// In " ab", a b (rank 0) wins over Ġ a (rank 1), and Ġ ab is not a
		// merge; in " ac", Ġ a merges.
		{"x ab", []int32{'x', 32, 256}},
		{"x ac", []int32{'x', 257, 'c'}},
		// The two UTF-8 bytes of é are printable byte characters.
		{"é", []int32{0xC3, 0xA9}},
	} {
		got := tok.Encode(c.text, true)
		if len(got) == 0 && len(c.want) == 0 {
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("Encode(%q) = %v, want %v", c.text, got, c.want)
		}
		if d := tok.Decode(got); d != c.text {
			t.Errorf("Decode(Encode(%q)) = %q", c.text, d)
		}
	}
	// Qwen 2 adds no BOS; Llama 3 does, and keeps a word that is a token
	// whole even when the merges would not build it.
	md = bpeVocab("llama-bpe", "a b", "b c")
	md["tokenizer.ggml.tokens"] = append(md["tokenizer.ggml.tokens"].([]string), "abc")
	md["tokenizer.ggml.token_type"] = append(md["tokenizer.ggml.token_type"].([]int64), typeNormal)
	tok, err = New(md)
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Encode("abc", true); !reflect.DeepEqual(got, []int32{tok.BOS(), 259}) {
		t.Errorf("llama-bpe Encode(abc) = %v", got)
	}
	if got := tok.Decode([]int32{tok.BOS(), 256}); got != "ab" {
		t.Errorf("Decode skips control tokens: %q", got)
	}
}

func TestBPERejects(t *testing.T) {
	md := bpeVocab("gpt-4o")
	if _, err := New(md); err == nil {
		t.Error("accepted an unknown pre-tokenizer")
	}
	md = bpeVocab("qwen2", "ab")
	if _, err := New(md); err == nil {
		t.Error("accepted a merge without a space")
	}
	md = bpeVocab("qwen2", "a b")
	delete(md, "tokenizer.ggml.merges")
	if _, err := New(md); err == nil {
		t.Error("accepted a vocabulary without merges")
	}
}
