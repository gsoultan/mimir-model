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

package tokenizer

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/gsoultan/mim"
)

// vocab builds a small vocabulary: the specials, all 256 byte tokens and
// the given pieces, with scores falling in list order.
func vocab(pieces ...string) mim.Metadata {
	tokens := []string{"<unk>", "<s>", "</s>"}
	types := []int64{2, 3, 3}
	scores := []float64{0, 0, 0}
	for b := 0; b < 256; b++ {
		tokens = append(tokens, fmt.Sprintf("<0x%02X>", b))
		types = append(types, 6)
		scores = append(scores, 0)
	}
	for i, p := range pieces {
		tokens = append(tokens, p)
		types = append(types, 1)
		scores = append(scores, -float64(i))
	}
	return mim.Metadata{
		"tokenizer.ggml.model":      "llama",
		"tokenizer.ggml.tokens":     tokens,
		"tokenizer.ggml.scores":     scores,
		"tokenizer.ggml.token_type": types,
	}
}

func TestEncode(t *testing.T) {
	// Pieces 259.. in order: ▁, a, b, ab, ▁ab, bc, c.
	tok, err := FromMetadata(vocab("▁", "a", "b", "ab", "▁ab", "bc", "c"))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		text string
		want []int32
	}{
		{"", []int32{1}},
		// ab merges first (higher score than bc), then ▁ab.
		{"abc", []int32{1, 263, 265}},
		// é is not in the vocabulary: its two UTF-8 bytes fall back.
		{"é", []int32{1, 259, 3 + 0xC3, 3 + 0xA9}},
	} {
		if got := tok.Encode(c.text, true); !reflect.DeepEqual(got, c.want) {
			t.Errorf("Encode(%q) = %v, want %v", c.text, got, c.want)
		}
		if got := tok.Decode(tok.Encode(c.text, true)); got != c.text {
			t.Errorf("Decode(Encode(%q)) = %q", c.text, got)
		}
	}
	if got := tok.Encode("a", false); !reflect.DeepEqual(got, []int32{259, 260}) {
		t.Errorf("without BOS: %v", got)
	}
}

func TestFromMetadataRejects(t *testing.T) {
	md := vocab()
	md["tokenizer.ggml.model"] = "gpt2"
	if _, err := FromMetadata(md); err == nil {
		t.Error("accepted a gpt2 tokenizer")
	}
	md = vocab()
	md["tokenizer.ggml.scores"] = []float64{1}
	if _, err := FromMetadata(md); err == nil {
		t.Error("accepted mismatched scores")
	}
	md = vocab()
	md["tokenizer.ggml.eos_token_id"] = int64(1 << 20)
	if _, err := FromMetadata(md); err == nil {
		t.Error("accepted an out-of-range special token")
	}
}
