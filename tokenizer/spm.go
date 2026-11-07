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

// Package tokenizer turns text into token ids and back for models imported
// with their tokenizer in the metadata section.
//
// It implements the SentencePiece-style tokenizer of GGUF files whose
// tokenizer.ggml.model is "llama" (Llama 1 and 2, TinyLlama, Mistral and
// others): the vocabulary and merge scores come from the metadata keys
// tokenizer.ggml.tokens, tokenizer.ggml.scores and tokenizer.ggml.token_type,
// and the algorithm follows llama.cpp's llm_tokenizer_spm, so the ids match
// what llama.cpp produces for the same model.
package tokenizer

import (
	"fmt"
	"strings"
	"unicode/utf8"

	mim "github.com/gsoultan/mimir-model"
)

// Token types of tokenizer.ggml.token_type.
const (
	typeNormal      = 1
	typeUnknown     = 2
	typeControl     = 3
	typeUserDefined = 4
	typeUnused      = 5
	typeByte        = 6
)

const space = "▁" // ▁, SentencePiece's word boundary marker

// SPM is a SentencePiece-style byte-fallback BPE tokenizer.
type SPM struct {
	tokens     []string
	scores     []float32
	types      []int64
	ids        map[string]int32
	bytes      [256]int32 // byte value to its <0xXX> token, -1 if absent
	bos, eos   int32
	unk        int32
	addBOS     bool
	addSpace   bool
	maxTextLen int
}

func mdInt(md mim.Metadata, key string, def int64) int64 {
	switch v := md[key].(type) {
	case int64:
		return v
	case uint64:
		return int64(v)
	}
	return def
}

func mdBool(md mim.Metadata, key string, def bool) bool {
	if v, ok := md[key].(bool); ok {
		return v
	}
	return def
}

// FromMetadata builds the tokenizer described by a file's metadata.
func FromMetadata(md mim.Metadata) (*SPM, error) {
	if m, _ := md["tokenizer.ggml.model"].(string); m != "llama" {
		return nil, fmt.Errorf("tokenizer model %q: only \"llama\" (SentencePiece) is supported", m)
	}
	tokens, _ := md["tokenizer.ggml.tokens"].([]string)
	if len(tokens) == 0 || len(tokens) > 1<<24 {
		return nil, fmt.Errorf("tokenizer.ggml.tokens is missing or invalid")
	}
	t := &SPM{
		tokens:   tokens,
		scores:   make([]float32, len(tokens)),
		types:    make([]int64, len(tokens)),
		ids:      make(map[string]int32, len(tokens)),
		bos:      int32(mdInt(md, "tokenizer.ggml.bos_token_id", 1)),
		eos:      int32(mdInt(md, "tokenizer.ggml.eos_token_id", 2)),
		unk:      int32(mdInt(md, "tokenizer.ggml.unknown_token_id", 0)),
		addBOS:   mdBool(md, "tokenizer.ggml.add_bos_token", true),
		addSpace: mdBool(md, "tokenizer.ggml.add_space_prefix", true),
	}
	if s, ok := md["tokenizer.ggml.scores"].([]float64); ok {
		if len(s) != len(tokens) {
			return nil, fmt.Errorf("%d scores for %d tokens", len(s), len(tokens))
		}
		for i, x := range s {
			t.scores[i] = float32(x)
		}
	}
	if ty, ok := md["tokenizer.ggml.token_type"].([]int64); ok {
		if len(ty) != len(tokens) {
			return nil, fmt.Errorf("%d token types for %d tokens", len(ty), len(tokens))
		}
		copy(t.types, ty)
	} else {
		for i := range t.types {
			t.types[i] = typeNormal
		}
	}
	for i := range t.bytes {
		t.bytes[i] = -1
	}
	for i, s := range tokens {
		if _, dup := t.ids[s]; !dup {
			t.ids[s] = int32(i)
		}
		t.maxTextLen = max(t.maxTextLen, len(s))
		if t.types[i] == typeByte {
			var b byte
			if _, err := fmt.Sscanf(s, "<0x%02X>", &b); err == nil {
				t.bytes[b] = int32(i)
			}
		}
	}
	for _, id := range []int32{t.bos, t.eos} {
		if id < 0 || int(id) >= len(tokens) {
			return nil, fmt.Errorf("special token id %d is out of range", id)
		}
	}
	return t, nil
}

// VocabSize returns the number of tokens.
func (t *SPM) VocabSize() int { return len(t.tokens) }

// BOS returns the beginning-of-sequence token id.
func (t *SPM) BOS() int32 { return t.bos }

// EOS returns the end-of-sequence token id.
func (t *SPM) EOS() int32 { return t.eos }

// symbol is a span of the input in the merge list.
type symbol struct {
	prev, next int
	start, end int // its bytes in the text; end < 0 once merged away
}

type bigram struct {
	left, right int
	score       float32
	size        int
}

// before says whether b merges ahead of o: the higher score first, and of
// equal scores the leftmost pair.
func (b bigram) before(o bigram) bool {
	return b.score > o.score || b.score == o.score && b.left < o.left
}

// bigrams is a max-heap by before.
type bigrams []bigram

func (h *bigrams) push(b bigram) {
	q := append(*h, b)
	for i := len(q) - 1; i > 0; {
		p := (i - 1) / 2
		if !q[i].before(q[p]) {
			break
		}
		q[i], q[p] = q[p], q[i]
		i = p
	}
	*h = q
}

func (h *bigrams) pop() bigram {
	q := *h
	top := q[0]
	last := len(q) - 1
	q[0] = q[last]
	q = q[:last]
	for i := 0; ; {
		c := 2*i + 1
		if c >= last {
			break
		}
		if r := c + 1; r < last && q[r].before(q[c]) {
			c = r
		}
		if !q[c].before(q[i]) {
			break
		}
		q[i], q[c] = q[c], q[i]
		i = c
	}
	*h = q
	return top
}

// Encode tokenizes text, adding the beginning-of-sequence token first when
// the model asks for it and bos is true.
func (t *SPM) Encode(text string, bos bool) []int32 {
	var out []int32
	if bos && t.addBOS {
		out = append(out, t.bos)
	}
	if text == "" {
		return out
	}
	if t.addSpace {
		text = " " + text
	}
	text = strings.ReplaceAll(text, " ", space)

	// One symbol per UTF-8 character; invalid bytes are one symbol each.
	// A symbol is a span of text, and adjacent ones merge into the span
	// over both, so no merge copies text.
	syms := make([]symbol, 0, len(text))
	for i := 0; i < len(text); {
		_, n := utf8.DecodeRuneInString(text[i:])
		syms = append(syms, symbol{prev: len(syms) - 1, next: len(syms) + 1, start: i, end: i + n})
		i += n
	}
	syms[len(syms)-1].next = -1

	var h bigrams
	add := func(l, r int) {
		if l < 0 || r < 0 {
			return
		}
		s := text[syms[l].start:syms[r].end]
		if len(s) > t.maxTextLen {
			return
		}
		id, ok := t.ids[s]
		if !ok {
			return
		}
		h.push(bigram{left: l, right: r, score: t.scores[id], size: len(s)})
	}
	for i := 1; i < len(syms); i++ {
		add(i-1, i)
	}
	for len(h) > 0 {
		bg := h.pop()
		l, r := &syms[bg.left], &syms[bg.right]
		// Skip pairs made stale by an earlier merge.
		if l.end < 0 || r.end < 0 || l.end-l.start+r.end-r.start != bg.size {
			continue
		}
		l.end = r.end
		r.end = -1
		l.next = r.next
		if r.next >= 0 {
			syms[r.next].prev = bg.left
		}
		add(l.prev, bg.left)
		add(bg.left, l.next)
	}
	for i := 0; i >= 0; i = syms[i].next {
		s := text[syms[i].start:syms[i].end]
		if id, ok := t.ids[s]; ok {
			out = append(out, id)
			continue
		}
		for j := 0; j < len(s); j++ {
			if id := t.bytes[s[j]]; id >= 0 {
				out = append(out, id)
			} else {
				out = append(out, t.unk)
			}
		}
	}
	return out
}

// Piece returns the bytes token id stands for in decoded text: the word
// marker becomes a space, byte tokens become their byte, and control
// tokens such as BOS and EOS decode to nothing.
func (t *SPM) Piece(id int32) []byte {
	if id < 0 || int(id) >= len(t.tokens) {
		return nil
	}
	s := t.tokens[id]
	switch t.types[id] {
	case typeNormal, typeUserDefined:
		return []byte(strings.ReplaceAll(s, space, " "))
	case typeUnknown:
		return []byte("▅") // ▅, as llama.cpp prints it
	case typeControl, typeUnused:
		return nil
	case typeByte:
		var b byte
		if _, err := fmt.Sscanf(s, "<0x%02X>", &b); err == nil {
			return []byte{b}
		}
	}
	return nil
}

// Decode turns token ids back into text. The space the tokenizer adds in
// front of the text is removed.
func (t *SPM) Decode(ids []int32) string {
	var b []byte
	for _, id := range ids {
		b = append(b, t.Piece(id)...)
	}
	s := string(b)
	if t.addSpace {
		s = strings.TrimPrefix(s, " ")
	}
	return s
}
