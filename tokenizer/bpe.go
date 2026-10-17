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
	"container/heap"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/gsoultan/mmf"
)

// Tokenizer is what a language model runner needs from a tokenizer.
type Tokenizer interface {
	Encode(text string, bos bool) []int32
	Decode(ids []int32) string
	Piece(id int32) []byte
	VocabSize() int
	BOS() int32
	EOS() int32
}

// New builds the tokenizer a file's metadata describes: SentencePiece for
// tokenizer.ggml.model "llama", byte-level BPE for "gpt2".
func New(md mmf.Metadata) (Tokenizer, error) {
	if m, _ := md["tokenizer.ggml.model"].(string); m == "gpt2" {
		return BPEFromMetadata(md)
	}
	return FromMetadata(md)
}

// Pre-tokenizers: how text is split into words before the merges run.
const (
	preLlama3 = iota // \p{N}{1,3}: digits in runs of up to three
	preQwen2         // \p{N}: one digit per word
)

// BPE is the byte-level BPE tokenizer of GGUF files whose
// tokenizer.ggml.model is "gpt2" (Llama 3, Qwen 2 and others). Text is
// split into words by the pre-tokenizer named in tokenizer.ggml.pre, each
// word's bytes are mapped to printable characters as GPT-2 does, and the
// pairs in tokenizer.ggml.merges are merged lowest rank first. It follows
// llama.cpp's llm_tokenizer_bpe, so the ids match llama.cpp's for the same
// model.
type BPE struct {
	tokens       []string
	types        []int64
	ids          map[string]int32
	ranks        map[string]int // "left right" to merge rank
	bos, eos     int32
	addBOS       bool
	pre          int
	ignoreMerges bool // a word that is a token is not split
}

// BPEFromMetadata builds a byte-level BPE tokenizer from a file's metadata.
func BPEFromMetadata(md mmf.Metadata) (*BPE, error) {
	if m, _ := md["tokenizer.ggml.model"].(string); m != "gpt2" {
		return nil, fmt.Errorf("tokenizer model %q: want \"gpt2\"", m)
	}
	tokens, _ := md["tokenizer.ggml.tokens"].([]string)
	if len(tokens) == 0 || len(tokens) > 1<<24 {
		return nil, fmt.Errorf("tokenizer.ggml.tokens is missing or invalid")
	}
	merges, _ := md["tokenizer.ggml.merges"].([]string)
	if len(merges) == 0 || len(merges) > 1<<24 {
		return nil, fmt.Errorf("tokenizer.ggml.merges is missing or invalid")
	}
	t := &BPE{
		tokens: tokens,
		types:  make([]int64, len(tokens)),
		ids:    make(map[string]int32, len(tokens)),
		ranks:  make(map[string]int, len(merges)),
		bos:    int32(mdInt(md, "tokenizer.ggml.bos_token_id", 1)),
		eos:    int32(mdInt(md, "tokenizer.ggml.eos_token_id", 2)),
	}
	switch pre, _ := md["tokenizer.ggml.pre"].(string); pre {
	case "llama3", "llama-v3", "llama-bpe":
		t.pre, t.ignoreMerges = preLlama3, true
		t.addBOS = mdBool(md, "tokenizer.ggml.add_bos_token", true)
	case "qwen2":
		t.pre = preQwen2
		t.addBOS = mdBool(md, "tokenizer.ggml.add_bos_token", false)
	default:
		return nil, fmt.Errorf("BPE pre-tokenizer %q is not supported", pre)
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
	for i, s := range tokens {
		if _, dup := t.ids[s]; !dup {
			t.ids[s] = int32(i)
		}
	}
	for i, m := range merges {
		// The pair splits at the first space after its first character,
		// as in llama.cpp, so a left part may not contain one.
		if k := strings.IndexByte(m[min(1, len(m)):], ' '); k < 0 {
			return nil, fmt.Errorf("merge %d %q has no space", i, m)
		}
		if _, dup := t.ranks[m]; !dup {
			t.ranks[m] = i
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
func (t *BPE) VocabSize() int { return len(t.tokens) }

// BOS returns the beginning-of-sequence token id.
func (t *BPE) BOS() int32 { return t.bos }

// EOS returns the end-of-sequence token id.
func (t *BPE) EOS() int32 { return t.eos }

// byteRune and runeByte are GPT-2's map between bytes and the printable
// characters that stand for them in the vocabulary.
var byteRune [256]rune
var runeByte = map[rune]byte{}

func init() {
	n := 0
	for b := 0; b < 256; b++ {
		if b >= '!' && b <= '~' || b >= 0xA1 && b <= 0xAC || b >= 0xAE {
			byteRune[b] = rune(b)
		} else {
			byteRune[b] = rune(256 + n)
			n++
		}
		runeByte[byteRune[b]] = byte(b)
	}
}

// split cuts text into the words of the Llama 3 and Qwen 2 pre-tokenizer,
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
//	 ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// with \p{N} for \p{N}{1,3} in Qwen 2's. It follows llama.cpp's
// hand-written matcher for this expression, which Go's regexp cannot run
// because of the lookahead.
func split(text string, pre int) []string {
	cpts := []rune(text)
	n := len(cpts)
	const none = -1
	get := func(i int) rune {
		if i < n {
			return cpts[i]
		}
		return none
	}
	isLetter := func(i int) bool { return i < n && unicode.IsLetter(cpts[i]) }
	isNumber := func(i int) bool { return i < n && unicode.IsNumber(cpts[i]) }
	isSpace := func(i int) bool { return i < n && unicode.IsSpace(cpts[i]) }
	// defined is false past the end, like llama.cpp's undefined flags.
	defined := func(i int) bool { return i < n }

	var words []string
	start := 0
	add := func(end int) int {
		l := end - start
		if l > 0 {
			words = append(words, string(cpts[start:end]))
		}
		start = end
		return l
	}
	for pos := 0; pos < n; {
		c := cpts[pos]
		if c == '\'' && pos+1 < n {
			nx := unicode.ToLower(cpts[pos+1])
			if nx == 's' || nx == 't' || nx == 'm' || nx == 'd' {
				pos += add(pos + 2)
				continue
			}
			if pos+2 < n {
				nn := unicode.ToLower(cpts[pos+2])
				if nx == 'r' && nn == 'e' || nx == 'v' && nn == 'e' || nx == 'l' && nn == 'l' {
					pos += add(pos + 3)
					continue
				}
			}
		}
		// [^\r\n\p{L}\p{N}]?\p{L}+
		if !(c == '\r' || c == '\n' || isNumber(pos)) {
			if isLetter(pos) || isLetter(pos+1) {
				pos++
				for isLetter(pos) {
					pos++
				}
				add(pos)
				continue
			}
		}
		// \p{N}{1,3} or \p{N}
		if isNumber(pos) {
			if pre == preQwen2 {
				pos++
				add(pos)
				continue
			}
			ini := pos
			for isNumber(pos) {
				pos++
				if pos-ini >= 3 {
					add(pos)
					ini = pos
				}
			}
			add(pos)
			continue
		}
		//  ?[^\s\p{L}\p{N}]+[\r\n]*
		p2 := pos
		if c == ' ' {
			p2 = pos + 1
		}
		other := func(i int) bool { return defined(i) && !isSpace(i) && !isLetter(i) && !isNumber(i) }
		if other(p2) {
			pos = p2
			for other(pos) {
				pos++
			}
			for c2 := get(pos); c2 == '\r' || c2 == '\n'; c2 = get(pos) {
				pos++
			}
			add(pos)
			continue
		}
		spaces, lastRN := 0, 0
		for isSpace(pos + spaces) {
			if c2 := cpts[pos+spaces]; c2 == '\r' || c2 == '\n' {
				lastRN = pos + spaces + 1
			}
			spaces++
		}
		switch {
		case lastRN > 0: // \s*[\r\n]+
			pos = lastRN
		case spaces > 1 && get(pos+spaces) != none: // \s+(?!\S)
			pos += spaces - 1
		case spaces > 0: // \s+
			pos += spaces
		default:
			pos++
		}
		add(pos)
	}
	return words
}

// bpeBigram is a mergeable pair of adjacent symbols.
type bpeBigram struct {
	left, right int
	rank        int
	text        string
}

// bpeQueue is a min-heap by rank; ties go to the leftmost pair.
type bpeQueue []bpeBigram

func (h bpeQueue) Len() int { return len(h) }
func (h bpeQueue) Less(i, j int) bool {
	return h[i].rank < h[j].rank || (h[i].rank == h[j].rank && h[i].left < h[j].left)
}
func (h bpeQueue) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *bpeQueue) Push(x any)   { *h = append(*h, x.(bpeBigram)) }
func (h *bpeQueue) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

// Encode tokenizes text, adding the beginning-of-sequence token first when
// the model asks for it and bos is true. Special tokens written in the text
// are tokenized as plain text, as llama.cpp does without parse_special.
func (t *BPE) Encode(text string, bos bool) []int32 {
	var out []int32
	if bos && t.addBOS {
		out = append(out, t.bos)
	}
	var sb strings.Builder
	for _, w := range split(text, t.pre) {
		sb.Reset()
		for _, b := range []byte(w) {
			sb.WriteRune(byteRune[b])
		}
		out = t.word(out, sb.String())
	}
	return out
}

// word appends the tokens of one byte-mapped word.
func (t *BPE) word(out []int32, w string) []int32 {
	if t.ignoreMerges {
		if id, ok := t.ids[w]; ok {
			return append(out, id)
		}
	}
	var syms []symbol
	for i := 0; i < len(w); {
		_, n := utf8.DecodeRuneInString(w[i:])
		syms = append(syms, symbol{prev: len(syms) - 1, next: len(syms) + 1, text: w[i : i+n]})
		i += n
	}
	if len(syms) == 0 {
		return out
	}
	syms[len(syms)-1].next = -1
	q := &bpeQueue{}
	add := func(l, r int) {
		if l < 0 || r < 0 {
			return
		}
		rank, ok := t.ranks[syms[l].text+" "+syms[r].text]
		if !ok {
			return
		}
		heap.Push(q, bpeBigram{left: l, right: r, rank: rank, text: syms[l].text + syms[r].text})
	}
	for i := 1; i < len(syms); i++ {
		add(i-1, i)
	}
	for q.Len() > 0 {
		bg := heap.Pop(q).(bpeBigram)
		l, r := &syms[bg.left], &syms[bg.right]
		// Skip pairs made stale by an earlier merge.
		if l.text == "" || r.text == "" || l.text+r.text != bg.text {
			continue
		}
		l.text = bg.text
		r.text = ""
		l.next = r.next
		if r.next >= 0 {
			syms[r.next].prev = bg.left
		}
		add(l.prev, bg.left)
		add(bg.left, l.next)
	}
	for i := 0; i >= 0; i = syms[i].next {
		s := syms[i].text
		if id, ok := t.ids[s]; ok {
			out = append(out, id)
			continue
		}
		// Not in the vocabulary: llama.cpp looks up each byte of the
		// mapped text on its own and drops those it does not find.
		for j := 0; j < len(s); j++ {
			if id, ok := t.ids[s[j:j+1]]; ok {
				out = append(out, id)
			}
		}
	}
	return out
}

// Piece returns the bytes token id stands for in decoded text. Control
// tokens such as BOS and EOS decode to nothing; user-defined tokens decode
// to their text as it is.
func (t *BPE) Piece(id int32) []byte {
	if id < 0 || int(id) >= len(t.tokens) {
		return nil
	}
	s := t.tokens[id]
	switch t.types[id] {
	case typeNormal:
		var b []byte
		for _, r := range s {
			if c, ok := runeByte[r]; ok {
				b = append(b, c)
			} else {
				b = utf8.AppendRune(b, r)
			}
		}
		return b
	case typeUserDefined:
		return []byte(s)
	}
	return nil
}

// Decode turns token ids back into text.
func (t *BPE) Decode(ids []int32) string {
	var b []byte
	for _, id := range ids {
		b = append(b, t.Piece(id)...)
	}
	return string(b)
}
