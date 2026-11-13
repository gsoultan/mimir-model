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
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	mim "github.com/gsoultan/mimir-model"
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
func New(md mim.Metadata) (Tokenizer, error) {
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

	// words remembers the tokens of words seen before (see remember), as
	// most text repeats its words; mu guards it.
	mu    sync.Mutex
	words map[string][]int32
}

// BPEFromMetadata builds a byte-level BPE tokenizer from a file's metadata.
func BPEFromMetadata(md mim.Metadata) (*BPE, error) {
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
	case "dbrx", "smaug-bpe":
		// Llama 3's expression, which Phi 4 and others name so, without
		// its rule that a word which is a token is not split.
		t.pre = preLlama3
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

// split cuts text into the words of the Llama 3 and Qwen 2 pre-tokenizer.
func split(text string, pre int) []string {
	var words []string
	splitWords(text, pre, func(w string) { words = append(words, w) })
	return words
}

// splitWords calls fn with each word of text, in order, cut as the Llama 3
// and Qwen 2 pre-tokenizer does,
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
//	 ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// with \p{N} for \p{N}{1,3} in Qwen 2's. It follows llama.cpp's
// hand-written matcher for this expression, which Go's regexp cannot run
// because of the lookahead. A word is a substring of text, so it costs no
// copy, unless text is not valid UTF-8: then each word is its code points
// written back, as []rune(text) reads them.
func splitWords(text string, pre int, fn func(w string)) {
	cpts := []rune(text)
	n := len(cpts)
	// offs[i] is where code point i starts in text.
	var offs []int
	if utf8.ValidString(text) {
		offs = make([]int, n+1)
		i := 0
		for off := range text {
			offs[i] = off
			i++
		}
		offs[n] = len(text)
	}
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

	start := 0
	add := func(end int) int {
		l := end - start
		if l > 0 {
			if offs != nil {
				fn(text[offs[start]:offs[end]])
			} else {
				fn(string(cpts[start:end]))
			}
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
}

// bpeSymbol is a span of a word in the merge list.
type bpeSymbol struct {
	prev, next int
	start, end int // its bytes in encoder.buf; end < 0 once merged away
}

// bpeBigram is a mergeable pair of adjacent symbols.
type bpeBigram struct {
	left, right int
	rank        int
	size        int // bytes of the pair, to notice one an earlier merge left stale
}

// before says whether b merges ahead of o: the lower rank first, and of
// equal ranks the leftmost pair.
func (b bpeBigram) before(o bpeBigram) bool {
	return b.rank < o.rank || b.rank == o.rank && b.left < o.left
}

// encoder holds what tokenizing a word needs, kept from one word to the
// next so that a text costs a few allocations rather than a few per
// word.
type encoder struct {
	buf  []byte      // the word's bytes mapped to characters (see byteRune)
	syms []bpeSymbol // the word's symbols, merged in place
	q    []bpeBigram // the pairs to merge, a min-heap by before
	key  []byte      // scratch for looking up a pair's rank
}

func (e *encoder) push(b bpeBigram) {
	e.q = append(e.q, b)
	for i := len(e.q) - 1; i > 0; {
		p := (i - 1) / 2
		if !e.q[i].before(e.q[p]) {
			break
		}
		e.q[i], e.q[p] = e.q[p], e.q[i]
		i = p
	}
}

func (e *encoder) pop() bpeBigram {
	top := e.q[0]
	last := len(e.q) - 1
	e.q[0] = e.q[last]
	e.q = e.q[:last]
	for i := 0; ; {
		c := 2*i + 1
		if c >= last {
			break
		}
		if r := c + 1; r < last && e.q[r].before(e.q[c]) {
			c = r
		}
		if !e.q[c].before(e.q[i]) {
			break
		}
		e.q[i], e.q[c] = e.q[c], e.q[i]
		i = c
	}
	return top
}

// add queues the pair of symbols l and r when the merges have it.
func (e *encoder) add(t *BPE, l, r int) {
	if l < 0 || r < 0 {
		return
	}
	ls, rs := &e.syms[l], &e.syms[r]
	e.key = append(append(append(e.key[:0], e.buf[ls.start:ls.end]...), ' '), e.buf[rs.start:rs.end]...)
	rank, ok := t.ranks[string(e.key)] // no copy: a lookup converts for free
	if !ok {
		return
	}
	e.push(bpeBigram{left: l, right: r, rank: rank, size: ls.end - ls.start + rs.end - rs.start})
}

// Encode tokenizes text, adding the beginning-of-sequence token first when
// the model asks for it and bos is true. Special tokens written in the text
// are tokenized as plain text, as llama.cpp does without parse_special.
func (t *BPE) Encode(text string, bos bool) []int32 {
	var out []int32
	if bos && t.addBOS {
		out = append(out, t.bos)
	}
	var e encoder
	splitWords(text, t.pre, func(w string) {
		if ids, ok := t.recall(w); ok {
			out = append(out, ids...)
			return
		}
		n := len(out)
		out = t.word(out, &e, w)
		t.remember(w, out[n:])
	})
	return out
}

// cacheWords is the most words Encode remembers the tokens of, and
// cacheWordLen the longest; past the first the cache starts over.
const (
	cacheWords   = 1 << 15
	cacheWordLen = 64
)

// recall returns the tokens of a word seen before.
func (t *BPE) recall(w string) ([]int32, bool) {
	if len(w) > cacheWordLen {
		return nil, false
	}
	t.mu.Lock()
	ids, ok := t.words[w]
	t.mu.Unlock()
	return ids, ok
}

// remember keeps the tokens of w for recall.
func (t *BPE) remember(w string, ids []int32) {
	if len(w) > cacheWordLen {
		return
	}
	ids = append([]int32(nil), ids...) // the caller's slice grows on
	t.mu.Lock()
	if t.words == nil || len(t.words) >= cacheWords {
		t.words = make(map[string][]int32, 1024)
	}
	t.words[strings.Clone(w)] = ids // w is a substring of a whole text
	t.mu.Unlock()
}

// word appends the tokens of one word.
func (t *BPE) word(out []int32, e *encoder, w string) []int32 {
	e.buf = e.buf[:0]
	for i := 0; i < len(w); i++ {
		e.buf = utf8.AppendRune(e.buf, byteRune[w[i]])
	}
	buf := e.buf
	if t.ignoreMerges {
		if id, ok := t.ids[string(buf)]; ok {
			return append(out, id)
		}
	}
	e.syms = e.syms[:0]
	for i := 0; i < len(buf); {
		_, n := utf8.DecodeRune(buf[i:])
		e.syms = append(e.syms, bpeSymbol{prev: len(e.syms) - 1, next: len(e.syms) + 1, start: i, end: i + n})
		i += n
	}
	syms := e.syms
	if len(syms) == 0 {
		return out
	}
	syms[len(syms)-1].next = -1
	e.q = e.q[:0]
	for i := 1; i < len(syms); i++ {
		e.add(t, i-1, i)
	}
	for len(e.q) > 0 {
		bg := e.pop()
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
		e.add(t, l.prev, bg.left)
		e.add(t, bg.left, l.next)
	}
	for i := 0; i >= 0; i = syms[i].next {
		s := buf[syms[i].start:syms[i].end]
		if id, ok := t.ids[string(s)]; ok {
			out = append(out, id)
			continue
		}
		// Not in the vocabulary: llama.cpp looks up each byte of the
		// mapped text on its own and drops those it does not find.
		for j := range s {
			if id, ok := t.ids[string(s[j:j+1])]; ok {
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
