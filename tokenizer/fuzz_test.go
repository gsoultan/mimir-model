package tokenizer

import (
	"strings"
	"testing"

	mim "github.com/gsoultan/mimir-model"
)

// FuzzMetadata builds a tokenizer from arbitrary metadata, as a hostile
// GGUF or .mim file could carry, and runs text through it: building may
// fail, but nothing may panic, and what Encode gives Decode must take.
func FuzzMetadata(f *testing.F) {
	f.Add("llama", "", "a b <0x41> ▁hello ▁ <s> </s>", "1 1 6 1 1 3 3", "0 0 -1 -2 -3 0 0", int64(5), int64(6), "hello world")
	f.Add("gpt2", "llama3", "a b Ġ ab Ġa <|begin_of_text|> <|end_of_text|>", "1 1 1 1 1 3 3", "", int64(5), int64(6), "ab a 123 x")
	f.Add("gpt2", "qwen2", "a b Ġ ab Ġa <|im_start|> <|im_end|>", "1 1 1 1 1 3 3", "", int64(5), int64(6), "ab a 123 x")
	f.Fuzz(func(t *testing.T, model, pre, tokens, types, scores string, bos, eos int64, text string) {
		md := mim.Metadata{
			"tokenizer.ggml.model":        model,
			"tokenizer.ggml.pre":          pre,
			"tokenizer.ggml.tokens":       strings.Split(tokens, " "),
			"tokenizer.ggml.bos_token_id": bos,
			"tokenizer.ggml.eos_token_id": eos,
			"tokenizer.ggml.merges":       merges(strings.Split(tokens, " ")),
		}
		if types != "" {
			var ty []int64
			for _, s := range strings.Split(types, " ") {
				var v int64
				for _, c := range s {
					v = v*10 + int64(c-'0')
				}
				ty = append(ty, v%7)
			}
			md["tokenizer.ggml.token_type"] = ty
		}
		if scores != "" {
			var sc []float64
			for i, s := range strings.Split(scores, " ") {
				sc = append(sc, float64(len(s)-i))
			}
			md["tokenizer.ggml.scores"] = sc
		}
		tok, err := New(md)
		if err != nil {
			return
		}
		ids := tok.Encode(text, true)
		for _, id := range ids {
			if id < 0 || int(id) >= tok.VocabSize() {
				t.Fatalf("Encode gave id %d of %d", id, tok.VocabSize())
			}
		}
		_ = tok.Decode(ids)
		_ = tok.Decode([]int32{-1, int32(tok.VocabSize()), 1 << 30})
		_ = tok.Piece(tok.BOS())
		_ = tok.Piece(tok.EOS())
	})
}

// merges pairs up neighbouring tokens as BPE merges.
func merges(tokens []string) []string {
	var m []string
	for i := 0; i+1 < len(tokens); i++ {
		m = append(m, tokens[i]+" "+tokens[i+1])
	}
	return m
}
