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
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"

	mim "github.com/gsoultan/mimir-model"
)

// HFFiles are the side files of a Hugging Face Transformers checkpoint,
// which its safetensors weights need to be a model: config.json and,
// when the model has them, generation_config.json,
// preprocessor_config.json and tokenizer.json. Missing files are nil.
type HFFiles struct {
	Config           []byte
	GenerationConfig []byte
	Preprocessor     []byte
	Tokenizer        []byte
}

// HFArchitectures names the Hugging Face model classes FromHuggingFace
// describes, by the architecture it records for each.
var HFArchitectures = map[string]string{
	"WhisperForConditionalGeneration": "whisper",
	"CLIPModel":                       "clip",
}

// FromHuggingFace adds to src, a safetensors checkpoint of Hugging Face
// Transformers, the metadata a runtime needs to run it, read from its
// side files: general.architecture, the hyperparameters under the
// architecture's prefix and the tokenizer under tokenizer.ggml, as GGUF
// names them. It knows speech to text with Whisper
// (WhisperForConditionalGeneration, "whisper") and images and text with
// CLIP (CLIPModel, "clip"), and returns the architecture. The weights
// keep their Hugging Face names.
//
// Whisper's tokenizer is byte-level BPE ("gpt2" with the "gpt2"
// pre-tokenizer), its added tokens (<|endoftext|>, the languages, tasks
// and timestamps) typed as control tokens; generation_config.json gives
// whisper.suppress_tokens, whisper.begin_suppress_tokens and
// whisper.multilingual. CLIP's
// tokenizer is "clip": lowercased words whose last piece ends in </w>.
// preprocessor_config.json gives CLIP's image size, mean and deviation.
func FromHuggingFace(src *Source, files HFFiles) (string, error) {
	if src.Format != "safetensors" {
		return "", fmt.Errorf("huggingface: a %s file: want safetensors", src.Format)
	}
	if files.Config == nil {
		return "", fmt.Errorf("huggingface: no config.json")
	}
	var cfg map[string]any
	if err := decodeJSON(files.Config, &cfg); err != nil {
		return "", fmt.Errorf("huggingface: config.json: %w", err)
	}
	arch := ""
	if as, ok := cfg["architectures"].([]any); ok {
		for _, a := range as {
			if s, _ := a.(string); HFArchitectures[s] != "" {
				arch = HFArchitectures[s]
				break
			}
		}
	}
	if arch == "" {
		switch cfg["model_type"] {
		case "whisper":
			arch = "whisper"
		case "clip":
			arch = "clip"
		default:
			return "", fmt.Errorf("huggingface: model %v of type %v: only Whisper and CLIP checkpoints are known", cfg["architectures"], cfg["model_type"])
		}
	}
	md := src.Metadata
	if md == nil {
		md = mim.Metadata{}
		src.Metadata = md
	}
	md["general.architecture"] = arch
	if name, _ := cfg["_name_or_path"].(string); name != "" {
		md["general.name"] = name
	}
	if files.Tokenizer == nil {
		return "", fmt.Errorf("huggingface: no tokenizer.json")
	}
	var err error
	switch arch {
	case "whisper":
		err = whisperMetadata(md, cfg, files)
	case "clip":
		err = clipMetadata(md, cfg, files)
	}
	if err != nil {
		return "", fmt.Errorf("huggingface: %s: %w", arch, err)
	}
	return arch, nil
}

// decodeJSON decodes b into v, keeping numbers exact.
func decodeJSON(b []byte, v any) error {
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	return d.Decode(v)
}

// hfConfig reads the fields of a config.json object, collecting errors.
type hfConfig struct {
	m    map[string]any
	path string
	errs []string
}

func (c *hfConfig) fail(key, format string, args ...any) {
	c.errs = append(c.errs, fmt.Sprintf("%s%s %s", c.path, key, fmt.Sprintf(format, args...)))
}

// int returns a positive integer field, or def when it is missing and
// def is not 0.
func (c *hfConfig) int(key string, def int64) int64 {
	v, ok := c.m[key]
	if !ok || v == nil {
		if def == 0 {
			c.fail(key, "is missing")
		}
		return def
	}
	n, ok := v.(json.Number)
	if !ok {
		c.fail(key, "= %v, want a number", v)
		return def
	}
	i, err := n.Int64()
	if err != nil || i <= 0 || i > 1<<24 {
		c.fail(key, "= %v, want a positive integer", n)
		return def
	}
	return i
}

func (c *hfConfig) float(key string, def float64) float64 {
	v, ok := c.m[key]
	if !ok || v == nil {
		return def
	}
	n, ok := v.(json.Number)
	if !ok {
		c.fail(key, "= %v, want a number", v)
		return def
	}
	x, err := n.Float64()
	if err != nil || math.IsNaN(x) || math.IsInf(x, 0) {
		c.fail(key, "= %v", n)
		return def
	}
	return x
}

func (c *hfConfig) str(key, def string) string {
	if s, ok := c.m[key].(string); ok {
		return s
	}
	return def
}

func (c *hfConfig) floats(key string, def []float64) []float64 {
	v, ok := c.m[key].([]any)
	if !ok {
		return def
	}
	out := make([]float64, len(v))
	for i, e := range v {
		n, ok := e.(json.Number)
		x, err := n.Float64()
		if !ok || err != nil {
			c.fail(key, "[%d] = %v", i, e)
			return def
		}
		out[i] = x
	}
	return out
}

func (c *hfConfig) ints(key string) []int64 {
	var v []any
	switch e := c.m[key].(type) {
	case []any:
		v = e
	case string:
		// Some generation configs hold the list as a string.
		if err := decodeJSON([]byte(e), &v); err != nil {
			c.fail(key, "= %q", e)
			return nil
		}
	default:
		return nil
	}
	out := make([]int64, 0, len(v))
	for i, e := range v {
		n, ok := e.(json.Number)
		x, err := n.Int64()
		if !ok || err != nil || x < 0 {
			c.fail(key, "[%d] = %v", i, e)
			return nil
		}
		out = append(out, x)
	}
	return out
}

// sub returns the object field key, for a nested config.
func (c *hfConfig) sub(key string) *hfConfig {
	m, ok := c.m[key].(map[string]any)
	if !ok {
		c.fail(key, "is missing")
		m = map[string]any{}
	}
	return &hfConfig{m: m, path: c.path + key + "."}
}

// optional decodes b, a side file, into a config, empty when b is nil.
func optional(name string, b []byte) (*hfConfig, error) {
	m := map[string]any{}
	if b != nil {
		if err := decodeJSON(b, &m); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
	}
	return &hfConfig{m: m}, nil
}

// hfTokenizer is the part of a tokenizer.json a BPE model needs.
type hfTokenizer struct {
	Model struct {
		Type            string           `json:"type"`
		Vocab           map[string]int64 `json:"vocab"`
		Merges          []json.RawMessage
		EndOfWordSuffix string `json:"end_of_word_suffix"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int64  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
}

// Token types of tokenizer.ggml.token_type, as GGUF numbers them.
const (
	tokenNormal  = 1
	tokenControl = 3
)

// bpeVocab reads a BPE tokenizer.json: the tokens by id, as many as size
// (or as the largest id holds when size is 0), their types, with every
// added token a control token, and the merges as "left right" strings.
func bpeVocab(b []byte, size int64) (tokens []string, types []int64, merges []string, t *hfTokenizer, err error) {
	t = &hfTokenizer{}
	if err := json.Unmarshal(b, t); err != nil {
		return nil, nil, nil, nil, fmt.Errorf("tokenizer.json: %w", err)
	}
	if t.Model.Type != "BPE" {
		return nil, nil, nil, nil, fmt.Errorf("tokenizer.json: a %q model, want BPE", t.Model.Type)
	}
	top := int64(-1)
	for _, id := range t.Model.Vocab {
		top = max(top, id)
	}
	for _, a := range t.AddedTokens {
		top = max(top, a.ID)
	}
	if size == 0 {
		size = top + 1
	}
	if top >= size || size > 1<<24 || len(t.Model.Vocab) == 0 {
		return nil, nil, nil, nil, fmt.Errorf("tokenizer.json: token ids up to %d for a vocabulary of %d", top, size)
	}
	tokens = make([]string, size)
	types = make([]int64, size)
	for s, id := range t.Model.Vocab {
		if id < 0 {
			return nil, nil, nil, nil, fmt.Errorf("tokenizer.json: token %q has id %d", s, id)
		}
		tokens[id], types[id] = s, tokenNormal
	}
	for _, a := range t.AddedTokens {
		if a.ID < 0 {
			return nil, nil, nil, nil, fmt.Errorf("tokenizer.json: added token %q has id %d", a.Content, a.ID)
		}
		tokens[a.ID], types[a.ID] = a.Content, tokenControl
	}
	for i := range tokens {
		if types[i] == 0 {
			// A gap the checkpoint's embeddings still have a row for.
			tokens[i], types[i] = fmt.Sprintf("<|unused_%d|>", i), tokenControl
		}
	}
	merges = make([]string, 0, len(t.Model.Merges))
	for i, raw := range t.Model.Merges {
		var s string
		if json.Unmarshal(raw, &s) == nil {
			merges = append(merges, s)
			continue
		}
		var pair []string
		if err := json.Unmarshal(raw, &pair); err != nil || len(pair) != 2 {
			return nil, nil, nil, nil, fmt.Errorf("tokenizer.json: merge %d is %s", i, raw)
		}
		merges = append(merges, pair[0]+" "+pair[1])
	}
	return tokens, types, merges, t, nil
}

// tokenID returns the id of the token s, or -1.
func tokenID(tokens []string, s string) int64 {
	return int64(slices.Index(tokens, s))
}

// whisperMetadata describes a Whisper checkpoint.
func whisperMetadata(md mim.Metadata, raw map[string]any, files HFFiles) error {
	c := &hfConfig{m: raw}
	gen, err := optional("generation_config.json", files.GenerationConfig)
	if err != nil {
		return err
	}
	pre, err := optional("preprocessor_config.json", files.Preprocessor)
	if err != nil {
		return err
	}
	vocab := c.int("vocab_size", 0)
	model := c.int("d_model", 0)
	set := func(key string, v any) { md["whisper."+key] = v }
	set("vocab_size", vocab)
	set("embedding_length", model)
	set("mel_bins", c.int("num_mel_bins", 80))
	set("encoder.block_count", c.int("encoder_layers", 0))
	set("encoder.head_count", c.int("encoder_attention_heads", 0))
	set("encoder.feed_forward_length", c.int("encoder_ffn_dim", 0))
	set("encoder.context_length", c.int("max_source_positions", 1500))
	set("decoder.block_count", c.int("decoder_layers", 0))
	set("decoder.head_count", c.int("decoder_attention_heads", 0))
	set("decoder.feed_forward_length", c.int("decoder_ffn_dim", 0))
	set("decoder.context_length", c.int("max_target_positions", 448))
	if act := c.str("activation_function", "gelu"); act != "gelu" {
		c.fail("activation_function", "= %q, want gelu", act)
	}
	set("sample_rate", pre.int("sampling_rate", 16000))
	set("fft_length", pre.int("n_fft", 400))
	set("hop_length", pre.int("hop_length", 160))
	set("chunk_length", pre.int("chunk_length", 30))
	if err := errorsOf(c, gen, pre); err != nil {
		return err
	}
	// The generation config's suppressed tokens, which the config itself
	// held in older checkpoints.
	suppress := gen.ints("suppress_tokens")
	if suppress == nil {
		suppress = c.ints("suppress_tokens")
	}
	begin := gen.ints("begin_suppress_tokens")
	if begin == nil {
		begin = c.ints("begin_suppress_tokens")
	}
	if err := errorsOf(c, gen); err != nil {
		return err
	}
	for _, list := range [][]int64{suppress, begin} {
		for _, id := range list {
			if id >= vocab {
				return fmt.Errorf("suppressed token %d of a vocabulary of %d", id, vocab)
			}
		}
	}
	// OpenAI tells a multilingual model by its vocabulary, the English-only
	// ones having one token fewer; the generation config says so outright.
	multi, ok := gen.m["is_multilingual"].(bool)
	if !ok {
		multi = vocab >= 51865
	}
	set("multilingual", multi)
	set("suppress_tokens", nonNil(suppress))
	set("begin_suppress_tokens", nonNil(begin))

	tokens, types, merges, _, err := bpeVocab(files.Tokenizer, vocab)
	if err != nil {
		return err
	}
	eos := tokenID(tokens, "<|endoftext|>")
	sot := tokenID(tokens, "<|startoftranscript|>")
	if eos < 0 || sot < 0 {
		return fmt.Errorf("tokenizer.json has no <|endoftext|> or <|startoftranscript|>")
	}
	md["tokenizer.ggml.model"] = "gpt2"
	md["tokenizer.ggml.pre"] = "gpt2"
	md["tokenizer.ggml.tokens"] = tokens
	md["tokenizer.ggml.token_type"] = types
	md["tokenizer.ggml.merges"] = merges
	md["tokenizer.ggml.bos_token_id"] = sot
	md["tokenizer.ggml.eos_token_id"] = eos
	md["tokenizer.ggml.add_bos_token"] = false
	return nil
}

// clipMetadata describes a CLIP checkpoint.
func clipMetadata(md mim.Metadata, raw map[string]any, files HFFiles) error {
	c := &hfConfig{m: raw}
	pre, err := optional("preprocessor_config.json", files.Preprocessor)
	if err != nil {
		return err
	}
	text, vision := c.sub("text_config"), c.sub("vision_config")
	if err := errorsOf(c); err != nil {
		return err
	}
	proj := c.int("projection_dim", 512)
	md["clip.projection_dim"] = proj
	for _, part := range []struct {
		name string
		c    *hfConfig
	}{{"text", text}, {"vision", vision}} {
		set := func(key string, v any) { md["clip."+part.name+"."+key] = v }
		p := part.c
		set("embedding_length", p.int("hidden_size", 0))
		set("block_count", p.int("num_hidden_layers", 0))
		set("head_count", p.int("num_attention_heads", 0))
		set("feed_forward_length", p.int("intermediate_size", 0))
		set("layer_norm_epsilon", p.float("layer_norm_eps", 1e-5))
		act := p.str("hidden_act", "quick_gelu")
		if act != "quick_gelu" && act != "gelu" {
			p.fail("hidden_act", "= %q, want quick_gelu or gelu", act)
		}
		set("activation", act)
	}
	md["clip.text.vocab_size"] = text.int("vocab_size", 0)
	md["clip.text.context_length"] = text.int("max_position_embeddings", 77)
	md["clip.vision.image_size"] = vision.int("image_size", 224)
	md["clip.vision.patch_size"] = vision.int("patch_size", 0)
	if ch := vision.int("num_channels", 3); ch != 3 {
		vision.fail("num_channels", "= %d, want 3", ch)
	}
	size := md["clip.vision.image_size"].(int64)
	// The preprocessor resizes the shorter side to size then crops the
	// middle crop_size square; either may be a number or an object.
	side := func(key string) int64 {
		switch v := pre.m[key].(type) {
		case json.Number:
			return pre.int(key, size)
		case map[string]any:
			s := &hfConfig{m: v, path: key + "."}
			for _, k := range []string{"shortest_edge", "height"} {
				if _, ok := v[k]; ok {
					n := s.int(k, size)
					pre.errs = append(pre.errs, s.errs...)
					return n
				}
			}
		}
		return size
	}
	md["clip.image.resize"] = side("size")
	md["clip.image.crop"] = side("crop_size")
	mean := pre.floats("image_mean", []float64{0.48145466, 0.4578275, 0.40821073})
	std := pre.floats("image_std", []float64{0.26862954, 0.26130258, 0.27577711})
	if len(mean) != 3 || len(std) != 3 || slices.ContainsFunc(std, func(x float64) bool { return !(x > 0) }) {
		pre.fail("image_mean", "and image_std %v %v: want three each", mean, std)
	}
	md["clip.image.mean"] = mean
	md["clip.image.std"] = std
	if crop := md["clip.image.crop"].(int64); crop != size {
		pre.fail("crop_size", "%d for images of %d", crop, size)
	}
	if err := errorsOf(c, text, vision, pre); err != nil {
		return err
	}

	tokens, types, merges, t, err := bpeVocab(files.Tokenizer, md["clip.text.vocab_size"].(int64))
	if err != nil {
		return err
	}
	if t.Model.EndOfWordSuffix != "</w>" {
		return fmt.Errorf("tokenizer.json: end-of-word suffix %q, want </w>", t.Model.EndOfWordSuffix)
	}
	bos, eos := tokenID(tokens, "<|startoftext|>"), tokenID(tokens, "<|endoftext|>")
	if bos < 0 || eos < 0 {
		return fmt.Errorf("tokenizer.json has no <|startoftext|> or <|endoftext|>")
	}
	md["tokenizer.ggml.model"] = "clip"
	md["tokenizer.ggml.tokens"] = tokens
	md["tokenizer.ggml.token_type"] = types
	md["tokenizer.ggml.merges"] = merges
	md["tokenizer.ggml.bos_token_id"] = bos
	md["tokenizer.ggml.eos_token_id"] = eos
	md["tokenizer.ggml.padding_token_id"] = eos
	return nil
}

// errorsOf joins the errors the configs collected.
func errorsOf(cs ...*hfConfig) error {
	var all []string
	for _, c := range cs {
		all = append(all, c.errs...)
		c.errs = nil
	}
	if len(all) == 0 {
		return nil
	}
	sort.Strings(all)
	return fmt.Errorf("%s", strings.Join(all, "; "))
}

func nonNil(v []int64) []int64 {
	if v == nil {
		return []int64{}
	}
	return v
}
