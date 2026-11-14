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
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// hfCheckpoint reads one of the tiny Hugging Face checkpoints of
// testdata/import: randomly initialized Whisper and CLIP models saved by
// Transformers, with their side files.
func hfCheckpoint(t *testing.T, dir string) (*Source, HFFiles) {
	t.Helper()
	read := func(name string) []byte {
		b, err := os.ReadFile(filepath.Join("..", "testdata", "import", dir, name))
		if os.IsNotExist(err) {
			return nil
		}
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	src, err := ParseSafetensors(read("model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	return src, HFFiles{
		Config:           read("config.json"),
		GenerationConfig: read("generation_config.json"),
		Preprocessor:     read("preprocessor_config.json"),
		Tokenizer:        read("tokenizer.json"),
	}
}

func TestFromHuggingFaceWhisper(t *testing.T) {
	src, files := hfCheckpoint(t, "hf-whisper")
	arch, err := FromHuggingFace(src, files)
	if err != nil || arch != "whisper" {
		t.Fatalf("%q, %v", arch, err)
	}
	md := src.Metadata
	for key, want := range map[string]any{
		"general.architecture":         "whisper",
		"whisper.embedding_length":     int64(16),
		"whisper.encoder.block_count":  int64(2),
		"whisper.decoder.head_count":   int64(2),
		"whisper.mel_bins":             int64(80),
		"whisper.multilingual":         true,
		"whisper.sample_rate":          int64(16000),
		"tokenizer.ggml.model":         "gpt2",
		"tokenizer.ggml.pre":           "gpt2",
		"tokenizer.ggml.eos_token_id":  int64(259),
		"tokenizer.ggml.bos_token_id":  int64(260),
		"tokenizer.ggml.add_bos_token": false,
	} {
		if md[key] != want {
			t.Errorf("%s = %v (%T), want %v", key, md[key], md[key], want)
		}
	}
	tokens := md["tokenizer.ggml.tokens"].([]string)
	types := md["tokenizer.ggml.token_type"].([]int64)
	if len(tokens) != 272 || tokens[262] != "<|de|>" || types[262] != tokenControl || types[3] != tokenNormal {
		t.Errorf("%d tokens, 262 %q of type %d", len(tokens), tokens[262], types[262])
	}
	if !slices.Equal(md["whisper.suppress_tokens"].([]int64), []int64{1, 2}) || !slices.Equal(md["tokenizer.ggml.merges"].([]string), []string{"Ġ t", "h e", "Ġt he"}) {
		t.Errorf("suppressed %v, merges %v", md["whisper.suppress_tokens"], md["tokenizer.ggml.merges"])
	}
}

func TestFromHuggingFaceCLIP(t *testing.T) {
	src, files := hfCheckpoint(t, "hf-clip")
	arch, err := FromHuggingFace(src, files)
	if err != nil || arch != "clip" {
		t.Fatalf("%q, %v", arch, err)
	}
	md := src.Metadata
	for key, want := range map[string]any{
		"clip.projection_dim":             int64(8),
		"clip.text.embedding_length":      int64(16),
		"clip.text.context_length":        int64(16),
		"clip.vision.patch_size":          int64(8),
		"clip.vision.image_size":          int64(32),
		"clip.vision.activation":          "quick_gelu",
		"clip.image.resize":               int64(32),
		"tokenizer.ggml.model":            "clip",
		"tokenizer.ggml.bos_token_id":     int64(516),
		"tokenizer.ggml.eos_token_id":     int64(517),
		"tokenizer.ggml.padding_token_id": int64(517),
	} {
		if md[key] != want {
			t.Errorf("%s = %v (%T), want %v", key, md[key], md[key], want)
		}
	}
	if m := md["clip.image.std"].([]float64); len(m) != 3 || m[0] != 0.26862954 {
		t.Errorf("std %v", m)
	}
}

func TestFromHuggingFaceRejects(t *testing.T) {
	for _, tc := range []struct {
		dir, what string
		edit      func(f *HFFiles)
		want      string
	}{
		{"hf-whisper", "no config", func(f *HFFiles) { f.Config = nil }, "no config.json"},
		{"hf-whisper", "no tokenizer", func(f *HFFiles) { f.Tokenizer = nil }, "no tokenizer.json"},
		{"hf-whisper", "a Llama", func(f *HFFiles) { f.Config = []byte(`{"architectures":["LlamaForCausalLM"],"model_type":"llama"}`) }, "only Whisper and CLIP"},
		{"hf-whisper", "a bad config", func(f *HFFiles) { f.Config = []byte(`{"model_type":"whisper","d_model":-3}`) }, "d_model"},
		{"hf-whisper", "a bad suppress list", func(f *HFFiles) { f.GenerationConfig = []byte(`{"suppress_tokens":[100000]}`) }, "suppressed token"},
		{"hf-whisper", "a WordPiece tokenizer", func(f *HFFiles) { f.Tokenizer = []byte(`{"model":{"type":"WordPiece"}}`) }, "want BPE"},
		{"hf-clip", "no end-of-word suffix", func(f *HFFiles) {
			f.Tokenizer = []byte(strings.Replace(string(f.Tokenizer), `"end_of_word_suffix": "</w>"`, `"end_of_word_suffix": ""`, 1))
		}, "suffix"},
		{"hf-clip", "a crop unlike the model", func(f *HFFiles) { f.Preprocessor = []byte(`{"crop_size":24}`) }, "crop_size"},
	} {
		src, files := hfCheckpoint(t, tc.dir)
		tc.edit(&files)
		if _, err := FromHuggingFace(src, files); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want %q", tc.what, err, tc.want)
		}
	}
	gguf := &Source{Format: "gguf"}
	if _, err := FromHuggingFace(gguf, HFFiles{Config: []byte(`{}`)}); err == nil {
		t.Error("a GGUF source was described")
	}
}
