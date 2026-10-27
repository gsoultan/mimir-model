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

package mim

import (
	"math"
	"strings"
)

// Metadata keys and suffixes of an adapter file (spec/adapters.md). They
// are llama.cpp's, so a GGUF LoRA adapter converts byte for byte.
const (
	KeyGeneralType   = "general.type"       // "adapter" in an adapter file
	KeyAdapterType   = "adapter.type"       // "lora"
	KeyLoRAAlpha     = "adapter.lora.alpha" // float
	KeyAdapterBase   = "adapter.base.sha256"
	LoRASuffixA      = ".lora_a"
	LoRASuffixB      = ".lora_b"
	maxAdapterTarget = 1 << 16 // longest name a target may have
)

// LoRA is a low-rank adapter (spec/adapters.md): for each target weight W
// of a base model, [out, in], a pair A, [r, in], and B, [out, r], that
// change it to W + Alpha/r · B·A.
type LoRA struct {
	// Alpha scales every pair by Alpha/r, r being the pair's rank.
	Alpha float64
	// Base is the lowercase hex SHA-256 of the base model's file, or ""
	// when the adapter doesn't say which file it was trained on.
	Base string
	// Layers are the adapted weights, sorted by Target.
	Layers []LoRALayer
}

// LoRALayer is the pair of one target weight.
type LoRALayer struct {
	Target string     // the base model's weight, such as blk.0.attn_q.weight
	A, B   TensorInfo // [Rank, In] and [Out, Rank]
	Rank   int
	In     int
	Out    int
}

// Scale is what the layer's B·A is multiplied by: Alpha/Rank.
func (l *LoRA) Scale(layer LoRALayer) float64 { return l.Alpha / float64(layer.Rank) }

// IsAdapter reports whether the file's metadata says it is an adapter,
// which holds no model of its own.
func IsAdapter(md Metadata) bool {
	t, _ := md[KeyGeneralType].(string)
	return t == "adapter"
}

// LoRA reads the file as a LoRA adapter and checks it against the rules of
// spec/adapters.md, with bad-adapter. It doesn't check it against a base
// model; see CheckBase.
func (f *File) LoRA() (*LoRA, error) {
	md, err := f.Metadata()
	if err != nil {
		return nil, err
	}
	if !IsAdapter(md) {
		return nil, errf(CodeBadAdapter, "%s is %q, want \"adapter\"", KeyGeneralType, md[KeyGeneralType])
	}
	if t, _ := md[KeyAdapterType].(string); t != "lora" {
		return nil, errf(CodeBadAdapter, "%s is %q, want \"lora\"", KeyAdapterType, md[KeyAdapterType])
	}
	a := &LoRA{}
	switch v := md[KeyLoRAAlpha].(type) {
	case float64:
		a.Alpha = v
	case int64:
		a.Alpha = float64(v)
	case uint64:
		a.Alpha = float64(v)
	default:
		return nil, errf(CodeBadAdapter, "%s is %v, want a number", KeyLoRAAlpha, md[KeyLoRAAlpha])
	}
	if !(a.Alpha > 0) || math.IsInf(a.Alpha, 0) {
		return nil, errf(CodeBadAdapter, "%s is %v, want a positive number", KeyLoRAAlpha, a.Alpha)
	}
	if v, ok := md[KeyAdapterBase]; ok {
		s, _ := v.(string)
		if !isHexSHA256(s) {
			return nil, errf(CodeBadAdapter, "%s is %q, want 64 lowercase hex digits", KeyAdapterBase, v)
		}
		a.Base = s
	}
	if _, ok := f.Section(SectionGraph); ok {
		return nil, errf(CodeBadAdapter, "an adapter has no graph")
	}
	// The tensor table is sorted by name, so the pairs come out sorted by
	// target.
	for i := range f.NumTensors() {
		t := f.TensorAt(i)
		if target, ok := strings.CutSuffix(t.Name, LoRASuffixB); ok && target != "" {
			if _, ok := f.Tensor(target + LoRASuffixA); !ok {
				return nil, errf(CodeBadAdapter, "%s has no %s", t.Name, LoRASuffixA)
			}
			continue
		}
		target, ok := strings.CutSuffix(t.Name, LoRASuffixA)
		if !ok || target == "" || len(target) > maxAdapterTarget {
			return nil, errf(CodeBadAdapter, "tensor %q is not one of a pair", t.Name)
		}
		ta := t
		tb, ok := f.Tensor(target + LoRASuffixB)
		if !ok {
			return nil, errf(CodeBadAdapter, "%s has no %s", ta.Name, LoRASuffixB)
		}
		for _, t := range []TensorInfo{ta, tb} {
			if t.DType != F32 && t.DType != F16 && t.DType != BF16 {
				return nil, errf(CodeBadAdapter, "%s is %s, want f32, f16 or bf16", t.Name, t.DType)
			}
			if len(t.Shape) != 2 || t.Shape[0] == 0 || t.Shape[1] == 0 || t.Shape[0] > math.MaxInt32 || t.Shape[1] > math.MaxInt32 {
				return nil, errf(CodeBadAdapter, "%s of shape %v, want a matrix", t.Name, t.Shape)
			}
		}
		if ta.Shape[0] != tb.Shape[1] {
			return nil, errf(CodeBadAdapter, "%s is %v and %s %v: the ranks differ", ta.Name, ta.Shape, tb.Name, tb.Shape)
		}
		a.Layers = append(a.Layers, LoRALayer{Target: target, A: ta, B: tb, Rank: int(ta.Shape[0]), In: int(ta.Shape[1]), Out: int(tb.Shape[0])})
	}
	if len(a.Layers) == 0 {
		return nil, errf(CodeBadAdapter, "no pairs")
	}
	return a, nil
}

// CheckBase checks that base is a model the adapter applies to: that each
// target is a matrix of base of the pair's shape, and that base is the
// file the adapter names, if it names one. sha256 is base's lowercase hex
// SHA-256, or "" to skip that check.
func (l *LoRA) CheckBase(base *File, sha256 string) error {
	if l.Base != "" && sha256 != "" && l.Base != sha256 {
		return errf(CodeBadAdapter, "trained on the file %s, not %s", l.Base, sha256)
	}
	for _, p := range l.Layers {
		w, ok := base.Tensor(p.Target)
		if !ok {
			return errf(CodeBadAdapter, "the base model has no %s", p.Target)
		}
		if len(w.Shape) != 2 || w.Shape[0] != uint64(p.Out) || w.Shape[1] != uint64(p.In) {
			return errf(CodeBadAdapter, "%s is %v in the base model; the pair makes it [%d %d]", p.Target, w.Shape, p.Out, p.In)
		}
	}
	return nil
}

// LoRAMetadata returns the metadata of an adapter file of scale alpha,
// trained on the base model whose file has the lowercase hex SHA-256
// base, or on an unnamed one if base is "". Add to it whatever else the
// file should say, such as general.name.
func LoRAMetadata(alpha float64, base string) Metadata {
	md := Metadata{KeyGeneralType: "adapter", KeyAdapterType: "lora", KeyLoRAAlpha: alpha}
	if base != "" {
		md[KeyAdapterBase] = base
	}
	return md
}

func isHexSHA256(s string) bool {
	if len(s) != 64 {
		return false
	}
	for i := 0; i < len(s); i++ {
		if c := s[i]; (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}
