// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0

package mim

import (
	"strings"
	"testing"
)

// pair is one adapted weight for loraFile: its target, the dtype of both
// tensors and their shapes.
type pair struct {
	target string
	dt     DType
	a, b   []uint64
}

// loraFile writes an adapter with metadata md and the tensors of pairs,
// plus the named extra tensors, and parses it.
func loraFile(t *testing.T, md Metadata, pairs []pair, extra ...string) (*File, error) {
	t.Helper()
	w := NewWriter()
	add := func(name string, dt DType, shape []uint64) {
		n := shape[0] * shape[1] / dt.BlockElems() * dt.BlockBytes()
		must(t, w.AddTensor(name, dt, shape, make([]byte, n)))
	}
	for _, p := range pairs {
		if p.a != nil {
			add(p.target+LoRASuffixA, p.dt, p.a)
		}
		if p.b != nil {
			add(p.target+LoRASuffixB, p.dt, p.b)
		}
	}
	for _, name := range extra {
		add(name, F32, []uint64{1, 1})
	}
	mb, err := md.Encode()
	must(t, err)
	must(t, w.AddSection(SectionMetadata, false, mb))
	return Parse(encode(t, w))
}

const baseSHA = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func samplePairs() []pair {
	return []pair{
		{"blk.0.attn_q.weight", F32, []uint64{4, 16}, []uint64{16, 4}},
		{"blk.0.attn_v.weight", F16, []uint64{4, 16}, []uint64{8, 4}},
		{"blk.1.attn_q.weight", BF16, []uint64{2, 16}, []uint64{16, 2}},
	}
}

func TestLoRA(t *testing.T) {
	f, err := loraFile(t, LoRAMetadata(16, baseSHA), samplePairs())
	must(t, err)
	a, err := f.LoRA()
	must(t, err)
	if a.Alpha != 16 || a.Base != baseSHA || len(a.Layers) != 3 {
		t.Fatalf("got alpha %v, base %q and %d layers", a.Alpha, a.Base, len(a.Layers))
	}
	v := a.Layers[1]
	if v.Target != "blk.0.attn_v.weight" || v.Rank != 4 || v.In != 16 || v.Out != 8 || v.A.DType != F16 {
		t.Errorf("layer 1 is %+v", v)
	}
	if s := a.Scale(a.Layers[2]); s != 8 {
		t.Errorf("scale of rank 2 = %v, want 8", s)
	}

	// An unnamed base is fine too.
	f, err = loraFile(t, LoRAMetadata(8, ""), samplePairs()[:1])
	must(t, err)
	if a, err := f.LoRA(); err != nil || a.Base != "" {
		t.Errorf("unnamed base: %+v, %v", a, err)
	}
}

// TestLoRARejects checks the rules of spec/adapters.md, with bad-adapter.
func TestLoRARejects(t *testing.T) {
	type tc struct {
		md    func(Metadata)
		pairs func([]pair) []pair
		extra []string
	}
	cases := map[string]tc{
		"not an adapter":  {md: func(m Metadata) { delete(m, KeyGeneralType) }},
		"a model":         {md: func(m Metadata) { m[KeyGeneralType] = "model" }},
		"not lora":        {md: func(m Metadata) { m[KeyAdapterType] = "ia3" }},
		"no alpha":        {md: func(m Metadata) { delete(m, KeyLoRAAlpha) }},
		"alpha as string": {md: func(m Metadata) { m[KeyLoRAAlpha] = "16" }},
		"zero alpha":      {md: func(m Metadata) { m[KeyLoRAAlpha] = 0.0 }},
		"bad base":        {md: func(m Metadata) { m[KeyAdapterBase] = strings.ToUpper(baseSHA) }},
		"no pairs":        {pairs: func([]pair) []pair { return nil }},
		"A alone":         {pairs: func(p []pair) []pair { p[0].b = nil; return p }},
		"B alone":         {pairs: func(p []pair) []pair { p[1].a = nil; return p }},
		"ranks differ":    {pairs: func(p []pair) []pair { p[0].b = []uint64{16, 3}; return p }},
		"quantized":       {pairs: func(p []pair) []pair { p[0].dt = Q8_0; p[0].a = []uint64{32, 32}; p[0].b = []uint64{32, 32}; return p }},
		"other tensor":    {extra: []string{"blk.0.attn_q.weight"}},
		"empty target":    {extra: []string{".lora_a"}},
	}
	for name, c := range cases {
		md := LoRAMetadata(16, baseSHA)
		if c.md != nil {
			c.md(md)
		}
		pairs := samplePairs()
		if c.pairs != nil {
			pairs = c.pairs(pairs)
		}
		f, err := loraFile(t, md, pairs, c.extra...)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if _, err := f.LoRA(); CodeOf(err) != CodeBadAdapter {
			t.Errorf("%s: got %v, want bad-adapter", name, err)
		}
	}
}

func TestLoRACheckBase(t *testing.T) {
	f, err := loraFile(t, LoRAMetadata(16, baseSHA), samplePairs())
	must(t, err)
	a, err := f.LoRA()
	must(t, err)
	base := func(vShape []uint64) *File {
		w := NewWriter()
		for _, x := range []struct {
			name  string
			shape []uint64
		}{
			{"blk.0.attn_q.weight", []uint64{16, 16}},
			{"blk.0.attn_v.weight", vShape},
			{"blk.1.attn_q.weight", []uint64{16, 16}},
		} {
			must(t, w.AddTensor(x.name, F32, x.shape, make([]byte, 4*elems(x.shape))))
		}
		b, err := Parse(encode(t, w))
		must(t, err)
		return b
	}
	if err := a.CheckBase(base([]uint64{8, 16}), baseSHA); err != nil {
		t.Errorf("matching base: %v", err)
	}
	if err := a.CheckBase(base([]uint64{8, 16}), ""); err != nil {
		t.Errorf("base of unknown hash: %v", err)
	}
	other := strings.Repeat("f", 64)
	if err := a.CheckBase(base([]uint64{8, 16}), other); CodeOf(err) != CodeBadAdapter {
		t.Errorf("other base file: got %v, want bad-adapter", err)
	}
	if err := a.CheckBase(base([]uint64{16, 16}), baseSHA); CodeOf(err) != CodeBadAdapter {
		t.Errorf("weight of another shape: got %v, want bad-adapter", err)
	}
	if err := a.CheckBase(base([]uint64{8}), baseSHA); CodeOf(err) != CodeBadAdapter {
		t.Errorf("weight of rank 1: got %v, want bad-adapter", err)
	}
}

func elems(shape []uint64) uint64 {
	n := uint64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}
