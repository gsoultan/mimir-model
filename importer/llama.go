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

package importer

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/gsoultan/mmf"
)

// Graph construction for decoder-only transformers in the Llama family.
//
// GGUF and safetensors carry weights but no graph, so an importer that
// wants a runnable file writes the architecture's forward pass itself, from
// the hyperparameters in the metadata. BuildLlama does this for GGUF files
// whose general.architecture is "llama" (Llama 1 to 3, TinyLlama, Mistral
// without sliding windows, and others using llama.cpp's llama layout).
//
// The graph takes one block of tokens per run:
//
//	inputs:    tokens    i32[1, T]   token ids
//	           positions i64[1, T]   their positions in the sequence
//	outputs:   logits    f32[1, T, V]
//	variables: blk.N.cache_k, blk.N.cache_v  f32[1, Hkv, C, D]
//
// T is a symbolic dimension bounded by the context length C. The key-value
// cache follows ADR 0003: each layer scatters its new keys and values into
// the cache variables at the given positions and attends over the whole
// cache, with a mask that admits key position j for the query at position
// p when j <= p. A prompt is one run with positions 0..T-1; each generated
// token is a run with T = 1.

// Context lengths beyond this are refused, so a typo cannot ask for an
// enormous zero-filled cache.
const maxLlamaContext = 1 << 20

// LlamaConfig holds the hyperparameters BuildLlama reads from the metadata.
type LlamaConfig struct {
	Layers     int
	Embed      int // E
	FFN        int
	Heads      int // Hq
	KVHeads    int // Hkv
	HeadDim    int // D = E / Hq
	RopeDims   int
	RopeTheta  float64
	NormEps    float64
	Vocab      int
	Context    int // C, the cache length
	MaxContext int // the model's trained context length
}

func metaInt(md mmf.Metadata, key string) (int64, bool) {
	switch v := md[key].(type) {
	case int64:
		return v, true
	case uint64:
		if v > math.MaxInt64 {
			return 0, false
		}
		return int64(v), true
	}
	return 0, false
}

func metaFloat(md mmf.Metadata, key string) (float64, bool) {
	switch v := md[key].(type) {
	case float64:
		return v, true
	case int64:
		return float64(v), true
	case uint64:
		return float64(v), true
	}
	return 0, false
}

// LlamaConfigOf reads the hyperparameters of a llama-architecture source.
// context is the cache length to build for; 0 means the model's own.
func LlamaConfigOf(s *Source, context int) (*LlamaConfig, error) {
	md := s.Metadata
	if arch, _ := md["general.architecture"].(string); arch != "llama" {
		return nil, fmt.Errorf("architecture %q: only \"llama\" graphs can be built", arch)
	}
	req := func(key string) (int, error) {
		v, ok := metaInt(md, "llama."+key)
		if !ok || v <= 0 || v > 1<<24 {
			return 0, fmt.Errorf("metadata llama.%s is missing or invalid", key)
		}
		return int(v), nil
	}
	c := &LlamaConfig{}
	var err error
	for _, f := range []struct {
		key string
		dst *int
	}{
		{"block_count", &c.Layers}, {"embedding_length", &c.Embed}, {"feed_forward_length", &c.FFN},
		{"attention.head_count", &c.Heads}, {"context_length", &c.MaxContext},
	} {
		if *f.dst, err = req(f.key); err != nil {
			return nil, err
		}
	}
	c.KVHeads = c.Heads
	if _, ok := md["llama.attention.head_count_kv"]; ok {
		if c.KVHeads, err = req("attention.head_count_kv"); err != nil {
			return nil, err
		}
	}
	if c.Embed%c.Heads != 0 || c.Heads%c.KVHeads != 0 {
		return nil, fmt.Errorf("embedding length %d, %d heads and %d key-value heads do not divide", c.Embed, c.Heads, c.KVHeads)
	}
	c.HeadDim = c.Embed / c.Heads
	c.RopeDims = c.HeadDim
	if _, ok := md["llama.rope.dimension_count"]; ok {
		if c.RopeDims, err = req("rope.dimension_count"); err != nil {
			return nil, err
		}
		if c.RopeDims%2 != 0 || c.RopeDims > c.HeadDim {
			return nil, fmt.Errorf("rope dimension count %d for head size %d", c.RopeDims, c.HeadDim)
		}
	}
	c.RopeTheta = 10000
	if v, ok := metaFloat(md, "llama.rope.freq_base"); ok {
		c.RopeTheta = v
	}
	if t, ok := md["llama.rope.scaling.type"].(string); ok && t != "none" {
		return nil, fmt.Errorf("rope scaling %q is not supported yet", t)
	}
	var ok bool
	if c.NormEps, ok = metaFloat(md, "llama.attention.layer_norm_rms_epsilon"); !ok || c.NormEps <= 0 {
		return nil, fmt.Errorf("metadata llama.attention.layer_norm_rms_epsilon is missing or invalid")
	}
	c.Context = c.MaxContext
	if context > 0 {
		c.Context = context
	}
	if c.Context > maxLlamaContext {
		return nil, fmt.Errorf("context length %d exceeds %d", c.Context, maxLlamaContext)
	}
	return c, nil
}

// llamaBuilder accumulates the graph. Value refs index s.Tensors until
// AddTo remaps them to the file's sorted tensor order.
type llamaBuilder struct {
	s       *Source
	g       *mmf.Graph
	byName  map[string]int
	weights map[string]uint32
	a8      bool
}

func (b *llamaBuilder) val(name string, dt mmf.DType, shape ...mmf.Dim) uint32 {
	b.g.Values = append(b.g.Values, mmf.Value{Name: name, DType: dt, Shape: shape})
	return uint32(len(b.g.Values) - 1)
}

func (b *llamaBuilder) op(opc mmf.Opcode, in []uint32, out uint32, attrs ...mmf.Attr) uint32 {
	main := &b.g.Subgraphs[0]
	main.Ops = append(main.Ops, mmf.Op{Opcode: opc, Inputs: in, Outputs: []uint32{out}, Attrs: attrs})
	return out
}

func (b *llamaBuilder) call(name string, in []uint32, out uint32, attrs ...mmf.Attr) uint32 {
	a := append([]mmf.Attr{
		{Key: "name", Type: mmf.AttrString, String: name},
		{Key: "version", Type: mmf.AttrInt, Int: 1},
	}, attrs...)
	return b.op(mmf.OpCall, in, out, a...)
}

// stored adds a value of the given kind backed by source tensor i.
func (b *llamaBuilder) stored(i int, kind mmf.ValueKind) uint32 {
	t := b.s.Tensors[i]
	dims := make([]mmf.Dim, len(t.Shape))
	for k, d := range t.Shape {
		dims[k] = mmf.Dim(d)
	}
	b.g.Values = append(b.g.Values, mmf.Value{Name: t.Name, DType: t.DType, Shape: dims, Kind: kind, Ref: uint32(i)})
	return uint32(len(b.g.Values) - 1)
}

// zeros adds a variable that starts as zeros.
func (b *llamaBuilder) zeros(name string, dt mmf.DType, shape ...mmf.Dim) uint32 {
	b.g.Values = append(b.g.Values, mmf.Value{Name: name, DType: dt, Shape: shape, Kind: mmf.KindVariable, Zero: true})
	return uint32(len(b.g.Values) - 1)
}

// weight returns the value for a source tensor, checking its shape.
func (b *llamaBuilder) weight(name string, shape ...int) (uint32, error) {
	if id, ok := b.weights[name]; ok {
		return id, nil
	}
	i, ok := b.byName[name]
	if !ok {
		return 0, fmt.Errorf("tensor %q is missing", name)
	}
	t := b.s.Tensors[i]
	if len(t.Shape) != len(shape) {
		return 0, fmt.Errorf("tensor %q is %v, want %v", name, t.Shape, shape)
	}
	for k := range shape {
		if t.Shape[k] != uint64(shape[k]) {
			return 0, fmt.Errorf("tensor %q is %v, want %v", name, t.Shape, shape)
		}
	}
	id := b.stored(i, mmf.KindWeight)
	b.weights[name] = id
	return id, nil
}

// addTensor appends a tensor created by the importer.
func (b *llamaBuilder) addTensor(name string, dt mmf.DType, shape []uint64, data []byte, kind mmf.ValueKind) (uint32, error) {
	if _, dup := b.byName[name]; dup {
		return 0, fmt.Errorf("tensor %q already exists in the source", name)
	}
	b.s.Tensors = append(b.s.Tensors, Tensor{Name: name, DType: dt, Shape: shape, Data: data})
	b.byName[name] = len(b.s.Tensors) - 1
	return b.stored(len(b.s.Tensors)-1, kind), nil
}

func ints(v ...int64) mmf.Attr            { return mmf.Attr{Type: mmf.AttrInts, Ints: v} }
func named(k string, a mmf.Attr) mmf.Attr { a.Key = k; return a }

// matmulOp emits x times w^T, with linear.v1 for f32 weights and
// dequant_matmul.v1 or dequant_matmul_a8.v1 for every other dtype.
func (b *llamaBuilder) matmulOp(name string, x, w uint32, shape ...mmf.Dim) uint32 {
	comp := "dequant_matmul"
	switch {
	case b.g.Values[w].DType == mmf.F32:
		comp = "linear"
	case b.a8:
		comp = "dequant_matmul_a8"
	}
	return b.call(comp, []uint32{x, w}, b.val(name, mmf.F32, shape...))
}

// f32Of casts a weight such as a norm scale to f32 when it is stored in
// another float dtype.
func (b *llamaBuilder) f32Of(w uint32) uint32 {
	v := b.g.Values[w]
	if v.DType == mmf.F32 {
		return w
	}
	return b.op(mmf.OpCast, []uint32{w}, b.val(v.Name+".f32", mmf.F32, v.Shape...),
		mmf.Attr{Key: "to", Type: mmf.AttrDType, DType: mmf.F32})
}

// LlamaOptions configures BuildLlama.
type LlamaOptions struct {
	// Context is the key-value cache length; 0 means the model's own.
	Context int
	// A8 writes the matrix products with quantized weights as
	// dequant_matmul_a8.v1, which lets a runtime round the activations to
	// 8-bit blocks as llama.cpp does, instead of dequant_matmul.v1.
	A8 bool
}

// BuildLlama adds a forward-pass graph for a llama-architecture source,
// with a key-value cache of the configured context length, and returns
// the configuration it used. The cache tensors and the key position table
// are added to s.Tensors; AddTo writes the graph section.
func BuildLlama(s *Source, o LlamaOptions) (*LlamaConfig, error) {
	c, err := LlamaConfigOf(s, o.Context)
	if err != nil {
		return nil, err
	}
	if s.Graph != nil {
		return nil, fmt.Errorf("source already has a graph")
	}
	b := &llamaBuilder{
		s:       s,
		g:       &mmf.Graph{PrimitiveVersion: mmf.CurrentPrimitive, StdlibVersion: mmf.CurrentStdlib, Subgraphs: []mmf.Subgraph{{}}},
		byName:  map[string]int{},
		weights: map[string]uint32{},
		a8:      o.A8,
	}
	for i, t := range s.Tensors {
		b.byName[t.Name] = i
	}
	emb, ok := b.byName["token_embd.weight"]
	if !ok || len(s.Tensors[emb].Shape) != 2 || s.Tensors[emb].Shape[1] != uint64(c.Embed) {
		return nil, fmt.Errorf("tensor token_embd.weight is missing or not [V, %d]", c.Embed)
	}
	c.Vocab = int(s.Tensors[emb].Shape[0])

	E, F, H, Hkv, D, C, V := mmf.Dim(c.Embed), mmf.Dim(c.FFN), mmf.Dim(c.Heads), mmf.Dim(c.KVHeads), mmf.Dim(c.HeadDim), mmf.Dim(c.Context), mmf.Dim(c.Vocab)
	b.g.SymDims = []mmf.SymDim{{Name: "T", Bound: int64(c.Context)}}
	T := mmf.Dim(-1)
	sym := int64(-1) // the -1 sentinel of reshape for a symbolic dimension

	main := &b.g.Subgraphs[0]
	tokens := b.val("tokens", mmf.I32, 1, T)
	positions := b.val("positions", mmf.I64, 1, T)
	b.g.Values[tokens].Kind = mmf.KindInput
	b.g.Values[positions].Kind = mmf.KindInput
	main.Inputs = []uint32{tokens, positions}

	// Attention mask, shared by every layer: key j is visible from the
	// query at position p when j <= p, that is when not p < j.
	kposData := make([]byte, 8*c.Context)
	for j := 0; j < c.Context; j++ {
		binary.LittleEndian.PutUint64(kposData[8*j:], uint64(j))
	}
	kpos, err := b.addTensor("mmf.kv_positions", mmf.I64, []uint64{1, 1, 1, uint64(c.Context)}, kposData, mmf.KindConst)
	if err != nil {
		return nil, err
	}
	qpos := b.op(mmf.OpReshape, []uint32{positions}, b.val("query_positions", mmf.I64, 1, 1, T, 1), named("shape", ints(1, 1, sym, 1)))
	after := b.op(mmf.OpCmplt, []uint32{qpos, kpos}, b.val("mask.future", mmf.Bool, 1, 1, T, C))
	truth := b.op(mmf.OpConst, nil, b.val("true", mmf.Bool, 1, 1, 1, 1), mmf.Attr{Key: "value", Type: mmf.AttrInt, Int: 1})
	mask := b.op(mmf.OpXor, []uint32{after, truth}, b.val("mask", mmf.Bool, 1, 1, T, C))
	cacheIdx := b.op(mmf.OpReshape, []uint32{positions}, b.val("cache_index", mmf.I64, T), named("shape", ints(sym)))

	table := b.stored(emb, mmf.KindWeight)
	edt := s.Tensors[emb].DType
	var x uint32
	if edt == mmf.F32 || edt.IsBlock() || edt == mmf.F8E4M3 || edt == mmf.F8E5M2 {
		x = b.call("embedding", []uint32{table, tokens}, b.val("embed", mmf.F32, 1, T, E))
	} else {
		raw := b.call("embedding", []uint32{table, tokens}, b.val("embed.raw", edt, 1, T, E))
		x = b.op(mmf.OpCast, []uint32{raw}, b.val("embed", mmf.F32, 1, T, E), mmf.Attr{Key: "to", Type: mmf.AttrDType, DType: mmf.F32})
	}

	eps := mmf.Attr{Key: "epsilon", Type: mmf.AttrFloat, Float: c.NormEps}
	rope := []mmf.Attr{
		{Key: "theta", Type: mmf.AttrFloat, Float: c.RopeTheta},
		{Key: "interleaved", Type: mmf.AttrInt, Int: 1}, // llama.cpp's NORM rope: adjacent pairs
		{Key: "rotary_dims", Type: mmf.AttrInt, Int: int64(c.RopeDims)},
	}
	cacheShape := []mmf.Dim{1, mmf.Dim(c.KVHeads), mmf.Dim(c.Context), mmf.Dim(c.HeadDim)}
	for l := 0; l < c.Layers; l++ {
		p := fmt.Sprintf("blk.%d.", l)
		w := func(name string, shape ...int) (uint32, error) { return b.weight(p+name, shape...) }
		attnNorm, err := w("attn_norm.weight", c.Embed)
		if err != nil {
			return nil, err
		}
		wq, err := w("attn_q.weight", c.Embed, c.Embed)
		if err != nil {
			return nil, err
		}
		wk, err := w("attn_k.weight", c.KVHeads*c.HeadDim, c.Embed)
		if err != nil {
			return nil, err
		}
		wv, err := w("attn_v.weight", c.KVHeads*c.HeadDim, c.Embed)
		if err != nil {
			return nil, err
		}
		wo, err := w("attn_output.weight", c.Embed, c.Embed)
		if err != nil {
			return nil, err
		}
		ffnNorm, err := w("ffn_norm.weight", c.Embed)
		if err != nil {
			return nil, err
		}
		wg, err := w("ffn_gate.weight", c.FFN, c.Embed)
		if err != nil {
			return nil, err
		}
		wu, err := w("ffn_up.weight", c.FFN, c.Embed)
		if err != nil {
			return nil, err
		}
		wd, err := w("ffn_down.weight", c.Embed, c.FFN)
		if err != nil {
			return nil, err
		}
		// The caches start as zeros and take no space in the file.
		cacheK := b.zeros(p+"cache_k", mmf.F32, cacheShape...)
		cacheV := b.zeros(p+"cache_v", mmf.F32, cacheShape...)

		h := b.call("rms_norm", []uint32{x, b.f32Of(attnNorm)}, b.val(p+"attn_in", mmf.F32, 1, T, E), eps)
		heads := func(name string, w uint32, n mmf.Dim) uint32 {
			y := b.matmulOp(p+name, h, w, 1, T, n*D)
			y = b.op(mmf.OpReshape, []uint32{y}, b.val(p+name+".heads", mmf.F32, 1, T, n, D), named("shape", ints(1, sym, int64(n), int64(D))))
			return b.op(mmf.OpPermute, []uint32{y}, b.val(p+name+".bhtd", mmf.F32, 1, n, T, D), named("perm", ints(0, 2, 1, 3)))
		}
		q := heads("q", wq, H)
		k := heads("k", wk, Hkv)
		v := heads("v", wv, Hkv)
		q = b.call("rope", []uint32{q, positions}, b.val(p+"q.rope", mmf.F32, 1, H, T, D), rope...)
		k = b.call("rope", []uint32{k, positions}, b.val(p+"k.rope", mmf.F32, 1, Hkv, T, D), rope...)
		scatter := mmf.Attr{Key: "axis", Type: mmf.AttrInt, Int: 2}
		newK := b.op(mmf.OpScatter, []uint32{cacheK, cacheIdx, k}, b.val(p+"cache_k.next", mmf.F32, 1, Hkv, C, D), scatter)
		newV := b.op(mmf.OpScatter, []uint32{cacheV, cacheIdx, v}, b.val(p+"cache_v.next", mmf.F32, 1, Hkv, C, D), scatter)
		b.g.Updates = append(b.g.Updates, mmf.Update{Variable: cacheK, Value: newK}, mmf.Update{Variable: cacheV, Value: newV})

		o := b.call("attention", []uint32{q, newK, newV, mask}, b.val(p+"attn", mmf.F32, 1, H, T, D),
			mmf.Attr{Key: "num_kv_heads", Type: mmf.AttrInt, Int: int64(c.KVHeads)})
		o = b.op(mmf.OpPermute, []uint32{o}, b.val(p+"attn.bthd", mmf.F32, 1, T, H, D), named("perm", ints(0, 2, 1, 3)))
		o = b.op(mmf.OpReshape, []uint32{o}, b.val(p+"attn.merged", mmf.F32, 1, T, E), named("shape", ints(1, sym, int64(E))))
		o = b.matmulOp(p+"attn_out", o, wo, 1, T, E)
		x = b.op(mmf.OpAdd, []uint32{x, o}, b.val(p+"resid_attn", mmf.F32, 1, T, E))

		h = b.call("rms_norm", []uint32{x, b.f32Of(ffnNorm)}, b.val(p+"ffn_in", mmf.F32, 1, T, E), eps)
		gate := b.matmulOp(p+"gate", h, wg, 1, T, F)
		up := b.matmulOp(p+"up", h, wu, 1, T, F)
		act := b.call("swiglu", []uint32{gate, up}, b.val(p+"swiglu", mmf.F32, 1, T, F))
		down := b.matmulOp(p+"down", act, wd, 1, T, E)
		x = b.op(mmf.OpAdd, []uint32{x, down}, b.val(p+"resid_ffn", mmf.F32, 1, T, E))
	}
	outNorm, err := b.weight("output_norm.weight", c.Embed)
	if err != nil {
		return nil, err
	}
	head := table // tied embeddings when there is no output.weight
	if _, ok := b.byName["output.weight"]; ok {
		if head, err = b.weight("output.weight", c.Vocab, c.Embed); err != nil {
			return nil, err
		}
	}
	x = b.call("rms_norm", []uint32{x, b.f32Of(outNorm)}, b.val("final_norm", mmf.F32, 1, T, E), eps)
	logits := b.matmulOp("logits", x, head, 1, T, V)
	b.g.Values[logits].Kind = mmf.KindOutput
	main.Outputs = []uint32{logits}

	if err := b.g.Validate(); err != nil {
		return nil, fmt.Errorf("built graph is invalid: %w", err)
	}
	s.Graph = b.g
	s.Metadata["mmf.context_length"] = int64(c.Context)
	return c, nil
}
