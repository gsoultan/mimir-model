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
	"encoding/binary"
	"fmt"
	"math"

	mim "github.com/gsoultan/mimir-model"
)

// Graph construction for decoder-only transformers in the Llama family.
//
// GGUF and safetensors carry weights but no graph, so an importer that
// wants a runnable file writes the architecture's forward pass itself, from
// the hyperparameters in the metadata. BuildLlama does this for GGUF files
// whose general.architecture is "llama" (Llama 1 to 3, TinyLlama, Mistral
// without sliding windows, and others using llama.cpp's llama layout) or
// "qwen2" (Qwen 2 and 2.5), which adds biases to the query, key and value
// projections and rotates the two halves of each head instead of adjacent
// pairs.
//
// The graph takes one block of tokens per run:
//
//	inputs:    tokens    i32[1, T]   token ids
//	           positions i64[1, T]   their positions in the sequence
//	outputs:   logits    f32[1, T, V]
//	variables: blk.N.cache_k, blk.N.cache_v  f16 or f32 [1, Hkv, C, D]
//
// T is a symbolic dimension bounded by the context length C. The key-value
// cache follows ADR 0003: each layer scatters its new keys and values into
// the cache variables at the given positions and attends over the whole
// cache, with a mask that admits key position j for the query at position
// p when j <= p. A prompt is one run with positions 0..T-1; each generated
// token is a run with T = 1. A cache in f16 holds the keys and values
// rounded to f16, which attention.v2 reads as they are.

// Context lengths beyond this are refused, so a typo cannot ask for an
// enormous zero-filled cache.
const maxLlamaContext = 1 << 20

// LlamaConfig holds the hyperparameters BuildLlama reads from the metadata.
type LlamaConfig struct {
	Arch       string // general.architecture: "llama" or "qwen2"
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
	Context    int // C, the cache length of each sequence
	MaxContext int // the model's trained context length
	Slots      int // sequences the cache holds at once
}

func metaInt(md mim.Metadata, key string) (int64, bool) {
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

func metaFloat(md mim.Metadata, key string) (float64, bool) {
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

// LlamaConfigOf reads the hyperparameters of a llama- or qwen2-architecture
// source. context is the cache length to build for; 0 means the model's own.
func LlamaConfigOf(s *Source, context int) (*LlamaConfig, error) {
	md := s.Metadata
	arch, _ := md["general.architecture"].(string)
	if arch != "llama" && arch != "qwen2" {
		return nil, fmt.Errorf("architecture %q: only \"llama\" and \"qwen2\" graphs can be built", arch)
	}
	req := func(key string) (int, error) {
		v, ok := metaInt(md, arch+"."+key)
		if !ok || v <= 0 || v > 1<<24 {
			return 0, fmt.Errorf("metadata %s.%s is missing or invalid", arch, key)
		}
		return int(v), nil
	}
	c := &LlamaConfig{Arch: arch}
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
	if _, ok := md[arch+".attention.head_count_kv"]; ok {
		if c.KVHeads, err = req("attention.head_count_kv"); err != nil {
			return nil, err
		}
	}
	if c.Embed%c.Heads != 0 || c.Heads%c.KVHeads != 0 {
		return nil, fmt.Errorf("embedding length %d, %d heads and %d key-value heads do not divide", c.Embed, c.Heads, c.KVHeads)
	}
	c.HeadDim = c.Embed / c.Heads
	c.RopeDims = c.HeadDim
	if _, ok := md[arch+".rope.dimension_count"]; ok {
		if c.RopeDims, err = req("rope.dimension_count"); err != nil {
			return nil, err
		}
		if c.RopeDims%2 != 0 || c.RopeDims > c.HeadDim {
			return nil, fmt.Errorf("rope dimension count %d for head size %d", c.RopeDims, c.HeadDim)
		}
	}
	c.RopeTheta = 10000
	if v, ok := metaFloat(md, arch+".rope.freq_base"); ok {
		c.RopeTheta = v
	}
	if t, ok := md[arch+".rope.scaling.type"].(string); ok && t != "none" {
		return nil, fmt.Errorf("rope scaling %q is not supported yet", t)
	}
	var ok bool
	if c.NormEps, ok = metaFloat(md, arch+".attention.layer_norm_rms_epsilon"); !ok || c.NormEps <= 0 {
		return nil, fmt.Errorf("metadata %s.attention.layer_norm_rms_epsilon is missing or invalid", arch)
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
	g       *mim.Graph
	byName  map[string]int
	weights map[string]uint32
	a8      bool
	cache   mim.DType // the key-value cache's dtype, F32 or F16
}

func (b *llamaBuilder) val(name string, dt mim.DType, shape ...mim.Dim) uint32 {
	b.g.Values = append(b.g.Values, mim.Value{Name: name, DType: dt, Shape: shape})
	return uint32(len(b.g.Values) - 1)
}

func (b *llamaBuilder) op(opc mim.Opcode, in []uint32, out uint32, attrs ...mim.Attr) uint32 {
	main := &b.g.Subgraphs[0]
	main.Ops = append(main.Ops, mim.Op{Opcode: opc, Inputs: in, Outputs: []uint32{out}, Attrs: attrs})
	return out
}

func (b *llamaBuilder) call(name string, in []uint32, out uint32, attrs ...mim.Attr) uint32 {
	return b.callVersion(name, 1, in, out, attrs...)
}

// callVersion is call for version v of the composite.
func (b *llamaBuilder) callVersion(name string, v int64, in []uint32, out uint32, attrs ...mim.Attr) uint32 {
	a := append([]mim.Attr{
		{Key: "name", Type: mim.AttrString, String: name},
		{Key: "version", Type: mim.AttrInt, Int: v},
	}, attrs...)
	return b.op(mim.OpCall, in, out, a...)
}

// stored adds a value of the given kind backed by source tensor i.
func (b *llamaBuilder) stored(i int, kind mim.ValueKind) uint32 {
	t := b.s.Tensors[i]
	dims := make([]mim.Dim, len(t.Shape))
	for k, d := range t.Shape {
		dims[k] = mim.Dim(d)
	}
	b.g.Values = append(b.g.Values, mim.Value{Name: t.Name, DType: t.DType, Shape: dims, Kind: kind, Ref: uint32(i)})
	return uint32(len(b.g.Values) - 1)
}

// zeros adds a variable that starts as zeros.
func (b *llamaBuilder) zeros(name string, dt mim.DType, shape ...mim.Dim) uint32 {
	b.g.Values = append(b.g.Values, mim.Value{Name: name, DType: dt, Shape: shape, Kind: mim.KindVariable, Zero: true})
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
	id := b.stored(i, mim.KindWeight)
	b.weights[name] = id
	return id, nil
}

// addTensor appends a tensor created by the importer.
func (b *llamaBuilder) addTensor(name string, dt mim.DType, shape []uint64, data []byte, kind mim.ValueKind) (uint32, error) {
	if _, dup := b.byName[name]; dup {
		return 0, fmt.Errorf("tensor %q already exists in the source", name)
	}
	b.s.Tensors = append(b.s.Tensors, Tensor{Name: name, DType: dt, Shape: shape, Data: data})
	b.byName[name] = len(b.s.Tensors) - 1
	return b.stored(len(b.s.Tensors)-1, kind), nil
}

func ints(v ...int64) mim.Attr            { return mim.Attr{Type: mim.AttrInts, Ints: v} }
func named(k string, a mim.Attr) mim.Attr { a.Key = k; return a }

// matmulOp emits x times w^T, with linear.v1 for f32 weights and
// dequant_matmul.v1 or dequant_matmul_a8 for every other dtype. With 8-bit
// activations, the K-quant weights get dequant_matmul_a8.v2 with blocks of
// 256, the Q8_K activations llama.cpp pairs with them, and the other
// dtypes dequant_matmul_a8.v1, the blocks of 32 of its Q8_0 activations.
func (b *llamaBuilder) matmulOp(name string, x, w uint32, shape ...mim.Dim) uint32 {
	return b.matmulBias(name, x, w, nil, shape...)
}

// matmulBias is matmulOp plus a bias, the optional third input of the
// composites, when bias is not nil.
func (b *llamaBuilder) matmulBias(name string, x, w uint32, bias *uint32, shape ...mim.Dim) uint32 {
	out := b.val(name, mim.F32, shape...)
	in := []uint32{x, w}
	if bias != nil {
		in = append(in, *bias)
	}
	switch dt := b.g.Values[w].DType; {
	case dt == mim.F32:
		return b.call("linear", in, out)
	case b.a8 && (dt == mim.Q4_K || dt == mim.Q6_K):
		return b.callVersion("dequant_matmul_a8", 2, in, out,
			mim.Attr{Key: "block", Type: mim.AttrInt, Int: 256})
	case b.a8:
		return b.call("dequant_matmul_a8", in, out)
	}
	return b.call("dequant_matmul", in, out)
}

// f32Of casts a weight such as a norm scale to f32 when it is stored in
// another float dtype.
func (b *llamaBuilder) f32Of(w uint32) uint32 {
	v := b.g.Values[w]
	if v.DType == mim.F32 {
		return w
	}
	return b.op(mim.OpCast, []uint32{w}, b.val(v.Name+".f32", mim.F32, v.Shape...),
		mim.Attr{Key: "to", Type: mim.AttrDType, DType: mim.F32})
}

// LlamaOptions configures BuildLlama.
type LlamaOptions struct {
	// Context is the key-value cache length; 0 means the model's own.
	Context int
	// A8 writes the matrix products with quantized weights as
	// dequant_matmul_a8 (v2 with blocks of 256 for K-quant weights, v1
	// otherwise), which lets a runtime round the activations to 8-bit
	// blocks as llama.cpp does, instead of dequant_matmul.v1.
	A8 bool
	// Cache is the dtype of the key-value cache: F16, which halves its
	// memory and has the graph call attention.v2, or F32 (also the zero
	// value), with attention.v1.
	Cache mim.DType
	// Slots is the number of sequences the key-value cache holds at once,
	// so that a runtime can serve several requests in one run; 0 or 1
	// means one. With more than one, the graph has a third input, slots,
	// and its cache holds Slots times Context positions.
	Slots int
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
		g:       &mim.Graph{PrimitiveVersion: mim.CurrentPrimitive, StdlibVersion: mim.CurrentStdlib, Subgraphs: []mim.Subgraph{{}}},
		byName:  map[string]int{},
		weights: map[string]uint32{},
		a8:      o.A8,
		cache:   mim.F32,
	}
	switch o.Cache {
	case 0, mim.F32:
	case mim.F16:
		b.cache = mim.F16
	default:
		return nil, fmt.Errorf("key-value cache dtype %s: want f32 or f16", o.Cache)
	}
	for i, t := range s.Tensors {
		b.byName[t.Name] = i
	}
	emb, ok := b.byName["token_embd.weight"]
	if !ok || len(s.Tensors[emb].Shape) != 2 || s.Tensors[emb].Shape[1] != uint64(c.Embed) {
		return nil, fmt.Errorf("tensor token_embd.weight is missing or not [V, %d]", c.Embed)
	}
	c.Vocab = int(s.Tensors[emb].Shape[0])

	c.Slots = max(1, o.Slots)
	if c.Slots > 1<<10 || c.Slots*c.Context > maxLlamaContext {
		return nil, fmt.Errorf("%d slots of %d positions: the cache would exceed %d", c.Slots, c.Context, maxLlamaContext)
	}
	cacheLen := c.Slots * c.Context
	E, F, H, Hkv, D, C, V := mim.Dim(c.Embed), mim.Dim(c.FFN), mim.Dim(c.Heads), mim.Dim(c.KVHeads), mim.Dim(c.HeadDim), mim.Dim(cacheLen), mim.Dim(c.Vocab)
	b.g.SymDims = []mim.SymDim{{Name: "T", Bound: int64(cacheLen)}}
	T := mim.Dim(-1)
	sym := int64(-1) // the -1 sentinel of reshape for a symbolic dimension

	main := &b.g.Subgraphs[0]
	tokens := b.val("tokens", mim.I32, 1, T)
	positions := b.val("positions", mim.I64, 1, T)
	b.g.Values[tokens].Kind = mim.KindInput
	b.g.Values[positions].Kind = mim.KindInput
	main.Inputs = []uint32{tokens, positions}

	// Attention mask, shared by every layer: key j is visible from the
	// query at position p when j <= p, that is when not p < j. With
	// several slots, the token at position p of the sequence in slot s
	// goes to cache position s*C+p, and sees keys s*C to s*C+p: the mask
	// also hides the keys j < s*C of the slots before.
	kposData := make([]byte, 8*cacheLen)
	for j := 0; j < cacheLen; j++ {
		binary.LittleEndian.PutUint64(kposData[8*j:], uint64(j))
	}
	kpos, err := b.addTensor("mim.kv_positions", mim.I64, []uint64{1, 1, 1, uint64(cacheLen)}, kposData, mim.KindConst)
	if err != nil {
		return nil, err
	}
	at := positions
	var start uint32
	if c.Slots > 1 {
		slots := b.val("slots", mim.I64, 1, T)
		b.g.Values[slots].Kind = mim.KindInput
		main.Inputs = append(main.Inputs, slots)
		size := b.op(mim.OpConst, nil, b.val("slot_size", mim.I64, 1, 1), mim.Attr{Key: "value", Type: mim.AttrInt, Int: int64(c.Context)})
		start = b.op(mim.OpMul, []uint32{slots, size}, b.val("slot_start", mim.I64, 1, T))
		at = b.op(mim.OpAdd, []uint32{start, positions}, b.val("slot_positions", mim.I64, 1, T))
	}
	qpos := b.op(mim.OpReshape, []uint32{at}, b.val("query_positions", mim.I64, 1, 1, T, 1), named("shape", ints(1, 1, sym, 1)))
	hidden := b.op(mim.OpCmplt, []uint32{qpos, kpos}, b.val("mask.future", mim.Bool, 1, 1, T, C))
	if c.Slots > 1 {
		qstart := b.op(mim.OpReshape, []uint32{start}, b.val("query_slot_start", mim.I64, 1, 1, T, 1), named("shape", ints(1, 1, sym, 1)))
		before := b.op(mim.OpCmplt, []uint32{kpos, qstart}, b.val("mask.other_slots", mim.Bool, 1, 1, T, C))
		hidden = b.op(mim.OpOr, []uint32{hidden, before}, b.val("mask.hidden", mim.Bool, 1, 1, T, C))
	}
	truth := b.op(mim.OpConst, nil, b.val("true", mim.Bool, 1, 1, 1, 1), mim.Attr{Key: "value", Type: mim.AttrInt, Int: 1})
	mask := b.op(mim.OpXor, []uint32{hidden, truth}, b.val("mask", mim.Bool, 1, 1, T, C))
	cacheIdx := b.op(mim.OpReshape, []uint32{at}, b.val("cache_index", mim.I64, T), named("shape", ints(sym)))

	table := b.stored(emb, mim.KindWeight)
	edt := s.Tensors[emb].DType
	var x uint32
	if edt == mim.F32 || edt.IsBlock() || edt == mim.F8E4M3 || edt == mim.F8E5M2 {
		x = b.call("embedding", []uint32{table, tokens}, b.val("embed", mim.F32, 1, T, E))
	} else {
		raw := b.call("embedding", []uint32{table, tokens}, b.val("embed.raw", edt, 1, T, E))
		x = b.op(mim.OpCast, []uint32{raw}, b.val("embed", mim.F32, 1, T, E), mim.Attr{Key: "to", Type: mim.AttrDType, DType: mim.F32})
	}

	eps := mim.Attr{Key: "epsilon", Type: mim.AttrFloat, Float: c.NormEps}
	// llama.cpp's NORM rope rotates adjacent pairs; qwen2 uses NEOX, which
	// rotates the first half of each head against the second.
	interleaved := int64(1)
	if c.Arch == "qwen2" {
		interleaved = 0
	}
	rope := []mim.Attr{
		{Key: "theta", Type: mim.AttrFloat, Float: c.RopeTheta},
		{Key: "interleaved", Type: mim.AttrInt, Int: interleaved},
		{Key: "rotary_dims", Type: mim.AttrInt, Int: int64(c.RopeDims)},
	}
	cacheShape := []mim.Dim{1, mim.Dim(c.KVHeads), C, mim.Dim(c.HeadDim)}
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
		cacheK := b.zeros(p+"cache_k", b.cache, cacheShape...)
		cacheV := b.zeros(p+"cache_v", b.cache, cacheShape...)

		h := b.call("rms_norm", []uint32{x, b.f32Of(attnNorm)}, b.val(p+"attn_in", mim.F32, 1, T, E), eps)
		heads := func(name string, wt uint32, n mim.Dim) (uint32, error) {
			var bias *uint32
			if c.Arch == "qwen2" {
				bv, err := w("attn_"+name+".bias", int(n*D))
				if err != nil {
					return 0, err
				}
				bv = b.f32Of(bv)
				bias = &bv
			}
			y := b.matmulBias(p+name, h, wt, bias, 1, T, n*D)
			y = b.op(mim.OpReshape, []uint32{y}, b.val(p+name+".heads", mim.F32, 1, T, n, D), named("shape", ints(1, sym, int64(n), int64(D))))
			return b.op(mim.OpPermute, []uint32{y}, b.val(p+name+".bhtd", mim.F32, 1, n, T, D), named("perm", ints(0, 2, 1, 3))), nil
		}
		q, err := heads("q", wq, H)
		if err != nil {
			return nil, err
		}
		k, err := heads("k", wk, Hkv)
		if err != nil {
			return nil, err
		}
		v, err := heads("v", wv, Hkv)
		if err != nil {
			return nil, err
		}
		q = b.call("rope", []uint32{q, positions}, b.val(p+"q.rope", mim.F32, 1, H, T, D), rope...)
		k = b.call("rope", []uint32{k, positions}, b.val(p+"k.rope", mim.F32, 1, Hkv, T, D), rope...)
		attnVersion := int64(1)
		if b.cache == mim.F16 {
			// The new keys and values are rounded to f16 as they go into the
			// cache, and attention.v2 reads them from it as they are.
			to := mim.Attr{Key: "to", Type: mim.AttrDType, DType: mim.F16}
			k = b.op(mim.OpCast, []uint32{k}, b.val(p+"k.f16", mim.F16, 1, Hkv, T, D), to)
			v = b.op(mim.OpCast, []uint32{v}, b.val(p+"v.f16", mim.F16, 1, Hkv, T, D), to)
			attnVersion = 2
		}
		scatter := mim.Attr{Key: "axis", Type: mim.AttrInt, Int: 2}
		newK := b.op(mim.OpScatter, []uint32{cacheK, cacheIdx, k}, b.val(p+"cache_k.next", b.cache, 1, Hkv, C, D), scatter)
		newV := b.op(mim.OpScatter, []uint32{cacheV, cacheIdx, v}, b.val(p+"cache_v.next", b.cache, 1, Hkv, C, D), scatter)
		b.g.Updates = append(b.g.Updates, mim.Update{Variable: cacheK, Value: newK}, mim.Update{Variable: cacheV, Value: newV})

		o := b.callVersion("attention", attnVersion, []uint32{q, newK, newV, mask}, b.val(p+"attn", mim.F32, 1, H, T, D),
			mim.Attr{Key: "num_kv_heads", Type: mim.AttrInt, Int: int64(c.KVHeads)})
		o = b.op(mim.OpPermute, []uint32{o}, b.val(p+"attn.bthd", mim.F32, 1, T, H, D), named("perm", ints(0, 2, 1, 3)))
		o = b.op(mim.OpReshape, []uint32{o}, b.val(p+"attn.merged", mim.F32, 1, T, E), named("shape", ints(1, sym, int64(E))))
		o = b.matmulOp(p+"attn_out", o, wo, 1, T, E)
		x = b.op(mim.OpAdd, []uint32{x, o}, b.val(p+"resid_attn", mim.F32, 1, T, E))

		h = b.call("rms_norm", []uint32{x, b.f32Of(ffnNorm)}, b.val(p+"ffn_in", mim.F32, 1, T, E), eps)
		gate := b.matmulOp(p+"gate", h, wg, 1, T, F)
		up := b.matmulOp(p+"up", h, wu, 1, T, F)
		act := b.call("swiglu", []uint32{gate, up}, b.val(p+"swiglu", mim.F32, 1, T, F))
		down := b.matmulOp(p+"down", act, wd, 1, T, E)
		x = b.op(mim.OpAdd, []uint32{x, down}, b.val(p+"resid_ffn", mim.F32, 1, T, E))
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
	x = b.call("rms_norm", []uint32{x, b.f32Of(outNorm)}, b.val("final_norm", mim.F32, 1, T, E), eps)
	logits := b.matmulOp("logits", x, head, 1, T, V)
	b.g.Values[logits].Kind = mim.KindOutput
	main.Outputs = []uint32{logits}

	if err := b.g.Validate(); err != nil {
		return nil, fmt.Errorf("built graph is invalid: %w", err)
	}
	s.Graph = b.g
	s.Metadata["mim.context_length"] = int64(c.Context)
	if c.Slots > 1 {
		s.Metadata["mim.slots"] = int64(c.Slots)
	}
	return c, nil
}
