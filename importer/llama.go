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
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	mim "github.com/gsoultan/mimir-model"
)

// Graph construction for decoder-only transformers in the Llama family.
//
// GGUF and safetensors carry weights but no graph, so an importer that
// wants a runnable file writes the architecture's forward pass itself, from
// the hyperparameters in the metadata. BuildLlama does this for GGUF files
// whose general.architecture is one of:
//
//   - "llama": Llama 1 to 3, TinyLlama, Mistral without sliding windows, and
//     others using llama.cpp's llama layout. Llama 3.1 and later divide
//     each rotary pair's frequency by its entry of rope_freqs.weight.
//   - "qwen2": Qwen 2 and 2.5, which add biases to the query, key and value
//     projections and rotate the two halves of each head instead of
//     adjacent pairs.
//   - "qwen3": Qwen 3, which drops the biases and normalizes each query and
//     key head (RMSNorm over its features) before rotating it, with heads
//     wider than embedding/heads.
//   - "gemma2" and "gemma3": Gemma 2 and 3, which scale the embeddings by
//     the square root of their width, use a GELU-gated feed-forward layer,
//     normalize the output of attention and of the feed-forward layer
//     before adding it, and let local layers attend to a sliding window
//     only. Gemma 2 soft-caps the attention scores and the logits; Gemma 3
//     normalizes query and key heads as Qwen 3 does and gives its local
//     layers their own rope base.
//   - "phi3": Phi 3, 3.5 and 4, whose query, key and value projections may
//     be one fused matrix and whose up projection holds the gate's rows
//     ahead of its own, which the builder splits apart (see splitPhi3),
//     and whose LongRoPE divides each pair's frequency by a factor and
//     scales the rotated features.
//
// llama.cpp's converter already adds 1 to Gemma's norm weights, which
// Gemma multiplies by 1 + w, so every family's RMSNorm here is x/rms·w.
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
// p when j <= p, and for a local layer with a window W only when also
// p - j < W. A prompt is one run with positions 0..T-1; each generated
// token is a run with T = 1. A cache in f16 holds the keys and values
// rounded to f16, which attention.v2 reads as they are.

// Context lengths beyond this are refused, so a typo cannot ask for an
// enormous zero-filled cache.
const maxLlamaContext = 1 << 20

// LlamaArchitectures lists the values of general.architecture BuildLlama
// builds graphs for.
var LlamaArchitectures = []string{"llama", "qwen2", "qwen3", "gemma2", "gemma3", "phi3"}

// LlamaConfig holds the hyperparameters BuildLlama reads from the metadata.
// The fields after Slots are zero for an architecture that lacks them.
type LlamaConfig struct {
	Arch       string // general.architecture, one of LlamaArchitectures
	Layers     int
	Embed      int // E
	FFN        int
	Heads      int // Hq
	KVHeads    int // Hkv
	HeadDim    int // D, E / Hq unless the file says otherwise
	RopeDims   int
	RopeTheta  float64
	NormEps    float64
	Vocab      int
	Context    int // C, the cache length of each sequence
	MaxContext int // the model's trained context length
	Slots      int // sequences the cache holds at once

	// RopeScale is the linear rope scaling of the global layers: their
	// frequencies are divided by it. 0 means none.
	RopeScale float64
	// RopeAttnFactor scales the rotated features of queries and keys, as
	// Phi 3's LongRoPE does. 0 means none.
	RopeAttnFactor float64
	// Window is the sliding window of the local layers, in positions, and
	// LocalEvery says which layers are local: all but every LocalEvery-th
	// (Gemma 2: 2, Gemma 3: 6). 0 means every layer is global.
	Window, LocalEvery int
	// LocalTheta is the rope base of the local layers; 0 means RopeTheta.
	// Local layers are never linearly scaled.
	LocalTheta float64
	// AttnScale is the attention score scale; 0 means 1/sqrt(D).
	AttnScale float64
	// AttnSoftcap and FinalSoftcap cap the attention scores and the
	// logits at ±cap with cap·tanh(x/cap), as Gemma 2 does; 0 means none.
	AttnSoftcap, FinalSoftcap float64
}

// Local reports whether layer l attends to a sliding window only.
func (c *LlamaConfig) Local(l int) bool {
	return c.Window > 0 && c.LocalEvery > 0 && l%c.LocalEvery < c.LocalEvery-1
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

// LlamaConfigOf reads the hyperparameters of a source of one of
// LlamaArchitectures. context is the cache length to build for; 0 means the
// model's own.
func LlamaConfigOf(s *Source, context int) (*LlamaConfig, error) {
	md := s.Metadata
	arch, _ := md["general.architecture"].(string)
	if !slices.Contains(LlamaArchitectures, arch) {
		return nil, fmt.Errorf("architecture %q: only %s graphs can be built", arch, strings.Join(quoted(LlamaArchitectures), ", "))
	}
	req := func(key string) (int, error) {
		v, ok := metaInt(md, arch+"."+key)
		if !ok || v <= 0 || v > 1<<24 {
			return 0, fmt.Errorf("metadata %s.%s is missing or invalid", arch, key)
		}
		return int(v), nil
	}
	opt := func(key string, def int) (int, error) {
		if _, ok := md[arch+"."+key]; !ok {
			return def, nil
		}
		return req(key)
	}
	positive := func(key string, def float64) (float64, error) {
		if _, ok := md[arch+"."+key]; !ok {
			return def, nil
		}
		v, ok := metaFloat(md, arch+"."+key)
		if !ok || !(v > 0) || math.IsInf(v, 0) {
			return 0, fmt.Errorf("metadata %s.%s is invalid", arch, key)
		}
		return v, nil
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
	if c.KVHeads, err = opt("attention.head_count_kv", c.Heads); err != nil {
		return nil, err
	}
	if c.Heads%c.KVHeads != 0 {
		return nil, fmt.Errorf("%d heads and %d key-value heads do not divide", c.Heads, c.KVHeads)
	}
	if _, ok := md[arch+".attention.key_length"]; ok {
		if c.HeadDim, err = req("attention.key_length"); err != nil {
			return nil, err
		}
	} else if c.Embed%c.Heads != 0 {
		return nil, fmt.Errorf("embedding length %d, %d heads and %d key-value heads do not divide", c.Embed, c.Heads, c.KVHeads)
	} else {
		c.HeadDim = c.Embed / c.Heads
	}
	if v, err := opt("attention.value_length", c.HeadDim); err != nil {
		return nil, err
	} else if v != c.HeadDim {
		return nil, fmt.Errorf("values of %d features for keys of %d", v, c.HeadDim)
	}
	if c.RopeDims, err = opt("rope.dimension_count", c.HeadDim); err != nil {
		return nil, err
	}
	if c.RopeDims%2 != 0 || c.RopeDims > c.HeadDim {
		return nil, fmt.Errorf("rope dimension count %d for head size %d", c.RopeDims, c.HeadDim)
	}
	if c.RopeTheta, err = positive("rope.freq_base", 10000); err != nil {
		return nil, err
	}
	// llama.cpp reads a missing scaling type as linear, with a factor of 1
	// unless the file gives one. Phi 3's LongRoPE is in its factor tensors
	// and attention factor, read when the graph is built.
	scaling, _ := md[arch+".rope.scaling.type"].(string)
	switch scaling {
	case "", "linear":
		f, err := positive("rope.scaling.factor", 0)
		if err != nil {
			return nil, err
		}
		if f == 0 {
			if f, err = positive("rope.scale_linear", 0); err != nil {
				return nil, err
			}
		}
		if f != 0 && f != 1 {
			c.RopeScale = f
		}
	case "none":
	case "longrope":
		if arch != "phi3" {
			return nil, fmt.Errorf("rope scaling %q is not supported for %s", scaling, arch)
		}
	default:
		return nil, fmt.Errorf("rope scaling %q is not supported yet", scaling)
	}
	if af, err := positive("rope.scaling.attn_factor", 1); err != nil {
		return nil, err
	} else if af != 1 {
		c.RopeAttnFactor = af
	}
	var ok bool
	if c.NormEps, ok = metaFloat(md, arch+".attention.layer_norm_rms_epsilon"); !ok || c.NormEps <= 0 {
		return nil, fmt.Errorf("metadata %s.attention.layer_norm_rms_epsilon is missing or invalid", arch)
	}

	switch arch {
	case "gemma2", "gemma3":
		// Gemma scales the queries by 1/sqrt(query_pre_attn_scalar), which
		// GGUF does not carry: llama.cpp takes it to be the head size, but
		// embedding/heads for the 27B models, which it knows by their
		// layer count.
		c.AttnScale = 1 / math.Sqrt(float64(c.HeadDim))
		if arch == "gemma2" && c.Layers == 46 || arch == "gemma3" && c.Layers == 62 {
			c.AttnScale = 1 / math.Sqrt(float64(c.Embed/c.Heads))
		}
		if arch == "gemma2" {
			c.LocalEvery, c.Window = 2, 4096
			if c.AttnSoftcap, err = positive("attn_logit_softcapping", 0); err != nil {
				return nil, err
			}
		} else {
			c.LocalEvery = 6
			c.LocalTheta = 10000
		}
		if c.FinalSoftcap, err = positive("final_logit_softcapping", 0); err != nil {
			return nil, err
		}
		if c.Window, err = opt("attention.sliding_window", c.Window); err != nil {
			return nil, err
		}
		if c.LocalEvery, err = opt("attention.sliding_window_pattern", c.LocalEvery); err != nil {
			return nil, fmt.Errorf("%w (a list of local layers is not supported)", err)
		}
		if arch == "gemma2" {
			c.LocalTheta = c.RopeTheta
		}
		if c.LocalTheta, err = positive("rope.freq_base_swa", c.LocalTheta); err != nil {
			return nil, err
		}
		if c.Window == 0 {
			c.LocalEvery, c.LocalTheta = 0, 0
		}
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

func quoted(ss []string) []string {
	out := make([]string, len(ss))
	for i, s := range ss {
		out[i] = strconv.Quote(s)
	}
	return out
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

// BuildLlama adds a forward-pass graph for a source of one of
// LlamaArchitectures, with a key-value cache of the configured context
// length, and returns the configuration it used. The cache tensors and the
// key position table are added to s.Tensors; AddTo writes the graph
// section.
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
	if c.Arch == "phi3" {
		if err := splitPhi3(s, c); err != nil {
			return nil, err
		}
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
	QD, KD := H*D, Hkv*D // widths of the queries and of the keys or values
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
	// A local layer's mask also hides the keys j <= p - W, that is those
	// with j < p - W + 1. Slots start a multiple of C apart and W < C
	// matters only within one, so this needs no slot term.
	localMask := mask
	if c.Window > 0 && c.Window < c.Context {
		off := b.op(mim.OpConst, nil, b.val("window_offset", mim.I64, 1, 1, 1, 1), mim.Attr{Key: "value", Type: mim.AttrInt, Int: int64(1 - c.Window)})
		edge := b.op(mim.OpAdd, []uint32{qpos, off}, b.val("window_start", mim.I64, 1, 1, T, 1))
		old := b.op(mim.OpCmplt, []uint32{kpos, edge}, b.val("mask.outside_window", mim.Bool, 1, 1, T, C))
		lh := b.op(mim.OpOr, []uint32{hidden, old}, b.val("mask.local_hidden", mim.Bool, 1, 1, T, C))
		localMask = b.op(mim.OpXor, []uint32{lh, truth}, b.val("mask.local", mim.Bool, 1, 1, T, C))
	}
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
	gemma := c.Arch == "gemma2" || c.Arch == "gemma3"
	if gemma {
		x = b.op(mim.OpMul, []uint32{x, b.scalar("embed_scale", math.Sqrt(float64(c.Embed)))}, b.val("embed.scaled", mim.F32, 1, T, E))
	}

	eps := mim.Attr{Key: "epsilon", Type: mim.AttrFloat, Float: c.NormEps}
	norm := func(name string, in, scale uint32, shape ...mim.Dim) uint32 {
		return b.call("rms_norm", []uint32{in, b.f32Of(scale)}, b.val(name, mim.F32, shape...), eps)
	}
	globalRope, localRope, err := b.ropes(c)
	if err != nil {
		return nil, err
	}
	cacheShape := []mim.Dim{1, Hkv, C, D}
	for l := 0; l < c.Layers; l++ {
		p := fmt.Sprintf("blk.%d.", l)
		var errs []error
		w := func(name string, shape ...int) uint32 {
			id, err := b.weight(p+name, shape...)
			errs = append(errs, err)
			return id
		}
		attnNorm := w("attn_norm.weight", c.Embed)
		wq := w("attn_q.weight", int(QD), c.Embed)
		wk := w("attn_k.weight", int(KD), c.Embed)
		wv := w("attn_v.weight", int(KD), c.Embed)
		wo := w("attn_output.weight", c.Embed, int(QD))
		ffnNorm := w("ffn_norm.weight", c.Embed)
		wg := w("ffn_gate.weight", c.FFN, c.Embed)
		wu := w("ffn_up.weight", c.FFN, c.Embed)
		wd := w("ffn_down.weight", c.Embed, c.FFN)
		var qNorm, kNorm, postAttn, postFFN uint32
		headNorms := c.Arch == "qwen3" || c.Arch == "gemma3"
		if headNorms {
			qNorm = w("attn_q_norm.weight", c.HeadDim)
			kNorm = w("attn_k_norm.weight", c.HeadDim)
		}
		if gemma {
			postAttn = w("post_attention_norm.weight", c.Embed)
			postFFN = w("post_ffw_norm.weight", c.Embed)
		}
		var biases [3]*uint32
		if c.Arch == "qwen2" {
			for i, name := range []string{"q", "k", "v"} {
				n := KD
				if i == 0 {
					n = QD
				}
				bv := b.f32Of(w("attn_"+name+".bias", int(n)))
				biases[i] = &bv
			}
		}
		if err := errors.Join(errs...); err != nil {
			return nil, err
		}
		// The caches start as zeros and take no space in the file.
		cacheK := b.zeros(p+"cache_k", b.cache, cacheShape...)
		cacheV := b.zeros(p+"cache_v", b.cache, cacheShape...)

		h := norm(p+"attn_in", x, attnNorm, 1, T, E)
		qf := b.matmulBias(p+"q", h, wq, biases[0], 1, T, QD) // [1, T, H*D]
		kf := b.matmulBias(p+"k", h, wk, biases[1], 1, T, KD)
		vf := b.matmulBias(p+"v", h, wv, biases[2], 1, T, KD)
		heads := func(name string, y uint32, n mim.Dim, scale uint32, normed bool) uint32 {
			y = b.op(mim.OpReshape, []uint32{y}, b.val(p+name+".heads", mim.F32, 1, T, n, D), named("shape", ints(1, sym, int64(n), int64(D))))
			if normed {
				y = norm(p+name+".normed", y, scale, 1, T, n, D)
			}
			return b.op(mim.OpPermute, []uint32{y}, b.val(p+name+".bhtd", mim.F32, 1, n, T, D), named("perm", ints(0, 2, 1, 3)))
		}
		q := heads("q", qf, H, qNorm, headNorms)
		k := heads("k", kf, Hkv, kNorm, headNorms)
		v := heads("v", vf, Hkv, 0, false)
		r := globalRope
		layerMask := mask
		if c.Local(l) {
			r, layerMask = localRope, localMask
		}
		q = b.rope(p+"q.rope", q, positions, r, 1, H, T, D)
		k = b.rope(p+"k.rope", k, positions, r, 1, Hkv, T, D)
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

		attrs := []mim.Attr{{Key: "num_kv_heads", Type: mim.AttrInt, Int: int64(c.KVHeads)}}
		if c.AttnScale > 0 {
			attrs = append(attrs, mim.Attr{Key: "scale", Type: mim.AttrFloat, Float: c.AttnScale})
		}
		if c.AttnSoftcap > 0 {
			attrs = append(attrs, mim.Attr{Key: "softcap", Type: mim.AttrFloat, Float: c.AttnSoftcap})
		}
		o := b.callVersion("attention", attnVersion, []uint32{q, newK, newV, layerMask}, b.val(p+"attn", mim.F32, 1, H, T, D), attrs...)
		o = b.op(mim.OpPermute, []uint32{o}, b.val(p+"attn.bthd", mim.F32, 1, T, H, D), named("perm", ints(0, 2, 1, 3)))
		o = b.op(mim.OpReshape, []uint32{o}, b.val(p+"attn.merged", mim.F32, 1, T, QD), named("shape", ints(1, sym, int64(QD))))
		o = b.matmulOp(p+"attn_out", o, wo, 1, T, E)
		if gemma {
			o = norm(p+"attn_post", o, postAttn, 1, T, E)
		}
		x = b.op(mim.OpAdd, []uint32{x, o}, b.val(p+"resid_attn", mim.F32, 1, T, E))

		h = norm(p+"ffn_in", x, ffnNorm, 1, T, E)
		gate := b.matmulOp(p+"gate", h, wg, 1, T, F)
		up := b.matmulOp(p+"up", h, wu, 1, T, F)
		var act uint32
		if gemma {
			act = b.call("geglu", []uint32{gate, up}, b.val(p+"geglu", mim.F32, 1, T, F),
				mim.Attr{Key: "approximate", Type: mim.AttrString, String: "tanh"})
		} else {
			act = b.call("swiglu", []uint32{gate, up}, b.val(p+"swiglu", mim.F32, 1, T, F))
		}
		down := b.matmulOp(p+"down", act, wd, 1, T, E)
		if gemma {
			down = norm(p+"ffn_post", down, postFFN, 1, T, E)
		}
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
	x = norm("final_norm", x, outNorm, 1, T, E)
	logits := b.matmulOp("logits", x, head, 1, T, V)
	if c.FinalSoftcap > 0 {
		capped := b.op(mim.OpMul, []uint32{logits, b.scalar("logit_softcap.inv", 1/c.FinalSoftcap)}, b.val("logits.scaled", mim.F32, 1, T, V))
		capped = b.call("tanh", []uint32{capped}, b.val("logits.tanh", mim.F32, 1, T, V))
		logits = b.op(mim.OpMul, []uint32{capped, b.scalar("logit_softcap", c.FinalSoftcap)}, b.val("logits.capped", mim.F32, 1, T, V))
	}
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

// scalar adds a constant f32 [1, 1, 1] of value x.
func (b *llamaBuilder) scalar(name string, x float64) uint32 {
	return b.op(mim.OpConst, nil, b.val(name, mim.F32, 1, 1, 1), mim.Attr{Key: "value", Type: mim.AttrFloat, Float: x})
}

// ropeSpec is how one kind of layer rotates its queries and keys: the
// attributes of its rope call, and the frequency factors it reads, if any.
type ropeSpec struct {
	attrs   []mim.Attr
	factors *uint32
	v2      bool
}

// ropes returns the rope of c's global layers and that of its local ones.
// llama.cpp's NORM rope, the llama architecture's, rotates adjacent pairs;
// the others use NEOX, which rotates the first half of each head against
// the second. A global layer is linearly scaled by RopeScale and divides
// its frequencies by rope_freqs.weight (Llama 3.1 and later, and Phi 3's
// factors, see splitPhi3); these call rope.v2, and plain rotations
// rope.v1.
func (b *llamaBuilder) ropes(c *LlamaConfig) (global, local ropeSpec, err error) {
	interleaved := int64(0)
	if c.Arch == "llama" {
		interleaved = 1
	}
	base := func(theta float64) []mim.Attr {
		return []mim.Attr{
			{Key: "theta", Type: mim.AttrFloat, Float: theta},
			{Key: "interleaved", Type: mim.AttrInt, Int: interleaved},
			{Key: "rotary_dims", Type: mim.AttrInt, Int: int64(c.RopeDims)},
		}
	}
	global.attrs = base(c.RopeTheta)
	local.attrs = global.attrs
	if c.LocalTheta > 0 {
		local.attrs = base(c.LocalTheta)
	}
	if c.RopeScale > 0 {
		global.v2 = true
		global.attrs = append(global.attrs, mim.Attr{Key: "freq_scale", Type: mim.AttrFloat, Float: 1 / c.RopeScale})
	}
	if c.RopeAttnFactor > 0 {
		global.v2 = true
		global.attrs = append(global.attrs, mim.Attr{Key: "attn_factor", Type: mim.AttrFloat, Float: c.RopeAttnFactor})
	}
	const name = "rope_freqs.weight"
	if _, ok := b.byName[name]; ok {
		f, err := b.weight(name, c.RopeDims/2)
		if err != nil {
			return global, local, err
		}
		if b.g.Values[f].DType != mim.F32 {
			return global, local, fmt.Errorf("tensor %q is %s, want f32", name, b.g.Values[f].DType)
		}
		global.v2, global.factors = true, &f
	}
	return global, local, nil
}

// rope rotates x, [1, n, T, D], at positions as r says.
func (b *llamaBuilder) rope(name string, x, positions uint32, r ropeSpec, shape ...mim.Dim) uint32 {
	in := []uint32{x, positions}
	if !r.v2 {
		return b.call("rope", in, b.val(name, mim.F32, shape...), r.attrs...)
	}
	if r.factors != nil {
		in = append(in, *r.factors)
	}
	return b.callVersion("rope", 2, in, b.val(name, mim.F32, shape...), r.attrs...)
}

// splitPhi3 puts Phi 3's weights in the layout of the other families, so
// that the graph, runtimes that load the weights by name and LoRA
// adapters see the same projections: each block's fused attn_qkv.weight
// becomes attn_q, attn_k and attn_v, and its ffn_up.weight, which holds
// the gate's rows ahead of the up projection's, becomes ffn_gate and
// ffn_up. A row of a block-quantized tensor is whole blocks, so each part
// is a slice of the fused tensor's bytes. Of the LongRoPE factors, it
// keeps those llama.cpp uses for c's cache length as rope_freqs.weight:
// rope_factors_long.weight past the context the model was first trained
// on, rope_factors_short.weight within it.
func splitPhi3(s *Source, c *LlamaConfig) error {
	qd, kd := c.Heads*c.HeadDim, c.KVHeads*c.HeadDim
	split := func(t Tensor, names []string, rows []int) ([]Tensor, error) {
		total := 0
		for _, r := range rows {
			total += r
		}
		if len(t.Shape) != 2 || t.Shape[0] != uint64(total) || t.Shape[1] != uint64(c.Embed) {
			return nil, fmt.Errorf("tensor %q is %v, want [%d %d]", t.Name, t.Shape, total, c.Embed)
		}
		rowBytes, err := mim.ByteSize(t.DType, []uint64{t.Shape[1]})
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", t.Name, err)
		}
		var out []Tensor
		at := uint64(0)
		for i, r := range rows {
			n := uint64(r) * rowBytes
			out = append(out, Tensor{Name: names[i], DType: t.DType, Shape: []uint64{uint64(r), t.Shape[1]}, Data: t.Data[at : at+n : at+n]})
			at += n
		}
		return out, nil
	}
	orig := c.MaxContext
	if v, ok := metaInt(s.Metadata, c.Arch+".rope.scaling.original_context_length"); ok && v > 0 && v < math.MaxInt32 {
		orig = int(v)
	}
	keep, drop := "rope_factors_short.weight", "rope_factors_long.weight"
	if c.Context > orig {
		keep, drop = drop, keep
	}
	names := map[string]bool{}
	for _, t := range s.Tensors {
		names[t.Name] = true
	}
	if names[drop] && !names[keep] {
		return fmt.Errorf("tensor %q is missing: the cache of %d positions needs it, next to %s", keep, c.Context, drop)
	}
	var out []Tensor
	for _, t := range s.Tensors {
		var p string
		var l int
		if _, err := fmt.Sscanf(t.Name, "blk.%d.", &l); err == nil {
			p = fmt.Sprintf("blk.%d.", l)
		}
		switch {
		case p != "" && t.Name == p+"attn_qkv.weight":
			parts, err := split(t, []string{p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"}, []int{qd, kd, kd})
			if err != nil {
				return err
			}
			out = append(out, parts...)
		case p != "" && t.Name == p+"ffn_up.weight" && !names[p+"ffn_gate.weight"]:
			parts, err := split(t, []string{p + "ffn_gate.weight", p + "ffn_up.weight"}, []int{c.FFN, c.FFN})
			if err != nil {
				return err
			}
			out = append(out, parts...)
		case t.Name == keep && !names["rope_freqs.weight"]:
			t.Name = "rope_freqs.weight"
			out = append(out, t)
		case t.Name == keep || t.Name == drop:
		default:
			out = append(out, t)
		}
	}
	seen := map[string]bool{}
	for _, t := range out {
		if seen[t.Name] {
			return fmt.Errorf("tensor %q is both fused and apart", t.Name)
		}
		seen[t.Name] = true
	}
	s.Tensors = out
	return nil
}
