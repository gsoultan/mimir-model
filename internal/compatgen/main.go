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

// Command compatgen writes the compatibility corpus of one release into a
// new directory under testdata/compat (spec/versioning.md section 4):
//
//	go run ./internal/compatgen -dir testdata/compat/0.1-0.1
//
// It refuses to write into a directory that exists, because a released
// directory never changes. Each file is written by the reference writer
// and described by an expected.json; graph outputs are those of the
// reference executor at the time of the release.
package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/gsoultan/mim"
	"github.com/gsoultan/mim/compat"
	"github.com/gsoultan/mim/importer"
	"github.com/gsoultan/mim/interp"
	"github.com/gsoultan/mim/ref"
)

func main() {
	dir := flag.String("dir", "", "corpus directory to create, for example testdata/compat/0.1-0.1")
	llama := flag.String("llama", "testdata/llama", "directory of the tiny Llama fixture")
	flag.Parse()
	if *dir == "" {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(*dir, *llama); err != nil {
		fmt.Fprintln(os.Stderr, "compatgen:", err)
		os.Exit(1)
	}
}

// sample is one corpus file before it is written.
type sample struct {
	name  string
	w     *mim.Writer
	runs  []map[string]*ref.Tensor
	check func(output string) (string, float64) // "bits" or "close" and its tolerance
	notes string
}

func run(dir, llamaDir string) error {
	if _, err := os.Stat(dir); err == nil {
		return fmt.Errorf("%s exists; a released corpus directory is never rewritten", dir)
	}
	samples := []func() (*sample, error){dtypesSample, graphSample, func() (*sample, error) { return llamaSample(llamaDir) }, trainingSample}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for _, mk := range samples {
		s, err := mk()
		if err != nil {
			return err
		}
		if err := write(dir, s); err != nil {
			return fmt.Errorf("%s: %w", s.name, err)
		}
	}
	return nil
}

func write(dir string, s *sample) error {
	var buf bytes.Buffer
	if _, err := s.w.WriteTo(&buf); err != nil {
		return err
	}
	file := s.name + ".mim"
	if err := os.WriteFile(filepath.Join(dir, file), buf.Bytes(), 0o644); err != nil {
		return err
	}
	f, err := mim.Parse(buf.Bytes())
	if err != nil {
		return err
	}
	e, err := compat.Describe(f)
	if err != nil {
		return err
	}
	e.File, e.Notes = file, s.notes
	if len(s.runs) > 0 {
		p, err := interp.Load(f)
		if err != nil {
			return err
		}
		p.FastPaths = true
		sess := p.NewSession()
		for _, in := range s.runs {
			out, err := sess.Run(in)
			if err != nil {
				return err
			}
			r := compat.Run{Inputs: map[string]compat.TensorData{}, Outputs: map[string]compat.Output{}}
			for k, t := range in {
				r.Inputs[k] = compat.TensorDataOf(t)
			}
			for k, t := range out {
				check, tol := s.check(k)
				r.Outputs[k] = compat.Output{TensorData: compat.TensorDataOf(t), Check: check, Tol: tol}
			}
			e.Runs = append(e.Runs, r)
		}
	}
	js, err := json.MarshalIndent(e, "", " ")
	if err != nil {
		return err
	}
	path := filepath.Join(dir, s.name+".expected.json")
	if err := os.WriteFile(path, append(js, '\n'), 0o644); err != nil {
		return err
	}
	// The new files must pass the checker that the tests run.
	return compat.Check(dir, e, true)
}

// pattern returns n deterministic bytes.
func pattern(n int, seed uint32) []byte {
	b := make([]byte, n)
	x := seed*2654435761 + 1
	for i := range b {
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		b[i] = byte(x >> 24)
	}
	return b
}

func f32bytes(v ...float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

// dtypesSample holds one tensor of every dtype, a tensor large enough for
// the 4 KiB alignment rule, every metadata type, and an optional section of
// an unknown kind that a reader must skip.
func dtypesSample() (*sample, error) {
	w := mim.NewWriter()
	all := []mim.DType{mim.F32, mim.F16, mim.BF16, mim.F64, mim.I8, mim.U8, mim.I16, mim.U16, mim.I32, mim.U32,
		mim.I64, mim.U64, mim.Bool, mim.F8E4M3, mim.F8E5M2, mim.Q8_0, mim.Q4_0, mim.Q4_K, mim.Q6_K, mim.MXFP4, mim.Affine4}
	// Added in reverse name order, so data order and index order differ.
	sort.Slice(all, func(i, j int) bool { return all[i].String() > all[j].String() })
	for i, dt := range all {
		shape := []uint64{2, 3}
		if dt.IsBlock() {
			shape = []uint64{2, dt.BlockElems()}
		}
		n, err := mim.ByteSize(dt, shape)
		if err != nil {
			return nil, err
		}
		data := pattern(int(n), uint32(i+1))
		if dt == mim.Bool {
			for k := range data {
				data[k] &= 1
			}
		}
		if err := w.AddTensor("dtype."+dt.String(), dt, shape, data); err != nil {
			return nil, err
		}
	}
	for _, t := range []struct {
		name  string
		dt    mim.DType
		shape []uint64
	}{
		{"scalar.f32", mim.F32, nil},
		{"empty.f32", mim.F32, []uint64{0, 4}},
		{"rank8.u8", mim.U8, []uint64{1, 2, 1, 2, 1, 2, 1, 2}},
		{"large.u8", mim.U8, []uint64{1 << 20}},
		{"unicode.名前", mim.I32, []uint64{2}},
	} {
		n, _ := mim.ByteSize(t.dt, t.shape)
		if err := w.AddTensor(t.name, t.dt, t.shape, pattern(int(n), 99)); err != nil {
			return nil, err
		}
	}
	md := mim.Metadata{
		"general.name":    "compat dtypes",
		"test.int":        int64(math.MinInt64),
		"test.uint":       uint64(math.MaxUint64),
		"test.float":      math.Copysign(0, -1),
		"test.float.nan":  math.NaN(),
		"test.bool":       true,
		"test.string":     "héllo, 世界",
		"test.bytes":      []byte{0, 1, 0xfe, 0xff},
		"test.ints":       []int64{-1, 0, math.MaxInt64},
		"test.floats":     []float64{math.Inf(1), 1.5, -2.25e-310},
		"test.strings":    []string{"", "a", "ü"},
		"test.empty.ints": []int64{},
	}
	enc, err := md.Encode()
	if err != nil {
		return nil, err
	}
	if err := w.AddSection(mim.SectionMetadata, false, enc); err != nil {
		return nil, err
	}
	if err := w.AddSection(77, false, []byte("an optional section of a kind this version does not define")); err != nil {
		return nil, err
	}
	return &sample{name: "dtypes", w: w,
		notes: "One tensor of every dtype, a 1 MiB tensor (4 KiB alignment), every metadata type, and an optional section of unknown kind 77 that a reader skips."}, nil
}

// graphBuilder builds the small graph of graphSample.
type graphBuilder struct {
	g       *mim.Graph
	tensors []stored
}

type stored struct {
	name  string
	dt    mim.DType
	shape []uint64
	data  []byte
}

func (b *graphBuilder) val(name string, dt mim.DType, kind mim.ValueKind, shape ...mim.Dim) uint32 {
	b.g.Values = append(b.g.Values, mim.Value{Name: name, DType: dt, Shape: shape, Kind: kind})
	return uint32(len(b.g.Values) - 1)
}

func (b *graphBuilder) stored(name string, kind mim.ValueKind, dt mim.DType, shape []uint64, data []byte) uint32 {
	dims := make([]mim.Dim, len(shape))
	for i, s := range shape {
		dims[i] = mim.Dim(s)
	}
	id := b.val(name, dt, kind, dims...)
	b.g.Values[id].Ref = uint32(len(b.tensors))
	b.tensors = append(b.tensors, stored{name, dt, shape, data})
	return id
}

func (b *graphBuilder) op(si int, opc mim.Opcode, in []uint32, out uint32, attrs ...mim.Attr) uint32 {
	b.g.Subgraphs[si].Ops = append(b.g.Subgraphs[si].Ops, mim.Op{Opcode: opc, Inputs: in, Outputs: []uint32{out}, Attrs: attrs})
	return out
}

func (b *graphBuilder) output(si int, id uint32) {
	if si == 0 {
		b.g.Values[id].Kind = mim.KindOutput
	}
	b.g.Subgraphs[si].Outputs = append(b.g.Subgraphs[si].Outputs, id)
}

func (b *graphBuilder) writer() (*mim.Writer, error) {
	order := make([]int, len(b.tensors))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool { return b.tensors[order[i]].name < b.tensors[order[j]].name })
	pos := make([]uint32, len(order))
	for p, i := range order {
		pos[i] = uint32(p)
	}
	for i, v := range b.g.Values {
		if (v.Kind == mim.KindWeight || v.Kind == mim.KindConst || v.Kind == mim.KindVariable) && !v.Zero {
			b.g.Values[i].Ref = pos[v.Ref]
		}
	}
	w := mim.NewWriter()
	for _, t := range b.tensors {
		if err := w.AddTensor(t.name, t.dt, t.shape, t.data); err != nil {
			return nil, err
		}
	}
	enc, err := b.g.Encode()
	if err != nil {
		return nil, err
	}
	return w, w.AddSection(mim.SectionGraph, true, enc)
}

func sAttr(k, v string) mim.Attr        { return mim.Attr{Key: k, Type: mim.AttrString, String: v} }
func iAttr(k string, v int64) mim.Attr  { return mim.Attr{Key: k, Type: mim.AttrInt, Int: v} }
func gAttr(k string, v uint32) mim.Attr { return mim.Attr{Key: k, Type: mim.AttrSubgraph, Subgraph: v} }
func call(name string, version int64, extra ...mim.Attr) []mim.Attr {
	return append([]mim.Attr{sAttr("name", name), iAttr("version", version)}, extra...)
}

func tensor(dt mim.DType, shape []int, bits ...uint64) *ref.Tensor {
	t, _ := ref.New(dt, shape)
	for i, b := range bits {
		t.SetBits(i, b)
	}
	return t
}

func f32t(shape []int, v ...float32) *ref.Tensor {
	bits := make([]uint64, len(v))
	for i, x := range v {
		bits[i] = uint64(math.Float32bits(x))
	}
	return tensor(mim.F32, shape, bits...)
}

func i64s(v ...int64) []uint64 {
	r := make([]uint64, len(v))
	for i, x := range v {
		r[i] = uint64(x)
	}
	return r
}

// graphSample is a small shape-polymorphic graph that touches the parts of
// the graph format: a symbolic dimension with a bound, weights, a const, a
// stored variable and a zero-initialised one with their updates, `while`,
// `if`, a model-local composite, standard-library calls, `dequantize` of a
// block dtype and casts to bf16 and i32. Inputs are small dyadic numbers,
// so every output but the softmax is exact in any summation order.
func graphSample() (*sample, error) {
	b := &graphBuilder{g: &mim.Graph{PrimitiveVersion: mim.CurrentPrimitive, StdlibVersion: mim.CurrentStdlib,
		Subgraphs: make([]mim.Subgraph, 6)}}
	b.g.SymDims = []mim.SymDim{{Name: "batch", Bound: 8}}
	B := mim.Dim(-1)

	x := b.val("x", mim.F32, mim.KindInput, B, 4)
	n := b.val("n", mim.I64, mim.KindInput)
	pos := b.val("pos", mim.I64, mim.KindInput, 1)
	kt := b.val("k_t", mim.F32, mim.KindInput, 1, 1, 2)
	ids := b.val("ids", mim.I32, mim.KindInput, 3)
	b.g.Subgraphs[0].Inputs = []uint32{x, n, pos, kt, ids}

	w := b.stored("w", mim.KindWeight, mim.F32, []uint64{3, 4}, f32bytes(0.5, -1, 2, 1.5, 0.25, -0.75, 1, 0, -2, 0.5, 0.5, 3))
	bias := b.stored("bias", mim.KindWeight, mim.F32, []uint64{3}, f32bytes(0.125, -0.5, 1))
	q := make([]byte, 2*34)
	for blk := 0; blk < 2; blk++ {
		binary.LittleEndian.PutUint16(q[34*blk:], 0x3800+uint16(blk)*0x400) // f16 0.5, then 1.0
		for j := 0; j < 32; j++ {
			q[34*blk+2+j] = byte(int8(j - 16 + 3*blk))
		}
	}
	wq := b.stored("wq", mim.KindWeight, mim.Q8_0, []uint64{2, 32}, q)
	table := make([]byte, 5*4*2)
	for i := 0; i < 20; i++ {
		binary.LittleEndian.PutUint16(table[2*i:], 0x3c00+uint16(i)*0x80) // 1, 1.0625, 1.125, ...
	}
	emb := b.stored("table", mim.KindWeight, mim.F16, []uint64{5, 4}, table)
	scale := b.stored("scale", mim.KindConst, mim.F32, []uint64{1, 1}, f32bytes(0.25))
	acc := b.stored("acc", mim.KindVariable, mim.F32, []uint64{1, 4}, f32bytes(1, 2, 3, 4))
	cache := b.val("cache", mim.F32, mim.KindVariable, 1, 4, 2)
	b.g.Values[cache].Zero = true

	h := b.op(0, mim.OpCall, []uint32{x, w, bias}, b.val("h", mim.F32, 0, B, 3), call("linear", 1)...)
	r := b.op(0, mim.OpCall, []uint32{h}, b.val("relu", mim.F32, 0, B, 3), call("compat/relu", 1)...)
	probs := b.op(0, mim.OpCall, []uint32{r}, b.val("probs", mim.F32, 0, B, 3), call("softmax", 1, iAttr("axis", -1))...)
	scaled := b.op(0, mim.OpMul, []uint32{h, scale}, b.val("scaled", mim.F32, 0, B, 3))
	dq := b.op(0, mim.OpDequantize, []uint32{wq}, b.val("dequant", mim.F32, 0, 2, 32))
	hb := b.op(0, mim.OpCast, []uint32{h}, b.val("h_bf16", mim.BF16, 0, B, 3), mim.Attr{Key: "to", Type: mim.AttrDType, DType: mim.BF16})
	hi := b.op(0, mim.OpCast, []uint32{scaled}, b.val("h_i32", mim.I32, 0, B, 3), mim.Attr{Key: "to", Type: mim.AttrDType, DType: mim.I32})
	e := b.op(0, mim.OpCall, []uint32{emb, ids}, b.val("emb", mim.F16, 0, 3, 4), call("embedding", 1)...)
	colsum := b.op(0, mim.OpReduce, []uint32{x}, b.val("colsum", mim.F32, 0, 1, 4), sAttr("mode", "add"), mim.Attr{Key: "axes", Type: mim.AttrInts, Ints: []int64{0}})
	accNew := b.op(0, mim.OpAdd, []uint32{acc, colsum}, b.val("acc_new", mim.F32, 0, 1, 4))
	cacheNew := b.op(0, mim.OpScatter, []uint32{cache, pos, kt}, b.val("cache_new", mim.F32, 0, 1, 4, 2), iAttr("axis", 1))

	// squares = sum of i*i for i in 0..n-1, in a while loop.
	zero := b.op(0, mim.OpConst, nil, b.val("zero", mim.I64, 0), iAttr("value", 0))
	i1, s1 := b.val("i_end", mim.I64, 0), b.val("squares", mim.I64, 0)
	b.g.Subgraphs[0].Ops = append(b.g.Subgraphs[0].Ops, mim.Op{Opcode: mim.OpWhile, Inputs: []uint32{zero, zero},
		Outputs: []uint32{i1, s1}, Attrs: []mim.Attr{gAttr("cond", 1), gAttr("body", 2)}})
	ci, cs := b.val("ci", mim.I64, 0), b.val("cs", mim.I64, 0)
	b.g.Subgraphs[1].Inputs = []uint32{ci, cs}
	b.output(1, b.op(1, mim.OpCmplt, []uint32{ci, n}, b.val("more", mim.Bool, 0)))
	bi, bs := b.val("bi", mim.I64, 0), b.val("bs", mim.I64, 0)
	b.g.Subgraphs[2].Inputs = []uint32{bi, bs}
	one := b.op(2, mim.OpConst, nil, b.val("one", mim.I64, 0), iAttr("value", 1))
	b.output(2, b.op(2, mim.OpAdd, []uint32{bi, one}, b.val("bi_next", mim.I64, 0)))
	sq := b.op(2, mim.OpMul, []uint32{bi, bi}, b.val("bi_sq", mim.I64, 0))
	b.output(2, b.op(2, mim.OpAdd, []uint32{bs, sq}, b.val("bs_next", mim.I64, 0)))

	// branch = -squares if n < 5 else squares + squares
	five := b.op(0, mim.OpConst, nil, b.val("five", mim.I64, 0), iAttr("value", 5))
	small := b.op(0, mim.OpCmplt, []uint32{n, five}, b.val("small", mim.Bool, 0))
	branch := b.val("branch", mim.I64, 0)
	b.g.Subgraphs[0].Ops = append(b.g.Subgraphs[0].Ops, mim.Op{Opcode: mim.OpIf, Inputs: []uint32{small, s1},
		Outputs: []uint32{branch}, Attrs: []mim.Attr{gAttr("then", 3), gAttr("else", 4)}})
	tx := b.val("tx", mim.I64, 0)
	b.g.Subgraphs[3].Inputs = []uint32{tx}
	b.output(3, b.op(3, mim.OpNeg, []uint32{tx}, b.val("ty", mim.I64, 0)))
	ex := b.val("ex", mim.I64, 0)
	b.g.Subgraphs[4].Inputs = []uint32{ex}
	b.output(4, b.op(4, mim.OpAdd, []uint32{ex, ex}, b.val("ey", mim.I64, 0)))

	// compat/relu.v1: max(x, 0), a model-local composite.
	rx := b.val("rx", mim.F32, 0, B, 3)
	b.g.Subgraphs[5].Inputs = []uint32{rx}
	rz := b.op(5, mim.OpConst, nil, b.val("rz", mim.F32, 0, 1, 1), mim.Attr{Key: "value", Type: mim.AttrFloat, Float: 0})
	b.output(5, b.op(5, mim.OpMax, []uint32{rx, rz}, b.val("ry", mim.F32, 0, B, 3)))
	b.g.Composites = []mim.Composite{{Name: "compat/relu", Version: 1, Subgraph: 5}}

	for _, id := range []uint32{probs, scaled, dq, hb, hi, e, accNew, cacheNew, s1, branch} {
		b.output(0, id)
	}
	b.g.Updates = []mim.Update{{Variable: acc, Value: accNew}, {Variable: cache, Value: cacheNew}}
	wr, err := b.writer()
	if err != nil {
		return nil, err
	}
	wr.DataAlignment = 4096
	wr.Hash = false

	in := func(batch int, nv, p int64, kv []float32, idv []int64) map[string]*ref.Tensor {
		xs := make([]float32, 4*batch)
		for i := range xs {
			xs[i] = float32((i*7)%11-5) * 0.25
		}
		idb := make([]uint64, len(idv))
		for i, v := range idv {
			idb[i] = uint64(uint32(int32(v)))
		}
		return map[string]*ref.Tensor{
			"x":   f32t([]int{batch, 4}, xs...),
			"n":   tensor(mim.I64, nil, i64s(nv)...),
			"pos": tensor(mim.I64, []int{1}, i64s(p)...),
			"k_t": f32t([]int{1, 1, 2}, kv...),
			"ids": tensor(mim.I32, []int{3}, idb...),
		}
	}
	return &sample{name: "graph", w: wr,
		runs: []map[string]*ref.Tensor{
			in(2, 4, 0, []float32{1, -1}, []int64{0, 4, 2}),
			in(1, 0, 3, []float32{2.5, 0.75}, []int64{1, 1, 3}),
			in(3, 7, 0, []float32{-8, 16}, []int64{4, 0, 0}),
		},
		check: func(out string) (string, float64) {
			if out == "probs" {
				return "close", 1e-6
			}
			return "bits", 0
		},
		notes: "Data alignment 4096 and no section hashes. Three runs in one session: the variables acc and cache carry over between runs.",
	}, nil
}

// llamaSample is the tiny Llama of testdata/llama imported with its graph,
// as `mim-import -graph -ctx 64 -a8=false` writes it. The runs are the
// prompt and then one decode step through the key-value cache.
func llamaSample(dir string) (*sample, error) {
	var fx struct {
		Model     string  `json:"model"`
		PromptIDs []int32 `json:"prompt_ids"`
		Greedy    []int32 `json:"greedy"`
	}
	js, err := os.ReadFile(filepath.Join(dir, "expected.json"))
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(js, &fx); err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(filepath.Join(dir, fx.Model))
	if err != nil {
		return nil, err
	}
	src, err := importer.ParseGGUF(raw)
	if err != nil {
		return nil, err
	}
	if _, err := importer.BuildLlama(src, importer.LlamaOptions{Context: 64}); err != nil {
		return nil, err
	}
	w := mim.NewWriter()
	if err := src.AddTo(w); err != nil {
		return nil, err
	}
	step := func(ids []int32, pos int) map[string]*ref.Tensor {
		tok := tensor(mim.I32, []int{1, len(ids)})
		p := tensor(mim.I64, []int{1, len(ids)})
		for i, id := range ids {
			tok.SetBits(i, uint64(uint32(id)))
			p.SetBits(i, uint64(pos+i))
		}
		return map[string]*ref.Tensor{"tokens": tok, "positions": p}
	}
	return &sample{name: "tiny-llama", w: w,
		runs: []map[string]*ref.Tensor{step(fx.PromptIDs, 0), step(fx.Greedy[:1], len(fx.PromptIDs))},
		check: func(string) (string, float64) {
			return "close", 2e-5
		},
		notes: "The Q4_K_M Llama of testdata/llama with its graph and a 64-position key-value cache. The first run is the prompt, the second one decode step at position 7. Logits are those of the Go executor; a runtime matches them within 2e-5 of the largest logit.",
	}, nil
}

// trainingSample is a checkpoint of a small linear layer mid-training:
// its weights, AdamW's moments for each, the random generator's state and
// the training section that ties them together (spec/training.md).
func trainingSample() (*sample, error) {
	w := mim.NewWriter()
	f32 := func(n int, seed uint32) []byte {
		b := make([]byte, 4*n)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(float32(int(seed)*31+i)/64-1))
		}
		return b
	}
	tensors := []struct {
		name  string
		dt    mim.DType
		shape []uint64
		data  []byte
	}{
		{"linear.weight", mim.F32, []uint64{4, 3}, f32(12, 1)},
		{"linear.bias", mim.F32, []uint64{3}, f32(3, 2)},
		{"linear.weight@m", mim.F32, []uint64{4, 3}, f32(12, 3)},
		{"linear.weight@v", mim.F32, []uint64{4, 3}, f32(12, 4)},
		{"linear.bias@m", mim.F32, []uint64{3}, f32(3, 5)},
		{"linear.bias@v", mim.F32, []uint64{3}, f32(3, 6)},
		{"train.rng", mim.U8, []uint64{16}, pattern(16, 7)},
	}
	for _, t := range tensors {
		if err := w.AddTensor(t.name, t.dt, t.shape, t.data); err != nil {
			return nil, err
		}
	}
	tr := &mim.Training{
		Optimizer: "adamw",
		Counters:  []mim.TrainCounter{{Name: "step", Value: 1200}, {Name: "epoch", Value: 3}},
		Hyper: []mim.TrainHyper{
			{Name: "lr", Value: 3e-4}, {Name: "beta1", Value: 0.9}, {Name: "beta2", Value: 0.999},
			{Name: "eps", Value: 1e-8}, {Name: "weight_decay", Value: 0.1},
			{Name: "weight_decay", Group: 1, Value: 0},
		},
		Params: []mim.TrainParam{
			{Tensor: "linear.weight", States: []mim.TrainState{{Slot: "m", Tensor: "linear.weight@m"}, {Slot: "v", Tensor: "linear.weight@v"}}},
			{Tensor: "linear.bias", Group: 1, States: []mim.TrainState{{Slot: "m", Tensor: "linear.bias@m"}, {Slot: "v", Tensor: "linear.bias@v"}}},
		},
		Extras: []mim.TrainExtra{{Name: "rng", Tensor: "train.rng"}},
	}
	b, err := tr.Encode()
	if err != nil {
		return nil, err
	}
	if err := w.AddSection(mim.SectionTraining, false, b); err != nil {
		return nil, err
	}
	meta, err := mim.Metadata{mim.KeyName: "compat training checkpoint"}.Encode()
	if err != nil {
		return nil, err
	}
	if err := w.AddSection(mim.SectionMetadata, false, meta); err != nil {
		return nil, err
	}
	return &sample{name: "training", w: w,
		notes: "A linear layer mid-training with AdamW: two parameter groups, m and v per parameter, and the random generator's state as an extra."}, nil
}
