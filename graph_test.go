// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mlm, created and owned by Gembit Soultan Shirazi.
// It is licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License in the LICENSE file at the root
// of this repository or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

package mlm

import (
	"reflect"
	"testing"
)

// sampleGraph builds: y = call softmax.v1(add(x, w)) with a while loop that
// counts i from 0 to n, and a KV-cache style variable update.
func sampleGraph() *Graph {
	g := &Graph{PrimitiveVersion: CurrentPrimitive, StdlibVersion: CurrentStdlib}
	g.SymDims = []SymDim{{"batch", -1}, {"seq", 8192}}
	B, S := Dim(-1), Dim(-2)
	g.Values = []Value{
		{Name: "x", DType: F32, Shape: []Dim{B, S, 16}, Kind: KindInput},          // 0
		{Name: "w", DType: F32, Shape: []Dim{1, 1, 16}, Kind: KindWeight, Ref: 0}, // 1
		{Name: "t", DType: F32, Shape: []Dim{B, S, 16}},                           // 2
		{Name: "y", DType: F32, Shape: []Dim{B, S, 16}, Kind: KindOutput},         // 3
		{Name: "n", DType: I64, Shape: nil, Kind: KindInput},                      // 4
		{Name: "i0", DType: I64, Shape: nil},                                      // 5
		{Name: "i", DType: I64, Shape: nil, Kind: KindOutput},                     // 6
		{Name: "ci", DType: I64, Shape: nil},                                      // 7 cond input
		{Name: "lt", DType: Bool, Shape: nil},                                     // 8 cond output
		{Name: "bi", DType: I64, Shape: nil},                                      // 9 body input
		{Name: "one", DType: I64, Shape: nil},                                     // 10
		{Name: "bo", DType: I64, Shape: nil},                                      // 11 body output
		{Name: "cache", DType: F16, Shape: []Dim{4, 8192, 16}, Kind: KindVariable, Ref: 1},
		{Name: "pos", DType: I64, Shape: []Dim{1}, Kind: KindInput},       // 13
		{Name: "kt", DType: F16, Shape: []Dim{4, 1, 16}, Kind: KindInput}, // 14
		{Name: "cache2", DType: F16, Shape: []Dim{4, 8192, 16}},           // 15
	}
	main := Subgraph{
		Inputs:  []uint32{0, 4, 13, 14},
		Outputs: []uint32{3, 6},
		Ops: []Op{
			{Opcode: OpAdd, Inputs: []uint32{0, 1}, Outputs: []uint32{2}},
			{Opcode: OpCall, Inputs: []uint32{2}, Outputs: []uint32{3}, Attrs: []Attr{
				{Key: "name", Type: AttrString, String: "softmax"},
				{Key: "version", Type: AttrInt, Int: 1},
				{Key: "axis", Type: AttrInt, Int: 2},
			}},
			{Opcode: OpConst, Outputs: []uint32{5}, Attrs: []Attr{{Key: "value", Type: AttrFloat, Float: 0}}},
			{Opcode: OpWhile, Inputs: []uint32{5}, Outputs: []uint32{6}, Attrs: []Attr{
				{Key: "cond", Type: AttrSubgraph, Subgraph: 1},
				{Key: "body", Type: AttrSubgraph, Subgraph: 2},
			}},
			{Opcode: OpScatter, Inputs: []uint32{12, 13, 14}, Outputs: []uint32{15}, Attrs: []Attr{{Key: "axis", Type: AttrInt, Int: 1}}},
		},
	}
	cond := Subgraph{Inputs: []uint32{7}, Outputs: []uint32{8}, Ops: []Op{
		{Opcode: OpCmplt, Inputs: []uint32{7, 4}, Outputs: []uint32{8}},
	}}
	body := Subgraph{Inputs: []uint32{9}, Outputs: []uint32{11}, Ops: []Op{
		{Opcode: OpConst, Outputs: []uint32{10}, Attrs: []Attr{{Key: "value", Type: AttrInt, Int: 1}}},
		{Opcode: OpAdd, Inputs: []uint32{9, 10}, Outputs: []uint32{11}},
	}}
	g.Subgraphs = []Subgraph{main, cond, body}
	g.Updates = []Update{{Variable: 12, Value: 15}}
	return g
}

func TestGraphRoundTrip(t *testing.T) {
	g := sampleGraph()
	must(t, g.Validate())
	b, err := g.Encode()
	must(t, err)
	got, err := DecodeGraph(b)
	must(t, err)
	// Normalise nil vs empty slices before comparing.
	b2, err := got.Encode()
	must(t, err)
	if !reflect.DeepEqual(b, b2) {
		t.Fatal("re-encode differs")
	}
	if len(got.Main().Ops) != 5 || got.Main().Ops[3].Attrs[1].Subgraph != 2 {
		t.Fatalf("decoded graph: %+v", got.Main())
	}
	if got.Values[0].Shape[1].SymIndex() != 1 || got.SymDims[1].Bound != 8192 {
		t.Fatal("symbolic dims lost")
	}
}

func TestGraphInFile(t *testing.T) {
	w := NewWriter()
	must(t, w.AddTensor("w", F32, []uint64{1, 1, 16}, make([]byte, 64)))
	must(t, w.AddTensor("cache", F16, []uint64{4, 8192, 16}, make([]byte, 4*8192*16*2)))
	g := sampleGraph()
	g.Values[1].Ref = 1 // "w" sorts after "cache"
	g.Values[12].Ref = 0
	gb, err := g.Encode()
	must(t, err)
	must(t, w.AddSection(SectionGraph, true, gb))
	f, err := Parse(encode(t, w))
	must(t, err)
	if _, err := f.Graph(); err != nil {
		t.Fatal(err)
	}
	// A weight whose shape disagrees with the tensor is rejected.
	g.Values[1].Shape[2] = 17
	gb, _ = g.Encode()
	w2 := NewWriter()
	must(t, w2.AddTensor("w", F32, []uint64{1, 1, 16}, make([]byte, 64)))
	must(t, w2.AddTensor("cache", F16, []uint64{4, 8192, 16}, make([]byte, 4*8192*16*2)))
	must(t, w2.AddSection(SectionGraph, true, gb))
	f2, err := Parse(encode(t, w2))
	must(t, err)
	if _, err := f2.Graph(); CodeOf(err) != CodeBadGraph {
		t.Fatalf("got %v", err)
	}
}

func TestGraphValidation(t *testing.T) {
	cases := map[string]func(g *Graph){
		"use-before-def":  func(g *Graph) { g.Main().Ops[0].Inputs[0] = 2 },
		"double-def":      func(g *Graph) { g.Main().Ops[2].Outputs[0] = 2 },
		"output-not-kind": func(g *Graph) { g.Main().Outputs[0] = 2 },
		"subgraph-twice":  func(g *Graph) { g.Main().Ops[3].Attrs[0].Subgraph = 2 },
		"update-not-var":  func(g *Graph) { g.Updates[0].Variable = 0 },
		"update-shape":    func(g *Graph) { g.Values[15].Shape[0] = 5 },
		"weight-produced": func(g *Graph) { g.Main().Ops[0].Outputs[0] = 1 },
		"missing-def":     func(g *Graph) { g.Values = append(g.Values, Value{Name: "z", DType: F32}) },
		"bad-opcode":      func(g *Graph) { g.Main().Ops[0].Opcode = 99 },
		"bad-dtype":       func(g *Graph) { g.Values[0].DType = 1000 },
		"bad-symdim":      func(g *Graph) { g.Values[0].Shape[0] = -9 },
		"zero-weight":     func(g *Graph) { g.Values[1].Zero = true; g.Values[1].Ref = 0 },
		"zero-ref":        func(g *Graph) { g.Values[12].Zero = true; g.Values[12].Ref = 1 },
		"zero-symbolic": func(g *Graph) {
			g.Values[12].Zero, g.Values[12].Ref, g.Values[12].Shape[1], g.Values[15].Shape[1] = true, 0, -2, -2
		},
		"zero-block": func(g *Graph) {
			g.Values[12].Zero, g.Values[12].Ref, g.Values[12].DType, g.Values[15].DType = true, 0, Q8_0, Q8_0
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			g := sampleGraph()
			mutate(g)
			b, err := g.Encode()
			if err == nil {
				_, err = DecodeGraph(b)
			}
			if err == nil {
				t.Fatal("accepted an invalid graph")
			}
			if CodeOf(err) == "" {
				t.Fatalf("error without code: %v", err)
			}
		})
	}
}

// TestGraphZeroVariable checks a variable that starts as zeros: it needs no
// tensor in the file, and its flag survives a round trip.
func TestGraphZeroVariable(t *testing.T) {
	g := sampleGraph()
	g.Values[12].Zero, g.Values[12].Ref = true, 0
	gb, err := g.Encode()
	must(t, err)
	w := NewWriter()
	must(t, w.AddTensor("w", F32, []uint64{1, 1, 16}, make([]byte, 64)))
	must(t, w.AddSection(SectionGraph, true, gb))
	f, err := Parse(encode(t, w))
	must(t, err)
	got, err := f.Graph()
	must(t, err)
	if !got.Values[12].Zero || got.Values[1].Zero {
		t.Fatal("zero flag lost")
	}
	// Flag bits other than bit 0 are reserved.
	values := le.Uint64(gb[16+16*tValues:])
	gb[values+32*12+14] = 3
	if _, err := DecodeGraph(gb); CodeOf(err) != CodeBadGraph {
		t.Fatalf("got %v", err)
	}
}

func FuzzDecodeGraph(f *testing.F) {
	b, _ := sampleGraph().Encode()
	f.Add(b)
	f.Fuzz(func(t *testing.T, b []byte) {
		g, err := DecodeGraph(b)
		if err != nil {
			if CodeOf(err) == "" {
				t.Fatal(err)
			}
			return
		}
		b2, err := g.Encode()
		if err != nil {
			t.Fatalf("re-encode failed: %v", err)
		}
		if _, err := DecodeGraph(b2); err != nil {
			t.Fatalf("re-decode failed: %v", err)
		}
	})
}
