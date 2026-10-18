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
	"strings"
	"unicode/utf8"

	mim "github.com/gsoultan/mimir-model"
)

// ONNX import covers the operators of a convolutional network in float32:
//
//	Conv, BatchNormalization (inference), Relu, Clip, Sigmoid, Tanh,
//	MaxPool, AveragePool, GlobalAveragePool, Add, Sub, Mul, Div, Gemm,
//	MatMul, Flatten, Reshape, Transpose, Concat, Softmax, Identity,
//	Dropout (inference) and Constant
//
// for opsets 7 to 22 of the default domain. Initializers become weights
// stored byte for byte; the operators become primitives and the
// standard-library composites conv2d.v1, linear.v1, matmul.v1, softmax.v1,
// sigmoid.v1 and tanh.v1. Pooling has no composite and is written in
// primitives, as conv2d.v1's body is. A dimension an input declares by name
// becomes a symbolic dimension, so a model with a free batch size keeps it.
// Anything outside this subset is rejected by name.

// ONNX TensorProto data types used here.
const (
	onnxFloat  = 1
	onnxInt64  = 7
	onnxDouble = 11
)

type onnxTensor struct {
	name   string
	dtype  int64
	dims   []int64
	f32    []float32
	f64    []float64
	i64    []int64
	raw    []byte
	hasRaw bool
	extern bool
}

type onnxAttr struct {
	name   string
	f      float32
	i      int64
	s      []byte
	t      *onnxTensor
	floats []float32
	ints   []int64
	has    map[uint64]bool // which fields were present
}

type onnxNode struct {
	inputs, outputs []string
	name, op        string
	domain          string
	attrs           map[string]*onnxAttr
}

type onnxDim struct {
	value int64
	param string
	known bool
}

type onnxValueInfo struct {
	name   string
	elem   int64
	dims   []onnxDim
	hasDim bool
}

type onnxGraph struct {
	name         string
	nodes        []*onnxNode
	initializers []*onnxTensor
	inputs       []onnxValueInfo
	outputs      []onnxValueInfo
}

type onnxModel struct {
	irVersion int64
	producer  string
	opset     int64
	graph     *onnxGraph
}

func pbString(f pbField) (string, error) {
	b, err := f.bytes()
	if err != nil {
		return "", err
	}
	if !utf8.Valid(b) {
		return "", fmt.Errorf("onnx: string field %d is not UTF-8", f.num)
	}
	return string(b), nil
}

func parseONNXModel(b []byte) (*onnxModel, error) {
	fs, err := pbFields(b)
	if err != nil {
		return nil, err
	}
	m := &onnxModel{opset: -1}
	for _, f := range fs {
		switch f.num {
		case 1:
			if m.irVersion, err = f.varint(); err != nil {
				return nil, err
			}
		case 2:
			if m.producer, err = pbString(f); err != nil {
				return nil, err
			}
		case 7:
			gb, err := f.bytes()
			if err != nil {
				return nil, err
			}
			if m.graph, err = parseONNXGraph(gb); err != nil {
				return nil, err
			}
		case 8:
			ob, err := f.bytes()
			if err != nil {
				return nil, err
			}
			ofs, err := pbFields(ob)
			if err != nil {
				return nil, err
			}
			domain, version := "", int64(0)
			for _, of := range ofs {
				switch of.num {
				case 1:
					if domain, err = pbString(of); err != nil {
						return nil, err
					}
				case 2:
					if version, err = of.varint(); err != nil {
						return nil, err
					}
				}
			}
			if domain == "" || domain == "ai.onnx" {
				m.opset = version
			}
		}
	}
	if m.graph == nil {
		return nil, fmt.Errorf("onnx: the model has no graph")
	}
	return m, nil
}

func parseONNXGraph(b []byte) (*onnxGraph, error) {
	fs, err := pbFields(b)
	if err != nil {
		return nil, err
	}
	g := &onnxGraph{}
	for _, f := range fs {
		switch f.num {
		case 1:
			nb, err := f.bytes()
			if err != nil {
				return nil, err
			}
			n, err := parseONNXNode(nb)
			if err != nil {
				return nil, err
			}
			g.nodes = append(g.nodes, n)
		case 2:
			if g.name, err = pbString(f); err != nil {
				return nil, err
			}
		case 5:
			tb, err := f.bytes()
			if err != nil {
				return nil, err
			}
			t, err := parseONNXTensor(tb)
			if err != nil {
				return nil, err
			}
			g.initializers = append(g.initializers, t)
		case 11, 12:
			vb, err := f.bytes()
			if err != nil {
				return nil, err
			}
			v, err := parseONNXValueInfo(vb)
			if err != nil {
				return nil, err
			}
			if f.num == 11 {
				g.inputs = append(g.inputs, v)
			} else {
				g.outputs = append(g.outputs, v)
			}
		}
	}
	return g, nil
}

func parseONNXNode(b []byte) (*onnxNode, error) {
	fs, err := pbFields(b)
	if err != nil {
		return nil, err
	}
	n := &onnxNode{attrs: map[string]*onnxAttr{}}
	for _, f := range fs {
		switch f.num {
		case 1, 2:
			s, err := pbString(f)
			if err != nil {
				return nil, err
			}
			if f.num == 1 {
				n.inputs = append(n.inputs, s)
			} else {
				n.outputs = append(n.outputs, s)
			}
		case 3:
			if n.name, err = pbString(f); err != nil {
				return nil, err
			}
		case 4:
			if n.op, err = pbString(f); err != nil {
				return nil, err
			}
		case 7:
			if n.domain, err = pbString(f); err != nil {
				return nil, err
			}
		case 5:
			ab, err := f.bytes()
			if err != nil {
				return nil, err
			}
			a, err := parseONNXAttr(ab)
			if err != nil {
				return nil, err
			}
			if _, dup := n.attrs[a.name]; dup {
				return nil, fmt.Errorf("onnx: node %q has attribute %q twice", n.name, a.name)
			}
			n.attrs[a.name] = a
		}
	}
	return n, nil
}

func parseONNXAttr(b []byte) (*onnxAttr, error) {
	fs, err := pbFields(b)
	if err != nil {
		return nil, err
	}
	a := &onnxAttr{has: map[uint64]bool{}}
	for _, f := range fs {
		a.has[f.num] = true
		switch f.num {
		case 1:
			if a.name, err = pbString(f); err != nil {
				return nil, err
			}
		case 2:
			v, err := f.float32s(nil)
			if err != nil || len(v) != 1 {
				return nil, fmt.Errorf("onnx: attribute float: %v", err)
			}
			a.f = v[0]
		case 3:
			if a.i, err = f.varint(); err != nil {
				return nil, err
			}
		case 4:
			if a.s, err = f.bytes(); err != nil {
				return nil, err
			}
		case 5:
			tb, err := f.bytes()
			if err != nil {
				return nil, err
			}
			if a.t, err = parseONNXTensor(tb); err != nil {
				return nil, err
			}
		case 6, 10, 11:
			return nil, fmt.Errorf("onnx: graph and sparse tensor attributes are not supported")
		case 7:
			if a.floats, err = f.float32s(a.floats); err != nil {
				return nil, err
			}
		case 8:
			if a.ints, err = f.int64s(a.ints); err != nil {
				return nil, err
			}
		case 21:
			return nil, fmt.Errorf("onnx: attribute references (functions) are not supported")
		}
	}
	return a, nil
}

func parseONNXTensor(b []byte) (*onnxTensor, error) {
	fs, err := pbFields(b)
	if err != nil {
		return nil, err
	}
	t := &onnxTensor{}
	for _, f := range fs {
		switch f.num {
		case 1:
			if t.dims, err = f.int64s(t.dims); err != nil {
				return nil, err
			}
		case 2:
			if t.dtype, err = f.varint(); err != nil {
				return nil, err
			}
		case 4:
			if t.f32, err = f.float32s(t.f32); err != nil {
				return nil, err
			}
		case 7:
			if t.i64, err = f.int64s(t.i64); err != nil {
				return nil, err
			}
		case 8:
			if t.name, err = pbString(f); err != nil {
				return nil, err
			}
		case 9:
			if t.raw, err = f.bytes(); err != nil {
				return nil, err
			}
			t.hasRaw = true
		case 10:
			if t.f64, err = f.float64s(t.f64); err != nil {
				return nil, err
			}
		case 13:
			t.extern = true
		case 14:
			loc, err := f.varint()
			if err != nil {
				return nil, err
			}
			t.extern = t.extern || loc == 1
		}
	}
	return t, nil
}

func parseONNXValueInfo(b []byte) (onnxValueInfo, error) {
	var v onnxValueInfo
	fs, err := pbFields(b)
	if err != nil {
		return v, err
	}
	for _, f := range fs {
		switch f.num {
		case 1:
			if v.name, err = pbString(f); err != nil {
				return v, err
			}
		case 2: // TypeProto
			tb, err := f.bytes()
			if err != nil {
				return v, err
			}
			tfs, err := pbFields(tb)
			if err != nil {
				return v, err
			}
			for _, tf := range tfs {
				if tf.num != 1 { // tensor_type
					continue
				}
				ttb, err := tf.bytes()
				if err != nil {
					return v, err
				}
				ttfs, err := pbFields(ttb)
				if err != nil {
					return v, err
				}
				for _, ttf := range ttfs {
					switch ttf.num {
					case 1:
						if v.elem, err = ttf.varint(); err != nil {
							return v, err
						}
					case 2:
						v.hasDim = true
						sb, err := ttf.bytes()
						if err != nil {
							return v, err
						}
						sfs, err := pbFields(sb)
						if err != nil {
							return v, err
						}
						for _, sf := range sfs {
							if sf.num != 1 {
								continue
							}
							db, err := sf.bytes()
							if err != nil {
								return v, err
							}
							dfs, err := pbFields(db)
							if err != nil {
								return v, err
							}
							var d onnxDim
							for _, df := range dfs {
								switch df.num {
								case 1:
									if d.value, err = df.varint(); err != nil {
										return v, err
									}
									d.known = true
								case 2:
									if d.param, err = pbString(df); err != nil {
										return v, err
									}
								}
							}
							v.dims = append(v.dims, d)
						}
					}
				}
			}
		}
	}
	return v, nil
}

// elems returns the number of elements of t's shape, checking the dims.
func (t *onnxTensor) elems() (int, error) {
	n := 1
	for _, d := range t.dims {
		if d < 0 || d > 1<<40 || (d > 0 && n > (1<<40)/int(d)) {
			return 0, fmt.Errorf("onnx: tensor %q has dims %v", t.name, t.dims)
		}
		n *= int(d)
	}
	return n, nil
}

// floats returns the values of a float or double tensor as float64.
func (t *onnxTensor) floats() ([]float64, error) {
	if t.extern {
		return nil, fmt.Errorf("onnx: tensor %q is stored in an external file, which is not supported", t.name)
	}
	n, err := t.elems()
	if err != nil {
		return nil, err
	}
	out := make([]float64, 0, n)
	switch {
	case t.dtype == onnxFloat && t.hasRaw:
		if len(t.raw) != 4*n {
			return nil, fmt.Errorf("onnx: tensor %q has %d raw bytes for %d floats", t.name, len(t.raw), n)
		}
		for i := 0; i < n; i++ {
			out = append(out, float64(math.Float32frombits(binary.LittleEndian.Uint32(t.raw[4*i:]))))
		}
	case t.dtype == onnxFloat:
		if len(t.f32) != n {
			return nil, fmt.Errorf("onnx: tensor %q has %d floats for %d elements", t.name, len(t.f32), n)
		}
		for _, v := range t.f32 {
			out = append(out, float64(v))
		}
	case t.dtype == onnxDouble && t.hasRaw:
		if len(t.raw) != 8*n {
			return nil, fmt.Errorf("onnx: tensor %q has %d raw bytes for %d doubles", t.name, len(t.raw), n)
		}
		for i := 0; i < n; i++ {
			out = append(out, math.Float64frombits(binary.LittleEndian.Uint64(t.raw[8*i:])))
		}
	case t.dtype == onnxDouble:
		if len(t.f64) != n {
			return nil, fmt.Errorf("onnx: tensor %q has %d doubles for %d elements", t.name, len(t.f64), n)
		}
		out = append(out, t.f64...)
	default:
		return nil, fmt.Errorf("onnx: tensor %q has data type %d, want float", t.name, t.dtype)
	}
	return out, nil
}

// int64s returns the values of an int64 tensor.
func (t *onnxTensor) int64s() ([]int64, error) {
	if t.extern || t.dtype != onnxInt64 {
		return nil, fmt.Errorf("onnx: tensor %q is not an embedded int64 tensor", t.name)
	}
	n, err := t.elems()
	if err != nil {
		return nil, err
	}
	if t.hasRaw {
		if len(t.raw) != 8*n {
			return nil, fmt.Errorf("onnx: tensor %q has %d raw bytes for %d int64s", t.name, len(t.raw), n)
		}
		out := make([]int64, n)
		for i := range out {
			out[i] = int64(binary.LittleEndian.Uint64(t.raw[8*i:]))
		}
		return out, nil
	}
	if len(t.i64) != n {
		return nil, fmt.Errorf("onnx: tensor %q has %d int64s for %d elements", t.name, len(t.i64), n)
	}
	return t.i64, nil
}

// f32Bytes returns the tensor's data as little-endian float32 bytes. Float
// tensors stored raw are copied as they are; doubles are rounded.
func (t *onnxTensor) f32Bytes() ([]byte, error) {
	if t.dtype == onnxFloat && t.hasRaw && !t.extern {
		if n, err := t.elems(); err != nil || len(t.raw) != 4*n {
			return nil, fmt.Errorf("onnx: tensor %q has %d raw bytes for dims %v", t.name, len(t.raw), t.dims)
		}
		return t.raw, nil
	}
	v, err := t.floats()
	if err != nil {
		return nil, err
	}
	return f32Data(v), nil
}

func f32Data(v []float64) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(float32(x)))
	}
	return b
}

// ParseONNX converts an ONNX model of the supported subset into a source
// with a graph.
func ParseONNX(b []byte) (*Source, error) {
	m, err := parseONNXModel(b)
	if err != nil {
		return nil, err
	}
	if m.opset < 7 || m.opset > 22 {
		return nil, fmt.Errorf("onnx: opset %d of the default domain is not supported (7 to 22 are)", m.opset)
	}
	c := &onnxConverter{
		m:      m,
		src:    &Source{Format: "onnx", Metadata: mim.Metadata{}},
		g:      &mim.Graph{PrimitiveVersion: mim.CurrentPrimitive, StdlibVersion: mim.CurrentStdlib, Subgraphs: []mim.Subgraph{{}}},
		vals:   map[string]uint32{},
		consts: map[string]*onnxTensor{},
		syms:   map[string]mim.Dim{},
		names:  map[string]bool{},
	}
	if err := c.convert(); err != nil {
		return nil, err
	}
	c.src.Graph = c.g
	md := c.src.Metadata
	md["onnx.ir_version"] = m.irVersion
	md["onnx.opset"] = m.opset
	if m.producer != "" {
		md["onnx.producer_name"] = m.producer
	}
	if m.graph.name != "" {
		md[mim.KeyName] = m.graph.name
	}
	return c.src, nil
}

type shape = []mim.Dim

type onnxConverter struct {
	m      *onnxModel
	src    *Source
	g      *mim.Graph
	vals   map[string]uint32      // ONNX value name to graph value
	consts map[string]*onnxTensor // initializers and Constant outputs
	syms   map[string]mim.Dim
	names  map[string]bool // tensor names in src
	node   *onnxNode       // the node being converted, for messages
}

func (c *onnxConverter) convert() error {
	g := c.m.graph
	for _, t := range g.initializers {
		if t.name == "" {
			return fmt.Errorf("onnx: an initializer has no name")
		}
		if _, dup := c.consts[t.name]; dup {
			return fmt.Errorf("onnx: initializer %q is defined twice", t.name)
		}
		c.consts[t.name] = t
	}
	main := &c.g.Subgraphs[0]
	for _, in := range g.inputs {
		if _, ok := c.consts[in.name]; ok {
			continue // an initializer listed as an input (IR version 3 and older)
		}
		if in.elem != onnxFloat {
			return fmt.Errorf("onnx: input %q has element type %d; only float inputs are supported", in.name, in.elem)
		}
		if !in.hasDim {
			return fmt.Errorf("onnx: input %q has no shape", in.name)
		}
		s := make(shape, len(in.dims))
		for k, d := range in.dims {
			switch {
			case d.known && d.value >= 0:
				s[k] = mim.Dim(d.value)
			default:
				name := d.param
				if name == "" {
					name = fmt.Sprintf("%s.d%d", in.name, k)
				}
				s[k] = c.sym(name)
			}
		}
		if _, dup := c.vals[in.name]; dup {
			return fmt.Errorf("onnx: input %q is listed twice", in.name)
		}
		id := c.val(in.name, s)
		c.g.Values[id].Kind = mim.KindInput
		c.vals[in.name] = id
		main.Inputs = append(main.Inputs, id)
	}
	for _, n := range g.nodes {
		c.node = n
		if n.domain != "" && n.domain != "ai.onnx" {
			return fmt.Errorf("onnx: node %q: operator domain %q is not supported", n.name, n.domain)
		}
		if err := c.convertNode(n); err != nil {
			return fmt.Errorf("onnx: node %q (%s): %w", n.name, n.op, err)
		}
	}
	for _, out := range g.outputs {
		id, ok := c.vals[out.name]
		if !ok {
			return fmt.Errorf("onnx: output %q is not computed by any supported node", out.name)
		}
		v := &c.g.Values[id]
		if v.Kind != mim.KindIntermediate {
			return fmt.Errorf("onnx: output %q is a graph input or a weight", out.name)
		}
		v.Kind = mim.KindOutput
		main.Outputs = append(main.Outputs, id)
	}
	if len(main.Outputs) == 0 {
		return fmt.Errorf("onnx: the graph has no outputs")
	}
	return c.g.Validate()
}

func (c *onnxConverter) sym(name string) mim.Dim {
	if d, ok := c.syms[name]; ok {
		return d
	}
	c.g.SymDims = append(c.g.SymDims, mim.SymDim{Name: name, Bound: -1})
	d := mim.Dim(-len(c.g.SymDims))
	c.syms[name] = d
	return d
}

func (c *onnxConverter) val(name string, s shape) uint32 {
	c.g.Values = append(c.g.Values, mim.Value{Name: name, DType: mim.F32, Shape: append(shape{}, s...)})
	return uint32(len(c.g.Values) - 1)
}

func (c *onnxConverter) shapeOf(id uint32) shape { return c.g.Values[id].Shape }

func (c *onnxConverter) op(opc mim.Opcode, in []uint32, name string, s shape, attrs ...mim.Attr) uint32 {
	out := c.val(name, s)
	main := &c.g.Subgraphs[0]
	main.Ops = append(main.Ops, mim.Op{Opcode: opc, Inputs: in, Outputs: []uint32{out}, Attrs: attrs})
	return out
}

func (c *onnxConverter) call(comp string, in []uint32, name string, s shape, attrs ...mim.Attr) uint32 {
	a := append([]mim.Attr{
		{Key: "name", Type: mim.AttrString, String: comp},
		{Key: "version", Type: mim.AttrInt, Int: 1},
	}, attrs...)
	return c.op(mim.OpCall, in, name, s, a...)
}

// tensorName returns a tensor name derived from want that no stored tensor
// or ONNX constant uses, for tensors the importer computes.
func (c *onnxConverter) tensorName(want string) string {
	name := want
	for i := 2; c.names[name] || c.consts[name] != nil; i++ {
		name = fmt.Sprintf("%s.%d", want, i)
	}
	return name
}

// stored adds a tensor to the source and a value of the given kind for it.
func (c *onnxConverter) stored(name string, kind mim.ValueKind, dims []int64, data []byte) (uint32, error) {
	if name == "" || len(name) > mim.MaxNameLength || c.names[name] {
		return 0, fmt.Errorf("tensor name %q is empty, too long or used twice", name)
	}
	us := make([]uint64, len(dims))
	s := make(shape, len(dims))
	for i, d := range dims {
		if d < 0 {
			return 0, fmt.Errorf("tensor %q has dims %v", name, dims)
		}
		us[i], s[i] = uint64(d), mim.Dim(d)
	}
	t := Tensor{Name: name, DType: mim.F32, Shape: us, Data: data}
	if err := checkTensor(t); err != nil {
		return 0, err
	}
	c.names[name] = true
	c.src.Tensors = append(c.src.Tensors, t)
	id := c.val(name, s)
	c.g.Values[id].Kind = kind
	c.g.Values[id].Ref = uint32(len(c.src.Tensors) - 1)
	return id, nil
}

// constF32 adds a constant tensor computed by the importer.
func (c *onnxConverter) constF32(name string, dims []int64, v []float64) (uint32, error) {
	return c.stored(c.tensorName(name), mim.KindConst, dims, f32Data(v))
}

// input returns the graph value for input i of the node, making an
// initializer or Constant output a stored tensor on first use.
func (c *onnxConverter) input(n *onnxNode, i int) (uint32, error) {
	if i >= len(n.inputs) || n.inputs[i] == "" {
		return 0, fmt.Errorf("input %d is missing", i)
	}
	name := n.inputs[i]
	if id, ok := c.vals[name]; ok {
		return id, nil
	}
	t, ok := c.consts[name]
	if !ok {
		return 0, fmt.Errorf("input %q is not defined", name)
	}
	if t.dtype != onnxFloat && t.dtype != onnxDouble {
		return 0, fmt.Errorf("constant %q of data type %d is used as a tensor; only float tensors are", name, t.dtype)
	}
	data, err := t.f32Bytes()
	if err != nil {
		return 0, err
	}
	kind := mim.KindWeight
	if !c.isInitializer(name) {
		kind = mim.KindConst
	}
	id, err := c.stored(name, kind, t.dims, data)
	if err != nil {
		return 0, err
	}
	c.vals[name] = id
	return id, nil
}

func (c *onnxConverter) isInitializer(name string) bool {
	for _, t := range c.m.graph.initializers {
		if t.name == name {
			return true
		}
	}
	return false
}

// constInput returns the constant tensor behind input i, for inputs that
// are read at import time (shapes, normalisation statistics, clip bounds).
func (c *onnxConverter) constInput(n *onnxNode, i int) (*onnxTensor, error) {
	if i >= len(n.inputs) || n.inputs[i] == "" {
		return nil, nil
	}
	t, ok := c.consts[n.inputs[i]]
	if !ok {
		return nil, fmt.Errorf("input %q must be an initializer or a Constant", n.inputs[i])
	}
	return t, nil
}

func (c *onnxConverter) output(n *onnxNode, id uint32) error {
	if len(n.outputs) == 0 || n.outputs[0] == "" {
		return fmt.Errorf("the node has no output")
	}
	for _, o := range n.outputs[1:] {
		if o != "" {
			return fmt.Errorf("optional output %q is not supported", o)
		}
	}
	if _, dup := c.vals[n.outputs[0]]; dup {
		return fmt.Errorf("value %q is defined twice", n.outputs[0])
	}
	c.vals[n.outputs[0]] = id
	return nil
}

func (a *onnxAttr) str() string { return string(a.s) }

func (c *onnxConverter) attrInt(n *onnxNode, key string, def int64) int64 {
	if a, ok := n.attrs[key]; ok {
		return a.i
	}
	return def
}

func (c *onnxConverter) attrFloat(n *onnxNode, key string, def float64) float64 {
	if a, ok := n.attrs[key]; ok {
		return float64(a.f)
	}
	return def
}

func (c *onnxConverter) attrInts(n *onnxNode, key string, def []int64) []int64 {
	if a, ok := n.attrs[key]; ok {
		return a.ints
	}
	return def
}

func (c *onnxConverter) attrString(n *onnxNode, key, def string) string {
	if a, ok := n.attrs[key]; ok {
		return a.str()
	}
	return def
}

// only rejects attributes outside known.
func only(n *onnxNode, known ...string) error {
	for k := range n.attrs {
		ok := false
		for _, w := range known {
			ok = ok || k == w
		}
		if !ok {
			return fmt.Errorf("attribute %q is not supported", k)
		}
	}
	return nil
}

func lit(s shape, what string) ([]int64, error) {
	out := make([]int64, len(s))
	for i, d := range s {
		if d.Symbolic() {
			return nil, fmt.Errorf("%s needs a known size on axis %d", what, i)
		}
		out[i] = int64(d)
	}
	return out, nil
}

func ones(r int) shape {
	s := make(shape, r)
	for i := range s {
		s[i] = 1
	}
	return s
}

// k emits the body macro k(v): a scalar constant of rank r.
func (c *onnxConverter) k(v float64, r int) uint32 {
	return c.op(mim.OpConst, nil, "", ones(r), mim.Attr{Key: "value", Type: mim.AttrFloat, Float: v})
}

func (c *onnxConverter) reshape(x uint32, name string, s shape) uint32 {
	spec := make([]int64, len(s))
	for i, d := range s {
		spec[i] = int64(d)
		if d.Symbolic() {
			spec[i] = -1
		}
	}
	return c.op(mim.OpReshape, []uint32{x}, name, s, named("shape", ints(spec...)))
}

func (c *onnxConverter) shrink(x uint32, ax int, b, e int64) uint32 {
	s := append(shape{}, c.shapeOf(x)...)
	begin := make([]int64, len(s))
	end := make([]int64, len(s))
	for i := range s {
		end[i] = -1
	}
	begin[ax], end[ax] = b, e
	s[ax] = mim.Dim(e - b)
	return c.op(mim.OpShrink, []uint32{x}, "", s, named("begin", ints(begin...)), named("end", ints(end...)))
}

// takeStrided emits the body macro of spec/stdlib.md section 2.2: elements
// start, start+step, ..., start+(n-1)*step of axis ax, which must have at
// least start + n*step elements.
func (c *onnxConverter) takeStrided(x uint32, ax int, start, n, step int64) uint32 {
	y := c.shrink(x, ax, start, start+n*step)
	if step == 1 {
		return y
	}
	s := c.shapeOf(y)
	split := append(append(append(shape{}, s[:ax]...), mim.Dim(n), mim.Dim(step)), s[ax+1:]...)
	y = c.reshape(y, "", split)
	y = c.shrink(y, ax+1, 0, 1)
	out := append(append(shape{}, s[:ax]...), s[ax+1:]...)
	out = append(out[:ax], append(shape{mim.Dim(n)}, out[ax:]...)...)
	return c.reshape(y, "", out)
}

func (c *onnxConverter) bin(opc mim.Opcode, a, b uint32, name string) (uint32, error) {
	sa, sb := c.shapeOf(a), c.shapeOf(b)
	// ONNX broadcasting aligns trailing axes; mim needs equal ranks.
	for len(sa) < len(sb) {
		sa = append(shape{1}, sa...)
	}
	for len(sb) < len(sa) {
		sb = append(shape{1}, sb...)
	}
	if len(sa) != len(c.shapeOf(a)) {
		a = c.reshape(a, "", sa)
	}
	if len(sb) != len(c.shapeOf(b)) {
		b = c.reshape(b, "", sb)
	}
	out := make(shape, len(sa))
	for i := range sa {
		switch {
		case sa[i] == sb[i]:
			out[i] = sa[i]
		case sa[i] == 1:
			out[i] = sb[i]
		case sb[i] == 1:
			out[i] = sa[i]
		case sa[i] == 0 || sb[i] == 0:
			out[i] = 0
		default:
			return 0, fmt.Errorf("shapes %v and %v do not broadcast", c.shapeOf(a), c.shapeOf(b))
		}
	}
	return c.op(opc, []uint32{a, b}, name, out), nil
}

func (c *onnxConverter) neg(x uint32) uint32 {
	return c.op(mim.OpNeg, []uint32{x}, "", c.shapeOf(x))
}

// minimum emits min(a, b) = neg(max(neg(a), neg(b))).
func (c *onnxConverter) minimum(a, b uint32, name string) (uint32, error) {
	m, err := c.bin(mim.OpMax, c.neg(a), c.neg(b), "")
	if err != nil {
		return 0, err
	}
	return c.op(mim.OpNeg, []uint32{m}, name, c.shapeOf(m)), nil
}

func normAxis(ax int64, r int) (int, error) {
	if ax < 0 {
		ax += int64(r)
	}
	if ax < 0 || ax >= int64(r) {
		return 0, fmt.Errorf("axis %d is out of range for rank %d", ax, r)
	}
	return int(ax), nil
}

func (c *onnxConverter) convertNode(n *onnxNode) error {
	out := ""
	if len(n.outputs) > 0 {
		out = n.outputs[0]
	}
	switch n.op {
	case "Constant":
		if err := only(n, "value"); err != nil {
			return err
		}
		a := n.attrs["value"]
		if a == nil || a.t == nil {
			return fmt.Errorf("only the tensor form of Constant is supported")
		}
		if len(n.outputs) != 1 {
			return fmt.Errorf("a Constant node has one output")
		}
		t := *a.t
		t.name = out
		if _, dup := c.consts[out]; dup {
			return fmt.Errorf("value %q is defined twice", out)
		}
		c.consts[out] = &t
		return nil
	case "Identity", "Dropout":
		if err := only(n, "ratio", "seed"); err != nil {
			return err
		}
		if n.op == "Dropout" && len(n.inputs) > 2 && n.inputs[2] != "" {
			t, err := c.constInput(n, 2)
			if err != nil {
				return err
			}
			if v, err := boolScalar(t); err != nil || v {
				return fmt.Errorf("training-mode Dropout is not supported")
			}
		}
		x, err := c.input(n, 0)
		if err != nil {
			return err
		}
		return c.output(n, c.reshape(x, out, c.shapeOf(x)))
	case "Relu":
		if err := only(n); err != nil {
			return err
		}
		x, err := c.input(n, 0)
		if err != nil {
			return err
		}
		y, err := c.bin(mim.OpMax, x, c.k(0, len(c.shapeOf(x))), out)
		if err != nil {
			return err
		}
		return c.output(n, y)
	case "Clip":
		return c.clip(n, out)
	case "Sigmoid", "Tanh":
		if err := only(n); err != nil {
			return err
		}
		x, err := c.input(n, 0)
		if err != nil {
			return err
		}
		return c.output(n, c.call(strings.ToLower(n.op), []uint32{x}, out, c.shapeOf(x)))
	case "Add", "Sub", "Mul", "Div":
		if err := only(n); err != nil {
			return err
		}
		a, err := c.input(n, 0)
		if err != nil {
			return err
		}
		b, err := c.input(n, 1)
		if err != nil {
			return err
		}
		opc := mim.OpAdd
		switch n.op {
		case "Sub":
			b = c.neg(b)
		case "Mul":
			opc = mim.OpMul
		case "Div":
			opc = mim.OpMul
			b = c.op(mim.OpRecip, []uint32{b}, "", c.shapeOf(b))
		}
		y, err := c.bin(opc, a, b, out)
		if err != nil {
			return err
		}
		return c.output(n, y)
	case "Conv":
		return c.conv(n, out)
	case "BatchNormalization":
		return c.batchNorm(n, out)
	case "MaxPool", "AveragePool":
		return c.pool(n, out)
	case "GlobalAveragePool", "GlobalMaxPool":
		if err := only(n); err != nil {
			return err
		}
		x, err := c.input(n, 0)
		if err != nil {
			return err
		}
		s := c.shapeOf(x)
		if len(s) < 3 {
			return fmt.Errorf("input rank %d, want at least 3", len(s))
		}
		axes := make([]int64, 0, len(s)-2)
		count := 1.0
		rs := append(shape{}, s...)
		for i := 2; i < len(s); i++ {
			if s[i].Symbolic() {
				return fmt.Errorf("pooling needs known spatial sizes, got %v", s)
			}
			axes = append(axes, int64(i))
			count *= float64(s[i])
			rs[i] = 1
		}
		mode := "add"
		if n.op == "GlobalMaxPool" {
			mode = "max"
		}
		name := out
		if mode == "add" {
			name = out + ".sum"
		}
		y := c.op(mim.OpReduce, []uint32{x}, name, rs, named("mode", mim.Attr{Type: mim.AttrString, String: mode}), named("axes", ints(axes...)))
		if mode == "add" {
			y = c.op(mim.OpMul, []uint32{y, c.k(1/count, len(rs))}, out, rs)
		}
		return c.output(n, y)
	case "Gemm":
		return c.gemm(n, out)
	case "MatMul":
		return c.matmul(n, out)
	case "Flatten":
		return c.flatten(n, out)
	case "Reshape":
		return c.reshapeNode(n, out)
	case "Transpose":
		if err := only(n, "perm"); err != nil {
			return err
		}
		x, err := c.input(n, 0)
		if err != nil {
			return err
		}
		s := c.shapeOf(x)
		perm := c.attrInts(n, "perm", nil)
		if perm == nil {
			for i := len(s) - 1; i >= 0; i-- {
				perm = append(perm, int64(i))
			}
		}
		if len(perm) != len(s) {
			return fmt.Errorf("perm %v for rank %d", perm, len(s))
		}
		ys := make(shape, len(s))
		seen := make([]bool, len(s))
		for i, p := range perm {
			if p < 0 || p >= int64(len(s)) || seen[p] {
				return fmt.Errorf("perm %v is not a permutation", perm)
			}
			seen[p] = true
			ys[i] = s[p]
		}
		return c.output(n, c.op(mim.OpPermute, []uint32{x}, out, ys, named("perm", ints(perm...))))
	case "Concat":
		return c.concat(n, out)
	case "Softmax":
		if err := only(n, "axis"); err != nil {
			return err
		}
		x, err := c.input(n, 0)
		if err != nil {
			return err
		}
		r := len(c.shapeOf(x))
		def := int64(-1)
		if c.m.opset < 13 {
			def = 1
		}
		ax, err := normAxis(c.attrInt(n, "axis", def), r)
		if err != nil {
			return err
		}
		if c.m.opset < 13 && ax != r-1 {
			// Before opset 13 Softmax flattens from axis on; that equals a
			// softmax over the last axis only when axis is the last one.
			return fmt.Errorf("before opset 13, Softmax is supported only over the last axis")
		}
		return c.output(n, c.call("softmax", []uint32{x}, out, c.shapeOf(x), named("axis", mim.Attr{Type: mim.AttrInt, Int: int64(ax)})))
	}
	return fmt.Errorf("operator %s is not supported", n.op)
}

func boolScalar(t *onnxTensor) (bool, error) {
	if t == nil {
		return false, nil
	}
	if t.dtype != 9 || t.extern {
		return false, fmt.Errorf("not a bool scalar")
	}
	if t.hasRaw {
		return len(t.raw) > 0 && t.raw[0] != 0, nil
	}
	// bool values are stored in int32_data, which this parser does not
	// keep; a missing raw payload means false only if there is no data.
	return false, fmt.Errorf("bool scalar without raw data")
}

func (c *onnxConverter) clip(n *onnxNode, out string) error {
	if err := only(n, "min", "max"); err != nil {
		return err
	}
	x, err := c.input(n, 0)
	if err != nil {
		return err
	}
	lo, hi := math.Inf(-1), math.Inf(1)
	if c.m.opset < 11 {
		lo = c.attrFloat(n, "min", lo)
		hi = c.attrFloat(n, "max", hi)
	} else {
		for i, dst := range []*float64{&lo, &hi} {
			t, err := c.constInput(n, i+1)
			if err != nil {
				return err
			}
			if t == nil {
				continue
			}
			v, err := t.floats()
			if err != nil || len(v) != 1 {
				return fmt.Errorf("clip bound %q must be a float scalar", n.inputs[i+1])
			}
			*dst = v[0]
		}
	}
	r := len(c.shapeOf(x))
	y := x
	if !math.IsInf(lo, -1) {
		name := out
		if !math.IsInf(hi, 1) {
			name = out + ".lo"
		}
		if y, err = c.bin(mim.OpMax, y, c.k(lo, r), name); err != nil {
			return err
		}
	}
	if !math.IsInf(hi, 1) {
		if y, err = c.minimum(y, c.k(hi, r), out); err != nil {
			return err
		}
	}
	if y == x {
		y = c.reshape(x, out, c.shapeOf(x))
	}
	return c.output(n, y)
}

// spatial reads the kernel, stride, dilation and padding attributes of a
// 2-D convolution or pooling over an input of spatial size h x w.
type spatial struct {
	kh, kw, sh, sw, dh, dw   int64
	top, bottom, left, right int64
	ho, wo                   int64
	padsGiven                bool
}

func (c *onnxConverter) spatial(n *onnxNode, h, w, kh, kw int64) (*spatial, error) {
	p := &spatial{kh: kh, kw: kw, sh: 1, sw: 1, dh: 1, dw: 1}
	if ks := c.attrInts(n, "kernel_shape", nil); ks != nil {
		if len(ks) != 2 || (kh > 0 && (ks[0] != kh || ks[1] != kw)) {
			return nil, fmt.Errorf("kernel_shape %v", ks)
		}
		p.kh, p.kw = ks[0], ks[1]
	}
	if s := c.attrInts(n, "strides", nil); s != nil {
		if len(s) != 2 {
			return nil, fmt.Errorf("strides %v", s)
		}
		p.sh, p.sw = s[0], s[1]
	}
	if d := c.attrInts(n, "dilations", nil); d != nil {
		if len(d) != 2 {
			return nil, fmt.Errorf("dilations %v", d)
		}
		p.dh, p.dw = d[0], d[1]
	}
	if min(p.kh, p.kw, p.sh, p.sw, p.dh, p.dw) <= 0 || max(p.kh, p.kw, p.sh, p.sw, p.dh, p.dw) > 1<<20 {
		return nil, fmt.Errorf("kernel %dx%d, strides %d,%d, dilations %d,%d", p.kh, p.kw, p.sh, p.sw, p.dh, p.dw)
	}
	if pads := c.attrInts(n, "pads", nil); pads != nil {
		if len(pads) != 4 {
			return nil, fmt.Errorf("pads %v", pads)
		}
		// ONNX orders pads as [h_begin, w_begin, h_end, w_end].
		p.top, p.left, p.bottom, p.right = pads[0], pads[1], pads[2], pads[3]
		p.padsGiven = true
	}
	if min(p.top, p.left, p.bottom, p.right) < 0 || max(p.top, p.left, p.bottom, p.right) > 1<<20 {
		return nil, fmt.Errorf("pads %d %d %d %d", p.top, p.left, p.bottom, p.right)
	}
	extH, extW := p.dh*(p.kh-1)+1, p.dw*(p.kw-1)+1
	switch ap := c.attrString(n, "auto_pad", "NOTSET"); ap {
	case "NOTSET":
	case "VALID":
		p.top, p.bottom, p.left, p.right = 0, 0, 0, 0
	case "SAME_UPPER", "SAME_LOWER":
		th := max((ceilDiv(h, p.sh)-1)*p.sh+extH-h, 0)
		tw := max((ceilDiv(w, p.sw)-1)*p.sw+extW-w, 0)
		p.top, p.bottom, p.left, p.right = th/2, th-th/2, tw/2, tw-tw/2
		if ap == "SAME_LOWER" {
			p.top, p.bottom, p.left, p.right = th-th/2, th/2, tw-tw/2, tw/2
		}
	default:
		return nil, fmt.Errorf("auto_pad %q", ap)
	}
	p.ho = (h+p.top+p.bottom-extH)/p.sh + 1
	p.wo = (w+p.left+p.right-extW)/p.sw + 1
	if h+p.top+p.bottom < extH || w+p.left+p.right < extW {
		return nil, fmt.Errorf("the kernel is larger than the padded input")
	}
	return p, nil
}

func ceilDiv(a, b int64) int64 { return (a + b - 1) / b }

func (c *onnxConverter) conv(n *onnxNode, out string) error {
	if err := only(n, "auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"); err != nil {
		return err
	}
	x, err := c.input(n, 0)
	if err != nil {
		return err
	}
	w, err := c.input(n, 1)
	if err != nil {
		return err
	}
	xs, ws := c.shapeOf(x), c.shapeOf(w)
	if len(xs) != 4 || len(ws) != 4 {
		return fmt.Errorf("only 2-D convolutions are supported, input %v, weight %v", xs, ws)
	}
	wl, err := lit(ws, "the weight")
	if err != nil {
		return err
	}
	sp, err := lit(xs[1:], "Conv")
	if err != nil {
		return err
	}
	g := c.attrInt(n, "group", 1)
	if g <= 0 || sp[0]%g != 0 || wl[0]%g != 0 || wl[1] != sp[0]/g {
		return fmt.Errorf("group %d for input channels %d and weight %v", g, sp[0], wl)
	}
	p, err := c.spatial(n, sp[1], sp[2], wl[2], wl[3])
	if err != nil {
		return err
	}
	in := []uint32{x, w}
	if len(n.inputs) > 2 && n.inputs[2] != "" {
		b, err := c.input(n, 2)
		if err != nil {
			return err
		}
		if bs := c.shapeOf(b); len(bs) != 1 || bs[0] != mim.Dim(wl[0]) {
			return fmt.Errorf("bias %v for %d output channels", bs, wl[0])
		}
		in = append(in, b)
	}
	ys := shape{xs[0], mim.Dim(wl[0]), mim.Dim(p.ho), mim.Dim(p.wo)}
	return c.output(n, c.call("conv2d", in, out, ys,
		named("strides", ints(p.sh, p.sw)), named("dilations", ints(p.dh, p.dw)),
		named("pads", ints(p.top, p.bottom, p.left, p.right)), named("groups", mim.Attr{Type: mim.AttrInt, Int: g})))
}

func (c *onnxConverter) batchNorm(n *onnxNode, out string) error {
	if err := only(n, "epsilon", "momentum", "spatial", "training_mode"); err != nil {
		return err
	}
	if c.attrInt(n, "training_mode", 0) != 0 || c.attrInt(n, "spatial", 1) != 1 {
		return fmt.Errorf("only inference-mode, spatial BatchNormalization is supported")
	}
	x, err := c.input(n, 0)
	if err != nil {
		return err
	}
	s := c.shapeOf(x)
	if len(s) < 2 || s[1].Symbolic() {
		return fmt.Errorf("input %v needs a known channel axis", s)
	}
	ch := int(s[1])
	var p [4][]float64 // scale, bias, mean, var
	for i := range p {
		t, err := c.constInput(n, i+1)
		if err != nil {
			return err
		}
		if t == nil {
			return fmt.Errorf("input %d is missing", i+1)
		}
		if p[i], err = t.floats(); err != nil {
			return err
		}
		if len(p[i]) != ch {
			return fmt.Errorf("input %q has %d values for %d channels", n.inputs[i+1], len(p[i]), ch)
		}
	}
	eps := c.attrFloat(n, "epsilon", 1e-5)
	// y = x * a + b with a = scale / sqrt(var + eps), b = bias - mean * a,
	// computed in float64 and rounded once.
	a, b := make([]float64, ch), make([]float64, ch)
	for i := range a {
		a[i] = p[0][i] / math.Sqrt(p[3][i]+eps)
		b[i] = p[1][i] - p[2][i]*a[i]
	}
	dims := make([]int64, len(s))
	for i := range dims {
		dims[i] = 1
	}
	dims[1] = int64(ch)
	base := n.name
	if base == "" {
		base = out
	}
	av, err := c.constF32(base+".bn_scale", dims, a)
	if err != nil {
		return err
	}
	bv, err := c.constF32(base+".bn_shift", dims, b)
	if err != nil {
		return err
	}
	y, err := c.bin(mim.OpMul, x, av, out+".scaled")
	if err != nil {
		return err
	}
	if y, err = c.bin(mim.OpAdd, y, bv, out); err != nil {
		return err
	}
	return c.output(n, y)
}

func (c *onnxConverter) pool(n *onnxNode, out string) error {
	known := []string{"auto_pad", "ceil_mode", "kernel_shape", "pads", "strides"}
	if n.op == "MaxPool" {
		known = append(known, "dilations", "storage_order")
	} else {
		known = append(known, "count_include_pad")
	}
	if err := only(n, known...); err != nil {
		return err
	}
	if c.attrInt(n, "storage_order", 0) != 0 {
		return fmt.Errorf("storage_order 1 is not supported")
	}
	if len(n.outputs) > 1 && n.outputs[1] != "" {
		return fmt.Errorf("the indices output of MaxPool is not supported")
	}
	if n.op == "AveragePool" && c.attrInts(n, "dilations", nil) != nil {
		return fmt.Errorf("dilated AveragePool is not supported")
	}
	if _, ok := n.attrs["kernel_shape"]; !ok {
		return fmt.Errorf("kernel_shape is required")
	}
	x, err := c.input(n, 0)
	if err != nil {
		return err
	}
	xs := c.shapeOf(x)
	if len(xs) != 4 {
		return fmt.Errorf("only 2-D pooling is supported, input %v", xs)
	}
	sp, err := lit(xs[2:], n.op)
	if err != nil {
		return err
	}
	p, err := c.spatial(n, sp[0], sp[1], 0, 0)
	if err != nil {
		return err
	}
	if c.attrInt(n, "ceil_mode", 0) != 0 {
		extH, extW := p.dh*(p.kh-1)+1, p.dw*(p.kw-1)+1
		if (sp[0]+p.top+p.bottom-extH)%p.sh != 0 || (sp[1]+p.left+p.right-extW)%p.sw != 0 {
			return fmt.Errorf("ceil_mode 1 with a partial last window is not supported")
		}
	}
	isMax := n.op == "MaxPool"
	padValue := 0.0
	if isMax {
		padValue = math.Inf(-1)
	}
	// As in conv2d.v1's body: pad, with s-1 extra elements at the end of
	// each spatial axis so every strided take stays in bounds, then combine
	// one strided view per kernel position.
	padded := x
	hp, wp := sp[0]+p.top+p.bottom+p.sh-1, sp[1]+p.left+p.right+p.sw-1
	if hp != sp[0] || wp != sp[1] {
		padded = c.op(mim.OpPad, []uint32{x}, out+".padded", shape{xs[0], xs[1], mim.Dim(hp), mim.Dim(wp)},
			named("pads", ints(0, 0, 0, 0, p.top, p.bottom+p.sh-1, p.left, p.right+p.sw-1)),
			mim.Attr{Key: "value", Type: mim.AttrFloat, Float: padValue})
	}
	ys := shape{xs[0], xs[1], mim.Dim(p.ho), mim.Dim(p.wo)}
	var acc uint32
	first := true
	for i := int64(0); i < p.kh; i++ {
		for j := int64(0); j < p.kw; j++ {
			v := c.takeStrided(padded, 2, i*p.dh, p.ho, p.sh)
			v = c.takeStrided(v, 3, j*p.dw, p.wo, p.sw)
			if first {
				acc, first = v, false
				continue
			}
			opc := mim.OpAdd
			if isMax {
				opc = mim.OpMax
			}
			acc = c.op(opc, []uint32{acc, v}, "", ys)
		}
	}
	if isMax {
		return c.output(n, c.reshape(acc, out, ys))
	}
	// The average divides by the window size, or with count_include_pad = 0
	// by the number of window elements inside the input.
	area := float64(p.kh * p.kw)
	padded0 := p.top+p.bottom+p.left+p.right > 0
	if c.attrInt(n, "count_include_pad", 0) != 0 || !padded0 {
		return c.output(n, c.op(mim.OpMul, []uint32{acc, c.k(1/area, 4)}, out, ys))
	}
	inv := make([]float64, p.ho*p.wo)
	for oy := int64(0); oy < p.ho; oy++ {
		for ox := int64(0); ox < p.wo; ox++ {
			cnt := 0
			for i := int64(0); i < p.kh; i++ {
				for j := int64(0); j < p.kw; j++ {
					yy, xx := oy*p.sh+i-p.top, ox*p.sw+j-p.left
					if yy >= 0 && yy < sp[0] && xx >= 0 && xx < sp[1] {
						cnt++
					}
				}
			}
			if cnt == 0 {
				return fmt.Errorf("a pooling window lies entirely in the padding")
			}
			inv[oy*p.wo+ox] = 1 / float64(cnt)
		}
	}
	base := n.name
	if base == "" {
		base = out
	}
	iv, err := c.constF32(base+".inv_count", []int64{1, 1, p.ho, p.wo}, inv)
	if err != nil {
		return err
	}
	return c.output(n, c.op(mim.OpMul, []uint32{acc, iv}, out, ys))
}

func (c *onnxConverter) gemm(n *onnxNode, out string) error {
	if err := only(n, "alpha", "beta", "transA", "transB"); err != nil {
		return err
	}
	a, err := c.input(n, 0)
	if err != nil {
		return err
	}
	as := c.shapeOf(a)
	if len(as) != 2 {
		return fmt.Errorf("input A must be rank 2, got %v", as)
	}
	if c.attrInt(n, "transA", 0) != 0 {
		as = shape{as[1], as[0]}
		a = c.op(mim.OpPermute, []uint32{a}, "", as, named("perm", ints(1, 0)))
	}
	// linear.v1 takes the weight as [N, K], which is B with transB = 1.
	var w uint32
	if c.attrInt(n, "transB", 0) != 0 {
		if w, err = c.input(n, 1); err != nil {
			return err
		}
	} else {
		if len(n.inputs) < 2 {
			return fmt.Errorf("input 1 is missing")
		}
		t, ok := c.consts[n.inputs[1]]
		if ok && t.dtype == onnxFloat && len(t.dims) == 2 {
			v, err := t.floats()
			if err != nil {
				return err
			}
			K, N := t.dims[0], t.dims[1]
			tr := make([]float64, len(v))
			for i := int64(0); i < K; i++ {
				for j := int64(0); j < N; j++ {
					tr[j*K+i] = v[i*N+j]
				}
			}
			if w, err = c.stored(c.tensorName(n.inputs[1]+".T"), mim.KindWeight, []int64{N, K}, f32Data(tr)); err != nil {
				return err
			}
		} else {
			if w, err = c.input(n, 1); err != nil {
				return err
			}
			bs := c.shapeOf(w)
			if len(bs) != 2 {
				return fmt.Errorf("input B must be rank 2, got %v", bs)
			}
			w = c.op(mim.OpPermute, []uint32{w}, "", shape{bs[1], bs[0]}, named("perm", ints(1, 0)))
		}
	}
	ws := c.shapeOf(w)
	if len(ws) != 2 || ws[1] != as[1] {
		return fmt.Errorf("inputs A %v and B %v do not multiply", as, ws)
	}
	ys := shape{as[0], ws[0]}
	alpha, beta := c.attrFloat(n, "alpha", 1), c.attrFloat(n, "beta", 1)
	in := []uint32{a, w}
	var cExtra uint32
	hasC := len(n.inputs) > 2 && n.inputs[2] != ""
	if hasC {
		cv, err := c.input(n, 2)
		if err != nil {
			return err
		}
		if cs := c.shapeOf(cv); len(cs) == 1 && cs[0] == ws[0] && alpha == 1 && beta == 1 {
			in = append(in, cv)
			hasC = false
		} else {
			cExtra = cv
		}
	}
	name := out
	if alpha != 1 || hasC {
		name = out + ".ab"
	}
	y := c.call("linear", in, name, ys)
	if alpha != 1 {
		nm := out
		if hasC {
			nm = out + ".alpha"
		}
		y = c.op(mim.OpMul, []uint32{y, c.k(alpha, 2)}, nm, ys)
	}
	if hasC {
		if beta != 1 {
			cExtra, err = c.bin(mim.OpMul, cExtra, c.k(beta, len(c.shapeOf(cExtra))), "")
			if err != nil {
				return err
			}
		}
		if y, err = c.bin(mim.OpAdd, y, cExtra, out); err != nil {
			return err
		}
	}
	return c.output(n, y)
}

func (c *onnxConverter) matmul(n *onnxNode, out string) error {
	if err := only(n); err != nil {
		return err
	}
	a, err := c.input(n, 0)
	if err != nil {
		return err
	}
	b, err := c.input(n, 1)
	if err != nil {
		return err
	}
	sa, sb := c.shapeOf(a), c.shapeOf(b)
	if len(sa) < 2 || len(sb) < 2 {
		return fmt.Errorf("1-D MatMul operands are not supported: %v and %v", sa, sb)
	}
	for len(sa) < len(sb) {
		sa = append(shape{1}, sa...)
	}
	for len(sb) < len(sa) {
		sb = append(shape{1}, sb...)
	}
	if len(sa) != len(c.shapeOf(a)) {
		a = c.reshape(a, "", sa)
	}
	if len(sb) != len(c.shapeOf(b)) {
		b = c.reshape(b, "", sb)
	}
	r := len(sa)
	if sa[r-1] != sb[r-2] {
		return fmt.Errorf("%v and %v do not multiply", c.shapeOf(a), c.shapeOf(b))
	}
	ys := make(shape, r)
	for i := 0; i < r-2; i++ {
		switch {
		case sa[i] == sb[i] || sb[i] == 1:
			ys[i] = sa[i]
		case sa[i] == 1:
			ys[i] = sb[i]
		default:
			return fmt.Errorf("batch dimensions of %v and %v do not broadcast", sa, sb)
		}
	}
	ys[r-2], ys[r-1] = sa[r-2], sb[r-1]
	return c.output(n, c.call("matmul", []uint32{a, b}, out, ys))
}

// product multiplies the dims of s. A group that holds one symbolic
// dimension and otherwise ones is that dimension.
func product(s shape) (mim.Dim, error) {
	p := mim.Dim(1)
	var sym []mim.Dim
	for _, d := range s {
		if d.Symbolic() {
			sym = append(sym, d)
			continue
		}
		p *= d
	}
	switch {
	case len(sym) == 0:
		return p, nil
	case len(sym) == 1 && p == 1:
		return sym[0], nil
	}
	return 0, fmt.Errorf("cannot multiply the symbolic dimensions of %v", s)
}

func (c *onnxConverter) flatten(n *onnxNode, out string) error {
	if err := only(n, "axis"); err != nil {
		return err
	}
	x, err := c.input(n, 0)
	if err != nil {
		return err
	}
	s := c.shapeOf(x)
	ax := c.attrInt(n, "axis", 1)
	if ax < 0 {
		ax += int64(len(s))
	}
	if ax < 0 || ax > int64(len(s)) {
		return fmt.Errorf("axis %d for rank %d", ax, len(s))
	}
	d0, err := product(s[:ax])
	if err != nil {
		return err
	}
	d1, err := product(s[ax:])
	if err != nil {
		return err
	}
	return c.output(n, c.reshape(x, out, shape{d0, d1}))
}

func (c *onnxConverter) reshapeNode(n *onnxNode, out string) error {
	if err := only(n, "allowzero", "shape"); err != nil {
		return err
	}
	if c.attrInt(n, "allowzero", 0) != 0 {
		return fmt.Errorf("allowzero 1 is not supported")
	}
	x, err := c.input(n, 0)
	if err != nil {
		return err
	}
	var spec []int64
	if c.m.opset < 5 {
		spec = c.attrInts(n, "shape", nil)
	} else {
		t, err := c.constInput(n, 1)
		if err != nil {
			return err
		}
		if t == nil {
			return fmt.Errorf("the shape input is missing")
		}
		if spec, err = t.int64s(); err != nil {
			return err
		}
	}
	s := c.shapeOf(x)
	ys := make(shape, len(spec))
	infer := -1
	used := map[int]bool{}
	for i, d := range spec {
		switch {
		case d == 0:
			if i >= len(s) {
				return fmt.Errorf("shape %v copies axis %d of rank %d", spec, i, len(s))
			}
			ys[i] = s[i]
			used[i] = true
		case d == -1:
			if infer >= 0 {
				return fmt.Errorf("shape %v has two -1 entries", spec)
			}
			infer = i
		case d > 0:
			ys[i] = mim.Dim(d)
		default:
			return fmt.Errorf("shape %v", spec)
		}
	}
	// Elements on both sides: literal products and symbolic dimensions.
	inLit, outLit := int64(1), int64(1)
	inSym := map[mim.Dim]int{}
	for _, d := range s {
		if d.Symbolic() {
			inSym[d]++
		} else {
			inLit *= int64(d)
		}
	}
	for i, d := range ys {
		if i == infer {
			continue
		}
		if d.Symbolic() {
			inSym[d]--
		} else {
			outLit *= int64(d)
		}
	}
	var rest []mim.Dim
	for d, k := range inSym {
		if k < 0 {
			return fmt.Errorf("shape %v uses a symbolic dimension of %v more often than the input", spec, s)
		}
		for ; k > 0; k-- {
			rest = append(rest, d)
		}
	}
	if infer >= 0 {
		switch {
		case len(rest) == 0 && outLit != 0 && inLit%outLit == 0:
			ys[infer] = mim.Dim(inLit / outLit)
		case len(rest) == 1 && inLit == outLit:
			ys[infer] = rest[0]
		default:
			return fmt.Errorf("cannot infer the -1 of shape %v for input %v", spec, s)
		}
	} else if len(rest) != 0 || inLit != outLit {
		return fmt.Errorf("shape %v does not match input %v", spec, s)
	}
	return c.output(n, c.reshape(x, out, ys))
}

func (c *onnxConverter) concat(n *onnxNode, out string) error {
	if err := only(n, "axis"); err != nil {
		return err
	}
	if len(n.inputs) == 0 {
		return fmt.Errorf("no inputs")
	}
	var xs []uint32
	for i := range n.inputs {
		x, err := c.input(n, i)
		if err != nil {
			return err
		}
		xs = append(xs, x)
	}
	s0 := c.shapeOf(xs[0])
	ax, err := normAxis(c.attrInt(n, "axis", 0), len(s0))
	if err != nil {
		return err
	}
	total := int64(0)
	for _, x := range xs {
		s := c.shapeOf(x)
		if len(s) != len(s0) {
			return fmt.Errorf("inputs of different ranks")
		}
		for i := range s {
			if i != ax && s[i] != s0[i] {
				return fmt.Errorf("inputs %v and %v differ off the concat axis", s0, s)
			}
		}
		if s[ax].Symbolic() {
			return fmt.Errorf("the concat axis needs known sizes")
		}
		total += int64(s[ax])
	}
	ys := append(shape{}, s0...)
	ys[ax] = mim.Dim(total)
	// The concat macro of spec/stdlib.md: each input padded with zeros to
	// the full length at its offset, then summed.
	var acc uint32
	off := int64(0)
	for i, x := range xs {
		l := int64(c.shapeOf(x)[ax])
		pads := make([]int64, 2*len(ys))
		pads[2*ax], pads[2*ax+1] = off, total-off-l
		v := x
		if pads[2*ax] != 0 || pads[2*ax+1] != 0 {
			v = c.op(mim.OpPad, []uint32{x}, "", ys, named("pads", ints(pads...)))
		}
		off += l
		if i == 0 {
			acc = v
			continue
		}
		name := ""
		if i == len(xs)-1 {
			name = out
		}
		acc = c.op(mim.OpAdd, []uint32{acc, v}, name, ys)
	}
	if len(xs) == 1 {
		acc = c.reshape(acc, out, ys)
	}
	return c.output(n, acc)
}
