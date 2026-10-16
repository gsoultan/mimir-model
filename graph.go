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

package mmf

import (
	"math"
	"unicode/utf8"
)

// Opcodes. Values are fixed by spec/primitives.md and never reused.
type Opcode uint16

const (
	OpNeg        Opcode = 1
	OpRecip      Opcode = 2
	OpSqrt       Opcode = 3
	OpExp2       Opcode = 4
	OpLog2       Opcode = 5
	OpSin        Opcode = 6
	OpTrunc      Opcode = 7
	OpCast       Opcode = 8
	OpBitcast    Opcode = 9
	OpAdd        Opcode = 10
	OpMul        Opcode = 11
	OpMax        Opcode = 12
	OpPow        Opcode = 13
	OpIdiv       Opcode = 14
	OpImod       Opcode = 15
	OpShl        Opcode = 16
	OpShr        Opcode = 17
	OpAnd        Opcode = 18
	OpOr         Opcode = 19
	OpXor        Opcode = 20
	OpCmplt      Opcode = 21
	OpCmpeq      Opcode = 22
	OpWhere      Opcode = 23
	OpReduce     Opcode = 24
	OpReshape    Opcode = 25
	OpExpand     Opcode = 26
	OpPermute    Opcode = 27
	OpPad        Opcode = 28
	OpShrink     Opcode = 29
	OpGather     Opcode = 30
	OpScatter    Opcode = 31
	OpConst      Opcode = 32
	OpDequantize Opcode = 33
	OpCall       Opcode = 34
	OpWhile      Opcode = 64
	OpIf         Opcode = 65
)

var opNames = map[Opcode]string{
	OpNeg: "neg", OpRecip: "recip", OpSqrt: "sqrt", OpExp2: "exp2", OpLog2: "log2",
	OpSin: "sin", OpTrunc: "trunc", OpCast: "cast", OpBitcast: "bitcast", OpAdd: "add",
	OpMul: "mul", OpMax: "max", OpPow: "pow", OpIdiv: "idiv", OpImod: "imod",
	OpShl: "shl", OpShr: "shr", OpAnd: "and", OpOr: "or", OpXor: "xor",
	OpCmplt: "cmplt", OpCmpeq: "cmpeq", OpWhere: "where", OpReduce: "reduce",
	OpReshape: "reshape", OpExpand: "expand", OpPermute: "permute", OpPad: "pad",
	OpShrink: "shrink", OpGather: "gather", OpScatter: "scatter", OpConst: "const",
	OpDequantize: "dequantize", OpCall: "call", OpWhile: "while", OpIf: "if",
}

// Valid reports whether op is defined by this version of the spec.
func (op Opcode) Valid() bool { _, ok := opNames[op]; return ok }

func (op Opcode) String() string {
	if s, ok := opNames[op]; ok {
		return s
	}
	return "invalid"
}

// ValueKind says where a value's data comes from.
type ValueKind uint8

const (
	KindIntermediate ValueKind = 0 // produced by an op
	KindInput        ValueKind = 1 // supplied by the caller
	KindOutput       ValueKind = 2 // produced by an op, returned to the caller
	KindWeight       ValueKind = 3 // a tensor in the weights section (Ref = tensor index)
	KindVariable     ValueKind = 4 // persistent state; initial value is tensor Ref
	KindConst        ValueKind = 5 // a tensor in the weights section that is not a weight
)

// Dim is one dimension of a shape: >= 0 is a literal size, < 0 refers to the
// symbolic dimension with index -1-Dim.
type Dim int64

// Symbolic reports whether d refers to a symbolic dimension.
func (d Dim) Symbolic() bool { return d < 0 }

// SymIndex returns the symbolic dimension index for a symbolic Dim.
func (d Dim) SymIndex() int { return int(-1 - d) }

// SymDim declares a symbolic dimension.
type SymDim struct {
	Name  string
	Bound int64 // -1 when unbounded
}

// Value is one tensor in the graph.
type Value struct {
	Name  string // "" when anonymous
	DType DType
	Shape []Dim
	Kind  ValueKind
	Ref   uint32 // tensor index for KindWeight, KindVariable, KindConst
	// Zero marks a variable whose initial value is all zero bytes rather
	// than a stored tensor. Its shape is literal and Ref is 0.
	Zero bool
}

// valueZero is the flag bit of Value.Zero in a value row's flags byte.
const valueZero = 1

func literal(shape []Dim) bool {
	for _, d := range shape {
		if d.Symbolic() {
			return false
		}
	}
	return true
}

// AttrType identifies an attribute's payload.
type AttrType uint8

const (
	AttrInt      AttrType = 1
	AttrInts     AttrType = 2
	AttrFloat    AttrType = 3
	AttrString   AttrType = 4
	AttrDType    AttrType = 5
	AttrSubgraph AttrType = 6
)

// Attr is one operation attribute.
type Attr struct {
	Key      string
	Type     AttrType
	Int      int64
	Ints     []int64
	Float    float64
	String   string
	DType    DType
	Subgraph uint32
}

// Op is one node. Inputs and Outputs are value indices.
type Op struct {
	Opcode  Opcode
	Inputs  []uint32
	Outputs []uint32
	Attrs   []Attr
}

// Subgraph is a contiguous range of ops with its own interface. Subgraph 0
// is the main graph.
type Subgraph struct {
	Ops     []Op
	Inputs  []uint32
	Outputs []uint32
}

// Composite is a model-local composite definition.
type Composite struct {
	Name     string
	Version  uint32
	Subgraph uint32
}

// Update assigns a new value to a variable after a run.
type Update struct {
	Variable uint32
	Value    uint32
}

// Graph is a decoded, validated graph section.
type Graph struct {
	PrimitiveVersion uint32
	StdlibVersion    uint32
	SymDims          []SymDim
	Values           []Value
	Subgraphs        []Subgraph
	Composites       []Composite
	Updates          []Update
}

// Main returns the top-level subgraph.
func (g *Graph) Main() *Subgraph { return &g.Subgraphs[0] }

const (
	graphMagic       = "MMFG"
	graphHeaderSize  = 16
	graphTables      = 11
	graphDirSize     = graphHeaderSize + graphTables*16
	MaxGraphRows     = 1 << 28
	CurrentPrimitive = 1
	CurrentStdlib    = 1
)

// Table order and row sizes in the graph section.
const (
	tPool       = iota // raw bytes
	tStrings           // 8: u32 offset, u32 length
	tSymDims           // 16: u32 name, u32 reserved, i64 bound
	tDims              // 8: i64
	tValues            // 32
	tOps               // 32
	tRuns              // 4: u32
	tAttrs             // 24
	tSubgraphs         // 32
	tComposites        // 16
	tUpdates           // 8
)

var graphRowSize = [graphTables]uint64{1, 8, 16, 8, 32, 32, 4, 24, 32, 16, 8}

// Encode serializes g. It does not validate; call Validate first or decode
// the result to check it.
func (g *Graph) Encode() ([]byte, error) {
	var tabs [graphTables][]byte
	strIdx := map[string]uint32{}
	str := func(s string) uint32 {
		if i, ok := strIdx[s]; ok {
			return i
		}
		if !utf8.ValidString(s) {
			return math.MaxUint32
		}
		i := uint32(len(tabs[tStrings]) / 8)
		strIdx[s] = i
		tabs[tStrings] = le.AppendUint32(tabs[tStrings], uint32(len(tabs[tPool])))
		tabs[tStrings] = le.AppendUint32(tabs[tStrings], uint32(len(s)))
		tabs[tPool] = append(tabs[tPool], s...)
		return i
	}
	dims := func(d []int64) uint32 {
		off := uint32(len(tabs[tDims]) / 8)
		for _, x := range d {
			tabs[tDims] = le.AppendUint64(tabs[tDims], uint64(x))
		}
		return off
	}
	run := func(ids []uint32) uint32 {
		off := uint32(len(tabs[tRuns]) / 4)
		for _, x := range ids {
			tabs[tRuns] = le.AppendUint32(tabs[tRuns], x)
		}
		return off
	}
	str("") // string 0 is the empty string
	for _, s := range g.SymDims {
		tabs[tSymDims] = le.AppendUint32(tabs[tSymDims], str(s.Name))
		tabs[tSymDims] = le.AppendUint32(tabs[tSymDims], 0)
		tabs[tSymDims] = le.AppendUint64(tabs[tSymDims], uint64(s.Bound))
	}
	for _, v := range g.Values {
		if len(v.Shape) > MaxRank {
			return nil, errf(CodeBadShape, "value %q rank %d", v.Name, len(v.Shape))
		}
		sh := make([]int64, len(v.Shape))
		for i, d := range v.Shape {
			sh[i] = int64(d)
		}
		e := make([]byte, 32)
		le.PutUint32(e[0:], str(v.Name))
		le.PutUint32(e[4:], uint32(v.DType))
		le.PutUint32(e[8:], dims(sh))
		e[12] = byte(len(v.Shape))
		e[13] = byte(v.Kind)
		if v.Zero {
			e[14] = valueZero
		}
		le.PutUint32(e[16:], v.Ref)
		tabs[tValues] = append(tabs[tValues], e...)
	}
	for _, sg := range g.Subgraphs {
		opBegin := uint32(len(tabs[tOps]) / 32)
		for _, op := range sg.Ops {
			attrOff := uint32(len(tabs[tAttrs]) / 24)
			for _, a := range op.Attrs {
				e := make([]byte, 24)
				le.PutUint32(e[0:], str(a.Key))
				e[4] = byte(a.Type)
				var count uint32
				var payload uint64
				switch a.Type {
				case AttrInt:
					payload = uint64(a.Int)
				case AttrInts:
					payload, count = uint64(dims(a.Ints)), uint32(len(a.Ints))
				case AttrFloat:
					payload = math.Float64bits(a.Float)
				case AttrString:
					payload = uint64(str(a.String))
				case AttrDType:
					payload = uint64(a.DType)
				case AttrSubgraph:
					payload = uint64(a.Subgraph)
				default:
					return nil, errf(CodeBadGraph, "attribute %q type %d", a.Key, a.Type)
				}
				le.PutUint32(e[8:], count)
				le.PutUint64(e[16:], payload)
				tabs[tAttrs] = append(tabs[tAttrs], e...)
			}
			e := make([]byte, 32)
			le.PutUint16(e[0:], uint16(op.Opcode))
			le.PutUint32(e[4:], run(op.Inputs))
			le.PutUint32(e[8:], uint32(len(op.Inputs)))
			le.PutUint32(e[12:], run(op.Outputs))
			le.PutUint32(e[16:], uint32(len(op.Outputs)))
			le.PutUint32(e[20:], attrOff)
			le.PutUint32(e[24:], uint32(len(op.Attrs)))
			tabs[tOps] = append(tabs[tOps], e...)
		}
		e := make([]byte, 32)
		le.PutUint32(e[0:], opBegin)
		le.PutUint32(e[4:], uint32(len(sg.Ops)))
		le.PutUint32(e[8:], run(sg.Inputs))
		le.PutUint32(e[12:], uint32(len(sg.Inputs)))
		le.PutUint32(e[16:], run(sg.Outputs))
		le.PutUint32(e[20:], uint32(len(sg.Outputs)))
		tabs[tSubgraphs] = append(tabs[tSubgraphs], e...)
	}
	for _, c := range g.Composites {
		tabs[tComposites] = le.AppendUint32(tabs[tComposites], str(c.Name))
		tabs[tComposites] = le.AppendUint32(tabs[tComposites], c.Version)
		tabs[tComposites] = le.AppendUint32(tabs[tComposites], c.Subgraph)
		tabs[tComposites] = le.AppendUint32(tabs[tComposites], 0)
	}
	for _, u := range g.Updates {
		tabs[tUpdates] = le.AppendUint32(tabs[tUpdates], u.Variable)
		tabs[tUpdates] = le.AppendUint32(tabs[tUpdates], u.Value)
	}
	if len(tabs[tStrings]) > 0 && strIdx[""] != 0 {
		return nil, errf(CodeBadGraph, "invalid UTF-8 in a name")
	}
	for s := range strIdx {
		if !utf8.ValidString(s) {
			return nil, errf(CodeBadGraph, "invalid UTF-8 in %q", s)
		}
	}

	out := make([]byte, graphDirSize)
	copy(out, graphMagic)
	le.PutUint16(out[4:], FormatMajor)
	le.PutUint16(out[6:], FormatMinor)
	le.PutUint32(out[8:], g.PrimitiveVersion)
	le.PutUint32(out[12:], g.StdlibVersion)
	for i, t := range tabs {
		off, _ := alignUp(uint64(len(out)), 8)
		out = append(out, make([]byte, int(off)-len(out))...)
		le.PutUint64(out[graphHeaderSize+16*i:], off)
		le.PutUint64(out[graphHeaderSize+16*i+8:], uint64(len(t))/graphRowSize[i])
		out = append(out, t...)
	}
	return out, nil
}

// DecodeGraph parses and validates a graph section.
func DecodeGraph(b []byte) (*Graph, error) {
	d := &graphDecoder{b: b}
	return d.decode()
}

type graphDecoder struct {
	b    []byte
	tabs [graphTables][]byte
	n    [graphTables]uint64
	g    *Graph
}

func (d *graphDecoder) decode() (*Graph, error) {
	b := d.b
	if len(b) < graphDirSize || string(b[:4]) != graphMagic {
		return nil, errf(CodeBadGraph, "missing header")
	}
	if le.Uint16(b[4:]) != FormatMajor {
		return nil, errf(CodeUnsupportedVersion, "graph format %d", le.Uint16(b[4:]))
	}
	g := &Graph{PrimitiveVersion: le.Uint32(b[8:]), StdlibVersion: le.Uint32(b[12:])}
	d.g = g
	if g.PrimitiveVersion > CurrentPrimitive || g.StdlibVersion > CurrentStdlib {
		return nil, errf(CodeUnsupportedVersion, "primitives v%d, stdlib v%d", g.PrimitiveVersion, g.StdlibVersion)
	}
	for i := 0; i < graphTables; i++ {
		off := le.Uint64(b[graphHeaderSize+16*i:])
		n := le.Uint64(b[graphHeaderSize+16*i+8:])
		if n > MaxGraphRows || off%8 != 0 || !inBounds(off, n*graphRowSize[i], uint64(len(b))) {
			return nil, errf(CodeBadGraph, "table %d out of bounds", i)
		}
		d.tabs[i] = b[off : off+n*graphRowSize[i]]
		d.n[i] = n
	}
	if d.n[tStrings] == 0 {
		return nil, errf(CodeBadGraph, "string table is empty")
	}
	strs := make([]string, d.n[tStrings])
	for i := range strs {
		e := d.tabs[tStrings][8*i:]
		off, ln := uint64(le.Uint32(e)), uint64(le.Uint32(e[4:]))
		if !inBounds(off, ln, d.n[tPool]) {
			return nil, errf(CodeBadGraph, "string %d outside pool", i)
		}
		s := d.tabs[tPool][off : off+ln]
		if !utf8.Valid(s) || len(s) > MaxNameLength {
			return nil, errf(CodeBadName, "string %d", i)
		}
		strs[i] = string(s)
	}
	str := func(i uint32, what string) (string, error) {
		if uint64(i) >= d.n[tStrings] {
			return "", errf(CodeBadGraph, "%s: string %d out of range", what, i)
		}
		return strs[i], nil
	}
	dimsAt := func(off, n uint64, what string) ([]int64, error) {
		if !inBounds(off, n, d.n[tDims]) {
			return nil, errf(CodeBadGraph, "%s: dims out of range", what)
		}
		out := make([]int64, n)
		for i := range out {
			out[i] = int64(le.Uint64(d.tabs[tDims][8*(int(off)+i):]))
		}
		return out, nil
	}
	runAt := func(off, n uint64, what string) ([]uint32, error) {
		if !inBounds(off, n, d.n[tRuns]) {
			return nil, errf(CodeBadGraph, "%s: run out of range", what)
		}
		out := make([]uint32, n)
		for i := range out {
			out[i] = le.Uint32(d.tabs[tRuns][4*(int(off)+i):])
			if uint64(out[i]) >= d.n[tValues] {
				return nil, errf(CodeBadGraph, "%s: value %d out of range", what, out[i])
			}
		}
		return out, nil
	}

	g.SymDims = make([]SymDim, d.n[tSymDims])
	for i := range g.SymDims {
		e := d.tabs[tSymDims][16*i:]
		name, err := str(le.Uint32(e), "symdim")
		if err != nil {
			return nil, err
		}
		if le.Uint32(e[4:]) != 0 {
			return nil, errf(CodeReservedNonZero, "symdim %d", i)
		}
		bound := int64(le.Uint64(e[8:]))
		if bound < -1 {
			return nil, errf(CodeBadGraph, "symdim %q bound %d", name, bound)
		}
		g.SymDims[i] = SymDim{name, bound}
	}
	g.Values = make([]Value, d.n[tValues])
	for i := range g.Values {
		e := d.tabs[tValues][32*i:]
		name, err := str(le.Uint32(e), "value")
		if err != nil {
			return nil, err
		}
		v := Value{Name: name, DType: DType(le.Uint32(e[4:])), Kind: ValueKind(e[13]), Ref: le.Uint32(e[16:]), Zero: e[14] == valueZero}
		rank := uint64(e[12])
		if rank > MaxRank || e[14]&^valueZero != 0 || e[15] != 0 || le.Uint64(e[20:]) != 0 || le.Uint32(e[28:]) != 0 {
			return nil, errf(CodeBadGraph, "value %d header", i)
		}
		if !v.DType.Valid() {
			return nil, errf(CodeBadDType, "value %d dtype %d", i, uint32(v.DType))
		}
		if v.Kind > KindConst {
			return nil, errf(CodeBadGraph, "value %d kind %d", i, v.Kind)
		}
		raw, err := dimsAt(uint64(le.Uint32(e[8:])), rank, "value")
		if err != nil {
			return nil, err
		}
		v.Shape = make([]Dim, rank)
		for k, x := range raw {
			if x < 0 && uint64(-1-x) >= d.n[tSymDims] {
				return nil, errf(CodeBadGraph, "value %d symbolic dim out of range", i)
			}
			v.Shape[k] = Dim(x)
		}
		g.Values[i] = v
	}
	// Ops are decoded per subgraph; every op must belong to exactly one.
	owner := make([]int32, d.n[tOps])
	for i := range owner {
		owner[i] = -1
	}
	if d.n[tSubgraphs] == 0 {
		return nil, errf(CodeBadGraph, "no main subgraph")
	}
	g.Subgraphs = make([]Subgraph, d.n[tSubgraphs])
	for si := range g.Subgraphs {
		e := d.tabs[tSubgraphs][32*si:]
		begin, count := uint64(le.Uint32(e)), uint64(le.Uint32(e[4:]))
		if !inBounds(begin, count, d.n[tOps]) || le.Uint64(e[24:]) != 0 {
			return nil, errf(CodeBadGraph, "subgraph %d op range", si)
		}
		sg := Subgraph{Ops: make([]Op, count)}
		var err error
		if sg.Inputs, err = runAt(uint64(le.Uint32(e[8:])), uint64(le.Uint32(e[12:])), "subgraph inputs"); err != nil {
			return nil, err
		}
		if sg.Outputs, err = runAt(uint64(le.Uint32(e[16:])), uint64(le.Uint32(e[20:])), "subgraph outputs"); err != nil {
			return nil, err
		}
		for k := range sg.Ops {
			oi := int(begin) + k
			if owner[oi] != -1 {
				return nil, errf(CodeBadGraph, "op %d in two subgraphs", oi)
			}
			owner[oi] = int32(si)
			oe := d.tabs[tOps][32*oi:]
			op := Op{Opcode: Opcode(le.Uint16(oe))}
			if le.Uint16(oe[2:]) != 0 || le.Uint32(oe[28:]) != 0 {
				return nil, errf(CodeReservedNonZero, "op %d", oi)
			}
			if !op.Opcode.Valid() {
				return nil, errf(CodeBadGraph, "op %d opcode %d", oi, uint16(op.Opcode))
			}
			if op.Inputs, err = runAt(uint64(le.Uint32(oe[4:])), uint64(le.Uint32(oe[8:])), "op inputs"); err != nil {
				return nil, err
			}
			if op.Outputs, err = runAt(uint64(le.Uint32(oe[12:])), uint64(le.Uint32(oe[16:])), "op outputs"); err != nil {
				return nil, err
			}
			aoff, an := uint64(le.Uint32(oe[20:])), uint64(le.Uint32(oe[24:]))
			if !inBounds(aoff, an, d.n[tAttrs]) {
				return nil, errf(CodeBadGraph, "op %d attrs out of range", oi)
			}
			op.Attrs = make([]Attr, an)
			for j := range op.Attrs {
				ae := d.tabs[tAttrs][24*(int(aoff)+j):]
				a := Attr{Type: AttrType(ae[4])}
				if a.Key, err = str(le.Uint32(ae), "attr"); err != nil {
					return nil, err
				}
				if ae[5] != 0 || ae[6] != 0 || ae[7] != 0 || le.Uint32(ae[12:]) != 0 {
					return nil, errf(CodeReservedNonZero, "attr %q", a.Key)
				}
				count, payload := uint64(le.Uint32(ae[8:])), le.Uint64(ae[16:])
				if a.Type != AttrInts && count != 0 {
					return nil, errf(CodeBadGraph, "attr %q: count on a scalar", a.Key)
				}
				switch a.Type {
				case AttrInt:
					a.Int = int64(payload)
				case AttrInts:
					if a.Ints, err = dimsAt(payload, count, "attr "+a.Key); err != nil {
						return nil, err
					}
				case AttrFloat:
					a.Float = math.Float64frombits(payload)
				case AttrString:
					if payload > math.MaxUint32 {
						return nil, errf(CodeBadGraph, "attr %q string", a.Key)
					}
					if a.String, err = str(uint32(payload), "attr "+a.Key); err != nil {
						return nil, err
					}
				case AttrDType:
					a.DType = DType(payload)
					if payload > math.MaxUint32 || !a.DType.Valid() {
						return nil, errf(CodeBadDType, "attr %q", a.Key)
					}
				case AttrSubgraph:
					if payload >= d.n[tSubgraphs] || payload == 0 {
						return nil, errf(CodeBadGraph, "attr %q subgraph %d", a.Key, payload)
					}
					a.Subgraph = uint32(payload)
				default:
					return nil, errf(CodeBadGraph, "attr %q type %d", a.Key, a.Type)
				}
				op.Attrs[j] = a
			}
			sg.Ops[k] = op
		}
		g.Subgraphs[si] = sg
	}
	for i, o := range owner {
		if o == -1 {
			return nil, errf(CodeBadGraph, "op %d belongs to no subgraph", i)
		}
	}
	g.Composites = make([]Composite, d.n[tComposites])
	for i := range g.Composites {
		e := d.tabs[tComposites][16*i:]
		name, err := str(le.Uint32(e), "composite")
		if err != nil {
			return nil, err
		}
		c := Composite{name, le.Uint32(e[4:]), le.Uint32(e[8:])}
		if le.Uint32(e[12:]) != 0 {
			return nil, errf(CodeReservedNonZero, "composite %q", name)
		}
		if c.Subgraph == 0 || uint64(c.Subgraph) >= d.n[tSubgraphs] {
			return nil, errf(CodeBadGraph, "composite %q subgraph %d", name, c.Subgraph)
		}
		g.Composites[i] = c
	}
	g.Updates = make([]Update, d.n[tUpdates])
	for i := range g.Updates {
		e := d.tabs[tUpdates][8*i:]
		u := Update{le.Uint32(e), le.Uint32(e[4:])}
		if uint64(u.Variable) >= d.n[tValues] || uint64(u.Value) >= d.n[tValues] {
			return nil, errf(CodeBadGraph, "update %d value out of range", i)
		}
		g.Updates[i] = u
	}
	if err := g.Validate(); err != nil {
		return nil, err
	}
	return g, nil
}

// Validate checks the structural rules of spec/graph.md: each value has
// exactly one definition, every use is defined before it, subgraphs are
// referenced exactly once, and updates target variables.
func (g *Graph) Validate() error {
	nv := uint32(len(g.Values))
	chk := func(id uint32) error {
		if id >= nv {
			return errf(CodeBadGraph, "value %d out of range", id)
		}
		return nil
	}
	for i, v := range g.Values {
		if v.Zero && (v.Kind != KindVariable || v.Ref != 0 || v.DType.IsBlock() || !literal(v.Shape)) {
			return errf(CodeBadGraph, "value %d (%q): only a variable with a literal shape and a plain dtype can start as zeros", i, v.Name)
		}
	}
	producer := make([]int32, nv) // -1 none, else 1
	for i := range producer {
		producer[i] = -1
	}
	refs := make([]int, len(g.Subgraphs))
	refs[0] = 1
	for _, c := range g.Composites {
		if int(c.Subgraph) >= len(g.Subgraphs) {
			return errf(CodeBadGraph, "composite %q subgraph out of range", c.Name)
		}
		refs[c.Subgraph]++
	}
	for si, sg := range g.Subgraphs {
		for _, id := range sg.Inputs {
			if err := chk(id); err != nil {
				return err
			}
			if si == 0 {
				continue // main inputs are KindInput, checked below
			}
			if producer[id] != -1 || g.Values[id].Kind != KindIntermediate {
				return errf(CodeBadGraph, "subgraph %d input %d is defined twice or is not an intermediate value", si, id)
			}
			producer[id] = 1
		}
		for oi, op := range sg.Ops {
			for _, id := range op.Outputs {
				if err := chk(id); err != nil {
					return err
				}
				if producer[id] != -1 || g.Values[id].Kind != KindIntermediate && g.Values[id].Kind != KindOutput {
					return errf(CodeBadGraph, "subgraph %d op %d output %d is defined twice or is not an op-produced value", si, oi, id)
				}
				producer[id] = 1
			}
			for _, a := range op.Attrs {
				if a.Type == AttrSubgraph {
					if int(a.Subgraph) >= len(g.Subgraphs) {
						return errf(CodeBadGraph, "attr %q subgraph out of range", a.Key)
					}
					refs[a.Subgraph]++
				}
			}
		}
	}
	for r, n := range refs {
		if n != 1 {
			return errf(CodeBadGraph, "subgraph %d is referenced %d times", r, n)
		}
	}
	for id, v := range g.Values {
		if v.Kind == KindIntermediate || v.Kind == KindOutput {
			if producer[id] == -1 {
				return errf(CodeBadGraph, "value %d (%q) has no definition", id, v.Name)
			}
		} else if producer[id] != -1 {
			return errf(CodeBadGraph, "value %d (%q) of kind %d is also produced by an op", id, v.Name, v.Kind)
		}
	}
	// Definition before use, with subgraph scoping.
	defined := make([]bool, nv)
	for id, v := range g.Values {
		if v.Kind == KindInput || v.Kind == KindWeight || v.Kind == KindVariable || v.Kind == KindConst {
			defined[id] = true
		}
	}
	var walk func(si int, defined []bool) error
	walk = func(si int, defined []bool) error {
		sg := &g.Subgraphs[si]
		local := append([]bool(nil), defined...)
		for _, id := range sg.Inputs {
			local[id] = true
		}
		for oi, op := range sg.Ops {
			for _, id := range op.Inputs {
				if !local[id] {
					return errf(CodeBadGraph, "subgraph %d op %d uses value %d before its definition", si, oi, id)
				}
			}
			for _, a := range op.Attrs {
				if a.Type == AttrSubgraph {
					if err := walk(int(a.Subgraph), local); err != nil {
						return err
					}
				}
			}
			for _, id := range op.Outputs {
				local[id] = true
			}
		}
		for _, id := range sg.Outputs {
			if !local[id] {
				return errf(CodeBadGraph, "subgraph %d output %d is never defined", si, id)
			}
		}
		return nil
	}
	if err := walk(0, defined); err != nil {
		return err
	}
	for _, c := range g.Composites {
		if err := walk(int(c.Subgraph), defined); err != nil {
			return err
		}
	}
	for _, id := range g.Subgraphs[0].Inputs {
		if g.Values[id].Kind != KindInput {
			return errf(CodeBadGraph, "main input %d is not of kind input", id)
		}
	}
	for _, id := range g.Subgraphs[0].Outputs {
		if g.Values[id].Kind != KindOutput {
			return errf(CodeBadGraph, "main output %d is not of kind output", id)
		}
	}
	seen := map[uint32]bool{}
	for _, u := range g.Updates {
		if err := chk(u.Variable); err != nil {
			return err
		}
		if err := chk(u.Value); err != nil {
			return err
		}
		v, nv := g.Values[u.Variable], g.Values[u.Value]
		if v.Kind != KindVariable || seen[u.Variable] {
			return errf(CodeBadGraph, "update target %d is not a variable or is updated twice", u.Variable)
		}
		seen[u.Variable] = true
		if v.DType != nv.DType || !sameShape(v.Shape, nv.Shape) {
			return errf(CodeBadGraph, "update of %q changes dtype or shape", v.Name)
		}
	}
	return nil
}

func sameShape(a, b []Dim) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Graph decodes the file's graph section. It also checks that every weight,
// variable and const value refers to an existing tensor with the declared
// dtype and a compatible shape.
func (f *File) Graph() (*Graph, error) {
	s, ok := f.Section(SectionGraph)
	if !ok {
		return nil, errf(CodeMissingSection, "no graph section")
	}
	g, err := DecodeGraph(f.SectionBytes(s))
	if err != nil {
		return nil, err
	}
	for i, v := range g.Values {
		if v.Kind != KindWeight && v.Kind != KindVariable && v.Kind != KindConst || v.Zero {
			continue
		}
		if int(v.Ref) >= f.NumTensors() {
			return nil, errf(CodeBadGraph, "value %d (%q) refers to tensor %d of %d", i, v.Name, v.Ref, f.NumTensors())
		}
		t := f.TensorAt(int(v.Ref))
		if t.DType != v.DType || len(t.Shape) != len(v.Shape) {
			return nil, errf(CodeBadGraph, "value %q does not match tensor %q", v.Name, t.Name)
		}
		for k, d := range v.Shape {
			if !d.Symbolic() && uint64(d) != t.Shape[k] {
				return nil, errf(CodeBadGraph, "value %q dim %d does not match tensor %q", v.Name, k, t.Name)
			}
		}
	}
	return g, nil
}
