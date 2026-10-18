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

package ref

import (
	"fmt"

	"github.com/gsoultan/mlm"
)

// attrSpec lists the attributes of a primitive: key to type, and whether the
// attribute is required.
type attrSpec map[string]struct {
	typ      []mlm.AttrType
	required bool
}

var (
	tInt    = []mlm.AttrType{mlm.AttrInt}
	tInts   = []mlm.AttrType{mlm.AttrInts}
	tStr    = []mlm.AttrType{mlm.AttrString}
	tDType  = []mlm.AttrType{mlm.AttrDType}
	tScalar = []mlm.AttrType{mlm.AttrFloat, mlm.AttrInt}
)

var primAttrs = map[mlm.Opcode]attrSpec{
	mlm.OpCast:    {"to": {tDType, true}},
	mlm.OpBitcast: {"to": {tDType, true}},
	mlm.OpReduce:  {"mode": {tStr, true}, "axes": {tInts, true}},
	mlm.OpReshape: {"shape": {tInts, true}},
	mlm.OpPermute: {"perm": {tInts, true}},
	mlm.OpPad:     {"pads": {tInts, true}, "value": {tScalar, false}},
	mlm.OpShrink:  {"begin": {tInts, true}, "end": {tInts, true}},
	mlm.OpGather:  {"axis": {tInt, true}},
	mlm.OpScatter: {"axis": {tInt, true}},
	mlm.OpConst:   {"value": {tScalar, true}},
}

// inputCount is the number of inputs of each primitive.
var inputCount = map[mlm.Opcode]int{
	mlm.OpWhere: 3, mlm.OpScatter: 3, mlm.OpConst: 0, mlm.OpGather: 2,
}

func numInputs(op mlm.Opcode) int {
	if n, ok := inputCount[op]; ok {
		return n
	}
	if op >= mlm.OpAdd && op <= mlm.OpCmpeq {
		return 2
	}
	return 1
}

// checkAttrs rejects unknown, duplicate, mistyped and missing attributes
// (spec/primitives.md section 4) and returns them by key.
func checkAttrs(op mlm.Opcode, attrs []mlm.Attr) (map[string]*mlm.Attr, error) {
	spec := primAttrs[op]
	got := make(map[string]*mlm.Attr, len(attrs))
	for i := range attrs {
		a := &attrs[i]
		s, ok := spec[a.Key]
		if !ok {
			return nil, fmt.Errorf("%s: unknown attribute %q", op, a.Key)
		}
		if got[a.Key] != nil {
			return nil, fmt.Errorf("%s: duplicate attribute %q", op, a.Key)
		}
		typeOK := false
		for _, t := range s.typ {
			typeOK = typeOK || a.Type == t
		}
		if !typeOK {
			return nil, fmt.Errorf("%s: attribute %q has the wrong type", op, a.Key)
		}
		got[a.Key] = a
	}
	for k, s := range spec {
		if s.required && got[k] == nil {
			return nil, fmt.Errorf("%s: missing attribute %q", op, k)
		}
	}
	return got, nil
}

// Eval runs one primitive (opcodes 1 to 33) on concrete inputs. outDType and
// outShape are the declared type of the output value; Eval checks them
// against the type and shape rules of spec/primitives.md and returns an
// error on any violation instead of computing a result.
func Eval(op mlm.Opcode, attrs []mlm.Attr, in []*Tensor, outDType mlm.DType, outShape []int) (*Tensor, error) {
	if op < mlm.OpNeg || op > mlm.OpDequantize {
		return nil, fmt.Errorf("opcode %d is not a primitive", op)
	}
	if len(in) != numInputs(op) {
		return nil, fmt.Errorf("%s: %d inputs, want %d", op, len(in), numInputs(op))
	}
	for i, t := range in {
		if t == nil {
			return nil, fmt.Errorf("%s: input %d is missing", op, i)
		}
		if err := t.Validate(); err != nil {
			return nil, fmt.Errorf("%s: input %d: %w", op, i, err)
		}
		if t.DType.IsBlock() && op != mlm.OpDequantize {
			return nil, fmt.Errorf("%s: block dtype %s is only valid as the input of dequantize", op, t.DType)
		}
		if isStorage(t.DType) && !storageOK(op, i) {
			return nil, fmt.Errorf("%s: storage float %s is not valid here", op, t.DType)
		}
	}
	if !outDType.Valid() {
		return nil, fmt.Errorf("%s: invalid output dtype %d", op, outDType)
	}
	for _, d := range outShape {
		if d < 0 {
			return nil, fmt.Errorf("%s: negative output dimension", op)
		}
	}
	a, err := checkAttrs(op, attrs)
	if err != nil {
		return nil, err
	}
	y, err := eval(op, a, in, outDType, outShape)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	if y.DType != outDType || !sameInts(y.Shape, outShape) {
		return nil, fmt.Errorf("%s: result is %s%v, declared %s%v", op, y.DType, y.Shape, outDType, outShape)
	}
	return y, nil
}

// storageOK reports whether input i of op may be a storage float
// (spec/primitives.md section 2.2).
func storageOK(op mlm.Opcode, i int) bool {
	switch op {
	case mlm.OpCast, mlm.OpDequantize, mlm.OpReshape, mlm.OpExpand, mlm.OpPermute, mlm.OpPad, mlm.OpShrink:
		return true
	case mlm.OpBitcast:
		return true
	case mlm.OpWhere:
		return i != 0
	case mlm.OpGather:
		return i == 0
	case mlm.OpScatter:
		return i == 0 || i == 2
	}
	return false
}

func isFloatOnly(op mlm.Opcode) bool {
	switch op {
	case mlm.OpRecip, mlm.OpSqrt, mlm.OpExp2, mlm.OpLog2, mlm.OpSin, mlm.OpPow:
		return true
	}
	return false
}

func intsOf(v []int64) []int {
	r := make([]int, len(v))
	for i, x := range v {
		r[i] = int(x)
	}
	return r
}

func eval(op mlm.Opcode, a map[string]*mlm.Attr, in []*Tensor, outDType mlm.DType, outShape []int) (*Tensor, error) {
	switch {
	case op >= mlm.OpNeg && op <= mlm.OpTrunc:
		x := in[0]
		if !(isArith(x.DType) || IsInt(x.DType) && (op == mlm.OpNeg || op == mlm.OpTrunc)) {
			return nil, fmt.Errorf("unsupported dtype %s", x.DType)
		}
		return evalUnary(op, x)

	case op >= mlm.OpAdd && op <= mlm.OpCmpeq:
		x, z := in[0], in[1]
		if x.DType != z.DType {
			return nil, fmt.Errorf("operand dtypes %s and %s differ", x.DType, z.DType)
		}
		dt := x.DType
		var ok bool
		switch {
		case isFloatOnly(op):
			ok = isArith(dt)
		case op >= mlm.OpIdiv && op <= mlm.OpShr:
			ok = IsInt(dt)
		case op >= mlm.OpAnd && op <= mlm.OpXor:
			ok = IsInt(dt) || dt == mlm.Bool
		case op == mlm.OpCmpeq:
			ok = isArith(dt) || IsInt(dt) || dt == mlm.Bool
		default: // add, mul, max, cmplt
			ok = isArith(dt) || IsInt(dt)
		}
		if !ok {
			return nil, fmt.Errorf("unsupported dtype %s", dt)
		}
		shape, err := broadcastShape(x.Shape, z.Shape)
		if err != nil {
			return nil, err
		}
		out := dt
		if op == mlm.OpCmplt || op == mlm.OpCmpeq {
			out = mlm.Bool
		}
		return evalBinary(op, x, z, out, shape)

	case op == mlm.OpWhere:
		c, x, z := in[0], in[1], in[2]
		if c.DType != mlm.Bool || x.DType != z.DType {
			return nil, fmt.Errorf("cond must be bool and a, b must share a dtype")
		}
		shape, err := broadcastShape(c.Shape, x.Shape, z.Shape)
		if err != nil {
			return nil, err
		}
		return evalWhere(c, x, z, shape)

	case op == mlm.OpCast:
		to := a["to"].DType
		if !to.Valid() || to.IsBlock() || in[0].DType.IsBlock() {
			return nil, fmt.Errorf("cannot cast %s to %s", in[0].DType, to)
		}
		return evalCast(in[0], to)

	case op == mlm.OpBitcast:
		to := a["to"].DType
		from := in[0].DType
		if !to.Valid() || to.IsBlock() || to == mlm.Bool || from == mlm.Bool || elemSize(to) != elemSize(from) {
			return nil, fmt.Errorf("cannot bitcast %s to %s", from, to)
		}
		return &Tensor{DType: to, Shape: append([]int{}, in[0].Shape...), Data: append([]byte{}, in[0].Data...)}, nil

	case op == mlm.OpReduce:
		x := in[0]
		if !isArith(x.DType) && !IsInt(x.DType) {
			return nil, fmt.Errorf("unsupported dtype %s", x.DType)
		}
		mode := a["mode"].String
		if mode != "add" && mode != "mul" && mode != "max" {
			return nil, fmt.Errorf("unknown mode %q", mode)
		}
		axes := intsOf(a["axes"].Ints)
		shape := append([]int{}, x.Shape...)
		seen := make([]bool, len(shape))
		for _, ax := range axes {
			if ax < 0 || ax >= len(shape) || seen[ax] {
				return nil, fmt.Errorf("invalid axes %v", axes)
			}
			seen[ax] = true
			shape[ax] = 1
		}
		return evalReduce(x, mode, axes, shape)

	case op == mlm.OpReshape:
		x := in[0]
		target := a["shape"].Ints
		if len(target) != len(outShape) {
			return nil, fmt.Errorf("shape attribute has rank %d, output %d", len(target), len(outShape))
		}
		for i, d := range target {
			if d != -1 && d != int64(outShape[i]) {
				return nil, fmt.Errorf("shape attribute %v does not match output %v", target, outShape)
			}
		}
		if NumElems(outShape) != x.NumElems() {
			return nil, fmt.Errorf("cannot reshape %v to %v", x.Shape, outShape)
		}
		return &Tensor{DType: x.DType, Shape: append([]int{}, outShape...), Data: append([]byte{}, x.Data...)}, nil

	case op == mlm.OpExpand:
		x := in[0]
		if len(outShape) != len(x.Shape) {
			return nil, fmt.Errorf("expand cannot change rank")
		}
		for i, d := range x.Shape {
			if d != 1 && d != outShape[i] {
				return nil, fmt.Errorf("cannot expand %v to %v", x.Shape, outShape)
			}
		}
		return evalExpand(x, outShape)

	case op == mlm.OpPermute:
		x := in[0]
		perm := intsOf(a["perm"].Ints)
		if len(perm) != len(x.Shape) {
			return nil, fmt.Errorf("perm %v has the wrong length", perm)
		}
		seen := make([]bool, len(perm))
		shape := make([]int, len(perm))
		for i, p := range perm {
			if p < 0 || p >= len(perm) || seen[p] {
				return nil, fmt.Errorf("perm %v is not a permutation", perm)
			}
			seen[p] = true
			shape[i] = x.Shape[p]
		}
		return evalPermute(x, perm, shape)

	case op == mlm.OpPad:
		x := in[0]
		pads := intsOf(a["pads"].Ints)
		if len(pads) != 2*len(x.Shape) {
			return nil, fmt.Errorf("pads %v has the wrong length", pads)
		}
		shape := make([]int, len(x.Shape))
		for i := range shape {
			if pads[2*i] < 0 || pads[2*i+1] < 0 {
				return nil, fmt.Errorf("negative padding %v", pads)
			}
			shape[i] = x.Shape[i] + pads[2*i] + pads[2*i+1]
		}
		var value uint64
		if v := a["value"]; v != nil {
			var err error
			if value, err = scalarBits(x.DType, v); err != nil {
				return nil, err
			}
		}
		return evalPad(x, pads, value, shape)

	case op == mlm.OpShrink:
		x := in[0]
		begin, end := intsOf(a["begin"].Ints), intsOf(a["end"].Ints)
		if len(begin) != len(x.Shape) || len(end) != len(x.Shape) {
			return nil, fmt.Errorf("begin and end must have rank %d", len(x.Shape))
		}
		shape := make([]int, len(x.Shape))
		for i := range shape {
			e := end[i]
			if e == -1 {
				e = x.Shape[i]
			}
			if begin[i] < 0 || begin[i] > e || e > x.Shape[i] {
				return nil, fmt.Errorf("range [%d, %d) is outside axis %d of size %d", begin[i], e, i, x.Shape[i])
			}
			shape[i] = e - begin[i]
		}
		return evalShrink(x, begin, shape)

	case op == mlm.OpGather, op == mlm.OpScatter:
		data, ind := in[0], in[1]
		axis := int(a["axis"].Int)
		if len(data.Shape) == 0 || axis < 0 || axis >= len(data.Shape) {
			return nil, fmt.Errorf("axis %d is invalid for rank %d", axis, len(data.Shape))
		}
		if ind.DType != mlm.I32 && ind.DType != mlm.I64 {
			return nil, fmt.Errorf("indices must be i32 or i64, got %s", ind.DType)
		}
		gs := gatherShape(data.Shape, ind.Shape, axis)
		if op == mlm.OpGather {
			return evalGather(data, ind, axis, gs)
		}
		upd := in[2]
		if upd.DType != data.DType || !sameInts(upd.Shape, gs) {
			return nil, fmt.Errorf("updates must be %s%v, got %s%v", data.DType, gs, upd.DType, upd.Shape)
		}
		return evalScatter(data, ind, upd, axis)

	case op == mlm.OpConst:
		if isStorage(outDType) || outDType.IsBlock() {
			return nil, fmt.Errorf("const cannot produce %s", outDType)
		}
		v, err := scalarBits(outDType, a["value"])
		if err != nil {
			return nil, err
		}
		return evalConst(outDType, outShape, v)

	case op == mlm.OpDequantize:
		x := in[0]
		ok := x.DType.IsBlock() || x.DType == mlm.F16 || x.DType == mlm.BF16 || isStorage(x.DType)
		if !ok || outDType != mlm.F32 && outDType != mlm.BF16 {
			return nil, fmt.Errorf("cannot dequantize %s to %s", x.DType, outDType)
		}
		return evalDequantize(x, outDType)
	}
	return nil, fmt.Errorf("not implemented")
}
