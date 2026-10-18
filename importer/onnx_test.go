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

package importer

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"flag"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/gsoultan/mlm"
	"github.com/gsoultan/mlm/interp"
	"github.com/gsoultan/mlm/ref"
)

var updateONNX = flag.Bool("update-onnx", false, "rewrite testdata/onnx/tiny-cnn.mim")

type onnxExpected struct {
	Model     string  `json:"model"`
	MIM       string  `json:"mim"`
	Tolerance float64 `json:"tolerance"`
	Runs      []struct {
		Inputs  map[string]onnxArray `json:"inputs"`
		Outputs map[string]onnxArray `json:"outputs"`
	} `json:"runs"`
}

type onnxArray struct {
	Shape []int  `json:"shape"`
	Data  string `json:"data"`
}

func (a onnxArray) tensor(t *testing.T) *ref.Tensor {
	t.Helper()
	b, err := base64.StdEncoding.DecodeString(a.Data)
	if err != nil {
		t.Fatal(err)
	}
	r, err := ref.New(mlm.F32, a.Shape)
	if err != nil || len(r.Data) != len(b) {
		t.Fatalf("array %v with %d bytes: %v", a.Shape, len(b), err)
	}
	copy(r.Data, b)
	return r
}

func onnxDir() string { return filepath.Join("..", "testdata", "onnx") }

func importONNX(t *testing.T, b []byte) []byte {
	t.Helper()
	s, err := ParseONNX(b)
	if err != nil {
		t.Fatal(err)
	}
	w := mlm.NewWriter()
	if err := s.AddTo(w); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := w.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestONNXTinyCNN(t *testing.T) {
	var e onnxExpected
	j, err := os.ReadFile(filepath.Join(onnxDir(), "expected.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(j, &e); err != nil {
		t.Fatal(err)
	}
	model, err := os.ReadFile(filepath.Join(onnxDir(), e.Model))
	if err != nil {
		t.Fatal(err)
	}
	if Detect(model) != "" {
		t.Fatalf("the ONNX model is detected as %q", Detect(model))
	}
	out := importONNX(t, model)
	mimPath := filepath.Join(onnxDir(), e.MIM)
	if *updateONNX {
		if err := os.WriteFile(mimPath, out, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// The import is deterministic and the committed file is its result,
	// which the Python runtime runs.
	if committed, err := os.ReadFile(mimPath); err != nil || !bytes.Equal(committed, out) {
		t.Fatalf("%s differs from a fresh import (run go test ./importer -run TestONNXTinyCNN -update-onnx): %v", mimPath, err)
	}

	f, err := mlm.Parse(out)
	if err != nil {
		t.Fatal(err)
	}
	md, err := f.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	if md["import.source_format"] != "onnx" || md[mlm.KeyName] != "tiny-cnn" || md["onnx.opset"] != int64(13) {
		t.Errorf("metadata %v", md)
	}
	if ti, ok := f.Tensor("conv1.weight"); !ok || ti.DType != mlm.F32 || len(ti.Shape) != 4 {
		t.Errorf("conv1.weight is %+v", ti)
	}
	for _, fast := range []bool{true, false} {
		p, err := interp.Load(f)
		if err != nil {
			t.Fatal(err)
		}
		p.FastPaths = fast
		s := p.NewSession()
		for ri, r := range e.Runs {
			in := map[string]*ref.Tensor{}
			for k, v := range r.Inputs {
				in[k] = v.tensor(t)
			}
			got, err := s.Run(in)
			if err != nil {
				t.Fatalf("fast %v run %d: %v", fast, ri, err)
			}
			for k, v := range r.Outputs {
				want := v.tensor(t)
				g := got[k]
				if g == nil || g.DType != mlm.F32 || !sameDims(g.Shape, want.Shape) {
					t.Fatalf("fast %v run %d: output %q is %v", fast, ri, k, g)
				}
				for i := 0; i < want.NumElems(); i++ {
					a := math.Float32frombits(binary.LittleEndian.Uint32(g.Data[4*i:]))
					b := math.Float32frombits(binary.LittleEndian.Uint32(want.Data[4*i:]))
					if d := math.Abs(float64(a) - float64(b)); !(d <= e.Tolerance) {
						t.Fatalf("fast %v run %d: %s[%d] = %g, onnxruntime %g", fast, ri, k, i, a, b)
					}
				}
			}
		}
	}
}

func sameDims(a, b []int) bool {
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

// pb builds protocol buffer messages for the rejection tests.
type pb []byte

func (b pb) varint(num int, v uint64) pb {
	b = binary.AppendUvarint(b, uint64(num)<<3)
	return binary.AppendUvarint(b, v)
}

func (b pb) bytes(num int, v []byte) pb {
	b = binary.AppendUvarint(b, uint64(num)<<3|2)
	b = binary.AppendUvarint(b, uint64(len(v)))
	return append(b, v...)
}

func (b pb) str(num int, s string) pb { return b.bytes(num, []byte(s)) }

func testModel(opset uint64, graph pb) []byte {
	op := pb(nil).varint(2, opset)
	return pb(nil).varint(1, 8).bytes(8, op).bytes(7, graph)
}

func floatInput(name string, dims ...uint64) pb {
	shape := pb(nil)
	for _, d := range dims {
		shape = shape.bytes(1, pb(nil).varint(1, d))
	}
	tt := pb(nil).varint(1, onnxFloat).bytes(2, shape)
	return pb(nil).str(1, name).bytes(2, pb(nil).bytes(1, tt))
}

func node(op string, in, out []string, attrs ...pb) pb {
	n := pb(nil)
	for _, s := range in {
		n = n.str(1, s)
	}
	for _, s := range out {
		n = n.str(2, s)
	}
	n = n.str(4, op)
	for _, a := range attrs {
		n = n.bytes(5, a)
	}
	return n
}

func TestONNXRejects(t *testing.T) {
	relu := pb(nil).bytes(1, node("Relu", []string{"x"}, []string{"y"})).
		bytes(11, floatInput("x", 1, 4)).bytes(12, pb(nil).str(1, "y"))
	if _, err := ParseONNX(testModel(13, relu)); err != nil {
		t.Fatalf("a valid model: %v", err)
	}
	cases := map[string][]byte{
		"no graph":   pb(nil).varint(1, 8),
		"opset":      testModel(3, relu),
		"truncated":  testModel(13, relu)[:20],
		"bad length": append(pb(nil).varint(1, 8), 0x3a, 0xff, 0xff, 0x03),
		"operator": testModel(13, pb(nil).bytes(1, node("LSTM", []string{"x"}, []string{"y"})).
			bytes(11, floatInput("x", 1, 4)).bytes(12, pb(nil).str(1, "y"))),
		"undefined input": testModel(13, pb(nil).bytes(1, node("Relu", []string{"z"}, []string{"y"})).
			bytes(11, floatInput("x", 1, 4)).bytes(12, pb(nil).str(1, "y"))),
		"unknown attribute": testModel(13, pb(nil).bytes(1, node("Relu", []string{"x"}, []string{"y"},
			pb(nil).str(1, "alpha").varint(20, 1))).
			bytes(11, floatInput("x", 1, 4)).bytes(12, pb(nil).str(1, "y"))),
		"missing output": testModel(13, pb(nil).bytes(1, node("Relu", []string{"x"}, []string{"y"})).
			bytes(11, floatInput("x", 1, 4)).bytes(12, pb(nil).str(1, "q"))),
		"domain": testModel(13, pb(nil).bytes(1, node("Relu", []string{"x"}, []string{"y"}).str(7, "com.microsoft")).
			bytes(11, floatInput("x", 1, 4)).bytes(12, pb(nil).str(1, "y"))),
	}
	for name, b := range cases {
		if _, err := ParseONNX(b); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	// Every truncation of the fixture is rejected or imported, never a panic.
	model, err := os.ReadFile(filepath.Join(onnxDir(), "tiny-cnn.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	for n := 0; n < len(model); n += 7 {
		ParseONNX(model[:n])
	}
	if _, err := ParseONNX(model[:len(model)/2]); err == nil {
		t.Error("half a model was accepted")
	}
}
