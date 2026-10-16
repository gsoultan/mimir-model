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

package interp

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/ref"
)

// The files in testdata/pytorch were exported from PyTorch by the Python
// exporter (tools/gen_pytorch_testdata.py); expected.json holds PyTorch's
// outputs. Running them here shows that what the exporter writes is a
// valid graph that a runtime sharing no code with it executes.

type torchArray struct {
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
	Data  string `json:"data"`
}

type torchExpected struct {
	Tolerance float64 `json:"tolerance"`
	Models    []struct {
		Name string `json:"name"`
		MMF  string `json:"mmf"`
		Runs []struct {
			Inputs  map[string]torchArray `json:"inputs"`
			Outputs map[string]torchArray `json:"outputs"`
		} `json:"runs"`
	} `json:"models"`
}

func (a torchArray) tensor(t *testing.T) *ref.Tensor {
	t.Helper()
	dt := map[string]mmf.DType{"f32": mmf.F32, "i64": mmf.I64}[a.DType]
	b, err := base64.StdEncoding.DecodeString(a.Data)
	if err != nil || dt == 0 {
		t.Fatalf("array %s%v: %v", a.DType, a.Shape, err)
	}
	r, err := ref.New(dt, a.Shape)
	if err != nil || len(r.Data) != len(b) {
		t.Fatalf("array %s%v with %d bytes: %v", a.DType, a.Shape, len(b), err)
	}
	copy(r.Data, b)
	return r
}

func TestPyTorchExports(t *testing.T) {
	dir := filepath.Join("..", "testdata", "pytorch")
	j, err := os.ReadFile(filepath.Join(dir, "expected.json"))
	if err != nil {
		t.Fatal(err)
	}
	var e torchExpected
	if err := json.Unmarshal(j, &e); err != nil {
		t.Fatal(err)
	}
	if len(e.Models) == 0 {
		t.Fatal("no models")
	}
	for _, m := range e.Models {
		t.Run(m.Name, func(t *testing.T) {
			f, err := mmf.Open(filepath.Join(dir, m.MMF))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			if err := f.VerifyHashes(); err != nil {
				t.Fatal(err)
			}
			md, err := f.Metadata()
			if err != nil {
				t.Fatal(err)
			}
			if v, ok := md["export.source_format"].(string); !ok || v != "pytorch" {
				t.Errorf("export.source_format = %v", md["export.source_format"])
			}
			p, err := Load(f)
			if err != nil {
				t.Fatal(err)
			}
			s := p.NewSession()
			for ri, run := range m.Runs {
				in := map[string]*ref.Tensor{}
				for k, a := range run.Inputs {
					in[k] = a.tensor(t)
				}
				out, err := s.Run(in)
				if err != nil {
					t.Fatalf("run %d: %v", ri, err)
				}
				if len(out) != len(run.Outputs) {
					t.Fatalf("run %d: %d outputs, want %d", ri, len(out), len(run.Outputs))
				}
				for k, a := range run.Outputs {
					want := a.tensor(t)
					got := out[k]
					if got == nil || got.DType != want.DType || !sameInts(got.Shape, want.Shape) {
						t.Fatalf("run %d: output %s is %v, want %s%v", ri, k, got, want.DType, want.Shape)
					}
					g, w := f32s(got), f32s(want)
					scale, diff := 1e-30, 0.0
					for i := range w {
						scale = max(scale, math.Abs(w[i]))
						diff = max(diff, math.Abs(g[i]-w[i]))
					}
					if diff > e.Tolerance*scale {
						t.Errorf("run %d: output %s differs from PyTorch by %g (largest output %g)", ri, k, diff, scale)
					}
				}
			}
		})
	}
}
