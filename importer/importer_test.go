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
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gsoultan/mmf"
)

type expTensor struct {
	DType  string   `json:"dtype"`
	Shape  []uint64 `json:"shape"`
	SHA256 string   `json:"sha256"`
	F32    string   `json:"f32"`
}

func readFixture(t testing.TB, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "testdata", "import", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func expected(t *testing.T) map[string]map[string]expTensor {
	var e map[string]map[string]expTensor
	if err := json.Unmarshal(readFixture(t, "expected.json"), &e); err != nil {
		t.Fatal(err)
	}
	return e
}

// importAndReopen converts src to mmf in memory and parses the result.
func importAndReopen(t *testing.T, src *Source) *mmf.File {
	t.Helper()
	w := mmf.NewWriter()
	if err := src.AddTo(w); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := w.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	f, err := mmf.Parse(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if err := f.VerifyHashes(); err != nil {
		t.Fatal(err)
	}
	return f
}

func checkFile(t *testing.T, f *mmf.File, want map[string]expTensor) {
	t.Helper()
	if f.NumTensors() != len(want) {
		t.Errorf("%d tensors, want %d", f.NumTensors(), len(want))
	}
	for name, e := range want {
		ti, ok := f.Tensor(name)
		if !ok {
			t.Errorf("tensor %q missing", name)
			continue
		}
		if ti.DType.String() != e.DType || len(ti.Shape) != len(e.Shape) {
			t.Errorf("%s: %s%v, want %s%v", name, ti.DType, ti.Shape, e.DType, e.Shape)
			continue
		}
		for i := range e.Shape {
			if ti.Shape[i] != e.Shape[i] {
				t.Errorf("%s: shape %v, want %v", name, ti.Shape, e.Shape)
			}
		}
		data := f.TensorData(ti)
		sum := sha256.Sum256(data)
		if hex.EncodeToString(sum[:]) != e.SHA256 {
			t.Errorf("%s: data differs from the source", name)
		}
		if e.F32 == "" {
			continue
		}
		// Block tensors decode to exactly what gguf's dequantizer gives.
		raw, _ := base64.StdEncoding.DecodeString(e.F32)
		got := make([]float32, len(raw)/4)
		if err := mmf.Dequantize(ti.DType, data, got); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if math.Float32bits(got[i]) != binary.LittleEndian.Uint32(raw[4*i:]) {
				t.Errorf("%s[%d] = %v, want %v", name, i, got[i], math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])))
				break
			}
		}
	}
}

func TestImportGGUF(t *testing.T) {
	b := readFixture(t, "tiny.gguf")
	if Detect(b) != "gguf" {
		t.Fatal("not detected as gguf")
	}
	src, err := Parse(b)
	if err != nil {
		t.Fatal(err)
	}
	f := importAndReopen(t, src)
	checkFile(t, f, expected(t)["gguf"])
	md, err := f.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	checks := map[string]any{
		"general.name":          "tiny",
		"general.architecture":  "llama",
		"tiny.block_count":      uint64(2),
		"tiny.rope.freq_base":   float64(10000),
		"tiny.flag":             true,
		"import.source_format":  "gguf",
		"import.gguf_version":   int64(3),
		"tokenizer.ggml.tokens": []string{"<s>", "</s>", "hello", "world"},
		"tokenizer.ggml.scores": []float64{0, -1, -2.5, -3.25},
	}
	for k, want := range checks {
		got, _ := json.Marshal(md[k])
		w, _ := json.Marshal(want)
		if string(got) != string(w) {
			t.Errorf("metadata %s = %s, want %s", k, got, w)
		}
	}
	if tt, ok := md["tokenizer.ggml.token_type"].([]int64); !ok || len(tt) != 4 || tt[0] != 3 {
		t.Errorf("token_type = %#v", md["tokenizer.ggml.token_type"])
	}
}

func TestImportSafetensors(t *testing.T) {
	b := readFixture(t, "tiny.safetensors")
	if Detect(b) != "safetensors" {
		t.Fatal("not detected as safetensors")
	}
	src, err := Parse(b)
	if err != nil {
		t.Fatal(err)
	}
	f := importAndReopen(t, src)
	checkFile(t, f, expected(t)["safetensors"])
	md, _ := f.Metadata()
	if md["safetensors.format"] != "pt" || md["import.source_format"] != "safetensors" {
		t.Errorf("metadata %v", md)
	}
}

func TestMergeShards(t *testing.T) {
	a, err := ParseSafetensors(readFixture(t, "tiny.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Merge(a, a); err == nil || !strings.Contains(err.Error(), "more than one source") {
		t.Errorf("duplicate tensors across shards: %v", err)
	}
	g, _ := ParseGGUF(readFixture(t, "tiny.gguf"))
	if _, err := Merge(a, g); err == nil {
		t.Error("merged gguf with safetensors")
	}
}

// ggufBuilder writes minimal GGUF files for the hostile-input tests.
type ggufBuilder struct{ bytes.Buffer }

func (b *ggufBuilder) u32(v uint32) { binary.Write(&b.Buffer, binary.LittleEndian, v) }
func (b *ggufBuilder) u64(v uint64) { binary.Write(&b.Buffer, binary.LittleEndian, v) }
func (b *ggufBuilder) str(s string) { b.u64(uint64(len(s))); b.WriteString(s) }

func header(nt, nkv uint64) *ggufBuilder {
	b := &ggufBuilder{}
	b.WriteString("GGUF")
	b.u32(3)
	b.u64(nt)
	b.u64(nkv)
	return b
}

func TestGGUFHostile(t *testing.T) {
	good := readFixture(t, "tiny.gguf")
	cases := map[string][]byte{}
	cases["empty"] = nil
	cases["truncated"] = good[:len(good)/2]
	cases["big endian"] = append([]byte("GGUF\x00\x00\x00\x03"), make([]byte, 16)...)
	cases["version 1"] = append([]byte("GGUF\x01\x00\x00\x00"), make([]byte, 16)...)
	cases["huge tensor count"] = header(1<<40, 0).Bytes()

	b := header(0, 1)
	b.str("k")
	b.u32(gString)
	b.u64(1 << 40)
	cases["huge string"] = b.Bytes()

	b = header(0, 1)
	b.str("k")
	b.u32(gArray)
	b.u32(gUint64)
	b.u64(1 << 60)
	cases["huge array"] = b.Bytes()

	b = header(0, 1)
	b.str("k")
	b.u32(99)
	cases["unknown value type"] = b.Bytes()

	b = header(1, 0)
	b.str("t")
	b.u32(2)
	b.u64(4)
	b.u64(4)
	b.u32(0)
	b.u64(1 << 20)
	cases["data outside file"] = b.Bytes()

	b = header(1, 0)
	b.str("t")
	b.u32(1)
	b.u64(32)
	b.u32(11) // Q3_K
	b.u64(0)
	cases["undefined block type"] = b.Bytes()

	b = header(1, 0)
	b.str("t")
	b.u32(1)
	b.u64(31)
	b.u32(8) // Q8_0 needs a multiple of 32
	b.u64(0)
	cases["partial block"] = b.Bytes()

	b = header(1, 0)
	b.str("t")
	b.u32(9)
	cases["rank too high"] = b.Bytes()

	b = header(2, 0)
	for i := 0; i < 2; i++ {
		b.str("t")
		b.u32(1)
		b.u64(1)
		b.u32(0)
		b.u64(0)
	}
	cases["duplicate tensor"] = b.Bytes()

	for name, in := range cases {
		if _, err := ParseGGUF(in); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func safetensors(header string, data []byte) []byte {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, uint64(len(header)))
	return append(append(b, header...), data...)
}

func TestSafetensorsHostile(t *testing.T) {
	cases := map[string][]byte{
		"empty":           nil,
		"header too long": safetensors("{}", nil)[:9],
		"bad json":        safetensors(`{"a":`, nil),
		"bad dtype":       safetensors(`{"a":{"dtype":"Q4","shape":[1],"data_offsets":[0,1]}}`, []byte{0}),
		"outside":         safetensors(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, make([]byte, 4)),
		"size mismatch":   safetensors(`{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}`, make([]byte, 8)),
		"reversed":        safetensors(`{"a":{"dtype":"U8","shape":[0],"data_offsets":[4,0]}}`, make([]byte, 8)),
		"overlap": safetensors(`{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},`+
			`"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}}`, make([]byte, 8)),
		"bad metadata": safetensors(`{"__metadata__":{"a":1}}`, nil),
	}
	big := make([]byte, 8)
	binary.LittleEndian.PutUint64(big, 1<<62)
	cases["huge header length"] = append(big, '{')
	for name, in := range cases {
		if _, err := ParseSafetensors(in); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func FuzzParseGGUF(f *testing.F) {
	f.Add(readFixture(f, "tiny.gguf"))
	f.Fuzz(func(t *testing.T, b []byte) {
		if src, err := ParseGGUF(b); err == nil {
			for _, tn := range src.Tensors {
				if err := checkTensor(tn); err != nil {
					t.Fatal(err)
				}
			}
		}
	})
}

func FuzzParseSafetensors(f *testing.F) {
	f.Add(readFixture(f, "tiny.safetensors"))
	f.Fuzz(func(t *testing.T, b []byte) {
		if src, err := ParseSafetensors(b); err == nil {
			for _, tn := range src.Tensors {
				if err := checkTensor(tn); err != nil {
					t.Fatal(err)
				}
			}
		}
	})
}
