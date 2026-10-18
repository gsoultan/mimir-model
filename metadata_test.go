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

package mim

import (
	"reflect"
	"testing"
)

func TestMetadataRoundTrip(t *testing.T) {
	m := Metadata{
		KeyName:            "tiny",
		KeyArchitecture:    "llama",
		"general.layers":   int64(32),
		"general.vocab":    uint64(32000),
		"rope.theta":       10000.0,
		"general.chat":     true,
		"blob":             []byte{1, 2, 3},
		KeyTokenizerTokens: []string{"<s>", "</s>", "héllo", ""},
		KeyTokenizerScores: []float64{0, -1.5, 2},
		"ids":              []int64{-1, 0, 1 << 40},
	}
	b, err := m.Encode()
	must(t, err)
	got, err := DecodeMetadata(b)
	must(t, err)
	if !reflect.DeepEqual(got, m) {
		t.Fatalf("round trip mismatch:\n%v\n%v", got, m)
	}
	// Reject corruption of every byte: no panic and an error code.
	for i := range b {
		c := append([]byte(nil), b...)
		c[i] ^= 0xFF
		if _, err := DecodeMetadata(c); err != nil && CodeOf(err) == "" {
			t.Fatalf("byte %d: error without code: %v", i, err)
		}
	}
}

func TestMetadataInFile(t *testing.T) {
	w := sampleWriter(t)
	mb, err := Metadata{KeyName: "x", KeySplitCount: int64(1)}.Encode()
	must(t, err)
	must(t, w.AddSection(SectionMetadata, false, mb))
	f, err := Parse(encode(t, w))
	must(t, err)
	m, err := f.Metadata()
	must(t, err)
	if m[KeyName] != "x" {
		t.Fatalf("got %v", m)
	}
	if _, err := (Metadata{"": 1}).Encode(); CodeOf(err) != CodeBadMetadata {
		t.Fatal("empty key accepted")
	}
	if _, err := (Metadata{"k": 1.5}).Encode(); err != nil {
		t.Fatal(err)
	}
	if _, err := (Metadata{"k": struct{}{}}).Encode(); CodeOf(err) != CodeBadMetadata {
		t.Fatal("bad type accepted")
	}
}

func FuzzDecodeMetadata(f *testing.F) {
	b, _ := Metadata{"a": int64(1), "b": []string{"x"}, "c": []int64{1, 2}}.Encode()
	f.Add(b)
	f.Fuzz(func(t *testing.T, b []byte) {
		m, err := DecodeMetadata(b)
		if err != nil {
			if CodeOf(err) == "" {
				t.Fatal(err)
			}
			return
		}
		if _, err := m.Encode(); err != nil {
			t.Fatalf("re-encode of decoded metadata failed: %v", err)
		}
	})
}
