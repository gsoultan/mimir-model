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

package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gsoultan/mim"
)

func TestInspectRuns(t *testing.T) {
	dir := t.TempDir()
	w := mim.NewWriter()
	if err := w.AddTensor("a", mim.F32, []uint64{2}, make([]byte, 8)); err != nil {
		t.Fatal(err)
	}
	mb, _ := mim.Metadata{mim.KeyName: "t"}.Encode()
	if err := w.AddSection(mim.SectionMetadata, false, mb); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "t.mim")
	if err := w.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := inspect(&out, path, true, false, true); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "hashes ok") || !strings.Contains(out.String(), "general.name = t") {
		t.Fatalf("unexpected output:\n%s", out.String())
	}
}
