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
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/gsoultan/mim"
)

// maxSafetensorsHeader is the header size limit of the reference
// safetensors implementation.
const maxSafetensorsHeader = 100 << 20

var safetensorsTypes = map[string]mim.DType{
	"F64": mim.F64, "F32": mim.F32, "F16": mim.F16, "BF16": mim.BF16,
	"I64": mim.I64, "I32": mim.I32, "I16": mim.I16, "I8": mim.I8,
	"U64": mim.U64, "U32": mim.U32, "U16": mim.U16, "U8": mim.U8,
	"BOOL": mim.Bool, "F8_E4M3": mim.F8E4M3, "F8_E5M2": mim.F8E5M2,
}

type stEntry struct {
	DType   string   `json:"dtype"`
	Shape   []uint64 `json:"shape"`
	Offsets []uint64 `json:"data_offsets"`
}

// ParseSafetensors parses a safetensors file. The returned tensors alias b.
// The free-form `__metadata__` strings are kept under the key prefix
// `safetensors.`.
func ParseSafetensors(b []byte) (*Source, error) {
	if len(b) < 8 {
		return nil, fmt.Errorf("safetensors: truncated header")
	}
	n := binary.LittleEndian.Uint64(b)
	if n > maxSafetensorsHeader || n > uint64(len(b))-8 {
		return nil, fmt.Errorf("safetensors: header length %d is invalid", n)
	}
	head := b[8 : 8+n]
	data := b[8+n:]
	var raw map[string]json.RawMessage
	dec := json.NewDecoder(bytes.NewReader(head))
	if err := dec.Decode(&raw); err != nil {
		return nil, fmt.Errorf("safetensors: header: %w", err)
	}
	src := &Source{Format: "safetensors", Metadata: mim.Metadata{}}
	type span struct {
		name       string
		begin, end uint64
	}
	var spans []span
	for name, msg := range raw {
		if name == "__metadata__" {
			var m map[string]string
			if err := json.Unmarshal(msg, &m); err != nil {
				return nil, fmt.Errorf("safetensors: __metadata__ must map strings to strings: %w", err)
			}
			for k, v := range m {
				if k == "" {
					return nil, fmt.Errorf("safetensors: empty metadata key")
				}
				src.Metadata["safetensors."+k] = v
			}
			continue
		}
		var e stEntry
		if err := json.Unmarshal(msg, &e); err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		dt, ok := safetensorsTypes[e.DType]
		if !ok {
			return nil, fmt.Errorf("safetensors: tensor %q has unsupported dtype %q", name, e.DType)
		}
		if len(e.Offsets) != 2 || e.Offsets[0] > e.Offsets[1] || e.Offsets[1] > uint64(len(data)) {
			return nil, fmt.Errorf("safetensors: tensor %q has invalid data_offsets %v", name, e.Offsets)
		}
		if e.Shape == nil {
			e.Shape = []uint64{}
		}
		t := Tensor{Name: name, DType: dt, Shape: e.Shape, Data: data[e.Offsets[0]:e.Offsets[1]]}
		if err := checkTensor(t); err != nil {
			return nil, fmt.Errorf("safetensors: %w", err)
		}
		src.Tensors = append(src.Tensors, t)
		spans = append(spans, span{name, e.Offsets[0], e.Offsets[1]})
	}
	// Tensors must not overlap. Holes are tolerated: they cost space, not
	// safety.
	sort.Slice(spans, func(i, j int) bool { return spans[i].begin < spans[j].begin })
	for i := 1; i < len(spans); i++ {
		if spans[i].begin < spans[i-1].end {
			return nil, fmt.Errorf("safetensors: tensors %q and %q overlap", spans[i-1].name, spans[i].name)
		}
	}
	sort.Slice(src.Tensors, func(i, j int) bool { return src.Tensors[i].Name < src.Tensors[j].Name })
	return src, nil
}
