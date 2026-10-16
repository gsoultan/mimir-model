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
	"encoding/binary"
	"testing"
)

// hostileCases mutates a valid file and checks the reader rejects it with the
// expected code. The same cases are written to the conformance corpus.
func hostileCases(t testing.TB) []struct {
	name string
	code Code
	data []byte
} {
	base := encode(t, sampleWriter(t))
	le := binary.LittleEndian
	const idx = PreambleSize
	const sec0 = idx + IndexHeaderSize
	row := func(i int) int { return sec0 + 1*SectionEntrySize + i*TensorEntrySize }
	mut := func(f func(b []byte) []byte) []byte {
		b := append([]byte(nil), base...)
		return f(b)
	}
	return []struct {
		name string
		code Code
		data []byte
	}{
		{"truncated-preamble", CodeTruncated, base[:40]},
		{"bad-magic", CodeBadMagic, mut(func(b []byte) []byte { b[0] = 'X'; return b })},
		{"big-endian", CodeBigEndian, mut(func(b []byte) []byte { binary.BigEndian.PutUint32(b[8:], ByteOrderMark); return b })},
		{"major-version", CodeUnsupportedVersion, mut(func(b []byte) []byte { le.PutUint16(b[4:], 1); return b })},
		{"reserved-preamble", CodeReservedNonZero, mut(func(b []byte) []byte { b[60] = 1; return b })},
		{"alignment-not-pow2", CodeBadAlignment, mut(func(b []byte) []byte { le.PutUint32(b[12:], 5000); return b })},
		{"truncated-file", CodeTruncated, base[:len(base)-1]},
		{"trailing-bytes", CodeBadLength, append(append([]byte(nil), base...), 0)},
		{"index-too-large", CodeIndexTooLarge, mut(func(b []byte) []byte { le.PutUint64(b[24:], MaxIndexLength+1); return b })},
		{"index-out-of-bounds", CodeOutOfBounds, mut(func(b []byte) []byte { le.PutUint64(b[16:], (uint64(len(b))+7)&^7); return b })},
		{"index-length-mismatch", CodeBadLength, mut(func(b []byte) []byte { le.PutUint32(b[idx+4:], 7); return b })},
		{"weights-disagree", CodeBadLength, mut(func(b []byte) []byte { le.PutUint64(b[sec0+16:], 0); return b })},
		{"unknown-required-section", CodeUnknownRequired, mut(func(b []byte) []byte { le.PutUint32(b[sec0:], 99); return b })},
		{"bad-hash-alg", CodeBadHashAlg, mut(func(b []byte) []byte { b[sec0+24] = 9; return b })},
		{"section-flags", CodeReservedNonZero, mut(func(b []byte) []byte { le.PutUint32(b[sec0+4:], 3); return b })},
		{"name-out-of-pool", CodeOutOfBounds, mut(func(b []byte) []byte { le.PutUint32(b[row(0)+4:], 1<<30); return b })},
		{"unsorted-names", CodeUnsorted, mut(func(b []byte) []byte {
			r0, r1 := row(0), row(1)
			tmp := append([]byte(nil), b[r0:r0+TensorEntrySize]...)
			copy(b[r0:], b[r1:r1+TensorEntrySize])
			copy(b[r1:], tmp)
			return b
		})},
		{"bad-dtype", CodeBadDType, mut(func(b []byte) []byte { le.PutUint32(b[row(0)+8:], 999); return b })},
		{"rank-too-big", CodeBadShape, mut(func(b []byte) []byte { b[row(0)+12] = 9; return b })},
		{"size-mismatch", CodeSizeMismatch, mut(func(b []byte) []byte { le.PutUint64(b[row(0)+16:], 5); return b })},
		{"shape-overflow", CodeOverflow, mut(func(b []byte) []byte {
			b[row(0)+12] = 2
			le.PutUint64(b[row(0)+16:], 1<<40)
			le.PutUint64(b[row(0)+24:], 1<<40)
			return b
		})},
		{"misaligned-tensor", CodeMisaligned, mut(func(b []byte) []byte { le.PutUint64(b[row(0)+80:], le.Uint64(b[row(0)+80:])+4); return b })},
		{"tensor-out-of-bounds", CodeOutOfBounds, mut(func(b []byte) []byte { le.PutUint64(b[row(0)+80:], 1<<40); return b })},
		{"overlapping-tensors", CodeOverlap, mut(func(b []byte) []byte {
			// Point the second tensor in data order at the first one's bytes.
			le.PutUint64(b[row(0)+80:], 0)
			le.PutUint64(b[row(1)+80:], 0)
			return b
		})},
		{"hole-in-weights", CodeNonCanonical, mut(func(b []byte) []byte {
			// Grow the weights section by one alignment unit with no tensor in it.
			p := le.Uint64(b[40:]) + 64
			le.PutUint64(b[40:], p)
			le.PutUint64(b[sec0+16:], p)
			le.PutUint64(b[48:], le.Uint64(b[48:])+64)
			return append(b, make([]byte, 64)...)
		})},
	}
}

func TestHostileFiles(t *testing.T) {
	for _, c := range hostileCases(t) {
		t.Run(c.name, func(t *testing.T) {
			_, err := Parse(c.data)
			if err == nil {
				t.Fatal("accepted a hostile file")
			}
			if got := CodeOf(err); got != c.code {
				t.Fatalf("got %s (%v), want %s", got, err, c.code)
			}
		})
	}
}

func TestHashMismatch(t *testing.T) {
	b := encode(t, sampleWriter(t))
	f, err := Parse(b)
	must(t, err)
	b[len(b)-1] ^= 1
	if CodeOf(f.VerifyHashes()) != CodeHashMismatch {
		t.Fatal("corruption not detected")
	}
}

func FuzzParse(f *testing.F) {
	f.Add(encode(f, sampleWriter(f)))
	f.Add(encode(f, NewWriter()))
	for _, c := range hostileCases(f) {
		f.Add(c.data)
	}
	f.Fuzz(func(t *testing.T, b []byte) {
		file, err := Parse(b)
		if err != nil {
			if CodeOf(err) == "" {
				t.Fatalf("error without code: %v", err)
			}
			return
		}
		for i := 0; i < file.NumTensors(); i++ {
			ti := file.TensorAt(i)
			_ = file.TensorData(ti)
			if got, ok := file.Tensor(ti.Name); !ok || got.Offset != ti.Offset {
				t.Fatalf("lookup of %q failed", ti.Name)
			}
		}
		_ = file.VerifyHashes()
	})
}
