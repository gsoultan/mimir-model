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

// Layout constants. Every number here is normative; see spec/container.md.
const (
	// Magic is the first four bytes of every mmf file.
	Magic = "MMF\x00"
	// ByteOrderMark is written as a little-endian u32 at offset 8. A reader
	// that sees 0x0D0C0B0A is looking at a big-endian file, which v0 forbids.
	ByteOrderMark uint32 = 0x0A0B0C0D

	FormatMajor uint16 = 0
	FormatMinor uint16 = 1

	PreambleSize     = 64
	IndexHeaderSize  = 16
	SectionEntrySize = 64
	TensorEntrySize  = 96

	MaxIndexLength = 256 << 20
	MaxRank        = 8
	MaxNameLength  = 4096

	DefaultDataAlignment = 16 << 10
	MinDataAlignment     = 4 << 10
	MaxDataAlignment     = 16 << 20

	TensorAlignment      = 64
	LargeTensorAlignment = 4 << 10
	LargeTensorThreshold = 1 << 20

	SectionAlignment = 8
)

// SectionKind identifies what a section holds.
type SectionKind uint32

const (
	SectionWeights     SectionKind = 1
	SectionMetadata    SectionKind = 2
	SectionGraph       SectionKind = 3
	SectionRepackCache SectionKind = 4 // reserved, layout not yet specified
	SectionTraining    SectionKind = 5 // reserved, layout not yet specified
)

func (k SectionKind) known() bool { return k >= SectionWeights && k <= SectionTraining }

func (k SectionKind) String() string {
	switch k {
	case SectionWeights:
		return "weights"
	case SectionMetadata:
		return "metadata"
	case SectionGraph:
		return "graph"
	case SectionRepackCache:
		return "repack-cache"
	case SectionTraining:
		return "training"
	}
	return "unknown"
}

// Section flags.
const (
	// FlagRequired marks a section a reader must understand. A reader that
	// does not know a required section's kind must reject the file.
	FlagRequired uint32 = 1
)

// HashAlg identifies the per-section hash algorithm.
type HashAlg uint8

const (
	HashNone   HashAlg = 0
	HashSHA256 HashAlg = 1
)

// tensorAlignment returns the alignment a tensor of the given byte length
// must start on, relative to the start of the weights section.
func tensorAlignment(length uint64) uint64 {
	if length >= LargeTensorThreshold {
		return LargeTensorAlignment
	}
	return TensorAlignment
}

func alignUp(n, a uint64) (uint64, bool) {
	r := (n + a - 1) &^ (a - 1)
	return r, r >= n
}

func isPow2(n uint64) bool { return n != 0 && n&(n-1) == 0 }
