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
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"sort"
	"sync"
	"unicode/utf8"
)

var le = binary.LittleEndian

// Preamble is the decoded fixed 64-byte header.
type Preamble struct {
	Major, Minor  uint16
	DataAlignment uint32
	IndexOffset   uint64
	IndexLength   uint64
	DataOffset    uint64
	DataLength    uint64
	FileLength    uint64
}

// Section describes one entry of the section table.
type Section struct {
	Kind    SectionKind
	Flags   uint32
	Offset  uint64
	Length  uint64
	HashAlg HashAlg
	Hash    [32]byte
}

// TensorInfo describes one tensor. Offset is relative to the start of the
// weights section.
type TensorInfo struct {
	Name   string
	DType  DType
	Shape  []uint64
	Offset uint64
	Length uint64
}

// File is a validated, read-only view of a mim file. Accessors read the
// underlying bytes in place; nothing is copied at open time except the small
// section table.
type File struct {
	data     []byte
	release  func() error
	pre      Preamble
	sections []Section
	tcount   uint32
	tensors  []byte // tensor table bytes
	pool     []byte // index string pool
}

// Parse validates b as a mim file and returns a view over it. b must stay
// alive and unmodified for as long as the File is used.
func Parse(b []byte) (*File, error) {
	f := &File{data: b}
	if err := f.parsePreamble(); err != nil {
		return nil, err
	}
	if err := f.parseIndex(); err != nil {
		return nil, err
	}
	if err := f.checkTensors(); err != nil {
		return nil, err
	}
	return f, nil
}

func (f *File) parsePreamble() error {
	b := f.data
	if len(b) < PreambleSize {
		return errf(CodeTruncated, "file is %d bytes, preamble needs %d", len(b), PreambleSize)
	}
	if m := string(b[0:4]); m != Magic && m != LegacyMagic {
		return errf(CodeBadMagic, "got %q", b[0:4])
	}
	p := &f.pre
	p.Major = le.Uint16(b[4:])
	p.Minor = le.Uint16(b[6:])
	switch bom := le.Uint32(b[8:]); bom {
	case ByteOrderMark:
	case 0x0D0C0B0A:
		return errf(CodeBigEndian, "big-endian files are not supported")
	default:
		return errf(CodeBadByteOrderMark, "got %#08x", bom)
	}
	if p.Major != FormatMajor {
		return errf(CodeUnsupportedVersion, "format %d.%d, reader supports %d.x", p.Major, p.Minor, FormatMajor)
	}
	p.DataAlignment = le.Uint32(b[12:])
	p.IndexOffset = le.Uint64(b[16:])
	p.IndexLength = le.Uint64(b[24:])
	p.DataOffset = le.Uint64(b[32:])
	p.DataLength = le.Uint64(b[40:])
	p.FileLength = le.Uint64(b[48:])
	for _, c := range b[56:64] {
		if c != 0 {
			return errf(CodeReservedNonZero, "preamble bytes 56..63")
		}
	}
	a := uint64(p.DataAlignment)
	if !isPow2(a) || a < MinDataAlignment || a > MaxDataAlignment {
		return errf(CodeBadAlignment, "data alignment %d", a)
	}
	if p.FileLength != uint64(len(b)) {
		if p.FileLength > uint64(len(b)) {
			return errf(CodeTruncated, "preamble says %d bytes, have %d", p.FileLength, len(b))
		}
		return errf(CodeBadLength, "preamble says %d bytes, have %d", p.FileLength, len(b))
	}
	if p.IndexLength > MaxIndexLength {
		return errf(CodeIndexTooLarge, "index is %d bytes, cap is %d", p.IndexLength, MaxIndexLength)
	}
	if p.IndexOffset < PreambleSize || p.IndexOffset%SectionAlignment != 0 {
		return errf(CodeMisaligned, "index offset %d", p.IndexOffset)
	}
	if !inBounds(p.IndexOffset, p.IndexLength, p.FileLength) {
		return errf(CodeOutOfBounds, "index [%d,+%d) outside file", p.IndexOffset, p.IndexLength)
	}
	if p.DataOffset%a != 0 {
		return errf(CodeMisaligned, "data offset %d not a multiple of %d", p.DataOffset, a)
	}
	if !inBounds(p.DataOffset, p.DataLength, p.FileLength) {
		return errf(CodeOutOfBounds, "weights [%d,+%d) outside file", p.DataOffset, p.DataLength)
	}
	return nil
}

func inBounds(off, n, size uint64) bool { return off <= size && n <= size-off }

func (f *File) parseIndex() error {
	p := f.pre
	idx := f.data[p.IndexOffset : p.IndexOffset+p.IndexLength]
	if len(idx) < IndexHeaderSize {
		return errf(CodeTruncated, "index header")
	}
	nsec := uint64(le.Uint32(idx[0:]))
	ntens := uint64(le.Uint32(idx[4:]))
	poolLen := le.Uint64(idx[8:])
	want := uint64(IndexHeaderSize) + nsec*SectionEntrySize + ntens*TensorEntrySize
	if poolLen > MaxIndexLength || want+poolLen != uint64(len(idx)) {
		return errf(CodeBadLength, "index length %d does not match %d sections, %d tensors, %d pool bytes", len(idx), nsec, ntens, poolLen)
	}
	secBytes := idx[IndexHeaderSize : IndexHeaderSize+nsec*SectionEntrySize]
	f.tensors = idx[IndexHeaderSize+nsec*SectionEntrySize : want]
	f.pool = idx[want:]
	f.tcount = uint32(ntens)

	type region struct {
		off, n uint64
		what   string
	}
	regions := []region{{0, PreambleSize, "preamble"}, {p.IndexOffset, p.IndexLength, "index"}}
	seen := map[SectionKind]bool{}
	f.sections = make([]Section, nsec)
	for i := range f.sections {
		e := secBytes[i*SectionEntrySize:]
		s := Section{
			Kind:    SectionKind(le.Uint32(e[0:])),
			Flags:   le.Uint32(e[4:]),
			Offset:  le.Uint64(e[8:]),
			Length:  le.Uint64(e[16:]),
			HashAlg: HashAlg(e[24]),
		}
		copy(s.Hash[:], e[32:64])
		for _, c := range e[25:32] {
			if c != 0 {
				return errf(CodeReservedNonZero, "section %d reserved bytes", i)
			}
		}
		if s.Flags&^FlagRequired != 0 {
			return errf(CodeReservedNonZero, "section %d flags %#x", i, s.Flags)
		}
		switch s.HashAlg {
		case HashNone:
			if s.Hash != ([32]byte{}) {
				return errf(CodeReservedNonZero, "section %d has a hash but no algorithm", i)
			}
		case HashSHA256:
		default:
			return errf(CodeBadHashAlg, "section %d algorithm %d", i, s.HashAlg)
		}
		if !s.Kind.known() {
			if s.Flags&FlagRequired != 0 {
				return errf(CodeUnknownRequired, "section kind %d", s.Kind)
			}
		} else {
			if seen[s.Kind] {
				return errf(CodeDuplicate, "two %s sections", s.Kind)
			}
			seen[s.Kind] = true
		}
		if !inBounds(s.Offset, s.Length, p.FileLength) {
			return errf(CodeOutOfBounds, "section %d [%d,+%d)", i, s.Offset, s.Length)
		}
		if s.Kind == SectionWeights {
			if s.Offset != p.DataOffset || s.Length != p.DataLength {
				return errf(CodeBadLength, "weights section disagrees with preamble")
			}
		} else if s.Offset%SectionAlignment != 0 {
			return errf(CodeMisaligned, "section %d offset %d", i, s.Offset)
		}
		f.sections[i] = s
		regions = append(regions, region{s.Offset, s.Length, s.Kind.String()})
	}
	if !seen[SectionWeights] {
		return errf(CodeMissingSection, "no weights section")
	}
	sort.Slice(regions, func(i, j int) bool {
		if regions[i].off != regions[j].off {
			return regions[i].off < regions[j].off
		}
		return regions[i].n < regions[j].n
	})
	for i := 1; i < len(regions); i++ {
		prev := regions[i-1]
		if regions[i].off < prev.off+prev.n {
			return errf(CodeOverlap, "%s overlaps %s", regions[i].what, prev.what)
		}
	}
	return nil
}

func (f *File) tensorRow(i int) []byte {
	return f.tensors[i*TensorEntrySize : (i+1)*TensorEntrySize]
}

func (f *File) rowName(r []byte) ([]byte, error) {
	off, n := uint64(le.Uint32(r[0:])), uint64(le.Uint32(r[4:]))
	if !inBounds(off, n, uint64(len(f.pool))) {
		return nil, errf(CodeOutOfBounds, "tensor name outside string pool")
	}
	return f.pool[off : off+n], nil
}

func (f *File) checkTensors() error {
	n := int(f.tcount)
	var prev []byte
	type span struct{ off, n uint64 }
	spans := make([]span, n)
	for i := 0; i < n; i++ {
		r := f.tensorRow(i)
		name, err := f.rowName(r)
		if err != nil {
			return err
		}
		if len(name) == 0 || len(name) > MaxNameLength || !utf8.Valid(name) {
			return errf(CodeBadName, "tensor %d name", i)
		}
		if i > 0 && bytes.Compare(prev, name) >= 0 {
			if bytes.Equal(prev, name) {
				return errf(CodeDuplicate, "tensor %q", name)
			}
			return errf(CodeUnsorted, "tensor %q after %q", name, prev)
		}
		prev = name
		ti, err := f.decodeRow(r, name)
		if err != nil {
			return err
		}
		if ti.Offset%tensorAlignment(ti.Length) != 0 {
			return errf(CodeMisaligned, "tensor %q offset %d", name, ti.Offset)
		}
		if !inBounds(ti.Offset, ti.Length, f.pre.DataLength) {
			return errf(CodeOutOfBounds, "tensor %q data", name)
		}
		spans[i] = span{ti.Offset, ti.Length}
	}
	// Canonical packing: ordered by offset, each tensor starts at the first
	// aligned position after the previous one, and the weights section ends
	// exactly at the last tensor. This rules out overlaps, holes and
	// polyglot files in one pass.
	sort.SliceStable(spans, func(i, j int) bool {
		if spans[i].off != spans[j].off {
			return spans[i].off < spans[j].off
		}
		return spans[i].n < spans[j].n
	})
	end := uint64(0)
	for _, s := range spans {
		want, _ := alignUp(end, tensorAlignment(s.n))
		if s.off != want {
			if s.off < end {
				return errf(CodeOverlap, "tensor at %d overlaps previous tensor ending at %d", s.off, end)
			}
			return errf(CodeNonCanonical, "tensor at %d, expected %d", s.off, want)
		}
		end = s.off + s.n
	}
	if end != f.pre.DataLength {
		return errf(CodeNonCanonical, "weights section is %d bytes, tensors end at %d", f.pre.DataLength, end)
	}
	return nil
}

func (f *File) decodeRow(r []byte, name []byte) (TensorInfo, error) {
	ti := TensorInfo{
		Name:   string(name),
		DType:  DType(le.Uint32(r[8:])),
		Offset: le.Uint64(r[80:]),
		Length: le.Uint64(r[88:]),
	}
	rank := int(r[12])
	if r[13] != 0 || r[14] != 0 || r[15] != 0 {
		return ti, errf(CodeReservedNonZero, "tensor %q reserved bytes", name)
	}
	if rank > MaxRank {
		return ti, errf(CodeBadShape, "tensor %q rank %d", name, rank)
	}
	ti.Shape = make([]uint64, rank)
	for d := 0; d < MaxRank; d++ {
		v := le.Uint64(r[16+8*d:])
		if d < rank {
			ti.Shape[d] = v
		} else if v != 0 {
			return ti, errf(CodeReservedNonZero, "tensor %q unused dims", name)
		}
	}
	size, err := ByteSize(ti.DType, ti.Shape)
	if err != nil {
		return ti, &Error{Code: CodeOf(err), Detail: "tensor " + string(name) + ": " + err.(*Error).Detail}
	}
	if size != ti.Length {
		return ti, errf(CodeSizeMismatch, "tensor %q: shape and dtype need %d bytes, entry says %d", name, size, ti.Length)
	}
	return ti, nil
}

// Preamble returns the decoded preamble.
func (f *File) Preamble() Preamble { return f.pre }

// Sections returns the section table.
func (f *File) Sections() []Section { return f.sections }

// Section returns the section of the given kind.
func (f *File) Section(k SectionKind) (Section, bool) {
	for _, s := range f.sections {
		if s.Kind == k {
			return s, true
		}
	}
	return Section{}, false
}

// SectionBytes returns the bytes of a section, without copying.
func (f *File) SectionBytes(s Section) []byte { return f.data[s.Offset : s.Offset+s.Length] }

// NumTensors returns the number of tensors.
func (f *File) NumTensors() int { return int(f.tcount) }

// TensorAt returns the i-th tensor in name order.
func (f *File) TensorAt(i int) TensorInfo {
	r := f.tensorRow(i)
	name, _ := f.rowName(r)
	ti, _ := f.decodeRow(r, name)
	return ti
}

// Tensor looks a tensor up by name with a binary search over the sorted table.
func (f *File) Tensor(name string) (TensorInfo, bool) {
	key := []byte(name)
	i := sort.Search(int(f.tcount), func(i int) bool {
		n, _ := f.rowName(f.tensorRow(i))
		return bytes.Compare(n, key) >= 0
	})
	if i < int(f.tcount) {
		if n, _ := f.rowName(f.tensorRow(i)); bytes.Equal(n, key) {
			return f.TensorAt(i), true
		}
	}
	return TensorInfo{}, false
}

// TensorData returns the raw bytes of a tensor, without copying. With an
// mmap-backed File the slice points into the page cache.
func (f *File) TensorData(t TensorInfo) []byte {
	base := f.pre.DataOffset + t.Offset
	return f.data[base : base+t.Length]
}

// Weights returns the whole weights section as one aligned byte range, for
// runtimes that upload or prefetch it in one call.
func (f *File) Weights() []byte {
	return f.data[f.pre.DataOffset : f.pre.DataOffset+f.pre.DataLength]
}

// Bytes returns the underlying file bytes.
func (f *File) Bytes() []byte { return f.data }

// VerifyHashes checks every section that carries a hash. It reads every byte
// of those sections, so it is not done by Parse. The sections are hashed
// side by side; each one's SHA-256 is a single stream, so the largest
// section, usually the weights, sets how long it takes.
func (f *File) VerifyHashes() error {
	bad := make([]bool, len(f.sections))
	var wg sync.WaitGroup
	for i, s := range f.sections {
		if s.HashAlg != HashSHA256 {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			bad[i] = sha256.Sum256(f.SectionBytes(s)) != s.Hash
		}()
	}
	wg.Wait()
	for i, s := range f.sections {
		if bad[i] {
			return errf(CodeHashMismatch, "%s section", s.Kind)
		}
	}
	return nil
}

// Close releases the mapping when the File came from Open.
func (f *File) Close() error {
	if f.release == nil {
		return nil
	}
	r := f.release
	f.release, f.data = nil, nil
	return r()
}
