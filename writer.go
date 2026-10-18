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

package mlm

import (
	"bufio"
	"crypto/sha256"
	"io"
	"os"
	"path/filepath"
	"sort"
	"unicode/utf8"
)

// Writer builds a mim file. Tensor data is referenced, not copied, so a
// writer can be fed slices of an mmap'd source file.
type Writer struct {
	DataAlignment uint32 // 0 means DefaultDataAlignment
	Hash          bool   // write SHA-256 hashes for every section

	tensors  []wtensor
	names    map[string]bool
	sections []wsection
}

type wtensor struct {
	TensorInfo
	data []byte
}

type wsection struct {
	kind  SectionKind
	flags uint32
	data  []byte
}

// NewWriter returns a writer with default alignment and hashing on.
func NewWriter() *Writer {
	return &Writer{Hash: true, names: map[string]bool{}}
}

// AddTensor adds a tensor. Data is stored in the order tensors are added; the
// index is sorted by name.
func (w *Writer) AddTensor(name string, dt DType, shape []uint64, data []byte) error {
	if len(name) == 0 || len(name) > MaxNameLength || !utf8.ValidString(name) {
		return errf(CodeBadName, "%q", name)
	}
	if w.names[name] {
		return errf(CodeDuplicate, "tensor %q", name)
	}
	if len(shape) > MaxRank {
		return errf(CodeBadShape, "tensor %q rank %d", name, len(shape))
	}
	size, err := ByteSize(dt, shape)
	if err != nil {
		return err
	}
	if size != uint64(len(data)) {
		return errf(CodeSizeMismatch, "tensor %q needs %d bytes, got %d", name, size, len(data))
	}
	w.names[name] = true
	w.tensors = append(w.tensors, wtensor{TensorInfo{Name: name, DType: dt, Shape: append([]uint64(nil), shape...), Length: size}, data})
	return nil
}

// AddSection adds an encoded section (metadata, graph, or another kind).
func (w *Writer) AddSection(kind SectionKind, required bool, data []byte) error {
	if kind == SectionWeights {
		return errf(CodeDuplicate, "the weights section is written from tensors")
	}
	for _, s := range w.sections {
		if s.kind == kind {
			return errf(CodeDuplicate, "two %s sections", kind)
		}
	}
	var flags uint32
	if required {
		flags = FlagRequired
	}
	w.sections = append(w.sections, wsection{kind, flags, data})
	return nil
}

// WriteTo writes the file.
func (w *Writer) WriteTo(out io.Writer) (int64, error) {
	align := uint64(w.DataAlignment)
	if align == 0 {
		align = DefaultDataAlignment
	}
	if !isPow2(align) || align < MinDataAlignment || align > MaxDataAlignment {
		return 0, errf(CodeBadAlignment, "data alignment %d", align)
	}

	// Weights layout: canonical packing in insertion order.
	end := uint64(0)
	for i := range w.tensors {
		t := &w.tensors[i]
		t.Offset, _ = alignUp(end, tensorAlignment(t.Length))
		end = t.Offset + t.Length
	}
	dataLen := end

	// Index: string pool holds tensor names in sorted order.
	order := make([]int, len(w.tensors))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return w.tensors[order[a]].Name < w.tensors[order[b]].Name })
	var pool []byte
	nameOff := make([]uint32, len(w.tensors))
	for _, i := range order {
		nameOff[i] = uint32(len(pool))
		pool = append(pool, w.tensors[i].Name...)
	}
	nsec := len(w.sections) + 1
	indexLen := uint64(IndexHeaderSize + nsec*SectionEntrySize + len(w.tensors)*TensorEntrySize + len(pool))
	if indexLen > MaxIndexLength {
		return 0, errf(CodeIndexTooLarge, "index would be %d bytes", indexLen)
	}

	// Section placement after the index, 64-byte aligned.
	pos, _ := alignUp(PreambleSize+indexLen, 64)
	offs := make([]uint64, len(w.sections))
	for i, s := range w.sections {
		offs[i] = pos
		pos, _ = alignUp(pos+uint64(len(s.data)), 64)
	}
	dataOff, _ := alignUp(pos, align)
	fileLen := dataOff + dataLen

	// Hashes.
	hashes := make([][32]byte, len(w.sections))
	var wh [32]byte
	if w.Hash {
		for i, s := range w.sections {
			hashes[i] = sha256.Sum256(s.data)
		}
		h := sha256.New()
		if err := w.writeWeights(h); err != nil {
			return 0, err
		}
		copy(wh[:], h.Sum(nil))
	}

	// Assemble header bytes.
	head := make([]byte, PreambleSize+indexLen)
	copy(head, Magic)
	le.PutUint16(head[4:], FormatMajor)
	le.PutUint16(head[6:], FormatMinor)
	le.PutUint32(head[8:], ByteOrderMark)
	le.PutUint32(head[12:], uint32(align))
	le.PutUint64(head[16:], PreambleSize)
	le.PutUint64(head[24:], indexLen)
	le.PutUint64(head[32:], dataOff)
	le.PutUint64(head[40:], dataLen)
	le.PutUint64(head[48:], fileLen)

	idx := head[PreambleSize:]
	le.PutUint32(idx[0:], uint32(nsec))
	le.PutUint32(idx[4:], uint32(len(w.tensors)))
	le.PutUint64(idx[8:], uint64(len(pool)))
	putSection := func(e []byte, kind SectionKind, flags uint32, off, n uint64, h [32]byte) {
		le.PutUint32(e[0:], uint32(kind))
		le.PutUint32(e[4:], flags)
		le.PutUint64(e[8:], off)
		le.PutUint64(e[16:], n)
		if w.Hash {
			e[24] = byte(HashSHA256)
			copy(e[32:], h[:])
		}
	}
	e := idx[IndexHeaderSize:]
	putSection(e, SectionWeights, FlagRequired, dataOff, dataLen, wh)
	for i, s := range w.sections {
		putSection(e[(i+1)*SectionEntrySize:], s.kind, s.flags, offs[i], uint64(len(s.data)), hashes[i])
	}
	trows := idx[IndexHeaderSize+nsec*SectionEntrySize:]
	for row, i := range order {
		t := w.tensors[i]
		r := trows[row*TensorEntrySize:]
		le.PutUint32(r[0:], nameOff[i])
		le.PutUint32(r[4:], uint32(len(t.Name)))
		le.PutUint32(r[8:], uint32(t.DType))
		r[12] = byte(len(t.Shape))
		for d, v := range t.Shape {
			le.PutUint64(r[16+8*d:], v)
		}
		le.PutUint64(r[80:], t.Offset)
		le.PutUint64(r[88:], t.Length)
	}
	copy(trows[len(w.tensors)*TensorEntrySize:], pool)

	cw := &countWriter{w: out}
	if _, err := cw.Write(head); err != nil {
		return cw.n, err
	}
	for i, s := range w.sections {
		if err := cw.pad(offs[i]); err != nil {
			return cw.n, err
		}
		if _, err := cw.Write(s.data); err != nil {
			return cw.n, err
		}
	}
	if err := cw.pad(dataOff); err != nil {
		return cw.n, err
	}
	if err := w.writeWeights(cw); err != nil {
		return cw.n, err
	}
	return cw.n, nil
}

func (w *Writer) writeWeights(out io.Writer) error {
	cw, ok := out.(*countWriter)
	base := int64(0)
	if !ok {
		cw = &countWriter{w: out}
	} else {
		base = cw.n
	}
	for _, t := range w.tensors {
		if err := cw.pad(uint64(base) + t.Offset); err != nil {
			return err
		}
		if _, err := cw.Write(t.data); err != nil {
			return err
		}
	}
	return nil
}

type countWriter struct {
	w io.Writer
	n int64
}

func (c *countWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n += int64(n)
	return n, err
}

var zeros [4096]byte

func (c *countWriter) pad(to uint64) error {
	for uint64(c.n) < to {
		n := to - uint64(c.n)
		if n > uint64(len(zeros)) {
			n = uint64(len(zeros))
		}
		if _, err := c.Write(zeros[:n]); err != nil {
			return err
		}
	}
	return nil
}

// WriteFile writes the file atomically: to a temporary file in the same
// directory, fsync, then rename over path.
func (w *Writer) WriteFile(path string) (err error) {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".mim-*")
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			tmp.Close()
			os.Remove(tmp.Name())
		}
	}()
	bw := bufio.NewWriterSize(tmp, 1<<20)
	if _, err = w.WriteTo(bw); err != nil {
		return err
	}
	if err = bw.Flush(); err != nil {
		return err
	}
	if err = tmp.Sync(); err != nil {
		return err
	}
	if err = tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}
