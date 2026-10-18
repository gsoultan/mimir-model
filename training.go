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
	"math"
	"unicode/utf8"
)

// Training is the training section (spec/training.md): what a framework
// needs, besides the weights, to resume training where it stopped. The
// optimizer state is held in tensors of the file's tensor table; the
// section names them and ties each to the parameter it belongs to.
type Training struct {
	// Optimizer names the optimizer, such as "adamw"; empty if unspecified.
	Optimizer string
	// Counters are progress counters, such as "step" and "epoch".
	Counters []TrainCounter
	// Hyper are the optimizer's hyperparameters, per parameter group.
	Hyper []TrainHyper
	// Params are the trainable tensors with their optimizer state.
	Params []TrainParam
	// Extras are state tensors not tied to one parameter, such as the
	// random generator's state.
	Extras []TrainExtra
}

// TrainCounter is a named progress counter.
type TrainCounter struct {
	Name  string
	Value uint64
}

// TrainHyper is a hyperparameter of one parameter group.
type TrainHyper struct {
	Name  string
	Group uint32
	Value float64
}

// TrainParam is a trainable tensor, its parameter group and its optimizer
// state.
type TrainParam struct {
	Tensor string
	Group  uint32
	States []TrainState
}

// TrainState is one slot of a parameter's optimizer state, such as Adam's
// "m", held in the named tensor.
type TrainState struct {
	Slot   string
	Tensor string
}

// TrainExtra is a named state tensor not tied to one parameter.
type TrainExtra struct {
	Name   string
	Tensor string
}

const (
	trainingMagic      = "MIMT"
	trainingMajor      = 0
	trainingMinor      = 1
	trainingTables     = 7
	trainingHeaderSize = 16 + 16*trainingTables
	maxTrainingRows    = 1 << 28
)

// Row sizes of the training tables, in directory order.
var trainingRowSize = [trainingTables]uint64{1, 8, 16, 16, 16, 8, 8}

const (
	ttPool = iota
	ttStrings
	ttCounters
	ttHyper
	ttParams
	ttStates
	ttExtras
)

// Encode validates t and returns it as a training section.
func (t *Training) Encode() ([]byte, error) {
	var pool []byte
	var strs []uint32 // (offset, length) pairs
	index := map[string]uint32{}
	str := func(s string) uint32 {
		if i, ok := index[s]; ok {
			return i
		}
		i := uint32(len(strs) / 2)
		index[s] = i
		strs = append(strs, uint32(len(pool)), uint32(len(s)))
		pool = append(pool, s...)
		return i
	}
	str("")
	opt := str(t.Optimizer)
	var tables [trainingTables][]byte
	for _, c := range t.Counters {
		r := make([]byte, 16)
		le.PutUint32(r, str(c.Name))
		le.PutUint64(r[8:], c.Value)
		tables[ttCounters] = append(tables[ttCounters], r...)
	}
	for _, h := range t.Hyper {
		r := make([]byte, 16)
		le.PutUint32(r, str(h.Name))
		le.PutUint32(r[4:], h.Group)
		le.PutUint64(r[8:], math.Float64bits(h.Value))
		tables[ttHyper] = append(tables[ttHyper], r...)
	}
	nstates := 0
	for _, p := range t.Params {
		r := make([]byte, 16)
		le.PutUint32(r, str(p.Tensor))
		le.PutUint32(r[4:], p.Group)
		le.PutUint32(r[8:], uint32(nstates))
		le.PutUint32(r[12:], uint32(len(p.States)))
		tables[ttParams] = append(tables[ttParams], r...)
		for _, s := range p.States {
			r := make([]byte, 8)
			le.PutUint32(r, str(s.Slot))
			le.PutUint32(r[4:], str(s.Tensor))
			tables[ttStates] = append(tables[ttStates], r...)
		}
		nstates += len(p.States)
	}
	for _, x := range t.Extras {
		r := make([]byte, 8)
		le.PutUint32(r, str(x.Name))
		le.PutUint32(r[4:], str(x.Tensor))
		tables[ttExtras] = append(tables[ttExtras], r...)
	}
	tables[ttPool] = pool
	for _, v := range strs {
		tables[ttStrings] = le.AppendUint32(tables[ttStrings], v)
	}

	out := make([]byte, trainingHeaderSize)
	copy(out, trainingMagic)
	le.PutUint16(out[4:], trainingMajor)
	le.PutUint16(out[6:], trainingMinor)
	le.PutUint32(out[8:], opt)
	for i, tb := range tables {
		for len(out)%8 != 0 {
			out = append(out, 0)
		}
		le.PutUint64(out[16+16*i:], uint64(len(out)))
		le.PutUint64(out[24+16*i:], uint64(len(tb))/trainingRowSize[i])
		out = append(out, tb...)
	}
	// Decoding checks every rule of spec/training.md, so t is valid
	// exactly when its encoding is.
	if _, err := DecodeTraining(out); err != nil {
		return nil, err
	}
	return out, nil
}

// DecodeTraining parses and validates a training section on its own. The
// tensors it names are checked against a file's tensor table by
// File.Training.
func DecodeTraining(b []byte) (*Training, error) {
	if len(b) < trainingHeaderSize || string(b[:4]) != trainingMagic {
		return nil, errf(CodeBadTraining, "missing header")
	}
	if major := le.Uint16(b[4:]); major != trainingMajor {
		return nil, errf(CodeBadTraining, "format major %d, want %d", major, trainingMajor)
	}
	if le.Uint32(b[12:]) != 0 {
		return nil, errf(CodeBadTraining, "reserved header bytes are not zero")
	}
	var tables [trainingTables][]byte
	for i := range tables {
		off, n := le.Uint64(b[16+16*i:]), le.Uint64(b[24+16*i:])
		if off%8 != 0 || n > maxTrainingRows || !inBounds(off, n*trainingRowSize[i], uint64(len(b))) {
			return nil, errf(CodeBadTraining, "table %d (offset %d, %d rows) outside the section or misaligned", i, off, n)
		}
		tables[i] = b[off : off+n*trainingRowSize[i]]
	}
	pool := tables[ttPool]
	nstr := uint32(len(tables[ttStrings]) / 8)
	strs := make([]string, nstr)
	for i := range strs {
		r := tables[ttStrings][8*i:]
		off, n := uint64(le.Uint32(r)), uint64(le.Uint32(r[4:]))
		if !inBounds(off, n, uint64(len(pool))) || n > MaxNameLength || !utf8.Valid(pool[off:off+n]) {
			return nil, errf(CodeBadTraining, "string %d is outside the pool, too long or not UTF-8", i)
		}
		strs[i] = string(pool[off : off+n])
	}
	if nstr == 0 || strs[0] != "" {
		return nil, errf(CodeBadTraining, "string 0 is not the empty string")
	}
	// name returns string i, which must not be empty unless empty is true.
	name := func(i uint32, empty bool, what string) (string, error) {
		if i >= nstr {
			return "", errf(CodeBadTraining, "%s: string %d of %d", what, i, nstr)
		}
		if strs[i] == "" && !empty {
			return "", errf(CodeBadTraining, "%s: empty name", what)
		}
		return strs[i], nil
	}

	t := &Training{}
	var err error
	if t.Optimizer, err = name(le.Uint32(b[8:]), true, "optimizer"); err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	for r := tables[ttCounters]; len(r) > 0; r = r[16:] {
		c := TrainCounter{Value: le.Uint64(r[8:])}
		if c.Name, err = name(le.Uint32(r), false, "counter"); err != nil {
			return nil, err
		}
		if le.Uint32(r[4:]) != 0 {
			return nil, errf(CodeBadTraining, "counter %q: reserved bytes are not zero", c.Name)
		}
		if seen[c.Name] {
			return nil, errf(CodeBadTraining, "counter %q appears twice", c.Name)
		}
		seen[c.Name] = true
		t.Counters = append(t.Counters, c)
	}
	type hkey struct {
		name  string
		group uint32
	}
	hseen := map[hkey]bool{}
	for r := tables[ttHyper]; len(r) > 0; r = r[16:] {
		h := TrainHyper{Group: le.Uint32(r[4:]), Value: math.Float64frombits(le.Uint64(r[8:]))}
		if h.Name, err = name(le.Uint32(r), false, "hyperparameter"); err != nil {
			return nil, err
		}
		k := hkey{h.Name, h.Group}
		if hseen[k] {
			return nil, errf(CodeBadTraining, "hyperparameter %q of group %d appears twice", h.Name, h.Group)
		}
		hseen[k] = true
		t.Hyper = append(t.Hyper, h)
	}
	// Every tensor the section names is named once: as a parameter, a
	// state or an extra.
	tensors := map[string]bool{}
	tensor := func(i uint32, what string) (string, error) {
		s, err := name(i, false, what)
		if err != nil {
			return "", err
		}
		if tensors[s] {
			return "", errf(CodeBadTraining, "%s: tensor %q is named twice", what, s)
		}
		tensors[s] = true
		return s, nil
	}
	states := tables[ttStates]
	nstates := uint32(len(states) / 8)
	next := uint32(0)
	for r := tables[ttParams]; len(r) > 0; r = r[16:] {
		p := TrainParam{Group: le.Uint32(r[4:])}
		if p.Tensor, err = tensor(le.Uint32(r), "parameter"); err != nil {
			return nil, err
		}
		first, n := le.Uint32(r[8:]), le.Uint32(r[12:])
		if first != next || n > nstates-first {
			return nil, errf(CodeBadTraining, "parameter %q: states %d to %d do not follow the previous parameter's", p.Tensor, first, uint64(first)+uint64(n))
		}
		next += n
		slots := map[string]bool{}
		for j := first; j < first+n; j++ {
			s := states[8*j:]
			var st TrainState
			if st.Slot, err = name(le.Uint32(s), false, "state slot"); err != nil {
				return nil, err
			}
			if slots[st.Slot] {
				return nil, errf(CodeBadTraining, "parameter %q: slot %q appears twice", p.Tensor, st.Slot)
			}
			slots[st.Slot] = true
			if st.Tensor, err = tensor(le.Uint32(s[4:]), "state"); err != nil {
				return nil, err
			}
			p.States = append(p.States, st)
		}
		t.Params = append(t.Params, p)
	}
	if next != nstates {
		return nil, errf(CodeBadTraining, "%d states belong to no parameter", nstates-next)
	}
	xseen := map[string]bool{}
	for r := tables[ttExtras]; len(r) > 0; r = r[8:] {
		var x TrainExtra
		if x.Name, err = name(le.Uint32(r), false, "extra"); err != nil {
			return nil, err
		}
		if xseen[x.Name] {
			return nil, errf(CodeBadTraining, "extra %q appears twice", x.Name)
		}
		xseen[x.Name] = true
		if x.Tensor, err = tensor(le.Uint32(r[4:]), "extra"); err != nil {
			return nil, err
		}
		t.Extras = append(t.Extras, x)
	}
	return t, nil
}

// Training decodes the file's training section and checks that every
// tensor it names is in the tensor table, or returns nil when the file
// has no training section.
func (f *File) Training() (*Training, error) {
	s, ok := f.Section(SectionTraining)
	if !ok {
		return nil, nil
	}
	t, err := DecodeTraining(f.SectionBytes(s))
	if err != nil {
		return nil, err
	}
	check := func(name string) error {
		if _, ok := f.Tensor(name); !ok {
			return errf(CodeBadTraining, "tensor %q is not in the tensor table", name)
		}
		return nil
	}
	for _, p := range t.Params {
		if err := check(p.Tensor); err != nil {
			return nil, err
		}
		for _, st := range p.States {
			if err := check(st.Tensor); err != nil {
				return nil, err
			}
		}
	}
	for _, x := range t.Extras {
		if err := check(x.Tensor); err != nil {
			return nil, err
		}
	}
	return t, nil
}
