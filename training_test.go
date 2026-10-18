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
	"reflect"
	"testing"
)

// sampleTraining is AdamW state for two parameters of sampleWriter's
// file, with the random generator's state as an extra.
func sampleTraining() *Training {
	return &Training{
		Optimizer: "adamw",
		Counters:  []TrainCounter{{"step", 1200}, {"epoch", 3}},
		Hyper: []TrainHyper{
			{"lr", 0, 3e-4}, {"beta1", 0, 0.9}, {"beta2", 0, 0.999}, {"eps", 0, 1e-8},
			{"weight_decay", 0, 0.1}, {"weight_decay", 1, 0},
		},
		Params: []TrainParam{
			{Tensor: "layer.1.w", Group: 0, States: []TrainState{{"m", "layer.1.w@m"}, {"v", "layer.1.w@v"}}},
			{Tensor: "bias", Group: 1, States: []TrainState{{"m", "bias@m"}, {"v", "bias@v"}}},
		},
		Extras: []TrainExtra{{"rng", "train.rng"}},
	}
}

// trainingWriter is sampleWriter with the state tensors and the training
// section of sampleTraining.
func trainingWriter(t testing.TB, tr *Training) *Writer {
	w := sampleWriter(t)
	for _, n := range []string{"layer.1.w@m", "layer.1.w@v"} {
		must(t, w.AddTensor(n, F32, []uint64{2, 3}, make([]byte, 24)))
	}
	for _, n := range []string{"bias@m", "bias@v"} {
		must(t, w.AddTensor(n, F32, []uint64{3}, make([]byte, 12)))
	}
	must(t, w.AddTensor("train.rng", U8, []uint64{32}, make([]byte, 32)))
	b, err := tr.Encode()
	must(t, err)
	must(t, w.AddSection(SectionTraining, false, b))
	return w
}

func TestTrainingRoundTrip(t *testing.T) {
	want := sampleTraining()
	f, err := Parse(encode(t, trainingWriter(t, want)))
	must(t, err)
	got, err := f.Training()
	must(t, err)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v\nwant %+v", got, want)
	}
	// A file without the section has no training state.
	f, err = Parse(encode(t, sampleWriter(t)))
	must(t, err)
	if got, err := f.Training(); got != nil || err != nil {
		t.Fatalf("got %v, %v for a file without a training section", got, err)
	}
	// Empty state is valid.
	b, err := (&Training{}).Encode()
	must(t, err)
	if _, err := DecodeTraining(b); err != nil {
		t.Fatal(err)
	}
}

// TestTrainingRejects checks that Encode refuses state that breaks the
// rules of spec/training.md, with bad-training.
func TestTrainingRejects(t *testing.T) {
	cases := map[string]func(*Training){
		"empty counter name":   func(tr *Training) { tr.Counters[0].Name = "" },
		"counter twice":        func(tr *Training) { tr.Counters[1].Name = "step" },
		"empty hyper name":     func(tr *Training) { tr.Hyper[0].Name = "" },
		"hyper twice":          func(tr *Training) { tr.Hyper[5].Group = 0 },
		"empty parameter":      func(tr *Training) { tr.Params[0].Tensor = "" },
		"parameter twice":      func(tr *Training) { tr.Params[1].Tensor = "layer.1.w" },
		"slot twice":           func(tr *Training) { tr.Params[0].States[1].Slot = "m" },
		"empty slot":           func(tr *Training) { tr.Params[0].States[1].Slot = "" },
		"state is a parameter": func(tr *Training) { tr.Params[0].States[0].Tensor = "bias" },
		"state shared":         func(tr *Training) { tr.Params[1].States[0].Tensor = "layer.1.w@m" },
		"extra twice":          func(tr *Training) { tr.Extras = append(tr.Extras, TrainExtra{"rng", "x"}) },
		"extra is a state":     func(tr *Training) { tr.Extras[0].Tensor = "bias@v" },
		"empty extra":          func(tr *Training) { tr.Extras[0].Name = "" },
		"invalid UTF-8":        func(tr *Training) { tr.Optimizer = "\xff" },
	}
	for name, mut := range cases {
		tr := sampleTraining()
		mut(tr)
		if _, err := tr.Encode(); CodeOf(err) != CodeBadTraining {
			t.Errorf("%s: got %v, want bad-training", name, err)
		}
	}
}

// TestTrainingHostile checks that DecodeTraining rejects damaged sections
// with bad-training.
func TestTrainingHostile(t *testing.T) {
	base, err := sampleTraining().Encode()
	must(t, err)
	dir := func(i int) int { return 16 + 16*i }
	mut := func(f func(b []byte)) []byte {
		b := append([]byte(nil), base...)
		f(b)
		return b
	}
	params := int(le.Uint64(base[dir(ttParams):]))
	cases := map[string][]byte{
		"short":            base[:trainingHeaderSize-1],
		"magic":            mut(func(b []byte) { b[3] = 'X' }),
		"major":            mut(func(b []byte) { le.PutUint16(b[4:], 1) }),
		"reserved":         mut(func(b []byte) { b[12] = 1 }),
		"misaligned table": mut(func(b []byte) { le.PutUint64(b[dir(ttCounters):], le.Uint64(b[dir(ttCounters):])+4) }),
		"table outside":    mut(func(b []byte) { le.PutUint64(b[dir(ttStates)+8:], 1<<20) }),
		"too many rows":    mut(func(b []byte) { le.PutUint64(b[dir(ttStrings)+8:], maxTrainingRows+1) }),
		"string 0":         mut(func(b []byte) { le.PutUint32(b[int(le.Uint64(b[dir(ttStrings):]))+4:], 1) }),
		"string index":     mut(func(b []byte) { le.PutUint32(b[8:], 1000) }),
		"state run gap":    mut(func(b []byte) { le.PutUint32(b[params+16+8:], 3) }),
		"orphan state":     mut(func(b []byte) { le.PutUint32(b[params+16+12:], 1) }),
		"counter reserved": mut(func(b []byte) { b[int(le.Uint64(b[dir(ttCounters):]))+4] = 1 }),
	}
	for name, b := range cases {
		if _, err := DecodeTraining(b); CodeOf(err) != CodeBadTraining {
			t.Errorf("%s: got %v, want bad-training", name, err)
		}
	}
	// A training section naming a tensor the file lacks.
	tr := sampleTraining()
	tr.Extras[0].Tensor = "missing"
	f, err := Parse(encode(t, trainingWriter(t, tr)))
	must(t, err)
	if _, err := f.Training(); CodeOf(err) != CodeBadTraining {
		t.Errorf("missing tensor: got %v, want bad-training", err)
	}
}

func FuzzDecodeTraining(f *testing.F) {
	b, err := sampleTraining().Encode()
	if err != nil {
		f.Fatal(err)
	}
	f.Add(b)
	f.Fuzz(func(t *testing.T, b []byte) {
		tr, err := DecodeTraining(b)
		if err != nil {
			return
		}
		// What decodes encodes, and decodes to the same state.
		e, err := tr.Encode()
		if err != nil {
			t.Fatalf("decoded state does not encode: %v", err)
		}
		tr2, err := DecodeTraining(e)
		if err != nil || !reflect.DeepEqual(tr, tr2) {
			t.Fatalf("re-decoding gave %+v, %v; want %+v", tr2, err, tr)
		}
	})
}
