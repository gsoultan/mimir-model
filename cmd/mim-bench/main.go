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

// Command mim-bench measures how fast model files load and how much memory
// loading takes, for mim, GGUF and safetensors files alike.
//
//	mim-bench [-runs 5] [-cold] model.gguf model.safetensors model.mim ...
//
// Each run is a fresh child process, so its memory figures start from
// zero. A run maps the file, parses its header (for mim: mim.Open, which
// also validates the index and the tensor table; with a graph, also
// decodes and resolves it with interp.Load), and then reads every tensor
// byte, the work a loader must do before the first token. mim files also
// decode their metadata, which the GGUF and safetensors parsers do anyway. The table
// reports the median of the runs:
//
//	open    time to map and parse the header (and the graph)
//	read    time to then touch every tensor byte, one byte per cache line
//	heap    bytes the Go heap allocated while opening
//	anon    resident anonymous memory after reading (heap, stacks, runtime)
//	mapped  resident file-backed memory after reading (the page cache)
//
// With -cold, each run first asks the kernel to drop the file from the page
// cache (Linux only), so read includes the disk.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
	"unsafe"

	"github.com/gsoultan/mim"
	"github.com/gsoultan/mim/importer"
	"github.com/gsoultan/mim/interp"
)

func main() {
	runs := flag.Int("runs", 5, "runs per file")
	cold := flag.Bool("cold", false, "drop each file from the page cache before each run (Linux)")
	child := flag.Bool("child", false, "internal: measure one load of one file and print it as JSON")
	flag.Parse()
	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: mim-bench [-runs 5] [-cold] file...")
		os.Exit(2)
	}
	var err error
	if *child {
		var r result
		if r, err = measure(flag.Arg(0), *cold); err == nil {
			err = json.NewEncoder(os.Stdout).Encode(r)
		}
	} else {
		err = run(os.Stdout, flag.Args(), *runs, *cold, spawn)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "mim-bench:", err)
		os.Exit(1)
	}
}

// result is one load of one file.
type result struct {
	Format  string `json:"format"`
	Size    int64  `json:"size"`
	Tensors int    `json:"tensors"`
	Graph   bool   `json:"graph"`
	OpenNs  int64  `json:"open_ns"`
	ReadNs  int64  `json:"read_ns"`
	Heap    uint64 `json:"heap"`
	Anon    int64  `json:"anon"`   // -1 if unknown
	Mapped  int64  `json:"mapped"` // -1 if unknown
	Sum     byte   `json:"sum"`
}

// spawn measures path in a child process.
func spawn(path string, cold bool) (result, error) {
	exe, err := os.Executable()
	if err != nil {
		return result{}, err
	}
	args := []string{"-child"}
	if cold {
		args = append(args, "-cold")
	}
	cmd := exec.Command(exe, append(args, path)...)
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		return result{}, fmt.Errorf("%s: %w", path, err)
	}
	var r result
	if err := json.Unmarshal(out, &r); err != nil {
		return result{}, fmt.Errorf("%s: %w", path, err)
	}
	return r, nil
}

func addr(b []byte) uintptr { return uintptr(unsafe.Pointer(unsafe.SliceData(b))) }

// touch reads one byte per cache line of b so that a mapped file is paged
// in, and returns their XOR so the reads cannot be optimised away.
func touch(b []byte) (s byte) {
	for i := 0; i < len(b); i += 64 {
		s ^= b[i]
	}
	return s
}

// measure loads path once.
func measure(path string, cold bool) (result, error) {
	st, err := os.Stat(path)
	if err != nil {
		return result{}, err
	}
	r := result{Size: st.Size()}
	if cold {
		if err := dropCache(path); err != nil {
			return result{}, err
		}
	}
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)

	var tensors [][]byte
	var release func() error
	start := time.Now()
	if strings.HasSuffix(path, ".mim") {
		r.Format = "mim"
		f, err := mim.Open(path)
		if err != nil {
			return result{}, err
		}
		release = f.Close
		// A loader needs the metadata (the tokenizer, the hyperparameters),
		// which GGUF and safetensors parsers decode up front.
		if _, err := f.Metadata(); err != nil {
			return result{}, err
		}
		if _, ok := f.Section(mim.SectionGraph); ok {
			if _, err := interp.Load(f); err != nil {
				return result{}, err
			}
			r.Graph = true
		}
		for i := 0; i < f.NumTensors(); i++ {
			tensors = append(tensors, f.TensorData(f.TensorAt(i)))
		}
	} else {
		b, rel, err := mapFile(path)
		if err != nil {
			return result{}, err
		}
		release = rel
		s, err := importer.Parse(b)
		if err != nil {
			return result{}, err
		}
		r.Format = s.Format
		for _, t := range s.Tensors {
			tensors = append(tensors, t.Data)
		}
	}
	r.OpenNs = int64(time.Since(start))
	runtime.ReadMemStats(&after)
	r.Heap = after.TotalAlloc - before.TotalAlloc

	// Read in file order whatever order the parser lists the tensors in
	// (the safetensors parser sorts them by name), so that every format
	// gets the kernel's sequential readahead.
	sort.Slice(tensors, func(i, j int) bool { return addr(tensors[i]) < addr(tensors[j]) })
	start = time.Now()
	for _, t := range tensors {
		r.Sum ^= touch(t)
	}
	r.ReadNs = int64(time.Since(start))
	r.Tensors = len(tensors)
	r.Anon, r.Mapped = residentMemory()
	return r, release()
}

// summary is the median of several runs of one file.
type summary struct {
	path string
	result
}

func median(rs []result, f func(result) int64) int64 {
	v := make([]int64, len(rs))
	for i, r := range rs {
		v[i] = f(r)
	}
	sort.Slice(v, func(a, b int) bool { return v[a] < v[b] })
	return v[len(v)/2]
}

func summarize(path string, rs []result) summary {
	s := summary{path: path, result: rs[0]}
	s.OpenNs = median(rs, func(r result) int64 { return r.OpenNs })
	s.ReadNs = median(rs, func(r result) int64 { return r.ReadNs })
	s.Heap = uint64(median(rs, func(r result) int64 { return int64(r.Heap) }))
	s.Anon = median(rs, func(r result) int64 { return r.Anon })
	s.Mapped = median(rs, func(r result) int64 { return r.Mapped })
	return s
}

func mib(n int64) string {
	switch {
	case n < 0:
		return "n/a"
	case n < 1<<20:
		return fmt.Sprintf("%.0f KiB", float64(n)/(1<<10))
	}
	return fmt.Sprintf("%.1f MiB", float64(n)/(1<<20))
}

func dur(ns int64) string {
	d := time.Duration(ns)
	switch {
	case d < time.Millisecond:
		return d.Round(time.Microsecond).String()
	case d < time.Second:
		return d.Round(10 * time.Microsecond).String()
	}
	return d.Round(time.Millisecond).String()
}

func run(w io.Writer, paths []string, runs int, cold bool, measureOne func(string, bool) (result, error)) error {
	if runs < 1 {
		return fmt.Errorf("-runs must be at least 1")
	}
	var all []summary
	for _, p := range paths {
		var rs []result
		for i := 0; i < runs; i++ {
			r, err := measureOne(p, cold)
			if err != nil {
				return err
			}
			if i > 0 && r.Sum != rs[0].Sum {
				return fmt.Errorf("%s: tensor bytes changed between runs", p)
			}
			rs = append(rs, r)
		}
		all = append(all, summarize(p, rs))
	}
	mode := "warm page cache"
	if cold {
		mode = "cold page cache"
	}
	fmt.Fprintf(w, "median of %d runs, %s, %s/%s, %d CPUs\n\n", runs, mode, runtime.GOOS, runtime.GOARCH, runtime.NumCPU())
	fmt.Fprintln(w, "| file | format | size | tensors | open | read | heap | anon | mapped |")
	fmt.Fprintln(w, "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
	for _, s := range all {
		format := s.Format
		if s.Graph {
			format += " + graph"
		}
		fmt.Fprintf(w, "| %s | %s | %s | %d | %s | %s | %s | %s | %s |\n",
			filepath.Base(s.path), format, mib(s.Size), s.Tensors, dur(s.OpenNs), dur(s.ReadNs),
			mib(int64(s.Heap)), mib(s.Anon), mib(s.Mapped))
	}
	return nil
}
