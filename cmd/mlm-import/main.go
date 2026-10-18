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

// Command mlm-import converts GGUF, safetensors and ONNX files into mim.
//
//	mlm-import [-o out.mim] [-bench] [-nohash] input.gguf
//	mlm-import model.onnx
//	mlm-import -graph [-ctx 2048] [-a8=false] [-kv f32] [-slots 1] tinyllama.gguf
//	mlm-import -o model.mim model-00001-of-00002.safetensors model-00002-of-00002.safetensors
//
// Several safetensors shards are merged into one file. Tensors are copied
// byte for byte. By default the output holds weights and metadata only;
// with -graph, a GGUF file of the llama or qwen2 architecture also gets a
// runnable forward-pass graph with a key-value cache of -ctx positions
// (default: the model's context length), which mlm-run can generate text
// with. Its
// matrix products with quantized weights are dequant_matmul_a8, which
// lets a runtime round the activations to 8-bit blocks as llama.cpp does:
// blocks of 256 for K-quant weights (v2) and of 32 for the others (v1).
// -a8=false writes dequant_matmul.v1, which keeps them in float32. The
// key-value cache is f16, as llama.cpp's is by default, and the graph's
// attention is attention.v2, which reads it as it is; -kv f32 keeps the
// cache in float32, with attention.v1, for twice the memory. -slots N
// makes the cache hold N sequences of -ctx positions each, so that a
// server can run several requests in one step.
// An ONNX model (a file ending in .onnx) of the convolutional subset that
// package importer documents is converted with its graph, so the output
// runs as it is.
// With -bench it reports the input and output sizes and the time to open
// each and read every tensor byte.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/gsoultan/mlm"
	"github.com/gsoultan/mlm/importer"
)

func main() {
	out := flag.String("o", "", "output file (default: first input with the extension replaced by .mim)")
	bench := flag.Bool("bench", false, "report sizes and load times of the input and the output")
	nohash := flag.Bool("nohash", false, "do not write section hashes")
	graph := flag.Bool("graph", false, "build a runnable graph (GGUF llama or qwen2 architecture)")
	ctx := flag.Int("ctx", 0, "with -graph, the key-value cache length (default: the model's context length)")
	a8 := flag.Bool("a8", true, "with -graph, let runtimes round activations to 8 bits in quantized matrix products (dequant_matmul_a8), as llama.cpp does")
	kv := flag.String("kv", "f16", "with -graph, the key-value cache dtype: f16 or f32")
	slots := flag.Int("slots", 1, "with -graph, the number of sequences the cache holds at once, for a server running several requests")
	flag.Parse()
	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: mlm-import [-o out.mim] [-bench] [-nohash] input...")
		os.Exit(2)
	}
	cache, ok := mlm.ParseDType(*kv)
	if !ok || (cache != mlm.F16 && cache != mlm.F32) {
		fmt.Fprintf(os.Stderr, "mlm-import: -kv %s: want f16 or f32\n", *kv)
		os.Exit(2)
	}
	o := options{out: *out, bench: *bench, hash: !*nohash, graph: *graph, ctx: *ctx, a8: *a8, cache: cache, slots: *slots}
	if err := run(os.Stdout, flag.Args(), o); err != nil {
		fmt.Fprintln(os.Stderr, "mlm-import:", err)
		os.Exit(1)
	}
}

// parse reads an input by its extension (.onnx) or its leading bytes.
func parse(path string, b []byte) (*importer.Source, error) {
	if strings.HasSuffix(strings.ToLower(path), ".onnx") {
		return importer.ParseONNX(b)
	}
	return importer.Parse(b)
}

func defaultOutput(in string) string {
	for _, ext := range []string{".gguf", ".safetensors", ".onnx"} {
		if strings.HasSuffix(in, ext) {
			return strings.TrimSuffix(in, ext) + ".mim"
		}
	}
	return in + ".mim"
}

type options struct {
	out         string
	bench, hash bool
	graph       bool
	ctx         int
	a8          bool
	cache       mlm.DType // the key-value cache's dtype
	slots       int       // sequences the cache holds
}

func run(w io.Writer, inputs []string, o options) (err error) {
	out, bench, hash := o.out, o.bench, o.hash
	if out == "" {
		out = defaultOutput(inputs[0])
	}
	var srcs []*importer.Source
	var inBytes int64
	start := time.Now()
	for _, p := range inputs {
		b, release, err := mapFile(p)
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, release()) }()
		inBytes += int64(len(b))
		s, err := parse(p, b)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		srcs = append(srcs, s)
	}
	src, err := importer.Merge(srcs...)
	if err != nil {
		return err
	}
	if src.Format == "onnx" {
		if len(srcs) > 1 {
			return fmt.Errorf("an ONNX model is converted on its own")
		}
		if bench {
			// The import computes some tensors (folded normalisations,
			// transposed weights), so the byte-for-byte check does not apply.
			return fmt.Errorf("-bench does not support ONNX inputs")
		}
	}
	graphNote := ""
	if o.graph {
		if src.Format != "gguf" {
			return fmt.Errorf("-graph needs a GGUF input")
		}
		c, err := importer.BuildLlama(src, importer.LlamaOptions{Context: o.ctx, A8: o.a8, Cache: o.cache, Slots: o.slots})
		if err != nil {
			return err
		}
		graphNote = fmt.Sprintf(", %s graph: %d layers, %d heads (%d key-value), context %d", c.Arch, c.Layers, c.Heads, c.KVHeads, c.Context)
		if c.Slots > 1 {
			graphNote += fmt.Sprintf(" in each of %d slots", c.Slots)
		}
	}
	mw := mlm.NewWriter()
	mw.Hash = hash
	if err := src.AddTo(mw); err != nil {
		return err
	}
	if err := mw.WriteFile(out); err != nil {
		return err
	}
	st, err := os.Stat(out)
	if err != nil {
		return err
	}
	fmt.Fprintf(w, "wrote %s: %d tensors from %s, %d bytes (%s)%s\n", out, len(src.Tensors), src.Format, st.Size(), time.Since(start).Round(time.Millisecond), graphNote)
	if !bench {
		return nil
	}
	srcLoad, srcSum, err := timeSource(inputs)
	if err != nil {
		return err
	}
	dstLoad, dstSum, err := timeMIM(out)
	if err != nil {
		return err
	}
	if srcSum != dstSum {
		return fmt.Errorf("tensor data read back from %s differs from the input", out)
	}
	fmt.Fprintf(w, "%-12s %14s %14s\n", "", "size (bytes)", "open+read")
	fmt.Fprintf(w, "%-12s %14d %14s\n", src.Format, inBytes, srcLoad.Round(time.Microsecond))
	fmt.Fprintf(w, "%-12s %14d %14s\n", "mim", st.Size(), dstLoad.Round(time.Microsecond))
	return nil
}

// touch reads one byte per cache line of b so that a mapped file is paged
// in, and returns their XOR as a checksum.
func touch(b []byte) (s byte) {
	for i := 0; i < len(b); i += 64 {
		s ^= b[i]
	}
	return s
}

// timeSource measures mapping and parsing the inputs and reading every
// tensor byte, the work a loader has to do before inference.
func timeSource(inputs []string) (time.Duration, byte, error) {
	var sum byte
	start := time.Now()
	for _, p := range inputs {
		b, release, err := mapFile(p)
		if err != nil {
			return 0, 0, err
		}
		s, err := parse(p, b)
		if err != nil {
			release()
			return 0, 0, err
		}
		for _, t := range s.Tensors {
			sum ^= touch(t.Data)
		}
		if err := release(); err != nil {
			return 0, 0, err
		}
	}
	return time.Since(start), sum, nil
}

// timeMIM measures mlm.Open (map and validate) and reading every tensor byte.
func timeMIM(path string) (time.Duration, byte, error) {
	var sum byte
	start := time.Now()
	f, err := mlm.Open(path)
	if err != nil {
		return 0, 0, err
	}
	for i := 0; i < f.NumTensors(); i++ {
		sum ^= touch(f.TensorData(f.TensorAt(i)))
	}
	if err := f.Close(); err != nil {
		return 0, 0, err
	}
	return time.Since(start), sum, nil
}
