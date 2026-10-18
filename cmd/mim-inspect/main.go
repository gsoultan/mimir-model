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

// Command mim-inspect prints the index, metadata and graph of a mim file.
//
//	mim-inspect [-verify] [-graph] [-meta] model.mim
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/gsoultan/mim"
)

func main() {
	flag.Usage = usage
	verify := flag.Bool("verify", false, "verify section hashes")
	graph := flag.Bool("graph", false, "print the graph")
	meta := flag.Bool("meta", false, "print metadata")
	gembit := flag.Bool("gembit", false, "")
	flag.Parse()
	if *gembit {
		fmt.Print(banner)
		return
	}
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: mim-inspect [-verify] [-graph] [-meta] file.mim")
		os.Exit(2)
	}
	if err := inspect(os.Stdout, flag.Arg(0), *verify, *graph, *meta); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// usage prints the visible flags only, so -gembit stays a secret.
func usage() {
	fmt.Fprintln(flag.CommandLine.Output(), "usage: mim-inspect [-verify] [-graph] [-meta] file.mim")
	flag.VisitAll(func(f *flag.Flag) {
		if f.Name == "gembit" {
			return
		}
		fmt.Fprintf(flag.CommandLine.Output(), "  -%s\n    \t%s\n", f.Name, f.Usage)
	})
}

// banner is printed by the hidden -gembit flag.
const banner = `
  ╔══════════════════════════════╗
  ║  mim — crafted with care by  ║
  ║    Gembit Soultan Shirazi    ║
  ╚══════════════════════════════╝
`

func inspect(w io.Writer, path string, verify, graph, meta bool) error {
	f, err := mim.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	p := f.Preamble()
	fmt.Fprintf(w, "mim %d.%d  file %d bytes  data alignment %d\n", p.Major, p.Minor, p.FileLength, p.DataAlignment)
	fmt.Fprintf(w, "index  @%d +%d\nweights @%d +%d\n", p.IndexOffset, p.IndexLength, p.DataOffset, p.DataLength)
	fmt.Fprintln(w, "sections:")
	for _, s := range f.Sections() {
		req := ""
		if s.Flags&mim.FlagRequired != 0 {
			req = " required"
		}
		h := "no hash"
		if s.HashAlg == mim.HashSHA256 {
			h = fmt.Sprintf("sha256 %x…", s.Hash[:4])
		}
		fmt.Fprintf(w, "  %-13s @%d +%d%s  %s\n", s.Kind, s.Offset, s.Length, req, h)
	}
	if verify {
		if err := f.VerifyHashes(); err != nil {
			return err
		}
		fmt.Fprintln(w, "hashes ok")
	}
	fmt.Fprintf(w, "tensors: %d\n", f.NumTensors())
	for i := 0; i < f.NumTensors(); i++ {
		t := f.TensorAt(i)
		fmt.Fprintf(w, "  %-40s %-12s %v  @%d +%d\n", t.Name, t.DType, t.Shape, t.Offset, t.Length)
	}
	if meta {
		m, err := f.Metadata()
		if err != nil {
			return err
		}
		keys := make([]string, 0, len(m))
		for k := range m {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		fmt.Fprintln(w, "metadata:")
		for _, k := range keys {
			v := fmt.Sprint(m[k])
			if len(v) > 100 {
				v = v[:100] + "…"
			}
			fmt.Fprintf(w, "  %s = %s\n", k, v)
		}
	}
	if graph {
		g, err := f.Graph()
		if err != nil {
			return err
		}
		printGraph(w, g)
	}
	return nil
}

func printGraph(w io.Writer, g *mim.Graph) {
	fmt.Fprintf(w, "graph: primitives v%d, stdlib v%d, %d values, %d subgraphs, %d composites\n",
		g.PrimitiveVersion, g.StdlibVersion, len(g.Values), len(g.Subgraphs), len(g.Composites))
	for i, s := range g.SymDims {
		fmt.Fprintf(w, "  symdim %d %s bound %d\n", i, s.Name, s.Bound)
	}
	val := func(id uint32) string {
		v := g.Values[id]
		name := v.Name
		if name == "" {
			name = fmt.Sprintf("%%%d", id)
		}
		dims := make([]string, len(v.Shape))
		for i, d := range v.Shape {
			if d.Symbolic() {
				dims[i] = g.SymDims[d.SymIndex()].Name
			} else {
				dims[i] = fmt.Sprint(int64(d))
			}
		}
		return fmt.Sprintf("%s:%s[%s]", name, v.DType, strings.Join(dims, ","))
	}
	list := func(ids []uint32) string {
		s := make([]string, len(ids))
		for i, id := range ids {
			s[i] = val(id)
		}
		return strings.Join(s, ", ")
	}
	for si, sg := range g.Subgraphs {
		fmt.Fprintf(w, "subgraph %d (%s) -> (%s)\n", si, list(sg.Inputs), list(sg.Outputs))
		for _, op := range sg.Ops {
			attrs := make([]string, len(op.Attrs))
			for i, a := range op.Attrs {
				switch a.Type {
				case mim.AttrInt:
					attrs[i] = fmt.Sprintf("%s=%d", a.Key, a.Int)
				case mim.AttrInts:
					attrs[i] = fmt.Sprintf("%s=%v", a.Key, a.Ints)
				case mim.AttrFloat:
					attrs[i] = fmt.Sprintf("%s=%g", a.Key, a.Float)
				case mim.AttrString:
					attrs[i] = fmt.Sprintf("%s=%q", a.Key, a.String)
				case mim.AttrDType:
					attrs[i] = fmt.Sprintf("%s=%s", a.Key, a.DType)
				case mim.AttrSubgraph:
					attrs[i] = fmt.Sprintf("%s=subgraph %d", a.Key, a.Subgraph)
				}
			}
			fmt.Fprintf(w, "  %s = %s[%s](%s)\n", list(op.Outputs), op.Opcode, strings.Join(attrs, ", "), list(op.Inputs))
		}
	}
	for _, c := range g.Composites {
		fmt.Fprintf(w, "composite %s.v%d = subgraph %d\n", c.Name, c.Version, c.Subgraph)
	}
	for _, u := range g.Updates {
		init := ""
		if g.Values[u.Variable].Zero {
			init = " (starts as zeros)"
		}
		fmt.Fprintf(w, "update %s <- %s%s\n", val(u.Variable), val(u.Value), init)
	}
}
