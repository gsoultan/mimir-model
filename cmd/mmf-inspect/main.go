// Command mmf-inspect prints the index, metadata and graph of an mmf file.
//
//	mmf-inspect [-verify] [-graph] [-meta] model.mmf
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/gsoultan/mmf"
)

func main() {
	verify := flag.Bool("verify", false, "verify section hashes")
	graph := flag.Bool("graph", false, "print the graph")
	meta := flag.Bool("meta", false, "print metadata")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: mmf-inspect [-verify] [-graph] [-meta] file.mmf")
		os.Exit(2)
	}
	if err := inspect(os.Stdout, flag.Arg(0), *verify, *graph, *meta); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func inspect(w io.Writer, path string, verify, graph, meta bool) error {
	f, err := mmf.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	p := f.Preamble()
	fmt.Fprintf(w, "mmf %d.%d  file %d bytes  data alignment %d\n", p.Major, p.Minor, p.FileLength, p.DataAlignment)
	fmt.Fprintf(w, "index  @%d +%d\nweights @%d +%d\n", p.IndexOffset, p.IndexLength, p.DataOffset, p.DataLength)
	fmt.Fprintln(w, "sections:")
	for _, s := range f.Sections() {
		req := ""
		if s.Flags&mmf.FlagRequired != 0 {
			req = " required"
		}
		h := "no hash"
		if s.HashAlg == mmf.HashSHA256 {
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

func printGraph(w io.Writer, g *mmf.Graph) {
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
				case mmf.AttrInt:
					attrs[i] = fmt.Sprintf("%s=%d", a.Key, a.Int)
				case mmf.AttrInts:
					attrs[i] = fmt.Sprintf("%s=%v", a.Key, a.Ints)
				case mmf.AttrFloat:
					attrs[i] = fmt.Sprintf("%s=%g", a.Key, a.Float)
				case mmf.AttrString:
					attrs[i] = fmt.Sprintf("%s=%q", a.Key, a.String)
				case mmf.AttrDType:
					attrs[i] = fmt.Sprintf("%s=%s", a.Key, a.DType)
				case mmf.AttrSubgraph:
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
