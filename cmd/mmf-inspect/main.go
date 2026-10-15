// Command mmf-inspect prints the index, metadata and graph of an mmf file.
//
//	mmf-inspect [-verify] [-graph] [-meta] model.mmf
package main

import (
	"flag"
	"fmt"
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
	f, err := mmf.Open(flag.Arg(0))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()

	p := f.Preamble()
	fmt.Printf("mmf %d.%d  file %d bytes  data alignment %d\n", p.Major, p.Minor, p.FileLength, p.DataAlignment)
	fmt.Printf("index  @%d +%d\nweights @%d +%d\n", p.IndexOffset, p.IndexLength, p.DataOffset, p.DataLength)
	fmt.Println("sections:")
	for _, s := range f.Sections() {
		req := ""
		if s.Flags&mmf.FlagRequired != 0 {
			req = " required"
		}
		h := "no hash"
		if s.HashAlg == mmf.HashSHA256 {
			h = fmt.Sprintf("sha256 %x…", s.Hash[:4])
		}
		fmt.Printf("  %-13s @%d +%d%s  %s\n", s.Kind, s.Offset, s.Length, req, h)
	}
	if *verify {
		if err := f.VerifyHashes(); err != nil {
			fmt.Fprintln(os.Stderr, "hash check failed:", err)
			os.Exit(1)
		}
		fmt.Println("hashes ok")
	}
	fmt.Printf("tensors: %d\n", f.NumTensors())
	for i := 0; i < f.NumTensors(); i++ {
		t := f.TensorAt(i)
		fmt.Printf("  %-40s %-12s %v  @%d +%d\n", t.Name, t.DType, t.Shape, t.Offset, t.Length)
	}
	if *meta {
		m, err := f.Metadata()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		keys := make([]string, 0, len(m))
		for k := range m {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		fmt.Println("metadata:")
		for _, k := range keys {
			v := fmt.Sprint(m[k])
			if len(v) > 100 {
				v = v[:100] + "…"
			}
			fmt.Printf("  %s = %s\n", k, v)
		}
	}
	if *graph {
		g, err := f.Graph()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		printGraph(g)
	}
}

func printGraph(g *mmf.Graph) {
	fmt.Printf("graph: primitives v%d, stdlib v%d, %d values, %d subgraphs, %d composites\n",
		g.PrimitiveVersion, g.StdlibVersion, len(g.Values), len(g.Subgraphs), len(g.Composites))
	for i, s := range g.SymDims {
		fmt.Printf("  symdim %d %s bound %d\n", i, s.Name, s.Bound)
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
		fmt.Printf("subgraph %d (%s) -> (%s)\n", si, list(sg.Inputs), list(sg.Outputs))
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
			fmt.Printf("  %s = %s[%s](%s)\n", list(op.Outputs), op.Opcode, strings.Join(attrs, ", "), list(op.Inputs))
		}
	}
	for _, c := range g.Composites {
		fmt.Printf("composite %s.v%d = subgraph %d\n", c.Name, c.Version, c.Subgraph)
	}
	for _, u := range g.Updates {
		fmt.Printf("update %s <- %s\n", val(u.Variable), val(u.Value))
	}
}
