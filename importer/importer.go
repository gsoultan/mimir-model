// Package importer converts GGUF and safetensors files into mmf.
//
// Tensors are copied byte for byte: every GGUF block type that mmf defines
// has the same canonical layout (ADR 0004), and safetensors stores plain
// little-endian arrays, so an import never re-quantizes or rounds. Metadata
// keys are carried over with their types. Neither source format carries a
// graph mmf can use, so an imported file holds weights and metadata only.
//
// Both parsers treat their input as untrusted: every length, count and
// offset is bounds-checked before use, and malformed input returns an error
// rather than panicking.
package importer

import (
	"fmt"
	"sort"

	"github.com/gsoultan/mmf"
)

// Tensor is one tensor of a source file. Data aliases the source bytes.
type Tensor struct {
	Name  string
	DType mmf.DType
	Shape []uint64 // row-major, outermost dimension first
	Data  []byte
}

// Source is a parsed GGUF or safetensors file.
type Source struct {
	Format   string // "gguf" or "safetensors"
	Tensors  []Tensor
	Metadata mmf.Metadata
	// Graph is an optional graph built by the importer (see BuildLlama).
	// The Ref of its weight, const and variable values indexes Tensors.
	Graph *mmf.Graph
}

// Detect returns "gguf", "safetensors" or "" for the leading bytes of a file.
func Detect(b []byte) string {
	if len(b) >= 4 && string(b[:4]) == "GGUF" {
		return "gguf"
	}
	// safetensors starts with a u64 header length followed by '{'.
	if len(b) >= 9 && b[8] == '{' {
		return "safetensors"
	}
	return ""
}

// Parse detects the format of b and parses it.
func Parse(b []byte) (*Source, error) {
	switch Detect(b) {
	case "gguf":
		return ParseGGUF(b)
	case "safetensors":
		return ParseSafetensors(b)
	}
	return nil, fmt.Errorf("unrecognised file: neither GGUF nor safetensors")
}

// Merge combines several sources, for example the shards of a sharded
// safetensors checkpoint. Tensor names must be unique across sources.
// Metadata from later sources overrides earlier keys.
func Merge(srcs ...*Source) (*Source, error) {
	if len(srcs) == 0 {
		return nil, fmt.Errorf("no sources")
	}
	out := &Source{Format: srcs[0].Format, Metadata: mmf.Metadata{}}
	seen := map[string]bool{}
	for _, s := range srcs {
		if s.Format != out.Format {
			return nil, fmt.Errorf("cannot merge %s and %s sources", out.Format, s.Format)
		}
		for _, t := range s.Tensors {
			if seen[t.Name] {
				return nil, fmt.Errorf("tensor %q appears in more than one source", t.Name)
			}
			seen[t.Name] = true
			out.Tensors = append(out.Tensors, t)
		}
		for k, v := range s.Metadata {
			out.Metadata[k] = v
		}
	}
	return out, nil
}

// AddTo adds the source's tensors, a metadata section and, when the source
// has one, its graph to w. Tensors are added in name order so the weights
// region is laid out deterministically.
func (s *Source) AddTo(w *mmf.Writer) error {
	order := make([]int, len(s.Tensors))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool { return s.Tensors[order[i]].Name < s.Tensors[order[j]].Name })
	pos := make([]uint32, len(order)) // source index to file tensor index
	for p, i := range order {
		t := s.Tensors[i]
		if err := w.AddTensor(t.Name, t.DType, t.Shape, t.Data); err != nil {
			return fmt.Errorf("tensor %q: %w", t.Name, err)
		}
		pos[i] = uint32(p)
	}
	if s.Graph != nil {
		g := *s.Graph
		g.Values = append([]mmf.Value(nil), s.Graph.Values...)
		for i, v := range g.Values {
			if v.Kind != mmf.KindWeight && v.Kind != mmf.KindConst && v.Kind != mmf.KindVariable {
				continue
			}
			if int(v.Ref) >= len(pos) {
				return fmt.Errorf("graph value %q refers to tensor %d of %d", v.Name, v.Ref, len(pos))
			}
			g.Values[i].Ref = pos[v.Ref]
		}
		enc, err := g.Encode()
		if err != nil {
			return err
		}
		if err := w.AddSection(mmf.SectionGraph, true, enc); err != nil {
			return err
		}
	}
	md := mmf.Metadata{}
	for k, v := range s.Metadata {
		md[k] = v
	}
	md["import.source_format"] = s.Format
	enc, err := md.Encode()
	if err != nil {
		return err
	}
	return w.AddSection(mmf.SectionMetadata, false, enc)
}

// checkTensor verifies that a tensor's data length matches its dtype and
// shape.
func checkTensor(t Tensor) error {
	if len(t.Shape) > mmf.MaxRank {
		return fmt.Errorf("tensor %q: rank %d exceeds %d", t.Name, len(t.Shape), mmf.MaxRank)
	}
	n, err := mmf.ByteSize(t.DType, t.Shape)
	if err != nil {
		return fmt.Errorf("tensor %q: %w", t.Name, err)
	}
	if n != uint64(len(t.Data)) {
		return fmt.Errorf("tensor %q: %s%v needs %d bytes, has %d", t.Name, t.DType, t.Shape, n, len(t.Data))
	}
	return nil
}
