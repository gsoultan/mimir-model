// Package compat checks a reader and the reference executor against the
// compatibility corpus of spec/versioning.md section 4.
//
// The corpus lives in testdata/compat, one directory per released version.
// Each directory holds files written by the reference writer of that version
// and, next to each file, an expected.json that lists what a reader must see
// in it: the container version, the sections, every tensor with a digest of
// its bytes, the metadata, and for files with a graph the outputs of a
// sequence of runs in one session. A directory is never changed once its
// version is released; the current reader must keep passing every one.
package compat

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/interp"
	"github.com/gsoultan/mmf/ref"
)

// Format and Version identify an expected.json file.
const (
	Format  = "mmf-compat"
	Version = 1
)

// Expected is the content of an expected.json file.
type Expected struct {
	Format    string     `json:"format"`
	Version   int        `json:"version"`
	File      string     `json:"file"`
	Container Container  `json:"container"`
	Sections  []Section  `json:"sections"`
	Tensors   []Tensor   `json:"tensors"`
	Metadata  []Meta     `json:"metadata"`
	Runs      []Run      `json:"runs,omitempty"`
	Notes     string     `json:"notes,omitempty"`
	Graph     *GraphInfo `json:"graph,omitempty"`
}

// Container is the preamble a reader must report.
type Container struct {
	Major         uint16 `json:"major"`
	Minor         uint16 `json:"minor"`
	DataAlignment uint32 `json:"data_alignment"`
}

// Section is one entry of the section table, in table order.
type Section struct {
	Kind     uint32 `json:"kind"`
	Required bool   `json:"required"`
	HashAlg  uint8  `json:"hash_algorithm"`
	Length   uint64 `json:"length"`
}

// Tensor is one entry of the tensor table, in name order. SHA256 is the
// digest of the tensor's bytes.
type Tensor struct {
	Name   string   `json:"name"`
	DType  string   `json:"dtype"`
	Shape  []uint64 `json:"shape"`
	Offset uint64   `json:"offset"`
	SHA256 string   `json:"sha256"`
}

// Meta is one metadata entry, in key order. Value is encoded by type: int
// and uint as JSON integers, float as the hex of its little-endian binary64
// bytes, bool, string, bytes as hex, ints as a list of integers, floats as
// a list of hex binary64s, strings as a list.
type Meta struct {
	Key   string          `json:"key"`
	Type  string          `json:"type"`
	Value json.RawMessage `json:"value"`
}

// GraphInfo summarises the graph section.
type GraphInfo struct {
	Inputs  []string `json:"inputs"`
	Outputs []string `json:"outputs"`
}

// Run is one run of the main graph. Runs of one file share a session, so
// variables carry over from one run to the next.
type Run struct {
	Inputs  map[string]TensorData `json:"inputs"`
	Outputs map[string]Output     `json:"outputs"`
}

// TensorData is a tensor written out in full, as in the conformance corpus:
// the packed little-endian element bytes as lowercase hex.
type TensorData struct {
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
	Data  string `json:"data"`
}

// Output is an expected graph output. Check is "bits", which requires the
// same bytes, or "close", which requires every element to be within Tol
// times the largest magnitude of the expected tensor.
type Output struct {
	TensorData
	Check string  `json:"check"`
	Tol   float64 `json:"tol,omitempty"`
}

// Describe builds the expected description of the container, sections,
// tensors and metadata of f.
func Describe(f *mmf.File) (*Expected, error) {
	p := f.Preamble()
	e := &Expected{Format: Format, Version: Version,
		Container: Container{p.Major, p.Minor, p.DataAlignment}}
	for _, s := range f.Sections() {
		e.Sections = append(e.Sections, Section{uint32(s.Kind), s.Flags&mmf.FlagRequired != 0, uint8(s.HashAlg), s.Length})
	}
	for i := 0; i < f.NumTensors(); i++ {
		t := f.TensorAt(i)
		sum := sha256.Sum256(f.TensorData(t))
		e.Tensors = append(e.Tensors, Tensor{t.Name, t.DType.String(), append([]uint64{}, t.Shape...), t.Offset, hex.EncodeToString(sum[:])})
	}
	e.Tensors = nonNil(e.Tensors)
	md, err := f.Metadata()
	if err != nil {
		return nil, err
	}
	keys := make([]string, 0, len(md))
	for k := range md {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		m, err := encodeMeta(k, md[k])
		if err != nil {
			return nil, err
		}
		e.Metadata = append(e.Metadata, m)
	}
	e.Metadata = nonNil(e.Metadata)
	if _, ok := f.Section(mmf.SectionGraph); ok {
		g, err := f.Graph()
		if err != nil {
			return nil, err
		}
		gi := &GraphInfo{Inputs: []string{}, Outputs: []string{}}
		for _, id := range g.Main().Inputs {
			gi.Inputs = append(gi.Inputs, g.Values[id].Name)
		}
		for _, id := range g.Main().Outputs {
			gi.Outputs = append(gi.Outputs, g.Values[id].Name)
		}
		e.Graph = gi
	}
	return e, nil
}

func nonNil[T any](s []T) []T {
	if s == nil {
		return []T{}
	}
	return s
}

func f64hex(v float64) string {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
	return hex.EncodeToString(b[:])
}

func encodeMeta(k string, v any) (Meta, error) {
	var typ string
	var val any
	switch x := v.(type) {
	case int64:
		typ, val = "int", x
	case uint64:
		typ, val = "uint", x
	case float64:
		typ, val = "float", f64hex(x)
	case bool:
		typ, val = "bool", x
	case string:
		typ, val = "string", x
	case []byte:
		typ, val = "bytes", hex.EncodeToString(x)
	case []int64:
		typ, val = "ints", nonNil(x)
	case []float64:
		h := make([]string, len(x))
		for i, f := range x {
			h[i] = f64hex(f)
		}
		typ, val = "floats", h
	case []string:
		typ, val = "strings", nonNil(x)
	default:
		return Meta{}, fmt.Errorf("metadata %q has unsupported type %T", k, v)
	}
	raw, err := json.Marshal(val)
	if err != nil {
		return Meta{}, err
	}
	return Meta{Key: k, Type: typ, Value: raw}, nil
}

// Load reads every expected.json under dir, in path order.
func Load(dir string) (map[string]*Expected, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*", "*.expected.json"))
	if err != nil {
		return nil, err
	}
	out := map[string]*Expected{}
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		var e Expected
		if err := json.Unmarshal(b, &e); err != nil {
			return nil, fmt.Errorf("%s: %w", p, err)
		}
		if e.Format != Format || e.Version != Version {
			return nil, fmt.Errorf("%s: format %q version %d", p, e.Format, e.Version)
		}
		out[p] = &e
	}
	return out, nil
}

// Check opens the file that e describes, in dir, and compares everything a
// reader sees with e. It then runs the graph, when there is one, with the
// reference executor and compares the outputs. fast turns on the
// executor's fast paths; without them every composite runs its canonical
// body.
func Check(dir string, e *Expected, fast bool) error {
	b, err := os.ReadFile(filepath.Join(dir, e.File))
	if err != nil {
		return err
	}
	f, err := mmf.Parse(b)
	if err != nil {
		return fmt.Errorf("open: %w", err)
	}
	if err := f.VerifyHashes(); err != nil {
		return err
	}
	got, err := Describe(f)
	if err != nil {
		return err
	}
	if got.Container != e.Container {
		return fmt.Errorf("container %+v, want %+v", got.Container, e.Container)
	}
	if !reflect.DeepEqual(got.Sections, nonNil(e.Sections)) {
		return fmt.Errorf("sections %+v, want %+v", got.Sections, e.Sections)
	}
	if len(got.Tensors) != len(e.Tensors) {
		return fmt.Errorf("%d tensors, want %d", len(got.Tensors), len(e.Tensors))
	}
	for i := range e.Tensors {
		if !reflect.DeepEqual(got.Tensors[i], e.Tensors[i]) {
			return fmt.Errorf("tensor %d is %+v, want %+v", i, got.Tensors[i], e.Tensors[i])
		}
	}
	if len(got.Metadata) != len(e.Metadata) {
		return fmt.Errorf("%d metadata entries, want %d", len(got.Metadata), len(e.Metadata))
	}
	for i := range e.Metadata {
		g, w := got.Metadata[i], e.Metadata[i]
		if g.Key != w.Key || g.Type != w.Type || !jsonEqual(g.Value, w.Value) {
			return fmt.Errorf("metadata %q (%s) = %s, want %q (%s) = %s", g.Key, g.Type, g.Value, w.Key, w.Type, w.Value)
		}
	}
	if !reflect.DeepEqual(got.Graph, e.Graph) {
		return fmt.Errorf("graph %+v, want %+v", got.Graph, e.Graph)
	}
	if len(e.Runs) == 0 {
		return nil
	}
	p, err := interp.Load(f)
	if err != nil {
		return err
	}
	p.FastPaths = fast
	s := p.NewSession()
	for ri, r := range e.Runs {
		in := map[string]*ref.Tensor{}
		for name, td := range r.Inputs {
			t, err := td.Tensor()
			if err != nil {
				return fmt.Errorf("run %d input %q: %w", ri, name, err)
			}
			in[name] = t
		}
		out, err := s.Run(in)
		if err != nil {
			return fmt.Errorf("run %d: %w", ri, err)
		}
		if len(out) != len(r.Outputs) {
			return fmt.Errorf("run %d: %d outputs, want %d", ri, len(out), len(r.Outputs))
		}
		for name, want := range r.Outputs {
			if err := CompareOutput(out[name], want); err != nil {
				return fmt.Errorf("run %d output %q: %w", ri, name, err)
			}
		}
	}
	return nil
}

func jsonEqual(a, b json.RawMessage) bool {
	var x, y any
	da, db := json.NewDecoder(bytes.NewReader(a)), json.NewDecoder(bytes.NewReader(b))
	da.UseNumber()
	db.UseNumber()
	if da.Decode(&x) != nil || db.Decode(&y) != nil {
		return false
	}
	return reflect.DeepEqual(x, y)
}

// Tensor decodes td.
func (td TensorData) Tensor() (*ref.Tensor, error) {
	dt, ok := mmf.ParseDType(td.DType)
	if !ok {
		return nil, fmt.Errorf("dtype %q", td.DType)
	}
	data, err := hex.DecodeString(td.Data)
	if err != nil {
		return nil, err
	}
	t := &ref.Tensor{DType: dt, Shape: append([]int{}, td.Shape...), Data: data}
	return t, t.Validate()
}

// TensorDataOf encodes t.
func TensorDataOf(t *ref.Tensor) TensorData {
	return TensorData{DType: t.DType.String(), Shape: nonNil(append([]int{}, t.Shape...)), Data: hex.EncodeToString(t.Data)}
}

// CompareOutput compares a run's output with its expected value.
func CompareOutput(got *ref.Tensor, want Output) error {
	if got == nil {
		return fmt.Errorf("missing")
	}
	w, err := want.Tensor()
	if err != nil {
		return err
	}
	if got.DType != w.DType || !reflect.DeepEqual(nonNil(got.Shape), nonNil(w.Shape)) {
		return fmt.Errorf("%s%v, want %s%v", got.DType, got.Shape, w.DType, w.Shape)
	}
	switch want.Check {
	case "bits":
		if !bytes.Equal(got.Data, w.Data) {
			return fmt.Errorf("bytes differ")
		}
		return nil
	case "close":
		if got.DType != mmf.F32 {
			return fmt.Errorf("check close needs f32, have %s", got.DType)
		}
		var big, worst float64
		for i := 0; i < w.NumElems(); i++ {
			big = math.Max(big, math.Abs(float64(math.Float32frombits(uint32(w.Bits(i))))))
		}
		for i := 0; i < w.NumElems(); i++ {
			g := float64(math.Float32frombits(uint32(got.Bits(i))))
			x := float64(math.Float32frombits(uint32(w.Bits(i))))
			d := math.Abs(g - x)
			if math.IsNaN(d) {
				return fmt.Errorf("element %d is %v, want %v", i, g, x)
			}
			worst = math.Max(worst, d)
		}
		if worst > want.Tol*big {
			return fmt.Errorf("max |difference| %g, allowed %g", worst, want.Tol*big)
		}
		return nil
	}
	return fmt.Errorf("unknown check %q", want.Check)
}
