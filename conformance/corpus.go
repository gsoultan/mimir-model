// Package conformance loads the mmf conformance corpus (spec/conformance.md)
// and checks a runtime's results against it.
//
// The corpus lives in testdata/conformance. Each file holds the cases of one
// primitive; each case gives the inputs, attributes and declared output of a
// single operation, the expected result and the rule used to compare a
// runtime's result with it. The expected results come from an independent
// float64 oracle (tools/gen_conformance.py), so the spec, not any one
// implementation, is the source of truth.
package conformance

import (
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/ref"
)

// Mode is a runtime's subnormal handling for float dtypes
// (spec/primitives.md section 3.4).
type Mode int

const (
	IEEE  Mode = iota // subnormals are kept
	Flush             // subnormal inputs and outputs become signed zeros
)

// Check is the comparison rule of a case.
type Check struct {
	Kind   string    // "bits", "exact", "ulp" or "tol"
	ULP    float64   // for "ulp"
	Ref    []float64 // for "tol": per-element reference values
	Bound  []float64 // for "tol": per-element absolute error bounds
	MaxAbs float64   // for "tol": when non-zero, |result| must not exceed it
}

// Expected is an expected output with its comparison rule.
type Expected struct {
	Tensor *ref.Tensor
	Check  Check
}

// Case is one conformance case: a single primitive applied to concrete
// inputs.
type Case struct {
	File     string
	Op       string
	Opcode   mmf.Opcode
	Name     string
	Attrs    []mmf.Attr
	Inputs   []*ref.Tensor
	Output   Expected  // IEEE mode
	FlushOut *Expected // flush mode, when it differs from IEEE mode
	DontCare []int     // flat output indices whose value is unspecified
}

// ID returns "op/name", unique within the corpus.
func (c *Case) ID() string { return c.Op + "/" + c.Name }

// Expected returns the expected output for mode.
func (c *Case) Expected(m Mode) Expected {
	if m == Flush && c.FlushOut != nil {
		return *c.FlushOut
	}
	return c.Output
}

type jsonTensor struct {
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
	Data  string `json:"data"`
}

type jsonCheck struct {
	Kind   string  `json:"kind"`
	ULP    float64 `json:"ulp"`
	Ref    string  `json:"ref"`
	Bound  string  `json:"bound"`
	MaxAbs float64 `json:"max_abs"`
}

type jsonAttr struct {
	Int    *int64  `json:"int"`
	Ints   []int64 `json:"ints"`
	Float  *string `json:"float"`
	String *string `json:"string"`
	DType  *string `json:"dtype"`
}

type jsonCase struct {
	Name     string              `json:"name"`
	Attrs    map[string]jsonAttr `json:"attrs"`
	Inputs   []jsonTensor        `json:"inputs"`
	Output   jsonTensor          `json:"output"`
	Check    jsonCheck           `json:"check"`
	FlushOut *struct {
		Data  string    `json:"data"`
		Check jsonCheck `json:"check"`
	} `json:"output_ftz"`
	DontCare []int `json:"dont_care"`
}

type jsonFile struct {
	Format  string     `json:"format"`
	Version int        `json:"version"`
	Op      string     `json:"op"`
	Opcode  int        `json:"opcode"`
	Cases   []jsonCase `json:"cases"`
}

// LoadDir loads every *.json corpus file in dir, in file name order.
func LoadDir(dir string) ([]Case, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("no corpus files in %s", dir)
	}
	sort.Strings(paths)
	var all []Case
	for _, p := range paths {
		cs, err := LoadFile(p)
		if err != nil {
			return nil, err
		}
		all = append(all, cs...)
	}
	return all, nil
}

// LoadFile loads one corpus file.
func LoadFile(path string) ([]Case, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f jsonFile
	if err := json.Unmarshal(b, &f); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if f.Format != "mmf-conformance" || f.Version != 1 {
		return nil, fmt.Errorf("%s: unsupported corpus format %q version %d", path, f.Format, f.Version)
	}
	op := mmf.Opcode(f.Opcode)
	if op.String() != f.Op {
		return nil, fmt.Errorf("%s: opcode %d is not %q", path, f.Opcode, f.Op)
	}
	cases := make([]Case, 0, len(f.Cases))
	for _, jc := range f.Cases {
		c, err := decodeCase(jc)
		if err != nil {
			return nil, fmt.Errorf("%s: case %q: %w", path, jc.Name, err)
		}
		c.File, c.Op, c.Opcode = filepath.Base(path), f.Op, op
		cases = append(cases, c)
	}
	return cases, nil
}

func decodeCase(jc jsonCase) (Case, error) {
	c := Case{Name: jc.Name, DontCare: jc.DontCare}
	keys := make([]string, 0, len(jc.Attrs))
	for k := range jc.Attrs {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		a, err := decodeAttr(k, jc.Attrs[k])
		if err != nil {
			return c, err
		}
		c.Attrs = append(c.Attrs, a)
	}
	for _, jt := range jc.Inputs {
		t, err := decodeTensor(jt)
		if err != nil {
			return c, err
		}
		c.Inputs = append(c.Inputs, t)
	}
	out, err := decodeTensor(jc.Output)
	if err != nil {
		return c, err
	}
	chk, err := decodeCheck(jc.Check, out.NumElems())
	if err != nil {
		return c, err
	}
	c.Output = Expected{out, chk}
	if jc.FlushOut != nil {
		ft, err := decodeTensor(jsonTensor{jc.Output.DType, jc.Output.Shape, jc.FlushOut.Data})
		if err != nil {
			return c, err
		}
		fc, err := decodeCheck(jc.FlushOut.Check, out.NumElems())
		if err != nil {
			return c, err
		}
		c.FlushOut = &Expected{ft, fc}
	}
	for _, i := range c.DontCare {
		if i < 0 || i >= out.NumElems() {
			return c, fmt.Errorf("dont_care index %d out of range", i)
		}
	}
	return c, nil
}

func decodeAttr(key string, ja jsonAttr) (mmf.Attr, error) {
	a := mmf.Attr{Key: key}
	switch {
	case ja.Int != nil:
		a.Type, a.Int = mmf.AttrInt, *ja.Int
	case ja.Ints != nil:
		a.Type, a.Ints = mmf.AttrInts, ja.Ints
	case ja.Float != nil:
		v, err := decodeF64s(*ja.Float)
		if err != nil || len(v) != 1 {
			return a, fmt.Errorf("attribute %q: bad float", key)
		}
		a.Type, a.Float = mmf.AttrFloat, v[0]
	case ja.String != nil:
		a.Type, a.String = mmf.AttrString, *ja.String
	case ja.DType != nil:
		dt, ok := mmf.ParseDType(*ja.DType)
		if !ok {
			return a, fmt.Errorf("attribute %q: unknown dtype %q", key, *ja.DType)
		}
		a.Type, a.DType = mmf.AttrDType, dt
	default:
		return a, fmt.Errorf("attribute %q has no value", key)
	}
	return a, nil
}

func decodeTensor(jt jsonTensor) (*ref.Tensor, error) {
	dt, ok := mmf.ParseDType(jt.DType)
	if !ok {
		return nil, fmt.Errorf("unknown dtype %q", jt.DType)
	}
	data, err := hex.DecodeString(jt.Data)
	if err != nil {
		return nil, err
	}
	shape := jt.Shape
	if shape == nil {
		shape = []int{}
	}
	t := &ref.Tensor{DType: dt, Shape: shape, Data: data}
	if err := t.Validate(); err != nil {
		return nil, err
	}
	return t, nil
}

func decodeF64s(s string) ([]float64, error) {
	b, err := hex.DecodeString(s)
	if err != nil || len(b)%8 != 0 {
		return nil, fmt.Errorf("bad float64 array")
	}
	v := make([]float64, len(b)/8)
	for i := range v {
		v[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:]))
	}
	return v, nil
}

func decodeCheck(jc jsonCheck, n int) (Check, error) {
	c := Check{Kind: jc.Kind, ULP: jc.ULP, MaxAbs: jc.MaxAbs}
	switch jc.Kind {
	case "bits", "exact":
	case "ulp":
		if jc.ULP <= 0 {
			return c, fmt.Errorf("ulp check without a bound")
		}
	case "tol":
		var err error
		if c.Ref, err = decodeF64s(jc.Ref); err != nil {
			return c, err
		}
		if c.Bound, err = decodeF64s(jc.Bound); err != nil {
			return c, err
		}
		if len(c.Ref) != n || len(c.Bound) != n {
			return c, fmt.Errorf("tol arrays have %d and %d entries for %d elements", len(c.Ref), len(c.Bound), n)
		}
	default:
		return c, fmt.Errorf("unknown check kind %q", jc.Kind)
	}
	return c, nil
}
