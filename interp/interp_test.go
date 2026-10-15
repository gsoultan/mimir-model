package interp

import (
	"bytes"
	"encoding/binary"
	"math"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/conformance"
	"github.com/gsoultan/mmf/ref"
)

// gb builds graphs for the tests.
type gb struct {
	g       *mmf.Graph
	tensors []tensorSpec
}

type tensorSpec struct {
	name  string
	dt    mmf.DType
	shape []uint64
	data  []byte
}

func newGB() *gb {
	return &gb{g: &mmf.Graph{PrimitiveVersion: mmf.CurrentPrimitive, StdlibVersion: mmf.CurrentStdlib,
		Subgraphs: []mmf.Subgraph{{}}}}
}

func (b *gb) sym(name string, bound int64) mmf.Dim {
	b.g.SymDims = append(b.g.SymDims, mmf.SymDim{Name: name, Bound: bound})
	return mmf.Dim(-len(b.g.SymDims))
}

func (b *gb) val(name string, dt mmf.DType, kind mmf.ValueKind, shape ...mmf.Dim) uint32 {
	b.g.Values = append(b.g.Values, mmf.Value{Name: name, DType: dt, Shape: shape, Kind: kind})
	return uint32(len(b.g.Values) - 1)
}

func (b *gb) input(si int, name string, dt mmf.DType, shape ...mmf.Dim) uint32 {
	id := b.val(name, dt, mmf.KindInput, shape...)
	b.g.Subgraphs[si].Inputs = append(b.g.Subgraphs[si].Inputs, id)
	return id
}

// stored adds a weight, const or variable value backed by a file tensor.
func (b *gb) stored(name string, kind mmf.ValueKind, dt mmf.DType, shape []uint64, data []byte) uint32 {
	dims := make([]mmf.Dim, len(shape))
	for i, s := range shape {
		dims[i] = mmf.Dim(s)
	}
	id := b.val(name, dt, kind, dims...)
	b.g.Values[id].Ref = uint32(len(b.tensors))
	b.tensors = append(b.tensors, tensorSpec{name, dt, shape, data})
	return id
}

func (b *gb) op(si int, opc mmf.Opcode, in []uint32, out []uint32, attrs ...mmf.Attr) {
	b.g.Subgraphs[si].Ops = append(b.g.Subgraphs[si].Ops, mmf.Op{Opcode: opc, Inputs: in, Outputs: out, Attrs: attrs})
}

func (b *gb) output(si int, id uint32) {
	if si == 0 {
		b.g.Values[id].Kind = mmf.KindOutput
	}
	b.g.Subgraphs[si].Outputs = append(b.g.Subgraphs[si].Outputs, id)
}

// load encodes the graph and tensors into an mmf file and loads it, so the
// tests go through the real reader.
func (b *gb) load(t testing.TB) *Program {
	t.Helper()
	// The file's tensor index is sorted by name, and a value's Ref is a
	// position in that index.
	order := make([]int, len(b.tensors))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool { return b.tensors[order[i]].name < b.tensors[order[j]].name })
	pos := make([]uint32, len(order))
	for p, i := range order {
		pos[i] = uint32(p)
	}
	for i := range b.g.Values {
		if k := b.g.Values[i].Kind; (k == mmf.KindWeight || k == mmf.KindConst || k == mmf.KindVariable) && !b.g.Values[i].Zero {
			b.g.Values[i].Ref = pos[b.g.Values[i].Ref]
		}
	}
	w := mmf.NewWriter()
	for _, ts := range b.tensors {
		if err := w.AddTensor(ts.name, ts.dt, ts.shape, ts.data); err != nil {
			t.Fatal(err)
		}
	}
	enc, err := b.g.Encode()
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AddSection(mmf.SectionGraph, true, enc); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := w.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	f, err := mmf.Parse(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	p, err := Load(f)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func sAttr(k, v string) mmf.Attr { return mmf.Attr{Key: k, Type: mmf.AttrString, String: v} }
func iAttr(k string, v int64) mmf.Attr {
	return mmf.Attr{Key: k, Type: mmf.AttrInt, Int: v}
}
func fAttr(k string, v float64) mmf.Attr { return mmf.Attr{Key: k, Type: mmf.AttrFloat, Float: v} }
func gAttr(k string, v int) mmf.Attr {
	return mmf.Attr{Key: k, Type: mmf.AttrSubgraph, Subgraph: uint32(v)}
}
func callAttrs(name string, extra ...mmf.Attr) []mmf.Attr {
	return append([]mmf.Attr{sAttr("name", name), iAttr("version", 1)}, extra...)
}

func f32t(shape []int, v ...float32) *ref.Tensor {
	t, _ := ref.New(mmf.F32, shape)
	for i, x := range v {
		t.SetBits(i, uint64(math.Float32bits(x)))
	}
	return t
}

func f32bytes(v ...float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

func i64t(shape []int, v ...int64) *ref.Tensor {
	t, _ := ref.New(mmf.I64, shape)
	for i, x := range v {
		t.SetBits(i, uint64(x))
	}
	return t
}

func f32s(t *ref.Tensor) []float64 {
	r := make([]float64, t.NumElems())
	for i := range r {
		r[i] = float64(math.Float32frombits(uint32(t.Bits(i))))
	}
	return r
}

// TestConformanceThroughGraphs runs every corpus case as a one-op graph
// written to an mmf file, so the executor, the graph encoding and the
// kernels are checked together against the oracle.
func TestConformanceThroughGraphs(t *testing.T) {
	cases, err := conformance.LoadDir(filepath.Join("..", "testdata", "conformance", "primitives"))
	if err != nil {
		t.Fatal(err)
	}
	for i := range cases {
		c := &cases[i]
		b := newGB()
		in := map[string]*ref.Tensor{}
		var ids []uint32
		for k, x := range c.Inputs {
			dims := make([]mmf.Dim, len(x.Shape))
			for j, d := range x.Shape {
				dims[j] = mmf.Dim(d)
			}
			name := string(rune('a' + k))
			ids = append(ids, b.input(0, name, x.DType, dims...))
			in[name] = x
		}
		w := c.Output.Tensor
		dims := make([]mmf.Dim, len(w.Shape))
		for j, d := range w.Shape {
			dims[j] = mmf.Dim(d)
		}
		out := b.val("y", w.DType, mmf.KindOutput, dims...)
		b.op(0, c.Opcode, ids, []uint32{out}, c.Attrs...)
		b.output(0, out)
		p := b.load(t)
		res, err := p.NewSession().Run(in)
		if err != nil {
			t.Errorf("%s: %v", c.ID(), err)
			continue
		}
		if err := c.Verify(res["y"], conformance.IEEE); err != nil {
			t.Errorf("%s: %v", c.ID(), err)
		}
	}
}

// TestKVCache follows the decode-step example of spec/primitives.md
// section 7: a variable cache updated by scatter, persisting across runs.
func TestKVCache(t *testing.T) {
	b := newGB()
	cache := b.stored("k_cache", mmf.KindVariable, mmf.F32, []uint64{1, 4, 2}, make([]byte, 32))
	pos := b.input(0, "pos", mmf.I64, 1)
	kt := b.input(0, "k_t", mmf.F32, 1, 1, 2)
	knew := b.val("k_new", mmf.F32, mmf.KindIntermediate, 1, 4, 2)
	sum := b.val("sum", mmf.F32, mmf.KindIntermediate, 1, 1, 1)
	b.op(0, mmf.OpScatter, []uint32{cache, pos, kt}, []uint32{knew}, iAttr("axis", 1))
	b.op(0, mmf.OpReduce, []uint32{knew}, []uint32{sum}, sAttr("mode", "add"), mmf.Attr{Key: "axes", Type: mmf.AttrInts, Ints: []int64{1, 2}})
	b.output(0, sum)
	b.g.Updates = []mmf.Update{{Variable: cache, Value: knew}}
	p := b.load(t)

	s := p.NewSession()
	for step := 0; step < 3; step++ {
		v := float32(step + 1)
		out, err := s.Run(map[string]*ref.Tensor{"pos": i64t([]int{1}, int64(step)), "k_t": f32t([]int{1, 1, 2}, v, 10*v)})
		if err != nil {
			t.Fatal(err)
		}
		// Each run sees the cache as it was at the start of the run, plus
		// this step's row: the sum of 11 * (1 + ... + step+1).
		want := 11 * float64((step+1)*(step+2)/2)
		if got := f32s(out["sum"])[0]; got != want {
			t.Errorf("step %d: sum %v, want %v", step, got, want)
		}
	}
	c, _ := s.Variable("k_cache")
	if got := f32s(c); got[0] != 1 || got[1] != 10 || got[4] != 3 || got[6] != 0 {
		t.Errorf("cache = %v", got)
	}
	// A failed run changes no variable.
	if _, err := s.Run(map[string]*ref.Tensor{"pos": i64t([]int{1}, 3), "k_t": f32t([]int{1, 2, 2})}); err == nil {
		t.Fatal("accepted a wrongly shaped input")
	}
	if c2, _ := s.Variable("k_cache"); c2 != c {
		t.Error("a failed run changed the cache")
	}
	// A new session starts from the initial value.
	c3, _ := p.NewSession().Variable("k_cache")
	if f32s(c3)[0] != 0 {
		t.Error("a new session does not start from the initial value")
	}
}

// TestWhileIf sums 0..n-1 in a while loop, then picks the sum or its
// negation with an if on a predicate input.
func TestWhileIf(t *testing.T) {
	b := newGB()
	b.g.Subgraphs = make([]mmf.Subgraph, 5)
	n := b.input(0, "n", mmf.I64)
	neg := b.input(0, "negate", mmf.Bool)
	zero := b.val("zero", mmf.I64, mmf.KindIntermediate)
	b.op(0, mmf.OpConst, nil, []uint32{zero}, iAttr("value", 0))
	i1, acc1 := b.val("i1", mmf.I64, 0), b.val("acc1", mmf.I64, 0)
	b.op(0, mmf.OpWhile, []uint32{zero, zero}, []uint32{i1, acc1}, gAttr("cond", 1), gAttr("body", 2))
	res := b.val("result", mmf.I64, 0)
	b.op(0, mmf.OpIf, []uint32{neg, acc1}, []uint32{res}, gAttr("then", 3), gAttr("else", 4))
	b.output(0, res)

	// cond: i < n, reading n from the enclosing graph
	ci, ca := b.val("ci", mmf.I64, 0), b.val("ca", mmf.I64, 0)
	b.g.Subgraphs[1].Inputs = []uint32{ci, ca}
	lt := b.val("lt", mmf.Bool, 0)
	b.op(1, mmf.OpCmplt, []uint32{ci, n}, []uint32{lt})
	b.output(1, lt)
	// body: (i + 1, acc + i)
	bi, ba := b.val("bi", mmf.I64, 0), b.val("ba", mmf.I64, 0)
	b.g.Subgraphs[2].Inputs = []uint32{bi, ba}
	one, ni, na := b.val("one", mmf.I64, 0), b.val("ni", mmf.I64, 0), b.val("na", mmf.I64, 0)
	b.op(2, mmf.OpConst, nil, []uint32{one}, iAttr("value", 1))
	b.op(2, mmf.OpAdd, []uint32{bi, one}, []uint32{ni})
	b.op(2, mmf.OpAdd, []uint32{ba, bi}, []uint32{na})
	b.output(2, ni)
	b.output(2, na)
	// then: -x; else: x + 0
	tx, ty := b.val("tx", mmf.I64, 0), b.val("ty", mmf.I64, 0)
	b.g.Subgraphs[3].Inputs = []uint32{tx}
	b.op(3, mmf.OpNeg, []uint32{tx}, []uint32{ty})
	b.output(3, ty)
	ex, ez, ey := b.val("ex", mmf.I64, 0), b.val("ez", mmf.I64, 0), b.val("ey", mmf.I64, 0)
	b.g.Subgraphs[4].Inputs = []uint32{ex}
	b.op(4, mmf.OpConst, nil, []uint32{ez}, iAttr("value", 0))
	b.op(4, mmf.OpAdd, []uint32{ex, ez}, []uint32{ey})
	b.output(4, ey)
	p := b.load(t)

	boolT := func(v bool) *ref.Tensor {
		x, _ := ref.New(mmf.Bool, nil)
		if v {
			x.Data[0] = 1
		}
		return x
	}
	for _, c := range []struct {
		n    int64
		neg  bool
		want int64
	}{{10, false, 45}, {10, true, -45}, {0, false, 0}, {1, true, 0}} {
		out, err := p.NewSession().Run(map[string]*ref.Tensor{"n": i64t(nil, c.n), "negate": boolT(c.neg)})
		if err != nil {
			t.Fatal(err)
		}
		if got := int64(out["result"].Bits(0)); got != c.want {
			t.Errorf("n=%d negate=%v: %d, want %d", c.n, c.neg, got, c.want)
		}
	}
	p.MaxIterations = 5
	if _, err := p.NewSession().Run(map[string]*ref.Tensor{"n": i64t(nil, 100), "negate": boolT(false)}); err == nil ||
		!strings.Contains(err.Error(), "iterations") {
		t.Errorf("iteration limit: %v", err)
	}
}

// TestMLP runs a small shape-polymorphic network with stdlib calls:
// softmax(rms_norm(silu(linear(x, w, bias)))) over a symbolic batch.
func TestMLP(t *testing.T) {
	b := newGB()
	B := b.sym("batch", 16)
	x := b.input(0, "x", mmf.F32, B, 3)
	wv := []float32{0.5, -1, 2, 1.5, 0.25, -0.75}
	bv := []float32{0.1, -0.2}
	w := b.stored("w", mmf.KindWeight, mmf.F32, []uint64{2, 3}, f32bytes(wv...))
	bias := b.stored("bias", mmf.KindWeight, mmf.F32, []uint64{2}, f32bytes(bv...))
	h1, h2, h3, y := b.val("h1", mmf.F32, 0, B, 2), b.val("h2", mmf.F32, 0, B, 2), b.val("h3", mmf.F32, 0, B, 2), b.val("y", mmf.F32, 0, B, 2)
	b.op(0, mmf.OpCall, []uint32{x, w, bias}, []uint32{h1}, callAttrs("linear")...)
	b.op(0, mmf.OpCall, []uint32{h1}, []uint32{h2}, callAttrs("silu")...)
	b.op(0, mmf.OpCall, []uint32{h2}, []uint32{h3}, callAttrs("rms_norm", fAttr("epsilon", 1e-5))...)
	b.op(0, mmf.OpCall, []uint32{h3}, []uint32{y}, callAttrs("softmax", iAttr("axis", -1))...)
	b.output(0, y)
	p := b.load(t)

	for _, batch := range []int{1, 4} {
		xs := make([]float32, 3*batch)
		for i := range xs {
			xs[i] = float32(i%5) - 1.5
		}
		out, err := p.NewSession().Run(map[string]*ref.Tensor{"x": f32t([]int{batch, 3}, xs...)})
		if err != nil {
			t.Fatal(err)
		}
		got := f32s(out["y"])
		for r := 0; r < batch; r++ {
			var h [2]float64
			for j := 0; j < 2; j++ {
				s := float64(bv[j])
				for k := 0; k < 3; k++ {
					s += float64(xs[3*r+k]) * float64(wv[3*j+k])
				}
				h[j] = s / (1 + math.Exp(-s))
			}
			ms := (h[0]*h[0] + h[1]*h[1]) / 2
			for j := range h {
				h[j] /= math.Sqrt(ms + 1e-5)
			}
			m := math.Max(h[0], h[1])
			e0, e1 := math.Exp(h[0]-m), math.Exp(h[1]-m)
			want := []float64{e0 / (e0 + e1), e1 / (e0 + e1)}
			for j := 0; j < 2; j++ {
				if d := math.Abs(got[2*r+j] - want[j]); d > 1e-6 {
					t.Errorf("batch %d row %d col %d: %v, want %v", batch, r, j, got[2*r+j], want[j])
				}
			}
		}
	}
	if _, err := p.NewSession().Run(map[string]*ref.Tensor{"x": f32t([]int{17, 3})}); err == nil {
		t.Error("accepted a batch above the symbolic bound")
	}
}

// TestMatmulBatchBroadcast checks matmul.v1 with a broadcast batch dimension
// and integer operands, which wrap.
func TestMatmulBatchBroadcast(t *testing.T) {
	b := newGB()
	a := b.input(0, "a", mmf.I64, 2, 1, 2)
	m := b.input(0, "m", mmf.I64, 1, 2, 2)
	y := b.val("y", mmf.I64, 0, 2, 1, 2)
	b.op(0, mmf.OpCall, []uint32{a, m}, []uint32{y}, callAttrs("matmul")...)
	b.output(0, y)
	p := b.load(t)
	out, err := p.NewSession().Run(map[string]*ref.Tensor{
		"a": i64t([]int{2, 1, 2}, 1, 2, 3, math.MaxInt64),
		"m": i64t([]int{1, 2, 2}, 5, 6, 7, 2),
	})
	if err != nil {
		t.Fatal(err)
	}
	mx := int64(math.MaxInt64)
	want := []int64{19, 10, 15 + 7*mx, 18 + 2*mx} // wraps modulo 2^64
	for i, w := range want {
		if got := int64(out["y"].Bits(i)); got != w {
			t.Errorf("y[%d] = %d, want %d", i, got, w)
		}
	}
}

// TestLocalComposite calls a model-local composite from the composites
// table.
func TestLocalComposite(t *testing.T) {
	b := newGB()
	b.g.Subgraphs = make([]mmf.Subgraph, 2)
	x := b.input(0, "x", mmf.F32, 3)
	y := b.val("y", mmf.F32, 0, 3)
	b.op(0, mmf.OpCall, []uint32{x}, []uint32{y}, sAttr("name", "acme/double"), iAttr("version", 2))
	b.output(0, y)
	cx, cy := b.val("cx", mmf.F32, 0, 3), b.val("cy", mmf.F32, 0, 3)
	b.g.Subgraphs[1].Inputs = []uint32{cx}
	b.op(1, mmf.OpAdd, []uint32{cx, cx}, []uint32{cy})
	b.output(1, cy)
	b.g.Composites = []mmf.Composite{{Name: "acme/double", Version: 2, Subgraph: 1}}
	p := b.load(t)
	out, err := p.NewSession().Run(map[string]*ref.Tensor{"x": f32t([]int{3}, 1, -2, 3.5)})
	if err != nil {
		t.Fatal(err)
	}
	if got := f32s(out["y"]); got[0] != 2 || got[1] != -4 || got[2] != 7 {
		t.Errorf("y = %v", got)
	}
}

func TestRunErrors(t *testing.T) {
	b := newGB()
	S := b.sym("seq", -1)
	x := b.input(0, "x", mmf.F32, S)
	z := b.input(0, "z", mmf.F32, S)
	y, u := b.val("y", mmf.F32, 0, S), b.val("u", mmf.F32, 0, S)
	b.op(0, mmf.OpAdd, []uint32{x, z}, []uint32{y})
	b.op(0, mmf.OpCall, []uint32{y}, []uint32{u}, callAttrs("argsort")...)
	b.output(0, u)
	p := b.load(t)
	s := p.NewSession()
	cases := map[string]map[string]*ref.Tensor{
		"missing input":      {"x": f32t([]int{2})},
		"extra input":        {"x": f32t([]int{2}), "z": f32t([]int{2}), "q": f32t([]int{2})},
		"inconsistent dims":  {"x": f32t([]int{2}), "z": f32t([]int{3})},
		"wrong dtype":        {"x": f32t([]int{2}), "z": i64t([]int{2})},
		"unsupported stdlib": {"x": f32t([]int{2}), "z": f32t([]int{2})},
	}
	for name, in := range cases {
		if _, err := s.Run(in); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// TestZeroVariable runs a cache that starts as zeros and is not stored in
// the file: each session gets its own zeroed buffer.
func TestZeroVariable(t *testing.T) {
	b := newGB()
	cache := b.val("k_cache", mmf.F32, mmf.KindVariable, 1, 4, 2)
	b.g.Values[cache].Zero = true
	pos := b.input(0, "pos", mmf.I64, 1)
	kt := b.input(0, "k_t", mmf.F32, 1, 1, 2)
	knew := b.val("k_new", mmf.F32, mmf.KindIntermediate, 1, 4, 2)
	b.op(0, mmf.OpScatter, []uint32{cache, pos, kt}, []uint32{knew}, iAttr("axis", 1))
	b.g.Updates = []mmf.Update{{Variable: cache, Value: knew}}
	p := b.load(t)
	p.FastPaths = true

	s1, s2 := p.NewSession(), p.NewSession()
	if _, err := s1.Run(map[string]*ref.Tensor{"pos": i64t([]int{1}, 2), "k_t": f32t([]int{1, 1, 2}, 5, 6)}); err != nil {
		t.Fatal(err)
	}
	c1, _ := s1.Variable("k_cache")
	c2, _ := s2.Variable("k_cache")
	if got := f32s(c1); got[4] != 5 || got[5] != 6 || got[0] != 0 {
		t.Errorf("cache = %v", got)
	}
	for _, v := range f32s(c2) {
		if v != 0 {
			t.Fatalf("another session's cache changed: %v", f32s(c2))
		}
	}
}

func TestZeroVariableTooLarge(t *testing.T) {
	b := newGB()
	cache := b.val("k_cache", mmf.F32, mmf.KindVariable, 1<<30, 1<<30)
	b.g.Values[cache].Zero = true
	x := b.input(0, "x", mmf.F32, 1<<30, 1<<30)
	b.g.Updates = []mmf.Update{{Variable: cache, Value: x}}
	if _, err := New(b.g, nil); err == nil || !strings.Contains(err.Error(), "larger than") {
		t.Fatalf("got %v", err)
	}
}
