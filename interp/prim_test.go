package interp

import (
	"bytes"
	"testing"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/ref"
)

// TestScatterInPlace runs a cache update with and without the primitive
// fast paths, which write the cache in place from the second run on, and
// checks that a run failing after the scatter leaves the cache unchanged.
func TestScatterInPlace(t *testing.T) {
	b := newGB()
	b.g.Subgraphs = make([]mmf.Subgraph, 3)
	cache := b.stored("k_cache", mmf.KindVariable, mmf.F32, []uint64{1, 4, 2}, make([]byte, 32))
	pos := b.input(0, "pos", mmf.I64, 1)
	kt := b.input(0, "k_t", mmf.F32, 1, 1, 2)
	fail := b.input(0, "fail", mmf.Bool)
	knew := b.val("k_new", mmf.F32, mmf.KindIntermediate, 1, 4, 2)
	sum := b.val("sum", mmf.F32, mmf.KindIntermediate, 1, 1, 1)
	res := b.val("res", mmf.F32, mmf.KindIntermediate, 1, 1, 1)
	b.op(0, mmf.OpScatter, []uint32{cache, pos, kt}, []uint32{knew}, iAttr("axis", 1))
	b.op(0, mmf.OpReduce, []uint32{knew}, []uint32{sum}, sAttr("mode", "add"), mmf.Attr{Key: "axes", Type: mmf.AttrInts, Ints: []int64{1, 2}})
	b.op(0, mmf.OpIf, []uint32{fail, sum}, []uint32{res}, gAttr("then", 1), gAttr("else", 2))
	b.output(0, res)
	// then: a composite this executor does not have, so the run fails
	// after the scatter. else: pass the sum through.
	subIn := func(si int, name string) uint32 {
		id := b.val(name, mmf.F32, mmf.KindIntermediate, 1, 1, 1)
		b.g.Subgraphs[si].Inputs = append(b.g.Subgraphs[si].Inputs, id)
		return id
	}
	x1 := subIn(1, "x1")
	y1 := b.val("y1", mmf.F32, mmf.KindIntermediate, 1, 1, 1)
	b.op(1, mmf.OpCall, []uint32{x1}, []uint32{y1}, callAttrs("no_such_composite")...)
	b.output(1, y1)
	x2 := subIn(2, "x2")
	b.output(2, x2)
	b.g.Updates = []mmf.Update{{Variable: cache, Value: knew}}
	p := b.load(t)
	if len(p.inplace) != 1 {
		t.Fatalf("%d scatters may run in place, want 1", len(p.inplace))
	}

	no := &ref.Tensor{DType: mmf.Bool, Shape: []int{}, Data: []byte{0}}
	yes := &ref.Tensor{DType: mmf.Bool, Shape: []int{}, Data: []byte{1}}
	step := func(s *Session, i int64, f *ref.Tensor) (map[string]*ref.Tensor, error) {
		v := float32(i + 1)
		return s.Run(map[string]*ref.Tensor{"pos": i64t([]int{1}, i%4), "k_t": f32t([]int{1, 1, 2}, v, -v), "fail": f})
	}
	p.FastPaths = false
	slow := p.NewSession()
	p.FastPaths = true
	fast := p.NewSession()
	for i := int64(0); i < 6; i++ {
		p.FastPaths = false
		want, err := step(slow, i, no)
		if err != nil {
			t.Fatal(err)
		}
		p.FastPaths = true
		before, _ := fast.Variable("k_cache")
		saved := append([]byte{}, before.Data...)
		if _, err := step(fast, i+100, yes); err == nil {
			t.Fatal("the failing branch did not fail")
		}
		if after, _ := fast.Variable("k_cache"); after != before || !bytes.Equal(after.Data, saved) {
			t.Fatalf("step %d: a failed run changed the cache", i)
		}
		got, err := step(fast, i, no)
		if err != nil {
			t.Fatal(err)
		}
		if f32s(got["res"])[0] != f32s(want["res"])[0] {
			t.Errorf("step %d: in place %v, reference %v", i, f32s(got["res"]), f32s(want["res"]))
		}
	}
	a, _ := slow.Variable("k_cache")
	c, _ := fast.Variable("k_cache")
	if !bytes.Equal(a.Data, c.Data) {
		t.Errorf("caches differ: %v and %v", f32s(a), f32s(c))
	}
	// The file's initial value was never written.
	if init, _ := p.NewSession().Variable("k_cache"); !bytes.Equal(init.Data, make([]byte, 32)) {
		t.Error("the initial value changed")
	}
}

func TestScatterInPlaceNeedsSoleUse(t *testing.T) {
	b := newGB()
	cache := b.stored("k_cache", mmf.KindVariable, mmf.F32, []uint64{1, 4, 2}, make([]byte, 32))
	pos := b.input(0, "pos", mmf.I64, 1)
	kt := b.input(0, "k_t", mmf.F32, 1, 1, 2)
	knew := b.val("k_new", mmf.F32, mmf.KindIntermediate, 1, 4, 2)
	old := b.val("old", mmf.F32, mmf.KindIntermediate, 1, 4, 2)
	b.op(0, mmf.OpScatter, []uint32{cache, pos, kt}, []uint32{knew}, iAttr("axis", 1))
	b.op(0, mmf.OpAdd, []uint32{cache, knew}, []uint32{old}) // a second use of the old value
	b.output(0, old)
	b.g.Updates = []mmf.Update{{Variable: cache, Value: knew}}
	if p := b.load(t); len(p.inplace) != 0 {
		t.Error("a scatter whose input has another use may run in place")
	}
}

func TestFastPermute(t *testing.T) {
	x := f32t([]int{1, 3, 1, 2}, 1, 2, 3, 4, 5, 6)
	perm := func(p ...int64) *mmf.Op {
		return &mmf.Op{Opcode: mmf.OpPermute, Attrs: []mmf.Attr{{Key: "perm", Type: mmf.AttrInts, Ints: p}}}
	}
	for _, c := range []struct {
		perm  []int64
		shape []int
		fast  bool
	}{
		{[]int64{0, 2, 1, 3}, []int{1, 1, 3, 2}, true},
		{[]int64{2, 1, 0, 3}, []int{1, 3, 1, 2}, true},
		{[]int64{0, 3, 2, 1}, []int{1, 2, 1, 3}, false},
	} {
		got, ok := fastPermute(perm(c.perm...), []*ref.Tensor{x}, mmf.F32, c.shape)
		if ok != c.fast {
			t.Errorf("perm %v: handled %v", c.perm, ok)
			continue
		}
		if !ok {
			continue
		}
		want, err := ref.Eval(mmf.OpPermute, perm(c.perm...).Attrs, []*ref.Tensor{x}, mmf.F32, c.shape)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got.Data, want.Data) {
			t.Errorf("perm %v: %v, want %v", c.perm, f32s(got), f32s(want))
		}
	}
}
