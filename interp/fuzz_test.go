package interp

import (
	"bytes"
	"sort"
	"testing"

	mim "github.com/gsoultan/mimir-model"
	"github.com/gsoultan/mimir-model/ref"
)

// FuzzRun builds a graph of one primitive or call from arbitrary bytes,
// with inputs and an output of arbitrary small shapes and dtypes and
// arbitrary attributes, loads it through the real reader and runs it on a
// session: nothing may panic, whatever the file declares. A file's graph
// passes Validate before it runs, so this is what a hostile upload can
// reach.
func FuzzRun(f *testing.F) {
	var codes []mim.Opcode
	for c := mim.OpNeg; c <= mim.OpCall; c++ {
		codes = append(codes, c)
	}
	names := []string{"matmul", "linear", "softmax", "rms_norm", "exp", "sigmoid", "conv2d", "silu", "embedding"}
	dtypes := []mim.DType{mim.F32, mim.F16, mim.BF16, mim.F64, mim.I8, mim.U8, mim.I32, mim.I64, mim.Bool}
	keys := []string{"axis", "axes", "perm", "pads", "begin", "end", "shape", "mode", "to", "value", "k",
		"strides", "dilations", "groups", "eps", "transpose", "dim", "size", "steps", "pad"}
	f.Add([]byte{0, 1, 2, 3, 4, 0, 2, 3, 4})
	f.Add([]byte{33, 3, 0, 2, 2, 3, 0, 2, 3, 3, 0, 1, 2, 2, 1, 7})
	f.Add([]byte{29, 1, 0, 2, 4, 4, 0, 2, 2, 2, 2, 4, 1, 1, 2, 5, 2, 3, 3})
	f.Add([]byte{23, 1, 0, 2, 4, 4, 0, 2, 2, 2, 2, 4, 1, 1, 2, 5, 2, 3, 3, 9, 1, 2, 3})
	f.Fuzz(func(t *testing.T, b []byte) {
		next := func() int {
			if len(b) == 0 {
				return 0
			}
			v := int(b[0])
			b = b[1:]
			return v
		}
		shape := func() []mim.Dim {
			s := make([]mim.Dim, next()%5)
			for i := range s {
				s[i] = mim.Dim(next() % 5)
			}
			return s
		}
		code := codes[next()%len(codes)]
		gb := newGB()
		in := make([]uint32, next()%4)
		feeds := map[string]*ref.Tensor{}
		for i := range in {
			name := string(rune('a' + i))
			dt := dtypes[next()%len(dtypes)]
			s := shape()
			if next()%3 == 0 {
				// A stored weight rather than an input, so the loader's
				// checks of file tensors against values are reached too.
				us := make([]uint64, len(s))
				n := 1
				for k, d := range s {
					us[k] = uint64(d)
					n *= int(d)
				}
				data := make([]byte, n*int(dt.BlockBytes()))
				for k := range data {
					data[k] = byte(next())
				}
				in[i] = gb.stored(name, mim.KindWeight, dt, us, data)
				continue
			}
			in[i] = gb.input(0, name, dt, s...)
			is := make([]int, len(s))
			for k, d := range s {
				is[k] = int(d)
			}
			x, err := ref.New(dt, is)
			if err != nil {
				return
			}
			for k := range x.Data {
				x.Data[k] = byte(next())
			}
			feeds[name] = x
		}
		var attrs []mim.Attr
		if code == mim.OpCall {
			attrs = callAttrs(names[next()%len(names)])
		}
		for range next() % 4 {
			key := keys[next()%len(keys)]
			dup := false
			for _, a := range attrs {
				dup = dup || a.Key == key
			}
			if dup {
				continue
			}
			switch next() % 4 {
			case 0:
				attrs = append(attrs, iAttr(key, int64(next()%7)-2))
			case 1:
				v := make([]int64, next()%7)
				for i := range v {
					v[i] = int64(next()%7) - 1
				}
				attrs = append(attrs, mim.Attr{Key: key, Type: mim.AttrInts, Ints: v})
			case 2:
				attrs = append(attrs, sAttr(key, []string{"add", "max", "mul", "none", "constant"}[next()%5]))
			default:
				attrs = append(attrs, fAttr(key, float64(next()%7)-2))
			}
		}
		out := gb.val("y", dtypes[next()%len(dtypes)], mim.KindIntermediate, shape()...)
		gb.op(0, code, in, []uint32{out}, attrs...)
		gb.output(0, out)
		p, ok := loadFuzz(t, gb)
		if !ok {
			return
		}
		p.FastPaths = next()%2 == 0
		p.MaxBytes = 64 << 20
		s := p.NewSession()
		res, err := s.Run(feeds)
		if err == nil {
			_ = res["y"]
		}
	})
}

// loadFuzz is gb.load for a graph that may be invalid: it returns false
// for a graph or file the reader or loader refuses.
func loadFuzz(t *testing.T, b *gb) (*Program, bool) {
	t.Helper()
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
		if k := b.g.Values[i].Kind; (k == mim.KindWeight || k == mim.KindConst || k == mim.KindVariable) && !b.g.Values[i].Zero {
			b.g.Values[i].Ref = pos[b.g.Values[i].Ref]
		}
	}
	if b.g.Validate() != nil {
		return nil, false
	}
	w := mim.NewWriter()
	for _, ts := range b.tensors {
		if err := w.AddTensor(ts.name, ts.dt, ts.shape, ts.data); err != nil {
			return nil, false
		}
	}
	enc, err := b.g.Encode()
	if err != nil {
		return nil, false
	}
	if err := w.AddSection(mim.SectionGraph, true, enc); err != nil {
		return nil, false
	}
	var buf bytes.Buffer
	if _, err := w.WriteTo(&buf); err != nil {
		return nil, false
	}
	f, err := mim.Parse(buf.Bytes())
	if err != nil {
		return nil, false
	}
	p, err := Load(f)
	if err != nil {
		return nil, false
	}
	return p, true
}
