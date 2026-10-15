// Command mmf-run generates text with a language model stored as an mmf
// file with a graph, such as one written by `mmf-import -graph`.
//
//	mmf-run -m tinyllama.mmf -p "Once upon a time" [-n 64] [-temp 0] [-seed 1] [-ref]
//
// The prompt is tokenized with the tokenizer in the file's metadata and run
// through the graph in one step; each new token is then one step with the
// key-value cache held in the graph's variables. With -temp 0 (the
// default) the most likely token is chosen; otherwise tokens are sampled
// from the 40 most likely at the given temperature. -ref runs every
// standard-library composite through its canonical body on the reference
// kernels instead of the executor's fast paths: exact to the spec, and
// very slow.
package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"
	"unicode/utf8"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/interp"
	"github.com/gsoultan/mmf/ref"
	"github.com/gsoultan/mmf/tokenizer"
)

func main() {
	model := flag.String("m", "", "model file (.mmf with a graph)")
	prompt := flag.String("p", "", "prompt")
	n := flag.Int("n", 64, "number of tokens to generate")
	temp := flag.Float64("temp", 0, "sampling temperature; 0 picks the most likely token")
	seed := flag.Int64("seed", 1, "random seed for sampling")
	useRef := flag.Bool("ref", false, "run composites through their canonical bodies (slow)")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "usage: mmf-run -m model.mmf -p prompt [-n 64] [-temp 0] [-seed 1] [-ref]")
		os.Exit(2)
	}
	cfg := config{prompt: *prompt, tokens: *n, temp: *temp, seed: *seed, ref: *useRef}
	if err := run(os.Stdout, os.Stderr, *model, cfg); err != nil {
		fmt.Fprintln(os.Stderr, "mmf-run:", err)
		os.Exit(1)
	}
}

type config struct {
	prompt string
	tokens int
	temp   float64
	seed   int64
	ref    bool
}

// model is a loaded language model.
type model struct {
	f       *mmf.File
	prog    *interp.Program
	tok     *tokenizer.SPM
	context int
}

func load(path string, useRef bool) (*model, error) {
	f, err := mmf.Open(path)
	if err != nil {
		return nil, err
	}
	m, err := newModel(f, useRef)
	if err != nil {
		f.Close()
		return nil, err
	}
	return m, nil
}

func newModel(f *mmf.File, useRef bool) (*model, error) {
	md, err := f.Metadata()
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.FromMetadata(md)
	if err != nil {
		return nil, err
	}
	p, err := interp.Load(f)
	if err != nil {
		return nil, err
	}
	p.FastPaths = !useRef
	g := p.Graph()
	main := g.Main()
	if len(main.Inputs) != 2 || len(main.Outputs) != 1 || g.Values[main.Inputs[0]].Name != "tokens" ||
		g.Values[main.Inputs[1]].Name != "positions" || len(g.SymDims) == 0 {
		return nil, fmt.Errorf("the graph does not have the tokens and positions inputs of a language model")
	}
	return &model{f: f, prog: p, tok: tok, context: int(g.SymDims[0].Bound)}, nil
}

// step runs ids at positions pos, pos+1, ... and returns the logits of the
// last one.
func step(s *interp.Session, ids []int32, pos int) ([]float32, error) {
	all, err := stepAll(s, ids, pos)
	if err != nil {
		return nil, err
	}
	v := len(all) / len(ids)
	return all[len(all)-v:], nil
}

// stepAll runs ids at positions pos, pos+1, ... and returns the logits of
// every position, row after row.
func stepAll(s *interp.Session, ids []int32, pos int) ([]float32, error) {
	tokens, _ := ref.New(mmf.I32, []int{1, len(ids)})
	positions, _ := ref.New(mmf.I64, []int{1, len(ids)})
	for i, id := range ids {
		tokens.SetBits(i, uint64(uint32(id)))
		positions.SetBits(i, uint64(pos+i))
	}
	out, err := s.Run(map[string]*ref.Tensor{"tokens": tokens, "positions": positions})
	if err != nil {
		return nil, err
	}
	var logits *ref.Tensor
	for _, t := range out {
		logits = t
	}
	if logits.DType != mmf.F32 || len(logits.Shape) != 3 {
		return nil, fmt.Errorf("logits are %s%v, want f32 [1, T, V]", logits.DType, logits.Shape)
	}
	all := make([]float32, logits.NumElems())
	for i := range all {
		all[i] = math.Float32frombits(uint32(logits.Bits(i)))
	}
	return all, nil
}

func argmax(v []float32) int32 {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return int32(best)
}

// sample draws from the topK most likely tokens at temperature temp.
func sample(v []float32, temp float64, topK int, rng *rand.Rand) int32 {
	idx := make([]int, len(v))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return v[idx[a]] > v[idx[b]] })
	idx = idx[:min(topK, len(idx))]
	w := make([]float64, len(idx))
	var sum float64
	for i, j := range idx {
		w[i] = math.Exp((float64(v[j]) - float64(v[idx[0]])) / temp)
		sum += w[i]
	}
	r := rng.Float64() * sum
	for i, x := range w {
		if r -= x; r <= 0 {
			return int32(idx[i])
		}
	}
	return int32(idx[len(idx)-1])
}

// generate runs the prompt and up to n new tokens, calling emit with each
// new token. It stops early at the end-of-sequence token or a full cache.
func (m *model) generate(prompt string, n int, temp float64, seed int64, emit func(id int32)) (promptTokens int, promptTime, genTime time.Duration, err error) {
	ids := m.tok.Encode(prompt, true)
	if len(ids) == 0 {
		return 0, 0, 0, fmt.Errorf("empty prompt")
	}
	if len(ids) >= m.context {
		return 0, 0, 0, fmt.Errorf("the prompt has %d tokens; the context holds %d", len(ids), m.context)
	}
	s := m.prog.NewSession()
	rng := rand.New(rand.NewSource(seed))
	start := time.Now()
	logits, err := step(s, ids, 0)
	if err != nil {
		return 0, 0, 0, err
	}
	promptTime = time.Since(start)
	start = time.Now()
	pos := len(ids)
	for i := 0; i < n; i++ {
		var next int32
		if temp <= 0 {
			next = argmax(logits)
		} else {
			next = sample(logits, temp, 40, rng)
		}
		if next == m.tok.EOS() {
			break
		}
		emit(next)
		if pos >= m.context || i == n-1 {
			break
		}
		if logits, err = step(s, []int32{next}, pos); err != nil {
			return 0, 0, 0, err
		}
		pos++
	}
	return len(ids), promptTime, time.Since(start), nil
}

func run(stdout, stderr io.Writer, path string, c config) error {
	start := time.Now()
	m, err := load(path, c.ref)
	if err != nil {
		return err
	}
	defer m.f.Close()
	loadTime := time.Since(start)

	fmt.Fprint(stdout, c.prompt)
	var pending []byte // bytes of a UTF-8 character split across tokens
	generated := 0
	np, pt, gt, err := m.generate(c.prompt, c.tokens, c.temp, c.seed, func(id int32) {
		generated++
		pending = append(pending, m.tok.Piece(id)...)
		n := 0
		for n < len(pending) {
			r, size := utf8.DecodeRune(pending[n:])
			if r == utf8.RuneError && !utf8.FullRune(pending[n:]) {
				break
			}
			n += size
		}
		stdout.Write(pending[:n])
		pending = pending[n:]
	})
	stdout.Write(pending)
	fmt.Fprintln(stdout)
	if err != nil {
		return err
	}
	rate := func(n int, d time.Duration) float64 {
		if d <= 0 {
			return 0
		}
		return float64(n) / d.Seconds()
	}
	fmt.Fprintf(stderr, "load %s; prompt %d tokens in %s (%.1f tok/s); generated %d tokens in %s (%.2f tok/s)\n",
		loadTime.Round(time.Millisecond), np, pt.Round(time.Millisecond), rate(np, pt),
		generated, gt.Round(time.Millisecond), rate(max(generated-1, 0), gt))
	return nil
}
