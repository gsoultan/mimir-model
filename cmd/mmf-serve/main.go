// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mmf, created and owned by Gembit Soultan Shirazi.
// It is licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License in the LICENSE file at the root
// of this repository or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

// Command mmf-serve serves text generation over HTTP with one loaded
// language model, running several requests at the same time.
//
//	mmf-import -graph -ctx 2048 -slots 4 model.gguf
//	mmf-serve -m model.mmf [-addr localhost:8080]
//	mmf-serve -m model.mmf -bench 4 [-p prompt] [-n 64]
//
// The model must be imported with -slots N: its key-value cache then
// holds N sequences, and N requests run together. Each step of the
// server runs one block of tokens through the graph: the next token of
// every request that is generating, and the prompts of requests that just
// started, so the matrix products read each weight once for all of them.
// Requests beyond N wait for a slot.
//
// POST /generate with a JSON body
//
//	{"prompt": "Once upon a time", "max_tokens": 64, "temperature": 0, "seed": 1}
//
// returns {"text": ..., "tokens": ..., "prompt_tokens": ..., "seconds": ...}
// when the request is done. Greedy decoding (temperature 0) is the
// default; otherwise tokens are sampled from the 40 most likely.
//
// -bench N runs N copies of the prompt at once in the process, then the
// same N one after another, and prints the generation speed of each.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/interp"
	"github.com/gsoultan/mmf/ref"
	"github.com/gsoultan/mmf/tokenizer"
)

func main() {
	model := flag.String("m", "", "model file (.mmf imported with -graph -slots N)")
	addr := flag.String("addr", "localhost:8080", "address to listen on")
	bench := flag.Int("bench", 0, "instead of serving, run this many requests at once and then one after another, and print the speeds")
	prompt := flag.String("p", "Once upon a time there was a little girl who lived in a village near the forest.", "with -bench, the prompt")
	n := flag.Int("n", 64, "with -bench, tokens to generate per request")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "usage: mmf-serve -m model.mmf [-addr localhost:8080] [-bench N [-p prompt] [-n 64]]")
		os.Exit(2)
	}
	e, err := open(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "mmf-serve:", err)
		os.Exit(1)
	}
	go e.loop()
	if *bench > 0 {
		if err := runBench(os.Stdout, e, *bench, *prompt, *n); err != nil {
			fmt.Fprintln(os.Stderr, "mmf-serve:", err)
			os.Exit(1)
		}
		return
	}
	fmt.Fprintf(os.Stderr, "mmf-serve: %d slots of %d positions, listening on %s\n", e.slots, e.context, *addr)
	if err := http.ListenAndServe(*addr, e.handler()); err != nil {
		fmt.Fprintln(os.Stderr, "mmf-serve:", err)
		os.Exit(1)
	}
}

// engine runs requests on one model, in the slots of its cache.
type engine struct {
	f       *mmf.File
	prog    *interp.Program
	tok     tokenizer.Tokenizer
	slots   int
	context int // positions per slot
	queue   chan *request
	// maxBlock bounds the tokens of one step, so that a long prompt does
	// not hold back the requests that are generating for long: a prompt
	// runs over several steps if need be.
	maxBlock int
}

// request is one generation request and, once admitted, its sequence.
type request struct {
	prompt []int32
	n      int
	temp   float64
	rng    *rand.Rand
	done   chan result

	slot    int
	pos     int     // tokens in the cache
	pending []int32 // tokens to run at the next step
	decode  bool    // the prompt is done: pending is the last token
	out     []int32
	start   time.Time
}

type result struct {
	tokens []int32
	err    error
}

func open(path string) (*engine, error) {
	f, err := mmf.Open(path)
	if err != nil {
		return nil, err
	}
	e, err := newEngine(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	return e, nil
}

func newEngine(f *mmf.File) (*engine, error) {
	md, err := f.Metadata()
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.New(md)
	if err != nil {
		return nil, err
	}
	p, err := interp.Load(f)
	if err != nil {
		return nil, err
	}
	p.FastPaths = true
	g := p.Graph()
	main := g.Main()
	slots, _ := md["mmf.slots"].(int64)
	ctx, _ := md["mmf.context_length"].(int64)
	if len(main.Inputs) != 3 || g.Values[main.Inputs[0]].Name != "tokens" || g.Values[main.Inputs[1]].Name != "positions" ||
		g.Values[main.Inputs[2]].Name != "slots" || slots < 2 || ctx < 1 {
		return nil, fmt.Errorf("the graph has no slots for several sequences: import the model with mmf-import -graph -slots N")
	}
	return &engine{
		f: f, prog: p, tok: tok, slots: int(slots), context: int(ctx),
		queue: make(chan *request), maxBlock: max(512, int(slots)),
	}, nil
}

// generate runs a request to the end: up to n tokens after the prompt,
// stopping early at the end-of-sequence token or a full slot.
func (e *engine) generate(prompt string, n int, temp float64, seed int64) ([]int32, int, error) {
	ids := e.tok.Encode(prompt, true)
	if len(ids) == 0 {
		return nil, 0, errors.New("empty prompt")
	}
	if len(ids) >= e.context {
		return nil, 0, fmt.Errorf("the prompt has %d tokens; a slot holds %d", len(ids), e.context)
	}
	r := &request{prompt: ids, n: n, temp: temp, rng: rand.New(rand.NewSource(seed)), done: make(chan result, 1)}
	e.queue <- r
	res := <-r.done
	return res.tokens, len(ids), res.err
}

// loop owns the session: it admits requests into free slots and runs a
// step for all the active ones until none is left, then waits.
func (e *engine) loop() {
	s := e.prog.NewSession()
	free := make([]int, e.slots)
	for i := range free {
		free[i] = e.slots - 1 - i
	}
	var active []*request
	admit := func(r *request) {
		if r.n <= 0 {
			r.done <- result{}
			return
		}
		r.slot, free = free[len(free)-1], free[:len(free)-1]
		r.pending, r.start = r.prompt, time.Now()
		active = append(active, r)
	}
	for {
		if len(active) == 0 {
			admit(<-e.queue)
		}
	more:
		for len(free) > 0 {
			select {
			case r := <-e.queue:
				admit(r)
			default:
				break more
			}
		}
		var tokens []int32
		var positions, slots []int64
		type part struct {
			r    *request
			last int // the row of its last token in the block, -1 if none
		}
		// The requests that are generating go first, so that prompts
		// cannot crowd them out of the block.
		sort.SliceStable(active, func(i, j int) bool { return active[i].decode && !active[j].decode })
		parts := make([]part, len(active))
		for i, r := range active {
			parts[i] = part{r, -1}
			k := len(r.pending)
			if room := e.maxBlock - len(tokens); k > room {
				k = max(0, room)
			}
			for j := range k {
				tokens = append(tokens, r.pending[j])
				positions = append(positions, int64(r.pos+j))
				slots = append(slots, int64(r.slot))
			}
			r.pending, r.pos = r.pending[k:], r.pos+k
			if k > 0 && len(r.pending) == 0 {
				parts[i].last = len(tokens) - 1
			}
		}
		logits, err := e.step(s, tokens, positions, slots)
		var keep []*request
		for _, p := range parts {
			r := p.r
			if err != nil {
				r.done <- result{err: err}
				free = append(free, r.slot)
				continue
			}
			if p.last < 0 {
				keep = append(keep, r) // the rest of its prompt runs next step
				continue
			}
			v := e.tok.VocabSize()
			row := logits[p.last*v : (p.last+1)*v]
			var next int32
			if r.temp <= 0 {
				next = argmax(row)
			} else {
				next = sample(row, r.temp, 40, r.rng)
			}
			stop := next == e.tok.EOS()
			if !stop {
				r.out = append(r.out, next)
			}
			if stop || len(r.out) >= r.n || r.pos >= e.context {
				r.done <- result{tokens: r.out}
				free = append(free, r.slot)
				continue
			}
			r.pending, r.decode = []int32{next}, true
			keep = append(keep, r)
		}
		active = keep
	}
}

// step runs one block of tokens, each at its position in its slot, and
// returns the logits of every row.
func (e *engine) step(s *interp.Session, ids []int32, positions, slots []int64) ([]float32, error) {
	t := len(ids)
	tokens, _ := ref.New(mmf.I32, []int{1, t})
	pos, _ := ref.New(mmf.I64, []int{1, t})
	sl, _ := ref.New(mmf.I64, []int{1, t})
	for i := range ids {
		tokens.SetBits(i, uint64(uint32(ids[i])))
		pos.SetBits(i, uint64(positions[i]))
		sl.SetBits(i, uint64(slots[i]))
	}
	out, err := s.Run(map[string]*ref.Tensor{"tokens": tokens, "positions": pos, "slots": sl})
	if err != nil {
		return nil, err
	}
	var logits *ref.Tensor
	for _, v := range out {
		logits = v
	}
	if logits.DType != mmf.F32 || len(logits.Shape) != 3 || logits.Shape[1] != t || logits.Shape[2] != e.tok.VocabSize() {
		return nil, fmt.Errorf("logits are %s%v, want f32 [1, %d, %d]", logits.DType, logits.Shape, t, e.tok.VocabSize())
	}
	v := make([]float32, logits.NumElems())
	for i := range v {
		v[i] = math.Float32frombits(uint32(logits.Bits(i)))
	}
	return v, nil
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

type generateRequest struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   *int    `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
	Seed        int64   `json:"seed"`
}

type generateResponse struct {
	Text         string  `json:"text"`
	Tokens       []int32 `json:"tokens"`
	PromptTokens int     `json:"prompt_tokens"`
	Seconds      float64 `json:"seconds"`
}

// maxBody bounds a request body.
const maxBody = 1 << 20

func (e *engine) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /generate", func(w http.ResponseWriter, r *http.Request) {
		var req generateRequest
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBody))
		if err == nil {
			err = json.Unmarshal(body, &req)
		}
		if err != nil {
			http.Error(w, "bad request body: "+err.Error(), http.StatusBadRequest)
			return
		}
		n := 64
		if req.MaxTokens != nil {
			n = *req.MaxTokens
		}
		if n < 0 || n > e.context {
			http.Error(w, fmt.Sprintf("max_tokens must be between 0 and %d", e.context), http.StatusBadRequest)
			return
		}
		if req.Temperature < 0 || math.IsNaN(req.Temperature) {
			http.Error(w, "temperature must not be negative", http.StatusBadRequest)
			return
		}
		start := time.Now()
		ids, np, err := e.generate(req.Prompt, n, req.Temperature, req.Seed)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(generateResponse{
			Text: e.tok.Decode(ids), Tokens: ids, PromptTokens: np, Seconds: time.Since(start).Seconds(),
		})
	})
	return mux
}

// runBench runs n requests at once, then n one after another, and prints
// the tokens generated per second of each.
func runBench(w io.Writer, e *engine, n int, prompt string, tokens int) error {
	measure := func(concurrent int) (int, time.Duration, error) {
		var wg sync.WaitGroup
		var mu sync.Mutex
		var total int
		var first error
		start := time.Now()
		for done := 0; done < n; done += concurrent {
			for range min(concurrent, n-done) {
				wg.Add(1)
				go func() {
					defer wg.Done()
					ids, _, err := e.generate(prompt, tokens, 0, 1)
					mu.Lock()
					defer mu.Unlock()
					total += len(ids)
					if err != nil && first == nil {
						first = err
					}
				}()
			}
			wg.Wait()
		}
		return total, time.Since(start), first
	}
	np := len(e.tok.Encode(prompt, true))
	for _, c := range []int{n, 1} {
		total, d, err := measure(c)
		if err != nil {
			return err
		}
		fmt.Fprintf(w, "%d requests, %d at a time: prompt %d tokens each, %d tokens generated in %.2fs, %.1f tok/s\n",
			n, c, np, total, d.Seconds(), float64(total)/d.Seconds())
	}
	return nil
}
