package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/importer"
)

func TestBench(t *testing.T) {
	dir := t.TempDir()
	src, err := os.ReadFile(filepath.Join("..", "..", "testdata", "llama", "tiny-q4km.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	gguf := filepath.Join(dir, "tiny.gguf")
	if err := os.WriteFile(gguf, src, 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := importer.Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := importer.BuildLlama(s, importer.LlamaOptions{Context: 16}); err != nil {
		t.Fatal(err)
	}
	w := mmf.NewWriter()
	if err := s.AddTo(w); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "tiny.mmf")
	if err := w.WriteFile(out); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := run(&buf, []string{gguf, out}, 3, false, measure); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"| tiny.gguf | gguf |", "| tiny.mmf | mmf + graph |"} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("no %q row in:\n%s", want, buf.String())
		}
	}
}

func TestBenchRejects(t *testing.T) {
	bad := filepath.Join(t.TempDir(), "x.mmf")
	os.WriteFile(bad, []byte("not a model"), 0o644)
	if err := run(&bytes.Buffer{}, []string{bad}, 1, false, measure); err == nil {
		t.Error("accepted a non-model file")
	}
}
