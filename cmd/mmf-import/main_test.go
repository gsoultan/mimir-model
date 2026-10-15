package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gsoultan/mmf"
)

func copyFixture(t *testing.T, dir, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "testdata", "import", name))
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestRun(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"tiny.gguf", "tiny.safetensors"} {
		in := copyFixture(t, dir, name)
		var out bytes.Buffer
		if err := run(&out, []string{in}, options{bench: true, hash: true}); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(out.String(), "open+read") {
			t.Errorf("no benchmark table in:\n%s", out.String())
		}
		f, err := mmf.Open(defaultOutput(in))
		if err != nil {
			t.Fatal(err)
		}
		if err := f.VerifyHashes(); err != nil {
			t.Error(err)
		}
		if f.NumTensors() == 0 {
			t.Error("no tensors")
		}
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	}
}

func TestRunRejects(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "x.bin")
	os.WriteFile(bad, []byte("not a model"), 0o644)
	if err := run(&bytes.Buffer{}, []string{bad}, options{out: filepath.Join(dir, "o.mmf"), hash: true}); err == nil {
		t.Error("accepted a non-model file")
	}
}
