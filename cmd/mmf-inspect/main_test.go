package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gsoultan/mmf"
)

func TestInspectRuns(t *testing.T) {
	dir := t.TempDir()
	w := mmf.NewWriter()
	if err := w.AddTensor("a", mmf.F32, []uint64{2}, make([]byte, 8)); err != nil {
		t.Fatal(err)
	}
	mb, _ := mmf.Metadata{mmf.KeyName: "t"}.Encode()
	if err := w.AddSection(mmf.SectionMetadata, false, mb); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "t.mmf")
	if err := w.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := inspect(&out, path, true, false, true); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "hashes ok") || !strings.Contains(out.String(), "general.name = t") {
		t.Fatalf("unexpected output:\n%s", out.String())
	}
}
