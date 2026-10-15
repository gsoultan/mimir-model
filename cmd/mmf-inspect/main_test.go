package main

import (
	"os/exec"
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
	bin := filepath.Join(dir, "inspect")
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	out, err := exec.Command(bin, "-verify", "-meta", path).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "hashes ok") || !strings.Contains(string(out), "general.name = t") {
		t.Fatalf("err %v\n%s", err, out)
	}
}
