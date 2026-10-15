package compat

import (
	"path/filepath"
	"testing"
)

// TestCorpus reads every file of every released version with the current
// reader and executor, with and without its fast paths. A failure here is
// a compatibility break, which needs a major version and an ADR
// (spec/versioning.md section 4), not a change to the corpus.
func TestCorpus(t *testing.T) {
	root := filepath.Join("..", "testdata", "compat")
	all, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) == 0 {
		t.Fatal("empty corpus")
	}
	for path, e := range all {
		t.Run(filepath.Base(filepath.Dir(path))+"/"+e.File, func(t *testing.T) {
			if err := Check(filepath.Dir(path), e, true); err != nil {
				t.Fatal(err)
			}
			if testing.Short() {
				return
			}
			if err := Check(filepath.Dir(path), e, false); err != nil {
				t.Fatal("canonical bodies:", err)
			}
		})
	}
}
