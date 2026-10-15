// Command mmf-import converts GGUF and safetensors files into mmf.
//
//	mmf-import [-o out.mmf] [-bench] [-nohash] input.gguf
//	mmf-import -o model.mmf model-00001-of-00002.safetensors model-00002-of-00002.safetensors
//
// Several safetensors shards are merged into one file. Tensors are copied
// byte for byte; the output holds weights and metadata, with no graph.
// With -bench it reports the input and output sizes and the time to open
// each and read every tensor byte.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/gsoultan/mmf"
	"github.com/gsoultan/mmf/importer"
)

func main() {
	out := flag.String("o", "", "output file (default: first input with the extension replaced by .mmf)")
	bench := flag.Bool("bench", false, "report sizes and load times of the input and the output")
	nohash := flag.Bool("nohash", false, "do not write section hashes")
	flag.Parse()
	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: mmf-import [-o out.mmf] [-bench] [-nohash] input...")
		os.Exit(2)
	}
	if err := run(os.Stdout, flag.Args(), *out, *bench, !*nohash); err != nil {
		fmt.Fprintln(os.Stderr, "mmf-import:", err)
		os.Exit(1)
	}
}

func defaultOutput(in string) string {
	for _, ext := range []string{".gguf", ".safetensors"} {
		if strings.HasSuffix(in, ext) {
			return strings.TrimSuffix(in, ext) + ".mmf"
		}
	}
	return in + ".mmf"
}

func run(w io.Writer, inputs []string, out string, bench, hash bool) (err error) {
	if out == "" {
		out = defaultOutput(inputs[0])
	}
	var srcs []*importer.Source
	var inBytes int64
	start := time.Now()
	for _, p := range inputs {
		b, release, err := mapFile(p)
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, release()) }()
		inBytes += int64(len(b))
		s, err := importer.Parse(b)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		srcs = append(srcs, s)
	}
	src, err := importer.Merge(srcs...)
	if err != nil {
		return err
	}
	mw := mmf.NewWriter()
	mw.Hash = hash
	if err := src.AddTo(mw); err != nil {
		return err
	}
	if err := mw.WriteFile(out); err != nil {
		return err
	}
	st, err := os.Stat(out)
	if err != nil {
		return err
	}
	fmt.Fprintf(w, "wrote %s: %d tensors from %s, %d bytes (%s)\n", out, len(src.Tensors), src.Format, st.Size(), time.Since(start).Round(time.Millisecond))
	if !bench {
		return nil
	}
	srcLoad, srcSum, err := timeSource(inputs)
	if err != nil {
		return err
	}
	dstLoad, dstSum, err := timeMMF(out)
	if err != nil {
		return err
	}
	if srcSum != dstSum {
		return fmt.Errorf("tensor data read back from %s differs from the input", out)
	}
	fmt.Fprintf(w, "%-12s %14s %14s\n", "", "size (bytes)", "open+read")
	fmt.Fprintf(w, "%-12s %14d %14s\n", src.Format, inBytes, srcLoad.Round(time.Microsecond))
	fmt.Fprintf(w, "%-12s %14d %14s\n", "mmf", st.Size(), dstLoad.Round(time.Microsecond))
	return nil
}

// touch reads one byte per cache line of b so that a mapped file is paged
// in, and returns their XOR as a checksum.
func touch(b []byte) (s byte) {
	for i := 0; i < len(b); i += 64 {
		s ^= b[i]
	}
	return s
}

// timeSource measures mapping and parsing the inputs and reading every
// tensor byte, the work a loader has to do before inference.
func timeSource(inputs []string) (time.Duration, byte, error) {
	var sum byte
	start := time.Now()
	for _, p := range inputs {
		b, release, err := mapFile(p)
		if err != nil {
			return 0, 0, err
		}
		s, err := importer.Parse(b)
		if err != nil {
			release()
			return 0, 0, err
		}
		for _, t := range s.Tensors {
			sum ^= touch(t.Data)
		}
		if err := release(); err != nil {
			return 0, 0, err
		}
	}
	return time.Since(start), sum, nil
}

// timeMMF measures mmf.Open (map and validate) and reading every tensor byte.
func timeMMF(path string) (time.Duration, byte, error) {
	var sum byte
	start := time.Now()
	f, err := mmf.Open(path)
	if err != nil {
		return 0, 0, err
	}
	for i := 0; i < f.NumTensors(); i++ {
		sum ^= touch(f.TensorData(f.TensorAt(i)))
	}
	if err := f.Close(); err != nil {
		return 0, 0, err
	}
	return time.Since(start), sum, nil
}
