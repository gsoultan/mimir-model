//go:build unix

package mmf

import (
	"os"
	"syscall"
)

// Open maps the file at path read-only and validates it. The mapping lives
// outside the Go heap, so the garbage collector never scans or moves it.
func Open(path string) (*File, error) {
	fd, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fd.Close()
	st, err := fd.Stat()
	if err != nil {
		return nil, err
	}
	size := st.Size()
	if size == 0 {
		return Parse(nil)
	}
	if int64(int(size)) != size {
		return nil, errf(CodeOverflow, "file too large to map")
	}
	data, err := syscall.Mmap(int(fd.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, err
	}
	f, err := Parse(data)
	if err != nil {
		syscall.Munmap(data)
		return nil, err
	}
	f.release = func() error { return syscall.Munmap(data) }
	return f, nil
}
