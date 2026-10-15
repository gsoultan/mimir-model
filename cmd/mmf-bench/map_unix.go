//go:build unix

package main

import (
	"os"
	"syscall"
)

// mapFile maps path read-only. Tensor data stays in the page cache and is
// never copied onto the Go heap.
func mapFile(path string) ([]byte, func() error, error) {
	fd, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer fd.Close()
	st, err := fd.Stat()
	if err != nil {
		return nil, nil, err
	}
	if st.Size() == 0 {
		return nil, func() error { return nil }, nil
	}
	b, err := syscall.Mmap(int(fd.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, err
	}
	return b, func() error { return syscall.Munmap(b) }, nil
}
