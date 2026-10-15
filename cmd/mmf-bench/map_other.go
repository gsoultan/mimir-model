//go:build !unix

package main

import "os"

// mapFile reads path into memory on platforms without a unix mmap.
func mapFile(path string) ([]byte, func() error, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	return b, func() error { return nil }, nil
}
