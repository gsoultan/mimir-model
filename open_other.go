//go:build !unix

package mmf

import "os"

// Open reads the file at path into memory and validates it. Platforms without
// a unix mmap fall back to a plain read.
func Open(path string) (*File, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(data)
}
