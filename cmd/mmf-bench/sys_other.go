//go:build !(linux && (amd64 || arm64))

package main

import "errors"

func dropCache(string) error {
	return errors.New("-cold is only supported on Linux")
}

func residentMemory() (anon, file int64) { return -1, -1 }
