// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mim, created and owned by Gembit Soultan Shirazi.
// It is licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License in the LICENSE file at the root
// of this repository or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

//go:build windows

package mim

import (
	"os"
	"syscall"
	"unsafe"
)

// Open maps the file at path read-only and validates it, as on unix: the
// mapping lives outside the Go heap and pages in as tensors are read. The
// file stays in use, and cannot be replaced or deleted, until Close.
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
	m, err := syscall.CreateFileMapping(syscall.Handle(fd.Fd()), nil, syscall.PAGE_READONLY, uint32(uint64(size)>>32), uint32(size), nil)
	if err != nil {
		return nil, os.NewSyscallError("CreateFileMapping", err)
	}
	// The view keeps the mapping alive; the handle is not needed past it.
	addr, err := syscall.MapViewOfFile(m, syscall.FILE_MAP_READ, 0, 0, uintptr(size))
	syscall.CloseHandle(m)
	if err != nil {
		return nil, os.NewSyscallError("MapViewOfFile", err)
	}
	// addr is memory the system mapped, which the garbage collector does
	// not move; reading it through a pointer, not a uintptr conversion,
	// keeps vet's unsafe.Pointer rule.
	data := unsafe.Slice(*(**byte)(unsafe.Pointer(&addr)), int(size))
	unmap := func() error { return os.NewSyscallError("UnmapViewOfFile", syscall.UnmapViewOfFile(addr)) }
	f, err := Parse(data)
	if err != nil {
		unmap()
		return nil, err
	}
	f.release = unmap
	return f, nil
}
