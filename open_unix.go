// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mlm, created and owned by Gembit Soultan Shirazi.
// It is licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License in the LICENSE file at the root
// of this repository or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

//go:build unix

package mlm

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
