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

//go:build linux && (amd64 || arm64)

package main

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"syscall"
)

// dropCache writes nothing and asks the kernel to evict path's clean pages
// from the page cache (POSIX_FADV_DONTNEED), which needs no privileges.
func dropCache(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	const fadvDontNeed = 4
	if _, _, e := syscall.Syscall6(syscall.SYS_FADVISE64, f.Fd(), 0, 0, fadvDontNeed, 0, 0); e != 0 {
		return e
	}
	return nil
}

// residentMemory returns the resident anonymous and file-backed memory of
// this process from /proc/self/status.
func residentMemory() (anon, file int64) {
	anon, file = -1, -1
	f, err := os.Open("/proc/self/status")
	if err != nil {
		return
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok || (k != "RssAnon" && k != "RssFile") {
			continue
		}
		n, err := strconv.ParseInt(strings.TrimSuffix(strings.TrimSpace(v), " kB"), 10, 64)
		if err != nil {
			continue
		}
		if k == "RssAnon" {
			anon = n << 10
		} else {
			file = n << 10
		}
	}
	return
}
