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

//go:build amd64 && !purego

package kernels

import "syscall"

// requestAMX asks Linux for the tile data state, which a process must do
// before its first tile instruction (arch_prctl ARCH_REQ_XCOMP_PERM).
func requestAMX() bool {
	const archReqXcompPerm, xfeatureXtiledata = 0x1023, 18
	_, _, e := syscall.RawSyscall(syscall.SYS_ARCH_PRCTL, archReqXcompPerm, xfeatureXtiledata, 0)
	return e == 0
}
