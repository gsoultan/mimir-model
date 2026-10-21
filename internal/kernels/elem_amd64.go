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

//go:noescape
func binF32(op int, y, x, z *float32, n, scalar int, nan float32)

//go:noescape
func poolF32(op int, y, x *float32, offs *int, taps, w, rows, stride, xrs, yrs int, nan float32)

//go:noescape
func poolGatherF32(op int, y, x *float32, idx *int32, np, taps, n int, nan float32)

//go:noescape
func rowsF32(op int, y, v *float32, rows, plane int, nan float32)

//go:noescape
func epilogueF32(y *float32, rows, plane int, ops *int, steps int, v *float32, vs int, nan float32)

//go:noescape
func padPlaneF32(y, x *float32, h, w, hp, wp, top, left int, v float32)
