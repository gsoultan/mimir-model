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

package mim

import "fmt"

// Code is a stable, machine-readable error code. The codes are part of the
// spec (spec/container.md, "Error codes") so that conformance tests can check
// that two readers reject a hostile file for the same reason.
type Code string

const (
	CodeTruncated          Code = "truncated"
	CodeBadMagic           Code = "bad-magic"
	CodeBigEndian          Code = "big-endian"
	CodeBadByteOrderMark   Code = "bad-byte-order-mark"
	CodeUnsupportedVersion Code = "unsupported-version"
	CodeReservedNonZero    Code = "reserved-non-zero"
	CodeBadAlignment       Code = "bad-alignment"
	CodeBadLength          Code = "bad-length"
	CodeIndexTooLarge      Code = "index-too-large"
	CodeOutOfBounds        Code = "out-of-bounds"
	CodeOverlap            Code = "overlap"
	CodeMisaligned         Code = "misaligned"
	CodeNonCanonical       Code = "non-canonical-layout"
	CodeSizeMismatch       Code = "size-mismatch"
	CodeOverflow           Code = "overflow"
	CodeBadDType           Code = "bad-dtype"
	CodeBadShape           Code = "bad-shape"
	CodeBadName            Code = "bad-name"
	CodeUnsorted           Code = "unsorted"
	CodeDuplicate          Code = "duplicate"
	CodeUnknownRequired    Code = "unknown-required-section"
	CodeMissingSection     Code = "missing-section"
	CodeHashMismatch       Code = "hash-mismatch"
	CodeBadHashAlg         Code = "bad-hash-algorithm"
	CodeBadMetadata        Code = "bad-metadata"
	CodeBadGraph           Code = "bad-graph"
	CodeBadTraining        Code = "bad-training"
	CodeNotFound           Code = "not-found"
	CodeUnsupportedDType   Code = "unsupported-dtype"
)

// Error is returned for every malformed-input condition. Callers can match on
// Code with errors.As.
type Error struct {
	Code   Code
	Detail string
}

func (e *Error) Error() string {
	if e.Detail == "" {
		return "mim: " + string(e.Code)
	}
	return "mim: " + string(e.Code) + ": " + e.Detail
}

func errf(code Code, format string, args ...any) error {
	return &Error{Code: code, Detail: fmt.Sprintf(format, args...)}
}

// CodeOf returns the Code of err, or "" when err is not an *Error.
func CodeOf(err error) Code {
	for err != nil {
		if e, ok := err.(*Error); ok {
			return e.Code
		}
		u, ok := err.(interface{ Unwrap() error })
		if !ok {
			return ""
		}
		err = u.Unwrap()
	}
	return ""
}
