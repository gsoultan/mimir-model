package mmf

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
		return "mmf: " + string(e.Code)
	}
	return "mmf: " + string(e.Code) + ": " + e.Detail
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
