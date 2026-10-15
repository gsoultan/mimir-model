package mmf

import "math/bits"

// DType identifies an element type. Values are fixed by spec/dtypes.md and are
// never reused.
type DType uint32

const (
	DTypeInvalid DType = 0

	F32     DType = 1
	F16     DType = 2
	BF16    DType = 3
	F64     DType = 4
	I8      DType = 5
	U8      DType = 6
	I16     DType = 7
	U16     DType = 8
	I32     DType = 9
	U32     DType = 10
	I64     DType = 11
	U64     DType = 12
	Bool    DType = 13
	F8E4M3  DType = 14
	F8E5M2  DType = 15
	Q8_0    DType = 64
	Q4_0    DType = 65
	Q4_K    DType = 66
	Q6_K    DType = 67
	MXFP4   DType = 68
	Affine4 DType = 69
)

type dtypeInfo struct {
	name       string
	blockElems uint64 // 1 for plain types
	blockBytes uint64 // bytes per block (element size for plain types)
	float      bool
}

var dtypes = map[DType]dtypeInfo{
	F32:     {"f32", 1, 4, true},
	F16:     {"f16", 1, 2, true},
	BF16:    {"bf16", 1, 2, true},
	F64:     {"f64", 1, 8, true},
	I8:      {"i8", 1, 1, false},
	U8:      {"u8", 1, 1, false},
	I16:     {"i16", 1, 2, false},
	U16:     {"u16", 1, 2, false},
	I32:     {"i32", 1, 4, false},
	U32:     {"u32", 1, 4, false},
	I64:     {"i64", 1, 8, false},
	U64:     {"u64", 1, 8, false},
	Bool:    {"bool", 1, 1, false},
	F8E4M3:  {"f8_e4m3", 1, 1, true},
	F8E5M2:  {"f8_e5m2", 1, 1, true},
	Q8_0:    {"q8_0", 32, 34, true},
	Q4_0:    {"q4_0", 32, 18, true},
	Q4_K:    {"q4_k", 256, 144, true},
	Q6_K:    {"q6_k", 256, 210, true},
	MXFP4:   {"mxfp4", 32, 17, true},
	Affine4: {"affine4_g64", 64, 36, true},
}

// Valid reports whether d is a dtype defined by this version of the spec.
func (d DType) Valid() bool { _, ok := dtypes[d]; return ok }

// String returns the spec name of d.
func (d DType) String() string {
	if i, ok := dtypes[d]; ok {
		return i.name
	}
	return "invalid"
}

// IsBlock reports whether d is a block-quantized dtype.
func (d DType) IsBlock() bool { return dtypes[d].blockElems > 1 }

// BlockElems is the number of elements per block (1 for plain dtypes).
func (d DType) BlockElems() uint64 { return dtypes[d].blockElems }

// BlockBytes is the number of bytes per block (the element size for plain dtypes).
func (d DType) BlockBytes() uint64 { return dtypes[d].blockBytes }

// ParseDType returns the dtype with the given spec name.
func ParseDType(name string) (DType, bool) {
	for d, i := range dtypes {
		if i.name == name {
			return d, true
		}
	}
	return DTypeInvalid, false
}

// ByteSize returns the number of bytes a tensor of this dtype and shape
// occupies, checking every multiplication for overflow. For block dtypes the
// last dimension must be a multiple of the block size and the rank at least 1.
func ByteSize(d DType, shape []uint64) (uint64, error) {
	info, ok := dtypes[d]
	if !ok {
		return 0, errf(CodeBadDType, "unknown dtype %d", uint32(d))
	}
	n := uint64(1)
	for _, s := range shape {
		hi, lo := bits.Mul64(n, s)
		if hi != 0 {
			return 0, errf(CodeOverflow, "element count overflows u64")
		}
		n = lo
	}
	if info.blockElems > 1 {
		if len(shape) == 0 {
			return 0, errf(CodeBadShape, "%s tensor must have rank >= 1", info.name)
		}
		if shape[len(shape)-1]%info.blockElems != 0 {
			return 0, errf(CodeBadShape, "%s: last dimension %d is not a multiple of %d", info.name, shape[len(shape)-1], info.blockElems)
		}
	}
	hi, lo := bits.Mul64(n/info.blockElems, info.blockBytes)
	if hi != 0 {
		return 0, errf(CodeOverflow, "byte size overflows u64")
	}
	return lo, nil
}
