package importer

import (
	"encoding/binary"
	"fmt"
	"math"
)

// pbField is one field of a protocol buffer message in wire format.
type pbField struct {
	num  uint64
	wire uint64 // 0 varint, 1 fixed64, 2 length-delimited, 5 fixed32
	v    uint64 // varint, fixed64 or fixed32 value
	b    []byte // length-delimited payload; aliases the input
}

// pbFields splits a message into its fields. It checks every length
// against the input and never allocates more than the input's size.
func pbFields(b []byte) ([]pbField, error) {
	var out []pbField
	for len(b) > 0 {
		key, n := binary.Uvarint(b)
		if n <= 0 {
			return nil, fmt.Errorf("protobuf: bad field key")
		}
		b = b[n:]
		f := pbField{num: key >> 3, wire: key & 7}
		if f.num == 0 {
			return nil, fmt.Errorf("protobuf: field number 0")
		}
		switch f.wire {
		case 0:
			v, n := binary.Uvarint(b)
			if n <= 0 {
				return nil, fmt.Errorf("protobuf: bad varint in field %d", f.num)
			}
			f.v, b = v, b[n:]
		case 1:
			if len(b) < 8 {
				return nil, fmt.Errorf("protobuf: truncated fixed64 in field %d", f.num)
			}
			f.v, b = binary.LittleEndian.Uint64(b), b[8:]
		case 2:
			l, n := binary.Uvarint(b)
			if n <= 0 || l > uint64(len(b)-n) {
				return nil, fmt.Errorf("protobuf: bad length in field %d", f.num)
			}
			f.b, b = b[n:n+int(l)], b[n+int(l):]
		case 5:
			if len(b) < 4 {
				return nil, fmt.Errorf("protobuf: truncated fixed32 in field %d", f.num)
			}
			f.v, b = uint64(binary.LittleEndian.Uint32(b)), b[4:]
		default:
			return nil, fmt.Errorf("protobuf: unsupported wire type %d in field %d", f.wire, f.num)
		}
		out = append(out, f)
	}
	return out, nil
}

// int64s reads a repeated int64 field occurrence, packed or not, appending
// to dst.
func (f pbField) int64s(dst []int64) ([]int64, error) {
	switch f.wire {
	case 0:
		return append(dst, int64(f.v)), nil
	case 2:
		b := f.b
		for len(b) > 0 {
			v, n := binary.Uvarint(b)
			if n <= 0 {
				return nil, fmt.Errorf("protobuf: bad packed varint in field %d", f.num)
			}
			dst, b = append(dst, int64(v)), b[n:]
		}
		return dst, nil
	}
	return nil, fmt.Errorf("protobuf: field %d is not an integer", f.num)
}

// float32s reads a repeated float field occurrence, packed or not.
func (f pbField) float32s(dst []float32) ([]float32, error) {
	switch f.wire {
	case 5:
		return append(dst, math.Float32frombits(uint32(f.v))), nil
	case 2:
		if len(f.b)%4 != 0 {
			return nil, fmt.Errorf("protobuf: packed floats in field %d are not a multiple of 4 bytes", f.num)
		}
		for i := 0; i < len(f.b); i += 4 {
			dst = append(dst, math.Float32frombits(binary.LittleEndian.Uint32(f.b[i:])))
		}
		return dst, nil
	}
	return nil, fmt.Errorf("protobuf: field %d is not a float", f.num)
}

// float64s reads a repeated double field occurrence, packed or not.
func (f pbField) float64s(dst []float64) ([]float64, error) {
	switch f.wire {
	case 1:
		return append(dst, math.Float64frombits(f.v)), nil
	case 2:
		if len(f.b)%8 != 0 {
			return nil, fmt.Errorf("protobuf: packed doubles in field %d are not a multiple of 8 bytes", f.num)
		}
		for i := 0; i < len(f.b); i += 8 {
			dst = append(dst, math.Float64frombits(binary.LittleEndian.Uint64(f.b[i:])))
		}
		return dst, nil
	}
	return nil, fmt.Errorf("protobuf: field %d is not a double", f.num)
}

func (f pbField) bytes() ([]byte, error) {
	if f.wire != 2 {
		return nil, fmt.Errorf("protobuf: field %d is not length-delimited", f.num)
	}
	return f.b, nil
}

func (f pbField) varint() (int64, error) {
	if f.wire != 0 {
		return 0, fmt.Errorf("protobuf: field %d is not a varint", f.num)
	}
	return int64(f.v), nil
}
