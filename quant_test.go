package mmf

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"testing"
)

// refF16 decodes a binary16 bit pattern with float64 arithmetic, independently
// of F16ToF32, so the two can be checked against each other exhaustively.
func refF16(h uint16) float64 {
	sign := 1.0
	if h&0x8000 != 0 {
		sign = -1
	}
	exp := int(h>>10) & 0x1F
	mant := float64(h & 0x3FF)
	switch exp {
	case 0:
		return sign * math.Ldexp(mant, -24)
	case 0x1F:
		if mant != 0 {
			return math.NaN()
		}
		return sign * math.Inf(1)
	}
	return sign * math.Ldexp(1024+mant, exp-25)
}

func sameF32(a, b float32) bool {
	if a != a && b != b {
		return true
	}
	return math.Float32bits(a) == math.Float32bits(b)
}

func TestF16ToF32Exhaustive(t *testing.T) {
	for i := 0; i < 1<<16; i++ {
		h := uint16(i)
		got := F16ToF32(h)
		want := float32(refF16(h))
		if !sameF32(got, want) {
			t.Fatalf("F16ToF32(%#04x) = %v (%#08x), want %v", h, got, math.Float32bits(got), want)
		}
		if got != got {
			continue
		}
		if back := F32ToF16(got); back != h {
			t.Fatalf("F32ToF16(F16ToF32(%#04x)) = %#04x", h, back)
		}
	}
	// NaN survives the round trip as NaN.
	for _, h := range []uint16{0x7C01, 0x7E00, 0xFE00, 0xFFFF} {
		if back := F32ToF16(F16ToF32(h)); back&0x7C00 != 0x7C00 || back&0x3FF == 0 {
			t.Errorf("NaN %#04x round-tripped to %#04x", h, back)
		}
	}
}

func TestF32ToF16Known(t *testing.T) {
	cases := []struct {
		in   float32
		want uint16
	}{
		{0, 0x0000},
		{float32(math.Copysign(0, -1)), 0x8000},
		{1, 0x3C00},
		{-2, 0xC000},
		{65504, 0x7BFF},
		{65519, 0x7BFF}, // below the midpoint to infinity
		{65520, 0x7C00}, // exactly the midpoint: ties to even (infinity)
		{1e10, 0x7C00},  // overflow
		{float32(math.Inf(1)), 0x7C00},
		{float32(math.Inf(-1)), 0xFC00},
		{5.960464477539063e-08, 0x0001},  // smallest subnormal 2^-24
		{2.9802322387695312e-08, 0x0000}, // 2^-25: tie, rounds to even (0)
		{math.Nextafter32(2.9802322387695312e-08, 1), 0x0001}, // just above 2^-25
		{8.940696716308594e-08, 0x0002},                       // 1.5*2^-24: tie, rounds to even (2)
		{6.103515625e-05, 0x0400},                             // smallest normal 2^-14
		{6.097555160522461e-05, 0x03FF},                       // largest subnormal
		{6.1005353927612305e-05, 0x0400},                      // midpoint between 0x03FF and 0x0400 -> even
		{1e-10, 0x0000},
		{1.00048828125, 0x3C00},                      // 1 + 2^-11: tie, rounds to even
		{1.00146484375, 0x3C02},                      // 1 + 3*2^-11: tie, rounds to even
		{1.000977, 0x3C01},                           // 1 + 2^-10
		{math.Nextafter32(1.00048828125, 2), 0x3C01}, // just above the tie
		{0.1, 0x2E66},
		{3.140625, 0x4248},
	}
	for _, c := range cases {
		if got := F32ToF16(c.in); got != c.want {
			t.Errorf("F32ToF16(%v) = %#04x, want %#04x", c.in, got, c.want)
		}
	}
	// NaN stays NaN and keeps its sign.
	if got := F32ToF16(float32(math.NaN())); got&0x7C00 != 0x7C00 || got&0x3FF == 0 {
		t.Errorf("F32ToF16(NaN) = %#04x", got)
	}
	if got := F32ToF16(math.Float32frombits(0xFFC00000)); got&0xFC00 != 0xFC00 || got&0x3FF == 0 {
		t.Errorf("F32ToF16(-NaN) = %#04x", got)
	}
}

// TestF32ToF16Ties checks round-to-nearest-even between every pair of adjacent
// finite binary16 values: the midpoint goes to the even neighbour, one float32
// ulp either side goes to the nearer neighbour.
func TestF32ToF16Ties(t *testing.T) {
	for i := 0; i < 0x7BFF; i++ {
		lo, hi := uint16(i), uint16(i+1)
		a, b := F16ToF32(lo), F16ToF32(hi)
		mid := float32((float64(a) + float64(b)) / 2) // exact: one extra bit fits in float32
		even := lo
		if lo&1 == 1 {
			even = hi
		}
		if got := F32ToF16(mid); got != even {
			t.Fatalf("midpoint of %#04x/%#04x (%v) -> %#04x, want %#04x", lo, hi, mid, got, even)
		}
		if got := F32ToF16(math.Nextafter32(mid, 0)); got != lo {
			t.Fatalf("below midpoint of %#04x/%#04x -> %#04x", lo, hi, got)
		}
		if got := F32ToF16(math.Nextafter32(mid, math.MaxFloat32)); got != hi {
			t.Fatalf("above midpoint of %#04x/%#04x -> %#04x", lo, hi, got)
		}
		if got := F32ToF16(-mid); got != even|0x8000 {
			t.Fatalf("negative midpoint of %#04x/%#04x -> %#04x", lo, hi, got)
		}
	}
}

func TestBF16(t *testing.T) {
	for i := 0; i < 1<<16; i++ {
		h := uint16(i)
		f := BF16ToF32(h)
		if math.Float32bits(f) != uint32(h)<<16 {
			t.Fatalf("BF16ToF32(%#04x) = %#08x", h, math.Float32bits(f))
		}
		if f != f {
			if back := F32ToBF16(f); back&0x7F80 != 0x7F80 || back&0x7F == 0 {
				t.Fatalf("NaN %#04x round-tripped to %#04x", h, back)
			}
			continue
		}
		if back := F32ToBF16(f); back != h {
			t.Fatalf("F32ToBF16(BF16ToF32(%#04x)) = %#04x", h, back)
		}
	}
	cases := []struct {
		in   uint32
		want uint16
	}{
		{0x3F808000, 0x3F80}, // tie, low half even: down
		{0x3F818000, 0x3F82}, // tie, low half odd: up
		{0x3F808001, 0x3F81}, // above tie: up
		{0x3F807FFF, 0x3F80}, // below tie: down
		{0x7F7FFFFF, 0x7F80}, // max float32 rounds up to +inf
		{0x7F7F7FFF, 0x7F7F}, // just below rounds down to max bf16
		{0x00000001, 0x0000}, // smallest subnormal flushes to 0 by rounding
		{0x80000001, 0x8000},
	}
	for _, c := range cases {
		if got := F32ToBF16(math.Float32frombits(c.in)); got != c.want {
			t.Errorf("F32ToBF16(%#08x) = %#04x, want %#04x", c.in, got, c.want)
		}
	}
}

func TestF8E4M3(t *testing.T) {
	for i := 0; i < 256; i++ {
		b := uint8(i)
		got := F8E4M3ToF32(b)
		sign := 1.0
		if b&0x80 != 0 {
			sign = -1
		}
		exp := int(b>>3) & 0xF
		mant := float64(b & 7)
		var want float64
		switch {
		case exp == 0xF && mant == 7:
			want = math.NaN()
		case exp == 0:
			want = sign * math.Ldexp(mant, -9)
		default:
			want = sign * math.Ldexp(8+mant, exp-10)
		}
		if !sameF32(got, float32(want)) {
			t.Errorf("F8E4M3ToF32(%#02x) = %v, want %v", b, got, want)
		}
	}
	known := map[uint8]float32{0x00: 0, 0x80: float32(math.Copysign(0, -1)), 0x38: 1, 0xB8: -1, 0x7E: 448, 0xFE: -448, 0x01: 1.0 / 512, 0x08: 1.0 / 64, 0x30: 0.5, 0x40: 2}
	for b, want := range known {
		if got := F8E4M3ToF32(b); !sameF32(got, want) {
			t.Errorf("F8E4M3ToF32(%#02x) = %v, want %v", b, got, want)
		}
	}
	for _, b := range []uint8{0x7F, 0xFF} {
		if got := F8E4M3ToF32(b); got == got {
			t.Errorf("F8E4M3ToF32(%#02x) = %v, want NaN", b, got)
		}
	}
}

func TestF8E5M2(t *testing.T) {
	for i := 0; i < 256; i++ {
		b := uint8(i)
		got := F8E5M2ToF32(b)
		sign := 1.0
		if b&0x80 != 0 {
			sign = -1
		}
		exp := int(b>>2) & 0x1F
		mant := float64(b & 3)
		var want float64
		switch {
		case exp == 0x1F && mant != 0:
			want = math.NaN()
		case exp == 0x1F:
			want = sign * math.Inf(1)
		case exp == 0:
			want = sign * math.Ldexp(mant, -16)
		default:
			want = sign * math.Ldexp(4+mant, exp-17)
		}
		if !sameF32(got, float32(want)) {
			t.Errorf("F8E5M2ToF32(%#02x) = %v, want %v", b, got, want)
		}
	}
	known := map[uint8]float32{0x3C: 1, 0xBC: -1, 0x7B: 57344, 0x7C: float32(math.Inf(1)), 0xFC: float32(math.Inf(-1)), 0x01: 1.0 / 65536, 0x04: 1.0 / 16384, 0x3E: 1.5}
	for b, want := range known {
		if got := F8E5M2ToF32(b); !sameF32(got, want) {
			t.Errorf("F8E5M2ToF32(%#02x) = %v, want %v", b, got, want)
		}
	}
}

func TestE8M0(t *testing.T) {
	for i := 0; i < 255; i++ {
		got := E8M0ToF32(uint8(i))
		want := float32(math.Ldexp(1, i-127))
		if !sameF32(got, want) {
			t.Errorf("E8M0ToF32(%d) = %v (%#08x), want %v", i, got, math.Float32bits(got), want)
		}
	}
	if got := E8M0ToF32(0xFF); got == got {
		t.Errorf("E8M0ToF32(0xFF) = %v, want NaN", got)
	}
	if E8M0ToF32(127) != 1 || E8M0ToF32(0) != math.Float32frombits(0x00400000) {
		t.Error("E8M0 anchors")
	}
}

func TestE2M1(t *testing.T) {
	want := [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, float32(math.Copysign(0, -1)), -0.5, -1, -1.5, -2, -3, -4, -6}
	for i := 0; i < 256; i++ {
		if got := E2M1ToF32(uint8(i)); !sameF32(got, want[i&0xF]) {
			t.Errorf("E2M1ToF32(%#02x) = %v, want %v", i, got, want[i&0xF])
		}
	}
}

type goldenVector struct {
	Bytes     string    `json:"bytes"`
	InputBits []uint32  `json:"input_bits"`
	Bits      []uint32  `json:"bits"`
	Values    []float32 `json:"values"`
}

func loadGolden(t *testing.T) map[string]goldenVector {
	t.Helper()
	data, err := os.ReadFile("testdata/quant_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g map[string]goldenVector
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// TestDequantizeGolden checks the block decoders bit for bit against the
// output of the Python gguf package's reference dequantizers.
func TestDequantizeGolden(t *testing.T) {
	g := loadGolden(t)
	for _, dt := range []DType{Q8_0, Q4_0, Q4_K, Q6_K, MXFP4} {
		v, ok := g[dt.String()]
		if !ok {
			t.Errorf("%s: no golden vector", dt)
			continue
		}
		raw, err := base64.StdEncoding.DecodeString(v.Bytes)
		if err != nil {
			t.Fatal(err)
		}
		if len(v.Bits) != len(v.Values) || len(v.Bits) == 0 {
			t.Fatalf("%s: malformed golden vector", dt)
		}
		for i, b := range v.Bits {
			if math.Float32bits(v.Values[i]) != b {
				t.Fatalf("%s: golden value %d (%v) does not match its bit pattern %#08x", dt, i, v.Values[i], b)
			}
		}
		got := make([]float32, len(v.Bits))
		if err := Dequantize(dt, raw, got); err != nil {
			t.Errorf("%s: %v", dt, err)
			continue
		}
		bad := 0
		for i := range got {
			if math.Float32bits(got[i]) != v.Bits[i] {
				if bad < 5 {
					t.Errorf("%s[%d] = %v (%#08x), want %v (%#08x)", dt, i, got[i], math.Float32bits(got[i]), v.Values[i], v.Bits[i])
				}
				bad++
			}
		}
		if bad > 0 {
			t.Errorf("%s: %d of %d values differ", dt, bad, len(got))
		}
	}
}

func TestDequantizeHandBuilt(t *testing.T) {
	f16 := func(f float32) []byte {
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], F32ToF16(f))
		return b[:]
	}
	t.Run("q4_0 nibble order", func(t *testing.T) {
		blk := append(f16(2), make([]byte, 16)...)
		blk[2] = 0x1F // element 0 = 15-8 = 7, element 16 = 1-8 = -7
		blk[3] = 0x80 // element 1 = 0-8 = -8, element 17 = 8-8 = 0
		got := make([]float32, 32)
		if err := Dequantize(Q4_0, blk, got); err != nil {
			t.Fatal(err)
		}
		for i, want := range map[int]float32{0: 14, 16: -14, 1: -16, 17: 0, 2: -16, 31: -16} {
			if got[i] != want {
				t.Errorf("y[%d] = %v, want %v", i, got[i], want)
			}
		}
	})
	t.Run("q8_0", func(t *testing.T) {
		blk := append(f16(0.5), make([]byte, 32)...)
		blk[2] = 0x7F
		blk[33] = 0x80
		got := make([]float32, 32)
		if err := Dequantize(Q8_0, blk, got); err != nil {
			t.Fatal(err)
		}
		if got[0] != 63.5 || got[31] != -64 || got[1] != 0 {
			t.Errorf("got %v %v %v", got[0], got[31], got[1])
		}
	})
	t.Run("mxfp4", func(t *testing.T) {
		blk := make([]byte, 17)
		blk[0] = 129 // scale 4
		blk[1] = 0x7F
		blk[2] = 0x89
		got := make([]float32, 32)
		if err := Dequantize(MXFP4, blk, got); err != nil {
			t.Fatal(err)
		}
		for i, want := range map[int]float32{0: -24, 16: 24, 1: -2, 17: 0, 2: 0} {
			if got[i] != want {
				t.Errorf("y[%d] = %v, want %v", i, got[i], want)
			}
		}
		// The negative-zero code decodes to +0, as in ggml.
		if math.Signbit(float64(got[16+1])) {
			t.Error("-0 code should decode to +0")
		}
		blk[0] = 0 // smallest scale 2^-127, times 6 is still exact
		if err := Dequantize(MXFP4, blk, got); err != nil {
			t.Fatal(err)
		}
		if want := float32(math.Ldexp(-6, -127)); got[0] != want {
			t.Errorf("y[0] = %v, want %v", got[0], want)
		}
	})
	t.Run("affine4", func(t *testing.T) {
		blk := append(append(f16(0.25), f16(-1)...), make([]byte, 32)...)
		blk[4] = 0xF3 // element 0 = 3, element 32 = 15
		got := make([]float32, 64)
		if err := Dequantize(Affine4, blk, got); err != nil {
			t.Fatal(err)
		}
		for i, want := range map[int]float32{0: -0.25, 32: 2.75, 1: -1, 63: -1} {
			if got[i] != want {
				t.Errorf("y[%d] = %v, want %v", i, got[i], want)
			}
		}
	})
	t.Run("q4_k scales", func(t *testing.T) {
		// d = 1, dmin = 0, all quants 1, scales: sub-block j has scale j+1.
		blk := make([]byte, 144)
		copy(blk[0:], f16(1))
		copy(blk[2:], f16(0))
		sc := blk[4:16]
		for j := 0; j < 4; j++ {
			sc[j] = byte(j + 1)
		}
		for j := 4; j < 8; j++ {
			v := byte(j + 1)
			sc[j+4] |= v & 0xF
			sc[j-4] |= (v >> 4) << 6
		}
		for i := range blk[16:] {
			blk[16+i] = 0x11
		}
		got := make([]float32, 256)
		if err := Dequantize(Q4_K, blk, got); err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if want := float32(i/32 + 1); v != want {
				t.Fatalf("y[%d] = %v, want %v", i, v, want)
			}
		}
		// Now a high scale (>= 16) to exercise the two top bits, and a min.
		blk = make([]byte, 144)
		copy(blk[0:], f16(1))
		copy(blk[2:], f16(0.5))
		sc = blk[4:16]
		sc[8] = 0x35 // sub-block 4: scale low nibble 5, min low nibble 3
		sc[0] = 0x80 // sub-block 4: scale high bits 10b -> 0x25 = 37
		sc[4] = 0x40 // sub-block 4: min high bits 01b -> 0x13 = 19
		blk[16+64] = 0x02
		if err := Dequantize(Q4_K, blk, got); err != nil {
			t.Fatal(err)
		}
		if want := float32(37*2 - 0.5*19); got[128] != want {
			t.Errorf("y[128] = %v, want %v", got[128], want)
		}
		if want := float32(-0.5 * 19); got[129] != want {
			t.Errorf("y[129] = %v, want %v", got[129], want)
		}
		if got[0] != 0 || got[160] != 0 {
			t.Errorf("y[0] = %v, y[160] = %v, want 0 (scale and min 0)", got[0], got[160])
		}
	})
	t.Run("q6_k", func(t *testing.T) {
		blk := make([]byte, 210)
		copy(blk[208:], f16(1))
		for i := 192; i < 208; i++ {
			blk[i] = 1
		}
		blk[0] = 0xF0   // element 0: low nibble 0; element 64: high nibble 15
		blk[128] = 0x01 // element 0: high 2 bits 01 -> q = 16-32 = -16
		blk[192] = 0xFE // scale -2
		got := make([]float32, 256)
		if err := Dequantize(Q6_K, blk, got); err != nil {
			t.Fatal(err)
		}
		if got[0] != 32 || got[64] != -17 || got[32] != -32 || got[255] != -32 {
			t.Errorf("got y[0]=%v y[64]=%v y[32]=%v y[255]=%v", got[0], got[64], got[32], got[255])
		}
	})
}

func TestDequantizePlain(t *testing.T) {
	le := binary.LittleEndian
	check := func(dt DType, src []byte, want ...float32) {
		t.Helper()
		got := make([]float32, len(want))
		if err := Dequantize(dt, src, got); err != nil {
			t.Errorf("%s: %v", dt, err)
			return
		}
		for i := range want {
			if !sameF32(got[i], want[i]) {
				t.Errorf("%s[%d] = %v, want %v", dt, i, got[i], want[i])
			}
		}
	}
	check(F32, le.AppendUint32(le.AppendUint32(nil, math.Float32bits(1.5)), 0xFFC00000), 1.5, float32(math.NaN()))
	check(F16, le.AppendUint16(le.AppendUint16(nil, 0x3C00), 0xC000), 1, -2)
	check(BF16, le.AppendUint16(le.AppendUint16(nil, 0x3F80), 0xC000), 1, -2)
	check(F64, le.AppendUint64(nil, math.Float64bits(0.1)), float32(0.1))
	check(I8, []byte{0xFF, 0x7F}, -1, 127)
	check(U8, []byte{0xFF, 0x7F}, 255, 127)
	check(I16, le.AppendUint16(nil, 0x8000), -32768)
	check(U16, le.AppendUint16(nil, 0x8000), 32768)
	check(I32, le.AppendUint32(nil, 0xFFFFFFFF), -1)
	check(U32, le.AppendUint32(nil, 0xFFFFFFFF), 4294967295)
	check(I64, le.AppendUint64(nil, 0xFFFFFFFFFFFFFFFF), -1)
	check(U64, le.AppendUint64(nil, 1<<40), 1<<40)
	check(Bool, []byte{0, 1, 2}, 0, 1, 1)
	check(F8E4M3, []byte{0x38, 0x7E}, 1, 448)
	check(F8E5M2, []byte{0x3C, 0x7C}, 1, float32(math.Inf(1)))
}

func TestDequantizeErrors(t *testing.T) {
	if err := Dequantize(F32, make([]byte, 7), make([]float32, 2)); CodeOf(err) != CodeSizeMismatch {
		t.Errorf("short src: %v", err)
	}
	if err := Dequantize(Q8_0, make([]byte, 34), make([]float32, 33)); CodeOf(err) != CodeSizeMismatch {
		t.Errorf("partial block: %v", err)
	}
	if err := Dequantize(Q8_0, make([]byte, 35), make([]float32, 32)); CodeOf(err) != CodeSizeMismatch {
		t.Errorf("long src: %v", err)
	}
	if err := Dequantize(DType(999), nil, nil); CodeOf(err) != CodeBadDType {
		t.Errorf("unknown dtype: %v", err)
	}
	if err := Dequantize(Q4_K, nil, nil); err != nil {
		t.Errorf("empty: %v", err)
	}
	if err := Quantize(Q4_K, make([]float32, 256), make([]byte, 144)); CodeOf(err) != CodeUnsupportedDType {
		t.Errorf("Quantize(Q4_K): %v", err)
	}
	if err := Quantize(F32, make([]float32, 1), make([]byte, 4)); CodeOf(err) != CodeUnsupportedDType {
		t.Errorf("Quantize(F32): %v", err)
	}
	if err := Quantize(Q8_0, make([]float32, 32), make([]byte, 33)); CodeOf(err) != CodeSizeMismatch {
		t.Errorf("Quantize short dst: %v", err)
	}
	if err := Quantize(Q8_0, make([]float32, 31), make([]byte, 34)); CodeOf(err) != CodeSizeMismatch {
		t.Errorf("Quantize partial block: %v", err)
	}
}

// TestQuantizeMatchesGGUF checks that the reference Q8_0 and Q4_0 encoders
// produce the same bytes as gguf's quantize() for the golden input.
func TestQuantizeMatchesGGUF(t *testing.T) {
	g := loadGolden(t)
	for _, dt := range []DType{Q8_0, Q4_0} {
		v := g[dt.String()]
		raw, _ := base64.StdEncoding.DecodeString(v.Bytes)
		if len(v.InputBits) != len(v.Values) {
			t.Fatalf("%s: golden vector has no input", dt)
		}
		in := make([]float32, len(v.InputBits))
		for i, b := range v.InputBits {
			in[i] = math.Float32frombits(b)
		}
		out := make([]byte, len(raw))
		if err := Quantize(dt, in, out); err != nil {
			t.Fatal(err)
		}
		for i := range out {
			if out[i] != raw[i] {
				t.Errorf("%s: byte %d (block %d) = %#02x, want %#02x", dt, i, i/int(dt.BlockBytes()), out[i], raw[i])
				break
			}
		}
	}
}

// TestQuantizeIdempotent checks that re-quantizing already-quantized data
// reproduces it exactly.
func TestQuantizeIdempotent(t *testing.T) {
	g := loadGolden(t)
	for _, dt := range []DType{Q8_0, Q4_0} {
		v := g[dt.String()]
		raw, _ := base64.StdEncoding.DecodeString(v.Bytes)
		out := make([]byte, len(raw))
		if err := Quantize(dt, v.Values, out); err != nil {
			t.Fatal(err)
		}
		back := make([]float32, len(v.Values))
		if err := Dequantize(dt, out, back); err != nil {
			t.Fatal(err)
		}
		for i := range back {
			if math.Float32bits(back[i]) != v.Bits[i] {
				t.Errorf("%s: re-quantizing dequantized data changed element %d: %v -> %v", dt, i, v.Values[i], back[i])
				break
			}
		}
	}
}

func TestQuantizeRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const n = 4096
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64()) * 3
	}
	cases := []struct {
		dt     DType
		relErr float64 // max |y-x| relative to the block's max |x|
	}{
		// Each bound is half a quantization step plus the error from rounding
		// the block parameters to f16 (relative 2^-11 per parameter).
		{Q8_0, 1.0/254 + 1.0/2048},
		{Q4_0, 1.0/8 + 1.0/2048}, // the grid is -8..7 steps of max/8, so values opposite in sign to max can clip by a full step
		{Affine4, 2.0/30 + 3.0/2048},
	}
	for _, c := range cases {
		be := int(c.dt.BlockElems())
		bb := int(c.dt.BlockBytes())
		enc := make([]byte, n/be*bb)
		if err := Quantize(c.dt, x, enc); err != nil {
			t.Fatal(err)
		}
		dec := make([]float32, n)
		if err := Dequantize(c.dt, enc, dec); err != nil {
			t.Fatal(err)
		}
		var worst float64
		for b := 0; b < n/be; b++ {
			var amax float64
			for _, v := range x[b*be : (b+1)*be] {
				amax = math.Max(amax, math.Abs(float64(v)))
			}
			for i := b * be; i < (b+1)*be; i++ {
				e := math.Abs(float64(dec[i]-x[i])) / amax
				if e > worst {
					worst = e
				}
				if e > c.relErr {
					t.Errorf("%s: element %d: x=%v y=%v (rel err %.4f > %.4f)", c.dt, i, x[i], dec[i], e, c.relErr)
				}
			}
		}
		t.Logf("%s: worst relative error %.5f (bound %.5f)", c.dt, worst, c.relErr)
	}

	// Degenerate blocks: all zero and constant.
	for _, dt := range []DType{Q8_0, Q4_0, Affine4} {
		be := int(dt.BlockElems())
		for _, fill := range []float32{0, 2.5, -7} {
			src := make([]float32, be)
			for i := range src {
				src[i] = fill
			}
			enc := make([]byte, dt.BlockBytes())
			if err := Quantize(dt, src, enc); err != nil {
				t.Fatal(err)
			}
			dec := make([]float32, be)
			if err := Dequantize(dt, enc, dec); err != nil {
				t.Fatal(err)
			}
			for i, v := range dec {
				if v != v || math.Abs(float64(v-fill)) > math.Abs(float64(fill))/8+1e-6 {
					t.Errorf("%s constant %v: element %d decoded to %v", dt, fill, i, v)
					break
				}
			}
		}
	}
}
