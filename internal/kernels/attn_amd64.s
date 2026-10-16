//go:build amd64 && !purego

#include "textflag.h"

// AVX2 kernels for the attention fast path (attn.go).

// func scoresF32AVX2(q, k *float32, d, n int, scale float32, out *float32)
//
// Sets out[j] = scale * (q . k[j]) for the n rows of k, d floats each, d a
// positive multiple of 8. Four rows go together, each with its own
// accumulator, and their sums are reduced together.
TEXT ·scoresF32AVX2(SB), NOSPLIT, $0-48
	MOVQ         q+0(FP), SI
	MOVQ         k+8(FP), DI
	MOVQ         d+16(FP), R9
	MOVQ         n+24(FP), CX
	VBROADCASTSS scale+32(FP), X15
	MOVQ         out+40(FP), R8
	SHLQ         $2, R9           // bytes per row
	LEAQ         (R9)(R9*2), R10  // three rows

sc4:
	CMPQ   CX, $4
	JL     sc1
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	XORQ   AX, AX

sc4d:
	VMOVUPS     (SI)(AX*1), Y4
	LEAQ        (DI)(AX*1), BX
	VFMADD231PS (BX), Y4, Y0
	VFMADD231PS (BX)(R9*1), Y4, Y1
	VFMADD231PS (BX)(R9*2), Y4, Y2
	VFMADD231PS (BX)(R10*1), Y4, Y3
	ADDQ        $32, AX
	CMPQ        AX, R9
	JL          sc4d

	VHADDPS      Y1, Y0, Y0
	VHADDPS      Y3, Y2, Y2
	VHADDPS      Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VMULPS       X15, X0, X0
	VMOVUPS      X0, (R8)
	ADDQ         $16, R8
	LEAQ         (DI)(R9*4), DI
	SUBQ         $4, CX
	JMP          sc4

sc1:
	TESTQ  CX, CX
	JZ     scdone
	VXORPS Y0, Y0, Y0
	XORQ   AX, AX

sc1d:
	VMOVUPS     (SI)(AX*1), Y4
	VFMADD231PS (DI)(AX*1), Y4, Y0
	ADDQ        $32, AX
	CMPQ        AX, R9
	JL          sc1d

	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VMULSS       X15, X0, X0
	VMOVSS       X0, (R8)
	ADDQ         $4, R8
	ADDQ         R9, DI
	DECQ         CX
	JMP          sc1

scdone:
	VZEROUPPER
	RET

// expc holds the constants of expSumF32AVX2, eight copies of each: log2(e),
// ln(2) in two parts, the polynomial of Cephes' expf, 1, the lower limit
// of the arguments whose exp is a normal float32, and the exponent bias.
DATA expc<>+0(SB)/4, $0x3fb8aa3b
DATA expc<>+4(SB)/4, $0x3fb8aa3b
DATA expc<>+8(SB)/4, $0x3fb8aa3b
DATA expc<>+12(SB)/4, $0x3fb8aa3b
DATA expc<>+16(SB)/4, $0x3fb8aa3b
DATA expc<>+20(SB)/4, $0x3fb8aa3b
DATA expc<>+24(SB)/4, $0x3fb8aa3b
DATA expc<>+28(SB)/4, $0x3fb8aa3b
DATA expc<>+32(SB)/4, $0x3f318000
DATA expc<>+36(SB)/4, $0x3f318000
DATA expc<>+40(SB)/4, $0x3f318000
DATA expc<>+44(SB)/4, $0x3f318000
DATA expc<>+48(SB)/4, $0x3f318000
DATA expc<>+52(SB)/4, $0x3f318000
DATA expc<>+56(SB)/4, $0x3f318000
DATA expc<>+60(SB)/4, $0x3f318000
DATA expc<>+64(SB)/4, $0xb95e8083
DATA expc<>+68(SB)/4, $0xb95e8083
DATA expc<>+72(SB)/4, $0xb95e8083
DATA expc<>+76(SB)/4, $0xb95e8083
DATA expc<>+80(SB)/4, $0xb95e8083
DATA expc<>+84(SB)/4, $0xb95e8083
DATA expc<>+88(SB)/4, $0xb95e8083
DATA expc<>+92(SB)/4, $0xb95e8083
DATA expc<>+96(SB)/4, $0x39506967
DATA expc<>+100(SB)/4, $0x39506967
DATA expc<>+104(SB)/4, $0x39506967
DATA expc<>+108(SB)/4, $0x39506967
DATA expc<>+112(SB)/4, $0x39506967
DATA expc<>+116(SB)/4, $0x39506967
DATA expc<>+120(SB)/4, $0x39506967
DATA expc<>+124(SB)/4, $0x39506967
DATA expc<>+128(SB)/4, $0x3ab743ce
DATA expc<>+132(SB)/4, $0x3ab743ce
DATA expc<>+136(SB)/4, $0x3ab743ce
DATA expc<>+140(SB)/4, $0x3ab743ce
DATA expc<>+144(SB)/4, $0x3ab743ce
DATA expc<>+148(SB)/4, $0x3ab743ce
DATA expc<>+152(SB)/4, $0x3ab743ce
DATA expc<>+156(SB)/4, $0x3ab743ce
DATA expc<>+160(SB)/4, $0x3c088908
DATA expc<>+164(SB)/4, $0x3c088908
DATA expc<>+168(SB)/4, $0x3c088908
DATA expc<>+172(SB)/4, $0x3c088908
DATA expc<>+176(SB)/4, $0x3c088908
DATA expc<>+180(SB)/4, $0x3c088908
DATA expc<>+184(SB)/4, $0x3c088908
DATA expc<>+188(SB)/4, $0x3c088908
DATA expc<>+192(SB)/4, $0x3d2aa9c1
DATA expc<>+196(SB)/4, $0x3d2aa9c1
DATA expc<>+200(SB)/4, $0x3d2aa9c1
DATA expc<>+204(SB)/4, $0x3d2aa9c1
DATA expc<>+208(SB)/4, $0x3d2aa9c1
DATA expc<>+212(SB)/4, $0x3d2aa9c1
DATA expc<>+216(SB)/4, $0x3d2aa9c1
DATA expc<>+220(SB)/4, $0x3d2aa9c1
DATA expc<>+224(SB)/4, $0x3e2aaaaa
DATA expc<>+228(SB)/4, $0x3e2aaaaa
DATA expc<>+232(SB)/4, $0x3e2aaaaa
DATA expc<>+236(SB)/4, $0x3e2aaaaa
DATA expc<>+240(SB)/4, $0x3e2aaaaa
DATA expc<>+244(SB)/4, $0x3e2aaaaa
DATA expc<>+248(SB)/4, $0x3e2aaaaa
DATA expc<>+252(SB)/4, $0x3e2aaaaa
DATA expc<>+256(SB)/4, $0x3f000000
DATA expc<>+260(SB)/4, $0x3f000000
DATA expc<>+264(SB)/4, $0x3f000000
DATA expc<>+268(SB)/4, $0x3f000000
DATA expc<>+272(SB)/4, $0x3f000000
DATA expc<>+276(SB)/4, $0x3f000000
DATA expc<>+280(SB)/4, $0x3f000000
DATA expc<>+284(SB)/4, $0x3f000000
DATA expc<>+288(SB)/4, $0x3f800000
DATA expc<>+292(SB)/4, $0x3f800000
DATA expc<>+296(SB)/4, $0x3f800000
DATA expc<>+300(SB)/4, $0x3f800000
DATA expc<>+304(SB)/4, $0x3f800000
DATA expc<>+308(SB)/4, $0x3f800000
DATA expc<>+312(SB)/4, $0x3f800000
DATA expc<>+316(SB)/4, $0x3f800000
DATA expc<>+320(SB)/4, $0xc2aeac4f
DATA expc<>+324(SB)/4, $0xc2aeac4f
DATA expc<>+328(SB)/4, $0xc2aeac4f
DATA expc<>+332(SB)/4, $0xc2aeac4f
DATA expc<>+336(SB)/4, $0xc2aeac4f
DATA expc<>+340(SB)/4, $0xc2aeac4f
DATA expc<>+344(SB)/4, $0xc2aeac4f
DATA expc<>+348(SB)/4, $0xc2aeac4f
DATA expc<>+352(SB)/4, $0x0000007f
DATA expc<>+356(SB)/4, $0x0000007f
DATA expc<>+360(SB)/4, $0x0000007f
DATA expc<>+364(SB)/4, $0x0000007f
DATA expc<>+368(SB)/4, $0x0000007f
DATA expc<>+372(SB)/4, $0x0000007f
DATA expc<>+376(SB)/4, $0x0000007f
DATA expc<>+380(SB)/4, $0x0000007f
GLOBL expc<>(SB), RODATA|NOPTR, $384

// func expSumF32AVX2(x *float32, n int, m float32) float32
//
// Sets x[i] = exp(x[i] - m) for n a positive multiple of 8 and returns the
// sum, as Cephes' expf computes it: t = x - m is split as t = f*ln(2) + r
// with f an integer and |r| <= ln(2)/2, exp(r) comes from a polynomial,
// and 2^f is put into the exponent. Arguments below the limit, -inf
// among them, give 0.
TEXT ·expSumF32AVX2(SB), NOSPLIT, $0-28
	MOVQ         x+0(FP), SI
	MOVQ         n+8(FP), CX
	VBROADCASTSS m+16(FP), Y14
	VMOVUPS      expc<>+320(SB), Y13
	VMOVUPS      expc<>+0(SB), Y12
	VMOVUPS      expc<>+32(SB), Y11
	VMOVUPS      expc<>+64(SB), Y10
	VMOVUPS      expc<>+288(SB), Y9
	VMOVUPS      expc<>+352(SB), Y8
	VXORPS       Y15, Y15, Y15 // sum

exploop:
	VMOVUPS      (SI), Y0
	VSUBPS       Y14, Y0, Y0      // t
	VCMPPS       $0x1D, Y13, Y0, Y5 // t >= lo
	VMAXPS       Y13, Y0, Y0
	VMULPS       Y12, Y0, Y1
	VROUNDPS     $0, Y1, Y1       // f
	VFNMADD231PS Y11, Y1, Y0
	VFNMADD231PS Y10, Y1, Y0      // r
	VMULPS       Y0, Y0, Y3       // r*r
	VMOVUPS      expc<>+96(SB), Y2
	VFMADD213PS  expc<>+128(SB), Y0, Y2
	VFMADD213PS  expc<>+160(SB), Y0, Y2
	VFMADD213PS  expc<>+192(SB), Y0, Y2
	VFMADD213PS  expc<>+224(SB), Y0, Y2
	VFMADD213PS  expc<>+256(SB), Y0, Y2
	VFMADD213PS  Y0, Y3, Y2
	VADDPS       Y9, Y2, Y2       // exp(r)
	VCVTPS2DQ    Y1, Y1
	VPADDD       Y8, Y1, Y1
	VPSLLD       $23, Y1, Y1      // 2^f
	VMULPS       Y1, Y2, Y2
	VANDPS       Y5, Y2, Y2
	VMOVUPS      Y2, (SI)
	VADDPS       Y2, Y15, Y15
	ADDQ         $32, SI
	SUBQ         $8, CX
	JNZ          exploop

	VEXTRACTF128 $1, Y15, X1
	VADDPS       X1, X15, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VZEROUPPER
	MOVSS        X0, ret+24(FP)
	RET

// func wsumF32AVX2(p, v *float32, n, d int, acc *float32)
//
// Adds p[j] * v[j] to acc for the n rows of v, d floats each, d a positive
// multiple of 8, with one fused multiply-add per element and row, in the
// order of the rows, as axpyF32AVX2 row by row would.
TEXT ·wsumF32AVX2(SB), NOSPLIT, $0-40
	MOVQ p+0(FP), R11
	MOVQ v+8(FP), R12
	MOVQ n+16(FP), R13
	MOVQ d+24(FP), R9
	MOVQ acc+32(FP), R8
	SHLQ $2, R9 // bytes per row
	XORQ AX, AX // offset of the current columns

ws32:
	MOVQ    R9, BX
	SUBQ    AX, BX
	CMPQ    BX, $128
	JL      ws8
	VMOVUPS (R8)(AX*1), Y0
	VMOVUPS 32(R8)(AX*1), Y1
	VMOVUPS 64(R8)(AX*1), Y2
	VMOVUPS 96(R8)(AX*1), Y3
	MOVQ    R11, SI
	LEAQ    (R12)(AX*1), DI
	MOVQ    R13, CX

ws32k:
	VBROADCASTSS (SI), Y4
	VFMADD231PS  (DI), Y4, Y0
	VFMADD231PS  32(DI), Y4, Y1
	VFMADD231PS  64(DI), Y4, Y2
	VFMADD231PS  96(DI), Y4, Y3
	ADDQ         $4, SI
	ADDQ         R9, DI
	DECQ         CX
	JNZ          ws32k

	VMOVUPS Y0, (R8)(AX*1)
	VMOVUPS Y1, 32(R8)(AX*1)
	VMOVUPS Y2, 64(R8)(AX*1)
	VMOVUPS Y3, 96(R8)(AX*1)
	ADDQ    $128, AX
	JMP     ws32

ws8:
	CMPQ    AX, R9
	JGE     wsdone
	VMOVUPS (R8)(AX*1), Y0
	MOVQ    R11, SI
	LEAQ    (R12)(AX*1), DI
	MOVQ    R13, CX

ws8k:
	VBROADCASTSS (SI), Y4
	VFMADD231PS  (DI), Y4, Y0
	ADDQ         $4, SI
	ADDQ         R9, DI
	DECQ         CX
	JNZ          ws8k

	VMOVUPS Y0, (R8)(AX*1)
	ADDQ    $32, AX
	JMP     ws8

wsdone:
	VZEROUPPER
	RET

// func scoresF16AVX2(q *float32, k *byte, d, n int, scale float32, out *float32)
//
// scoresF32AVX2 for keys stored as f16, converted to float32 (exactly) as
// they are loaded. Needs F16C.
TEXT ·scoresF16AVX2(SB), NOSPLIT, $0-48
	MOVQ         q+0(FP), SI
	MOVQ         k+8(FP), DI
	MOVQ         d+16(FP), R9
	MOVQ         n+24(FP), CX
	VBROADCASTSS scale+32(FP), X15
	MOVQ         out+40(FP), R8
	SHLQ         $1, R9           // bytes per row
	LEAQ         (R9)(R9*2), R10  // three rows

sh4:
	CMPQ   CX, $4
	JL     sh1
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	XORQ   AX, AX

sh4d:
	VMOVUPS     (SI)(AX*2), Y4
	LEAQ        (DI)(AX*1), BX
	VCVTPH2PS   (BX), Y5
	VCVTPH2PS   (BX)(R9*1), Y6
	VCVTPH2PS   (BX)(R9*2), Y7
	VCVTPH2PS   (BX)(R10*1), Y8
	VFMADD231PS Y5, Y4, Y0
	VFMADD231PS Y6, Y4, Y1
	VFMADD231PS Y7, Y4, Y2
	VFMADD231PS Y8, Y4, Y3
	ADDQ        $16, AX
	CMPQ        AX, R9
	JL          sh4d

	VHADDPS      Y1, Y0, Y0
	VHADDPS      Y3, Y2, Y2
	VHADDPS      Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VMULPS       X15, X0, X0
	VMOVUPS      X0, (R8)
	ADDQ         $16, R8
	LEAQ         (DI)(R9*4), DI
	SUBQ         $4, CX
	JMP          sh4

sh1:
	TESTQ  CX, CX
	JZ     shdone
	VXORPS Y0, Y0, Y0
	XORQ   AX, AX

sh1d:
	VMOVUPS     (SI)(AX*2), Y4
	VCVTPH2PS   (DI)(AX*1), Y5
	VFMADD231PS Y5, Y4, Y0
	ADDQ        $16, AX
	CMPQ        AX, R9
	JL          sh1d

	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VMULSS       X15, X0, X0
	VMOVSS       X0, (R8)
	ADDQ         $4, R8
	ADDQ         R9, DI
	DECQ         CX
	JMP          sh1

shdone:
	VZEROUPPER
	RET

// func wsumF16AVX2(p *float32, v *byte, n, d int, acc *float32)
//
// wsumF32AVX2 for values stored as f16. Needs F16C.
TEXT ·wsumF16AVX2(SB), NOSPLIT, $0-40
	MOVQ p+0(FP), R11
	MOVQ v+8(FP), R12
	MOVQ n+16(FP), R13
	MOVQ d+24(FP), R9
	MOVQ acc+32(FP), R8
	SHLQ $1, R9 // bytes per row
	XORQ AX, AX // offset of the current columns in a row

wh32:
	MOVQ    R9, BX
	SUBQ    AX, BX
	CMPQ    BX, $64
	JL      wh8
	VMOVUPS (R8)(AX*2), Y0
	VMOVUPS 32(R8)(AX*2), Y1
	VMOVUPS 64(R8)(AX*2), Y2
	VMOVUPS 96(R8)(AX*2), Y3
	MOVQ    R11, SI
	LEAQ    (R12)(AX*1), DI
	MOVQ    R13, CX

wh32k:
	VBROADCASTSS (SI), Y4
	VCVTPH2PS    (DI), Y5
	VCVTPH2PS    16(DI), Y6
	VCVTPH2PS    32(DI), Y7
	VCVTPH2PS    48(DI), Y8
	VFMADD231PS  Y5, Y4, Y0
	VFMADD231PS  Y6, Y4, Y1
	VFMADD231PS  Y7, Y4, Y2
	VFMADD231PS  Y8, Y4, Y3
	ADDQ         $4, SI
	ADDQ         R9, DI
	DECQ         CX
	JNZ          wh32k

	VMOVUPS Y0, (R8)(AX*2)
	VMOVUPS Y1, 32(R8)(AX*2)
	VMOVUPS Y2, 64(R8)(AX*2)
	VMOVUPS Y3, 96(R8)(AX*2)
	ADDQ    $64, AX
	JMP     wh32

wh8:
	CMPQ    AX, R9
	JGE     whdone
	VMOVUPS (R8)(AX*2), Y0
	MOVQ    R11, SI
	LEAQ    (R12)(AX*1), DI
	MOVQ    R13, CX

wh8k:
	VBROADCASTSS (SI), Y4
	VCVTPH2PS    (DI), Y5
	VFMADD231PS  Y5, Y4, Y0
	ADDQ         $4, SI
	ADDQ         R9, DI
	DECQ         CX
	JNZ          wh8k

	VMOVUPS Y0, (R8)(AX*2)
	ADDQ    $16, AX
	JMP     wh8

whdone:
	VZEROUPPER
	RET
