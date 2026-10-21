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

// halff is one half, and signf masks the sign bit of a float32 in.
DATA halff<>+0(SB)/4, $0x3f000000
GLOBL halff<>(SB), RODATA, $4
DATA signf<>+0(SB)/4, $0x80000000
GLOBL signf<>(SB), RODATA, $4

// func siluMulF32AVX2(a, b, out *float32, n int)
//
// Sets out[i] to silu(a[i]) * b[i] for n a positive multiple of 8, with
// silu(x) = x / (1+e) for x >= 0 and (x*h)*h / (1+e) below, where
// h = exp(-|x|/2) is computed as in expSumF32AVX2 and e = h*h. Halving the
// exponent keeps h normal for any x whose silu is a normal float, so the
// result is as accurate down to the smallest normals; h is at most 1, so
// nothing overflows.
TEXT ·siluMulF32AVX2(SB), NOSPLIT, $0-32
	MOVQ         a+0(FP), SI
	MOVQ         b+8(FP), DI
	MOVQ         out+16(FP), DX
	MOVQ         n+24(FP), CX
	VMOVUPS      expc<>+320(SB), Y13
	VMOVUPS      expc<>+0(SB), Y12
	VMOVUPS      expc<>+32(SB), Y11
	VMOVUPS      expc<>+64(SB), Y10
	VMOVUPS      expc<>+288(SB), Y9
	VMOVUPS      expc<>+352(SB), Y8
	VBROADCASTSS signf<>(SB), Y7
	VXORPS       Y6, Y6, Y6
	VBROADCASTSS halff<>(SB), Y15

siluloop:
	VMOVUPS      (SI), Y4         // x
	VORPS        Y7, Y4, Y0
	VMULPS       Y15, Y0, Y0      // t = -|x|/2
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
	VANDPS       Y5, Y2, Y2       // h
	VMULPS       Y2, Y2, Y3
	VADDPS       Y9, Y3, Y3       // 1+e
	VMULPS       Y4, Y2, Y1
	VMULPS       Y2, Y1, Y1       // (x*h)*h
	VCMPPS       $0x1D, Y6, Y4, Y5 // x >= 0
	VBLENDVPS    Y5, Y4, Y1, Y1   // x or (x*h)*h
	VDIVPS       Y3, Y1, Y1
	VMULPS       (DI), Y1, Y1
	VMOVUPS      Y1, (DX)
	ADDQ         $32, SI
	ADDQ         $32, DI
	ADDQ         $32, DX
	SUBQ         $8, CX
	JNZ          siluloop
	VZEROUPPER
	RET

// func sigmoidF32AVX2(x, y *float32, n int, nan float32)
//
// Sets y[i] = 1/(1+exp(-x[i])) for n a positive multiple of 8, with
// e = exp(-|x|) computed as expSumF32AVX2 does (so never above 1): the
// result is 1/(1+e) for x >= 0 and e/(1+e) otherwise, which keeps the
// relative error within a few units in the last place for every x. A
// NaN gives nan; infinities give 0 and 1.
TEXT ·sigmoidF32AVX2(SB), NOSPLIT, $0-28
	MOVQ         x+0(FP), SI
	MOVQ         y+8(FP), DI
	MOVQ         n+16(FP), CX
	VBROADCASTSS nan+24(FP), Y14
	VMOVUPS      expc<>+320(SB), Y13
	VMOVUPS      expc<>+0(SB), Y12
	VMOVUPS      expc<>+32(SB), Y11
	VMOVUPS      expc<>+64(SB), Y10
	VMOVUPS      expc<>+288(SB), Y9
	VMOVUPS      expc<>+352(SB), Y8
	VBROADCASTSS signf<>(SB), Y7
	VXORPS       Y6, Y6, Y6

sigloop:
	VMOVUPS      (SI), Y4         // x
	VANDNPS      Y4, Y7, Y0       // |x|
	VXORPS       Y7, Y0, Y0       // t = -|x|
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
	VANDPS       Y5, Y2, Y2       // e = exp(-|x|)
	VADDPS       Y9, Y2, Y3       // 1+e
	VDIVPS       Y3, Y9, Y1       // s = 1/(1+e)
	VMULPS       Y2, Y1, Y3       // e/(1+e)
	VCMPPS       $0x1D, Y6, Y4, Y5 // x >= 0
	VBLENDVPS    Y5, Y1, Y3, Y1   // s or e/(1+e)
	VCMPPS       $3, Y4, Y4, Y5   // x is NaN
	VBLENDVPS    Y5, Y14, Y1, Y1
	VMOVUPS      Y1, (DI)
	ADDQ         $32, SI
	ADDQ         $32, DI
	SUBQ         $8, CX
	JNZ          sigloop
	VZEROUPPER
	RET

// func tanhF32AVX2(x, y *float32, n int, nan float32)
//
// Sets y[i] = tanh(x[i]) for n a positive multiple of 8, as
// sign(x) (1-e)/(1+e) with e = exp(-2|x|) computed as expSumF32AVX2
// does: the absolute error is within a few units in the last place of
// 1. A NaN gives nan; infinities give -1 and 1.
TEXT ·tanhF32AVX2(SB), NOSPLIT, $0-28
	MOVQ         x+0(FP), SI
	MOVQ         y+8(FP), DI
	MOVQ         n+16(FP), CX
	VBROADCASTSS nan+24(FP), Y14
	VMOVUPS      expc<>+320(SB), Y13
	VMOVUPS      expc<>+0(SB), Y12
	VMOVUPS      expc<>+32(SB), Y11
	VMOVUPS      expc<>+64(SB), Y10
	VMOVUPS      expc<>+288(SB), Y9
	VMOVUPS      expc<>+352(SB), Y8
	VBROADCASTSS signf<>(SB), Y7

tanhloop:
	VMOVUPS      (SI), Y4         // x
	VANDNPS      Y4, Y7, Y0       // |x|
	VADDPS       Y0, Y0, Y0       // 2|x|
	VXORPS       Y7, Y0, Y0       // t = -2|x|
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
	VANDPS       Y5, Y2, Y2       // e = exp(-2|x|)
	VSUBPS       Y2, Y9, Y1       // 1-e
	VADDPS       Y9, Y2, Y3       // 1+e
	VDIVPS       Y3, Y1, Y1       // (1-e)/(1+e), never negative
	VANDPS       Y7, Y4, Y5       // the sign of x
	VORPS        Y5, Y1, Y1
	VCMPPS       $3, Y4, Y4, Y5   // x is NaN
	VBLENDVPS    Y5, Y14, Y1, Y1
	VMOVUPS      Y1, (DI)
	ADDQ         $32, SI
	ADDQ         $32, DI
	SUBQ         $8, CX
	JNZ          tanhloop
	VZEROUPPER
	RET
// func sigmoidF32AVX512(x, y *float32, n int, nan float32)
//
// sigmoidF32AVX2 sixteen at a time, for n a positive multiple of 16.
TEXT ·sigmoidF32AVX512(SB), NOSPLIT, $0-28
	MOVQ         x+0(FP), SI
	MOVQ         y+8(FP), DI
	MOVQ         n+16(FP), CX
	VBROADCASTSS nan+24(FP), Z14
	VBROADCASTSS expc<>+320(SB), Z13
	VBROADCASTSS expc<>+0(SB), Z12
	VBROADCASTSS expc<>+32(SB), Z11
	VBROADCASTSS expc<>+64(SB), Z10
	VBROADCASTSS expc<>+288(SB), Z9
	VBROADCASTSS expc<>+352(SB), Z8
	VBROADCASTSS signf<>(SB), Z7
	VXORPS       Z6, Z6, Z6

sigmoidF32AVX512loop:
	VMOVUPS      (SI), Z4         // x
	VANDNPS      Z4, Z7, Z0       // |x|
	VXORPS       Z7, Z0, Z0       // t = -|x|
	VCMPPS       $0x1D, Z13, Z0, K5 // t >= lo
	VMAXPS       Z13, Z0, Z0
	VMULPS       Z12, Z0, Z1
	VRNDSCALEPS  $0, Z1, Z1       // f
	VFNMADD231PS Z11, Z1, Z0
	VFNMADD231PS Z10, Z1, Z0      // r
	VMULPS       Z0, Z0, Z3       // r*r
	VBROADCASTSS expc<>+96(SB), Z2
	VFMADD213PS.BCST expc<>+128(SB), Z0, Z2
	VFMADD213PS.BCST expc<>+160(SB), Z0, Z2
	VFMADD213PS.BCST expc<>+192(SB), Z0, Z2
	VFMADD213PS.BCST expc<>+224(SB), Z0, Z2
	VFMADD213PS.BCST expc<>+256(SB), Z0, Z2
	VFMADD213PS  Z0, Z3, Z2
	VADDPS       Z9, Z2, Z2       // exp(r)
	VCVTPS2DQ    Z1, Z1
	VPADDD       Z8, Z1, Z1
	VPSLLD       $23, Z1, Z1      // 2^f
	VMULPS       Z1, Z2, Z2
	VMOVAPS.Z    Z2, K5, Z2       // e = exp(-|x|)
	VADDPS       Z9, Z2, Z3       // 1+e
	VDIVPS       Z3, Z9, Z1       // s = 1/(1+e)
	VMULPS       Z2, Z1, Z3       // e/(1+e)
	VCMPPS       $0x1D, Z6, Z4, K5 // x >= 0
	KNOTW        K5, K5
	VMOVAPS      Z3, K5, Z1       // s or e/(1+e)
	VCMPPS       $3, Z4, Z4, K5   // x is NaN
	VMOVAPS      Z14, K5, Z1
	VMOVUPS      Z1, (DI)
	ADDQ         $64, SI
	ADDQ         $64, DI
	SUBQ         $16, CX
	JNZ          sigmoidF32AVX512loop
	VZEROUPPER
	RET

// func tanhF32AVX512(x, y *float32, n int, nan float32)
//
// tanhF32AVX2 sixteen at a time, for n a positive multiple of 16.
TEXT ·tanhF32AVX512(SB), NOSPLIT, $0-28
	MOVQ         x+0(FP), SI
	MOVQ         y+8(FP), DI
	MOVQ         n+16(FP), CX
	VBROADCASTSS nan+24(FP), Z14
	VBROADCASTSS expc<>+320(SB), Z13
	VBROADCASTSS expc<>+0(SB), Z12
	VBROADCASTSS expc<>+32(SB), Z11
	VBROADCASTSS expc<>+64(SB), Z10
	VBROADCASTSS expc<>+288(SB), Z9
	VBROADCASTSS expc<>+352(SB), Z8
	VBROADCASTSS signf<>(SB), Z7
	VXORPS       Z6, Z6, Z6

tanhF32AVX512loop:
	VMOVUPS      (SI), Z4         // x
	VANDNPS      Z4, Z7, Z0       // |x|
	VADDPS       Z0, Z0, Z0       // 2|x|
	VXORPS       Z7, Z0, Z0       // t = -2|x|
	VCMPPS       $0x1D, Z13, Z0, K5 // t >= lo
	VMAXPS       Z13, Z0, Z0
	VMULPS       Z12, Z0, Z1
	VRNDSCALEPS  $0, Z1, Z1       // f
	VFNMADD231PS Z11, Z1, Z0
	VFNMADD231PS Z10, Z1, Z0      // r
	VMULPS       Z0, Z0, Z3       // r*r
	VBROADCASTSS expc<>+96(SB), Z2
	VFMADD213PS.BCST expc<>+128(SB), Z0, Z2
	VFMADD213PS.BCST expc<>+160(SB), Z0, Z2
	VFMADD213PS.BCST expc<>+192(SB), Z0, Z2
	VFMADD213PS.BCST expc<>+224(SB), Z0, Z2
	VFMADD213PS.BCST expc<>+256(SB), Z0, Z2
	VFMADD213PS  Z0, Z3, Z2
	VADDPS       Z9, Z2, Z2       // exp(r)
	VCVTPS2DQ    Z1, Z1
	VPADDD       Z8, Z1, Z1
	VPSLLD       $23, Z1, Z1      // 2^f
	VMULPS       Z1, Z2, Z2
	VMOVAPS.Z    Z2, K5, Z2       // e = exp(-2|x|)
	VSUBPS       Z2, Z9, Z1       // 1-e
	VADDPS       Z9, Z2, Z3       // 1+e
	VDIVPS       Z3, Z1, Z1       // (1-e)/(1+e), never negative
	VANDPS       Z7, Z4, Z5       // the sign of x
	VORPS        Z5, Z1, Z1
	VCMPPS       $3, Z4, Z4, K5   // x is NaN
	VMOVAPS      Z14, K5, Z1
	VMOVUPS      Z1, (DI)
	ADDQ         $64, SI
	ADDQ         $64, DI
	SUBQ         $16, CX
	JNZ          tanhF32AVX512loop
	VZEROUPPER
	RET

