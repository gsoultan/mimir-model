//go:build amd64 && !purego

#include "textflag.h"

// AVX2 and FMA kernels. Each weight is decoded with the same float32
// operations, in the same order, as mmf.Dequantize, and then multiplied
// into the activations with fused multiply-adds into four accumulators.

// func cpuid(eax, ecx uint32) (a, b, c, d uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL eax+0(FP), AX
	MOVL ecx+4(FP), CX
	CPUID
	MOVL AX, a+8(FP)
	MOVL BX, b+12(FP)
	MOVL CX, c+16(FP)
	MOVL DX, d+20(FP)
	RET

// func xgetbv() (eax, edx uint32)
TEXT ·xgetbv(SB), NOSPLIT, $0-8
	MOVL $0, CX
	XGETBV
	MOVL AX, eax+0(FP)
	MOVL DX, edx+4(FP)
	RET

// HSUM adds the four accumulators Y0..Y3 and leaves the sum of the eight
// lanes in X0.
#define HSUM \
	VADDPS       Y1, Y0, Y0; \
	VADDPS       Y3, Y2, Y2; \
	VADDPS       Y2, Y0, Y0; \
	VEXTRACTF128 $1, Y0, X1; \
	VADDPS       X1, X0, X0; \
	VHADDPS      X0, X0, X0; \
	VHADDPS      X0, X0, X0; \
	VZEROUPPER

#define ZERO4 \
	VXORPS Y0, Y0, Y0; \
	VXORPS Y1, Y1, Y1; \
	VXORPS Y2, Y2, Y2; \
	VXORPS Y3, Y3, Y3

// func dotF32AVX2(a, b *float32, n int) float32
// n is a positive multiple of 8.
TEXT ·dotF32AVX2(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), SI
	MOVQ b+8(FP), DI
	MOVQ n+16(FP), CX
	ZERO4

f32loop32:
	CMPQ        CX, $32
	JL          f32loop8
	VMOVUPS     (SI), Y4
	VMOVUPS     32(SI), Y5
	VMOVUPS     64(SI), Y6
	VMOVUPS     96(SI), Y7
	VFMADD231PS (DI), Y4, Y0
	VFMADD231PS 32(DI), Y5, Y1
	VFMADD231PS 64(DI), Y6, Y2
	VFMADD231PS 96(DI), Y7, Y3
	ADDQ        $128, SI
	ADDQ        $128, DI
	SUBQ        $32, CX
	JMP         f32loop32

f32loop8:
	CMPQ        CX, $8
	JL          f32done
	VMOVUPS     (SI), Y4
	VFMADD231PS (DI), Y4, Y0
	ADDQ        $32, SI
	ADDQ        $32, DI
	SUBQ        $8, CX
	JMP         f32loop8

f32done:
	HSUM
	MOVSS X0, ret+24(FP)
	RET

// func axpyF32AVX2(alpha float32, x, y *float32, n int)
// n is a positive multiple of 8.
TEXT ·axpyF32AVX2(SB), NOSPLIT, $0-32
	VBROADCASTSS alpha+0(FP), Y15
	MOVQ         x+8(FP), SI
	MOVQ         y+16(FP), DI
	MOVQ         n+24(FP), CX

axpyloop:
	VMOVUPS     (DI), Y0
	VFMADD231PS (SI), Y15, Y0
	VMOVUPS     Y0, (DI)
	ADDQ        $32, SI
	ADDQ        $32, DI
	SUBQ        $8, CX
	JNZ         axpyloop
	VZEROUPPER
	RET

// Q4_K: 144-byte blocks of 256 weights (f16 d, f16 dmin, 12 scale bytes,
// 128 bytes of nibbles). tab holds 16 floats per block, the eight
// sub-block scales d*sc and then the eight offsets dmin*m. The 32 bytes
// of chunk c hold sub-block 2c in their low nibbles and 2c+1 in their high
// nibbles; w = d*sc*q - dmin*m.

// Q4STEP decodes the 8 low and 8 high nibbles of the bytes at qo and
// accumulates them against x at byte offsets lo and hi.
#define Q4STEP(qo, lo, hi, accA, accB) \
	VPMOVZXBD   qo(R8), Y8; \
	VPAND       Y15, Y8, Y9; \
	VPSRLD      $4, Y8, Y10; \
	VCVTDQ2PS   Y9, Y9; \
	VCVTDQ2PS   Y10, Y10; \
	VMULPS      Y4, Y9, Y9; \
	VSUBPS      Y5, Y9, Y9; \
	VMULPS      Y6, Y10, Y10; \
	VSUBPS      Y7, Y10, Y10; \
	VFMADD231PS lo(R9), Y9, accA; \
	VFMADD231PS hi(R9), Y10, accB

// func dotQ4KAVX2(q *byte, x, tab *float32, nb int) float32
// nb is positive.
TEXT ·dotQ4KAVX2(SB), NOSPLIT, $0-36
	MOVQ q+0(FP), SI
	MOVQ x+8(FP), DI
	MOVQ tab+16(FP), DX
	MOVQ nb+24(FP), CX
	ZERO4
	VPCMPEQD Y15, Y15, Y15
	VPSRLD   $28, Y15, Y15 // 0x0000000F

q4block:
	// Prefetch the weights two rows of K = 2048 ahead (16 blocks of 144
	// bytes): the hardware prefetcher alone leaves each core well short of
	// the memory bandwidth when every core streams its own rows.
	PREFETCHT0 2304(SI)
	PREFETCHT0 2368(SI)
	PREFETCHT0 2432(SI)
	LEAQ 16(SI), R8 // the nibbles of chunk 0
	MOVQ DI, R9     // x of chunk 0
	MOVQ DX, R10    // the scales of chunk 0
	MOVQ $4, BX

q4chunk:
	VBROADCASTSS (R10), Y4  // d*sc of sub-block 2c
	VBROADCASTSS 32(R10), Y5 // dmin*m of sub-block 2c
	VBROADCASTSS 4(R10), Y6  // d*sc of sub-block 2c+1
	VBROADCASTSS 36(R10), Y7 // dmin*m of sub-block 2c+1
	Q4STEP(0, 0, 128, Y0, Y1)
	Q4STEP(8, 32, 160, Y2, Y3)
	Q4STEP(16, 64, 192, Y0, Y1)
	Q4STEP(24, 96, 224, Y2, Y3)
	ADDQ $32, R8
	ADDQ $256, R9
	ADDQ $8, R10
	DECQ BX
	JNZ  q4chunk

	ADDQ $144, SI
	ADDQ $1024, DI
	ADDQ $64, DX
	DECQ CX
	JNZ  q4block

	HSUM
	MOVSS X0, ret+32(FP)
	RET

// Q6_K: 210-byte blocks of 256 weights (128 bytes ql, 64 bytes qh, 16
// int8 scales, f16 d). tab holds the 16 products d*scale per block. Each
// half of a block covers 128 weights from 64 bytes of ql and 32 of qh; for
// l in 0..31 of a half:
//
//	q1 = ql[l]&15    | (qh[l]&3)<<4       weight l      scale l/16
//	q2 = ql[l+32]&15 | (qh[l]>>2&3)<<4    weight l+32   scale l/16+2
//	q3 = ql[l]>>4    | (qh[l]>>4&3)<<4    weight l+64   scale l/16+4
//	q4 = ql[l+32]>>4 | (qh[l]>>6&3)<<4    weight l+96   scale l/16+6
//
// and w = (d*scale)*(q-32).

// Q6ONE finishes one of the four quants: OR in the high bits, subtract 32,
// convert, scale and accumulate.
#define Q6ONE(lowbits, highbits, scale, xo, acc) \
	VPOR        highbits, lowbits, lowbits; \
	VPSUBD      Y13, lowbits, lowbits; \
	VCVTDQ2PS   lowbits, lowbits; \
	VMULPS      scale, lowbits, lowbits; \
	VFMADD231PS xo(R9), lowbits, acc

// Q6STEP decodes weights l..l+7 of each quarter of a half block.
#define Q6STEP(l, l32, x1, x2, x3, x4) \
	VPMOVZXBD l(R8), Y8; \
	VPMOVZXBD l32(R8), Y9; \
	VPMOVZXBD l(R11), Y10; \
	VPAND     Y15, Y8, Y11; \
	VPSLLD    $4, Y10, Y12; \
	VPAND     Y14, Y12, Y12; \
	Q6ONE(Y11, Y12, Y4, x1, Y0); \
	VPAND     Y15, Y9, Y11; \
	VPSLLD    $2, Y10, Y12; \
	VPAND     Y14, Y12, Y12; \
	Q6ONE(Y11, Y12, Y5, x2, Y1); \
	VPSRLD    $4, Y8, Y11; \
	VPAND     Y14, Y10, Y12; \
	Q6ONE(Y11, Y12, Y6, x3, Y2); \
	VPSRLD    $4, Y9, Y11; \
	VPSRLD    $2, Y10, Y12; \
	VPAND     Y14, Y12, Y12; \
	Q6ONE(Y11, Y12, Y7, x4, Y3)

// Q6SCALES broadcasts the four scales of sub-row s (0 or 1) of a half.
#define Q6SCALES(s0, s2, s4, s6) \
	VBROADCASTSS s0(R10), Y4; \
	VBROADCASTSS s2(R10), Y5; \
	VBROADCASTSS s4(R10), Y6; \
	VBROADCASTSS s6(R10), Y7

// func dotQ6KAVX2(q *byte, x, tab *float32, nb int) float32
// nb is positive.
TEXT ·dotQ6KAVX2(SB), NOSPLIT, $0-36
	MOVQ q+0(FP), SI
	MOVQ x+8(FP), DI
	MOVQ tab+16(FP), DX
	MOVQ nb+24(FP), CX
	ZERO4
	VPCMPEQD Y15, Y15, Y15
	VPSRLD   $28, Y15, Y15 // 0x0F
	VPCMPEQD Y14, Y14, Y14
	VPSRLD   $30, Y14, Y14
	VPSLLD   $4, Y14, Y14  // 0x30
	VPCMPEQD Y13, Y13, Y13
	VPSRLD   $31, Y13, Y13
	VPSLLD   $5, Y13, Y13  // 32

q6block:
	// Prefetch two rows of K = 2048 ahead (16 blocks of 210 bytes).
	PREFETCHT0 3360(SI)
	PREFETCHT0 3424(SI)
	PREFETCHT0 3488(SI)
	PREFETCHT0 3552(SI)
	MOVQ SI, R8      // ql of half 0
	LEAQ 128(SI), R11 // qh of half 0
	MOVQ DI, R9      // x of half 0
	MOVQ DX, R10     // scales of half 0
	MOVQ $2, BX

q6half:
	Q6SCALES(0, 8, 16, 24)
	Q6STEP(0, 32, 0, 128, 256, 384)
	Q6STEP(8, 40, 32, 160, 288, 416)
	Q6SCALES(4, 12, 20, 28)
	Q6STEP(16, 48, 64, 192, 320, 448)
	Q6STEP(24, 56, 96, 224, 352, 480)
	ADDQ $64, R8
	ADDQ $32, R11
	ADDQ $512, R9
	ADDQ $32, R10
	DECQ BX
	JNZ  q6half

	ADDQ $210, SI
	ADDQ $1024, DI
	ADDQ $64, DX
	DECQ CX
	JNZ  q6block

	HSUM
	MOVSS X0, ret+32(FP)
	RET

// Q8_0: 34-byte blocks of 32 weights (f16 d, 32 int8). tab holds d per
// block; w = d*q.
#define Q8STEP(qo, xo, acc) \
	VPMOVSXBD   qo(SI), Y8; \
	VCVTDQ2PS   Y8, Y8; \
	VMULPS      Y4, Y8, Y8; \
	VFMADD231PS xo(DI), Y8, acc

// func dotQ8_0AVX2(q *byte, x, tab *float32, nb int) float32
// nb is positive.
TEXT ·dotQ8_0AVX2(SB), NOSPLIT, $0-36
	MOVQ q+0(FP), SI
	MOVQ x+8(FP), DI
	MOVQ tab+16(FP), DX
	MOVQ nb+24(FP), CX
	ZERO4

q8block:
	VBROADCASTSS (DX), Y4
	Q8STEP(2, 0, Y0)
	Q8STEP(10, 32, Y1)
	Q8STEP(18, 64, Y2)
	Q8STEP(26, 96, Y3)
	ADDQ $34, SI
	ADDQ $128, DI
	ADDQ $4, DX
	DECQ CX
	JNZ  q8block

	HSUM
	MOVSS X0, ret+32(FP)
	RET

// func dotF16AVX2(w *byte, x *float32, n int) float32
// n is a positive multiple of 8. Needs F16C.
TEXT ·dotF16AVX2(SB), NOSPLIT, $0-28
	MOVQ w+0(FP), SI
	MOVQ x+8(FP), DI
	MOVQ n+16(FP), CX
	ZERO4

f16loop32:
	CMPQ        CX, $32
	JL          f16loop8
	VCVTPH2PS   (SI), Y4
	VCVTPH2PS   16(SI), Y5
	VCVTPH2PS   32(SI), Y6
	VCVTPH2PS   48(SI), Y7
	VFMADD231PS (DI), Y4, Y0
	VFMADD231PS 32(DI), Y5, Y1
	VFMADD231PS 64(DI), Y6, Y2
	VFMADD231PS 96(DI), Y7, Y3
	ADDQ        $64, SI
	ADDQ        $128, DI
	SUBQ        $32, CX
	JMP         f16loop32

f16loop8:
	CMPQ        CX, $8
	JL          f16done
	VCVTPH2PS   (SI), Y4
	VFMADD231PS (DI), Y4, Y0
	ADDQ        $16, SI
	ADDQ        $32, DI
	SUBQ        $8, CX
	JMP         f16loop8

f16done:
	HSUM
	MOVSS X0, ret+24(FP)
	RET

// func dotBF16AVX2(w *byte, x *float32, n int) float32
// n is a positive multiple of 8. A bf16 is the high half of a float32.
TEXT ·dotBF16AVX2(SB), NOSPLIT, $0-28
	MOVQ w+0(FP), SI
	MOVQ x+8(FP), DI
	MOVQ n+16(FP), CX
	ZERO4

bf16loop32:
	CMPQ        CX, $32
	JL          bf16loop8
	VPMOVZXWD   (SI), Y4
	VPMOVZXWD   16(SI), Y5
	VPMOVZXWD   32(SI), Y6
	VPMOVZXWD   48(SI), Y7
	VPSLLD      $16, Y4, Y4
	VPSLLD      $16, Y5, Y5
	VPSLLD      $16, Y6, Y6
	VPSLLD      $16, Y7, Y7
	VFMADD231PS (DI), Y4, Y0
	VFMADD231PS 32(DI), Y5, Y1
	VFMADD231PS 64(DI), Y6, Y2
	VFMADD231PS 96(DI), Y7, Y3
	ADDQ        $64, SI
	ADDQ        $128, DI
	SUBQ        $32, CX
	JMP         bf16loop32

bf16loop8:
	CMPQ        CX, $8
	JL          bf16done
	VPMOVZXWD   (SI), Y4
	VPSLLD      $16, Y4, Y4
	VFMADD231PS (DI), Y4, Y0
	ADDQ        $16, SI
	ADDQ        $32, DI
	SUBQ        $8, CX
	JMP         bf16loop8

bf16done:
	HSUM
	MOVSS X0, ret+24(FP)
	RET

// 8-bit activation kernels (q8.go). The activations come as int8 quants
// in blocks of 32 with a float32 scale per block. Each block's integer dot
// product with the weight quants is exact in int32; it is converted to
// float32 and scaled.

// BYTES0F sets reg to 0x0F in every byte, and BYTES03 to 0x03.
#define BYTES0F(reg) \
	VPCMPEQW  reg, reg, reg; \
	VPSRLW    $12, reg, reg; \
	VPACKUSWB reg, reg, reg

#define BYTES03(reg) \
	VPCMPEQW  reg, reg, reg; \
	VPSRLW    $14, reg, reg; \
	VPACKUSWB reg, reg, reg

// HSUM1 leaves the sum of the eight lanes of Y0 in X0.
#define HSUM1 \
	VEXTRACTF128 $1, Y0, X1; \
	VADDPS       X1, X0, X0; \
	VHADDPS      X0, X0, X0; \
	VHADDPS      X0, X0, X0; \
	VZEROUPPER

// F16FINITE jumps to bad if the f16 at mem is infinite or NaN.
#define F16FINITE(mem, bad) \
	MOVWLZX mem, AX; \
	ANDL    $0x7C00, AX; \
	CMPL    AX, $0x7C00; \
	JEQ     bad

// F16SCALAR converts the f16 at mem to float32 in the low lane of xreg.
#define F16SCALAR(mem, xreg) \
	MOVWLZX   mem, AX; \
	VMOVD     AX, xreg; \
	VCVTPH2PS xreg, xreg

// func dotQ4KQ8AVX2(q *byte, xq *int8, xd, xdbs *float32, nb int) (float32, bool)
//
// The result is false, and the sum meaningless, if a block scale is not
// finite.
//
// With sub-block scales sc_s and offsets m_s, and activation scales dx_s
// and quants y:
//
//	sum w*x = d * sum_s dx_s * sc_s * sum q*y  -  sum_s dmin*m_s * dx_s * sum y
//
// xdbs holds dx_s * sum y per block of 32. The 6-bit scales and offsets are
// unpacked as in ggml (get_scale_min_k4, on 32-bit words); the scales go to
// the stack as int16 pairs for VPMADDWD.
TEXT ·dotQ4KQ8AVX2(SB), NOSPLIT, $32-45
	MOVQ   q+0(FP), SI
	MOVQ   xq+8(FP), DI
	MOVQ   xd+16(FP), R13
	MOVQ   xdbs+24(FP), R14
	MOVQ   nb+32(FP), CX
	VXORPS Y2, Y2, Y2 // d * scaled integer dot products
	VXORPS Y3, Y3, Y3 // offsets
	BYTES0F(Y15)

q4q8block:
	// Prefetch the weights two rows of K = 2048 ahead (16 blocks of 144
	// bytes): the hardware prefetcher alone leaves each core well short of
	// the memory bandwidth when every core streams its own rows.
	PREFETCHT0 2304(SI)
	PREFETCHT0 2368(SI)
	PREFETCHT0 2432(SI)
	F16FINITE(0(SI), q4q8bad)
	F16FINITE(2(SI), q4q8bad)
	F16SCALAR(0(SI), X13)
	VBROADCASTSS X13, Y13 // d
	F16SCALAR(2(SI), X12)
	VBROADCASTSS X12, Y12 // dmin

	MOVL 4(SI), R8  // scale bytes 0..3
	MOVL 8(SI), R9  // 4..7
	MOVL 12(SI), R10 // 8..11
	MOVL R10, AX
	SHRL $4, AX
	ANDL $0x0f0f0f0f, AX
	MOVL R9, BX
	SHRL $6, BX
	ANDL $0x03030303, BX
	SHLL $4, BX
	ORL  BX, AX          // m 4..7
	MOVL R9, R11
	ANDL $0x3f3f3f3f, R11 // m 0..3
	ANDL $0x0f0f0f0f, R10
	MOVL R8, BX
	SHRL $6, BX
	ANDL $0x03030303, BX
	SHLL $4, BX
	ORL  BX, R10         // sc 4..7
	ANDL $0x3f3f3f3f, R8 // sc 0..3

	SHLQ        $32, AX
	ORQ         R11, AX
	VMOVQ       AX, X9
	VPMOVZXBD   X9, Y9
	VCVTDQ2PS   Y9, Y9
	VMULPS      Y12, Y9, Y9 // dmin*m
	VFMADD231PS (R14), Y9, Y3

	SHLQ       $32, R10
	ORQ        R8, R10
	VMOVQ      R10, X10
	VPMOVZXBW  X10, X10
	VPUNPCKLWD X10, X10, X11
	VPUNPCKHWD X10, X10, X10
	VMOVDQU    X11, 0(SP)
	VMOVDQU    X10, 16(SP)

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	LEAQ   16(SI), R8 // the nibbles of chunk 0
	LEAQ   0(SP), R10 // the scale pairs of chunk 0
	MOVQ   $4, BX

q4q8chunk:
	VMOVDQU      (R8), Y8
	VPAND        Y15, Y8, Y9   // sub-block 2c
	VPSRLW       $4, Y8, Y10
	VPAND        Y15, Y10, Y10 // sub-block 2c+1
	VPMADDUBSW   (DI), Y9, Y9
	VPMADDUBSW   32(DI), Y10, Y10
	VPBROADCASTD (R10), Y11
	VPBROADCASTD 4(R10), Y12
	VPMADDWD     Y11, Y9, Y9
	VPMADDWD     Y12, Y10, Y10
	VCVTDQ2PS    Y9, Y9
	VCVTDQ2PS    Y10, Y10
	VBROADCASTSS (R13), Y11
	VBROADCASTSS 4(R13), Y12
	VFMADD231PS  Y11, Y9, Y0
	VFMADD231PS  Y12, Y10, Y1
	ADDQ         $32, R8
	ADDQ         $64, DI
	ADDQ         $8, R10
	ADDQ         $8, R13
	DECQ         BX
	JNZ          q4q8chunk

	VADDPS      Y1, Y0, Y0
	VFMADD231PS Y13, Y0, Y2
	ADDQ        $144, SI
	ADDQ        $32, R14
	DECQ        CX
	JNZ         q4q8block

	VSUBPS Y3, Y2, Y0
	HSUM1
	MOVSS  X0, ret+40(FP)
	MOVB   $1, ret1+44(FP)
	RET

q4q8bad:
	VZEROUPPER
	MOVB $0, ret1+44(FP)
	RET

// Q4Q8ROW multiplies the two sub-blocks of one chunk, held as unsigned
// nibbles in Y9 and Y10 with their scale pairs in Y11 and Y12, by one row
// of activations: quants at x0 and x1, block scales at d0 and d1. It is
// the inner step of dotQ4KQ8AVX2, instruction for instruction.
#define Q4Q8ROW(x0, x1, d0, d1, accA, accB) \
	VPMADDUBSW   x0, Y9, Y8; \
	VPMADDUBSW   x1, Y10, Y13; \
	VPMADDWD     Y11, Y8, Y8; \
	VPMADDWD     Y12, Y13, Y13; \
	VCVTDQ2PS    Y8, Y8; \
	VCVTDQ2PS    Y13, Y13; \
	VBROADCASTSS d0, Y14; \
	VFMADD231PS  Y14, Y8, accA; \
	VBROADCASTSS d1, Y14; \
	VFMADD231PS  Y14, Y13, accB

// Q4Q8OFF adds dmin*m (Y8) times the row's xdbs at mem to the row's
// offsets, kept on the stack at off.
#define Q4Q8OFF(mem, off) \
	VMOVUPS     off(SP), Y13; \
	VFMADD231PS mem, Y8, Y13; \
	VMOVUPS     Y13, off(SP)

// Q4Q8END adds d (Y8) times the row's block sum accA+accB to its total,
// kept on the stack at tot.
#define Q4Q8END(accA, accB, tot) \
	VADDPS      accB, accA, accA; \
	VMOVUPS     tot(SP), Y13; \
	VFMADD231PS Y8, accA, Y13; \
	VMOVUPS     Y13, tot(SP)

// Q4Q8OUT stores the dot product of one row, total minus offsets, at
// mem.
#define Q4Q8OUT(tot, off, mem) \
	VMOVUPS tot(SP), Y0; \
	VSUBPS  off(SP), Y0, Y0; \
	HSUM1; \
	MOVSS   X0, mem

// func dotQ4KQ8x4AVX2(q *byte, xq *int8, xd, xdbs *float32, xstride, dstride, nb int, out *[4]float32) bool
//
// dotQ4KQ8AVX2 for one row of weights and four rows of activations, the
// quants xstride bytes apart and the scales dstride bytes apart. Each
// weight chunk is decoded once for the four rows. The eight block
// accumulators fill the registers, so the totals and offsets of the rows
// live on the stack: totals at 64(SP), offsets at 192(SP), 32 bytes per
// row; d at 32(SP) and the scale pairs at 0(SP).
TEXT ·dotQ4KQ8x4AVX2(SB), NOSPLIT, $320-65
	MOVQ    q+0(FP), SI
	MOVQ    xq+8(FP), DI
	MOVQ    xd+16(FP), R13
	MOVQ    xdbs+24(FP), R14
	MOVQ    xstride+32(FP), R11
	MOVQ    dstride+40(FP), DX
	MOVQ    nb+48(FP), CX
	LEAQ    (R11)(R11*2), R12
	LEAQ    (DX)(DX*2), R9
	VXORPS  Y0, Y0, Y0
	VMOVUPS Y0, 64(SP)
	VMOVUPS Y0, 96(SP)
	VMOVUPS Y0, 128(SP)
	VMOVUPS Y0, 160(SP)
	VMOVUPS Y0, 192(SP)
	VMOVUPS Y0, 224(SP)
	VMOVUPS Y0, 256(SP)
	VMOVUPS Y0, 288(SP)
	BYTES0F(Y15)

x4block:
	PREFETCHT0 2304(SI)
	PREFETCHT0 2368(SI)
	PREFETCHT0 2432(SI)
	F16FINITE(0(SI), x4bad)
	F16FINITE(2(SI), x4bad)
	F16SCALAR(0(SI), X13)
	VMOVSS X13, 32(SP) // d
	F16SCALAR(2(SI), X12)
	VBROADCASTSS X12, Y12 // dmin

	// The scales and offsets, unpacked as in dotQ4KQ8AVX2 with fewer
	// registers.
	MOVL 12(SI), AX
	SHRL $4, AX
	ANDL $0x0f0f0f0f, AX
	MOVL 8(SI), BX
	SHRL $6, BX
	ANDL $0x03030303, BX
	SHLL $4, BX
	ORL  BX, AX // m 4..7
	MOVL 8(SI), BX
	ANDL $0x3f3f3f3f, BX // m 0..3
	SHLQ $32, AX
	ORQ  BX, AX

	MOVL 12(SI), R8
	ANDL $0x0f0f0f0f, R8
	MOVL 4(SI), BX
	SHRL $6, BX
	ANDL $0x03030303, BX
	SHLL $4, BX
	ORL  BX, R8 // sc 4..7
	MOVL 4(SI), BX
	ANDL $0x3f3f3f3f, BX // sc 0..3
	SHLQ $32, R8
	ORQ  BX, R8

	VMOVQ     AX, X8
	VPMOVZXBD X8, Y8
	VCVTDQ2PS Y8, Y8
	VMULPS    Y12, Y8, Y8 // dmin*m
	Q4Q8OFF((R14), 192)
	Q4Q8OFF((R14)(DX*1), 224)
	Q4Q8OFF((R14)(DX*2), 256)
	Q4Q8OFF((R14)(R9*1), 288)

	VMOVQ      R8, X10
	VPMOVZXBW  X10, X10
	VPUNPCKLWD X10, X10, X11
	VPUNPCKHWD X10, X10, X10
	VMOVDQU    X11, 0(SP)
	VMOVDQU    X10, 16(SP)

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7
	LEAQ   16(SI), R8 // the nibbles of chunk 0
	LEAQ   0(SP), R10 // the scale pairs of chunk 0
	MOVQ   $4, BX

x4chunk:
	VMOVDQU      (R8), Y8
	VPAND        Y15, Y8, Y9   // sub-block 2c
	VPSRLW       $4, Y8, Y10
	VPAND        Y15, Y10, Y10 // sub-block 2c+1
	VPBROADCASTD (R10), Y11
	VPBROADCASTD 4(R10), Y12
	Q4Q8ROW((DI), 32(DI), (R13), 4(R13), Y0, Y1)
	Q4Q8ROW((DI)(R11*1), 32(DI)(R11*1), (R13)(DX*1), 4(R13)(DX*1), Y2, Y3)
	Q4Q8ROW((DI)(R11*2), 32(DI)(R11*2), (R13)(DX*2), 4(R13)(DX*2), Y4, Y5)
	Q4Q8ROW((DI)(R12*1), 32(DI)(R12*1), (R13)(R9*1), 4(R13)(R9*1), Y6, Y7)
	ADDQ         $32, R8
	ADDQ         $64, DI
	ADDQ         $8, R10
	ADDQ         $8, R13
	DECQ         BX
	JNZ          x4chunk

	VBROADCASTSS 32(SP), Y8 // d
	Q4Q8END(Y0, Y1, 64)
	Q4Q8END(Y2, Y3, 96)
	Q4Q8END(Y4, Y5, 128)
	Q4Q8END(Y6, Y7, 160)
	ADDQ $144, SI
	ADDQ $32, R14
	DECQ CX
	JNZ  x4block

	MOVQ out+56(FP), AX
	Q4Q8OUT(64, 192, 0(AX))
	Q4Q8OUT(96, 224, 4(AX))
	Q4Q8OUT(128, 256, 8(AX))
	Q4Q8OUT(160, 288, 12(AX))
	MOVB $1, ret+64(FP)
	RET

x4bad:
	VZEROUPPER
	MOVB $0, ret+64(FP)
	RET

// Q6Q8DOT finishes one quarter of a Q6_K half block: qv holds its 32
// unsigned 6-bit quants. It subtracts the bias of 32 using m32, applies the
// two scales (one per 16 weights) picked from Y7 by shuffle mask sm,
// converts, and accumulates with the activation scale at do.
#define Q6Q8DOT(qv, xo, sm, do, acc) \
	VPMADDUBSW   xo(DI), qv, qv; \
	VPSUBW       xo(R14), qv, qv; \
	VPSHUFB      sm, Y7, Y10; \
	VPMADDWD     Y10, qv, qv; \
	VCVTDQ2PS    qv, qv; \
	VBROADCASTSS do(R13), Y11; \
	VFMADD231PS  Y11, qv, acc

// q6scales holds, for each quarter j of a half block, the VPSHUFB mask
// that fills the low 128 bits with int16 scale 2j and the high 128 bits
// with scale 2j+1.
DATA q6scales<>+0(SB)/8, $0x0100010001000100
DATA q6scales<>+8(SB)/8, $0x0100010001000100
DATA q6scales<>+16(SB)/8, $0x0302030203020302
DATA q6scales<>+24(SB)/8, $0x0302030203020302
DATA q6scales<>+32(SB)/8, $0x0504050405040504
DATA q6scales<>+40(SB)/8, $0x0504050405040504
DATA q6scales<>+48(SB)/8, $0x0706070607060706
DATA q6scales<>+56(SB)/8, $0x0706070607060706
DATA q6scales<>+64(SB)/8, $0x0908090809080908
DATA q6scales<>+72(SB)/8, $0x0908090809080908
DATA q6scales<>+80(SB)/8, $0x0b0a0b0a0b0a0b0a
DATA q6scales<>+88(SB)/8, $0x0b0a0b0a0b0a0b0a
DATA q6scales<>+96(SB)/8, $0x0d0c0d0c0d0c0d0c
DATA q6scales<>+104(SB)/8, $0x0d0c0d0c0d0c0d0c
DATA q6scales<>+112(SB)/8, $0x0f0e0f0e0f0e0f0e
DATA q6scales<>+120(SB)/8, $0x0f0e0f0e0f0e0f0e
GLOBL q6scales<>(SB), RODATA|NOPTR, $128

// func dotQ6KQ8AVX2(q *byte, xq *int8, xd *float32, m32 *int16, nb int) (float32, bool)
//
// m32 holds 32 times the sum of each pair of activation quants.
TEXT ·dotQ6KQ8AVX2(SB), NOSPLIT, $0-45
	MOVQ   q+0(FP), SI
	MOVQ   xq+8(FP), DI
	MOVQ   xd+16(FP), R13
	MOVQ   m32+24(FP), R14
	MOVQ   nb+32(FP), CX
	VXORPS Y2, Y2, Y2
	BYTES0F(Y15)
	BYTES03(Y14)
	VPSLLW $4, Y14, Y14 // 0x30

q6q8block:
	// Prefetch two rows of K = 2048 ahead (16 blocks of 210 bytes).
	PREFETCHT0 3360(SI)
	PREFETCHT0 3424(SI)
	PREFETCHT0 3488(SI)
	PREFETCHT0 3552(SI)
	F16FINITE(208(SI), q6q8bad)
	F16SCALAR(208(SI), X13)
	VBROADCASTSS X13, Y13 // d
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	MOVQ   SI, R8       // ql of half 0
	LEAQ   128(SI), R9  // qh of half 0
	LEAQ   192(SI), R10 // scales of half 0
	MOVQ   $2, BX

q6q8half:
	VPMOVSXBW   (R10), X7
	VINSERTI128 $1, X7, Y7, Y7 // the half's eight scales in both lanes
	VMOVDQU     (R8), Y4      // ql[0:32]
	VMOVDQU     32(R8), Y5    // ql[32:64]
	VMOVDQU     (R9), Y6      // qh[0:32]

	// The 16-bit shifts move bits across bytes, but only into bits the
	// masks clear.
	VPAND  Y15, Y4, Y8
	VPSLLW $4, Y6, Y9
	VPAND  Y14, Y9, Y9
	VPOR   Y9, Y8, Y8
	Q6Q8DOT(Y8, 0, q6scales<>+0(SB), 0, Y0)

	VPAND  Y15, Y5, Y8
	VPSLLW $2, Y6, Y9
	VPAND  Y14, Y9, Y9
	VPOR   Y9, Y8, Y8
	Q6Q8DOT(Y8, 32, q6scales<>+32(SB), 4, Y1)

	VPSRLW $4, Y4, Y8
	VPAND  Y15, Y8, Y8
	VPAND  Y14, Y6, Y9
	VPOR   Y9, Y8, Y8
	Q6Q8DOT(Y8, 64, q6scales<>+64(SB), 8, Y0)

	VPSRLW $4, Y5, Y8
	VPAND  Y15, Y8, Y8
	VPSRLW $2, Y6, Y9
	VPAND  Y14, Y9, Y9
	VPOR   Y9, Y8, Y8
	Q6Q8DOT(Y8, 96, q6scales<>+96(SB), 12, Y1)

	ADDQ $64, R8
	ADDQ $32, R9
	ADDQ $8, R10
	ADDQ $128, DI
	ADDQ $128, R14
	ADDQ $16, R13
	DECQ BX
	JNZ  q6q8half

	VADDPS      Y1, Y0, Y0
	VFMADD231PS Y13, Y0, Y2
	ADDQ        $210, SI
	DECQ        CX
	JNZ         q6q8block

	VMOVAPS Y2, Y0
	HSUM1
	MOVSS   X0, ret+40(FP)
	MOVB    $1, ret1+44(FP)
	RET

q6q8bad:
	VZEROUPPER
	MOVB $0, ret1+44(FP)
	RET

// Q6Q8ROW is the per-row part of Q6Q8DOT for one quarter, with the quants
// in Y8 kept for the other rows and the shuffled scales in Y10: quants at
// xq, pair sums at m, block scale at d.
#define Q6Q8ROW(xq, m, d, acc) \
	VPMADDUBSW   xq, Y8, Y9; \
	VPSUBW       m, Y9, Y9; \
	VPMADDWD     Y10, Y9, Y9; \
	VCVTDQ2PS    Y9, Y9; \
	VBROADCASTSS d, Y11; \
	VFMADD231PS  Y11, Y9, acc

// Q6Q8ROWS runs Q6Q8ROW for the four rows of one quarter, at byte offset
// xo into the quants and pair sums and do into the scales.
#define Q6Q8ROWS(xo, do, acc0, acc1, acc2, acc3) \
	Q6Q8ROW(xo(DI), xo(R14), do(R13), acc0); \
	Q6Q8ROW(xo(DI)(R11*1), xo(R14)(R11*1), do(R13)(DX*1), acc1); \
	Q6Q8ROW(xo(DI)(R11*2), xo(R14)(R11*2), do(R13)(DX*2), acc2); \
	Q6Q8ROW(xo(DI)(R12*1), xo(R14)(R12*1), do(R13)(R9*1), acc3)

// Q6Q8END adds d (Y9) times the row's block sum accA+accB to its total,
// kept on the stack at tot.
#define Q6Q8END(accA, accB, tot) \
	VADDPS      accB, accA, accA; \
	VMOVUPS     tot(SP), Y13; \
	VFMADD231PS Y9, accA, Y13; \
	VMOVUPS     Y13, tot(SP)

// func dotQ6KQ8x4AVX2(q *byte, xq *int8, xd *float32, m32 *int16, xstride, dstride, nb int, out *[4]float32) bool
//
// dotQ6KQ8AVX2 for one row of weights and four rows of activations, laid
// out as for dotQ4KQ8x4AVX2; the pair sums m32 have the stride of the
// quants. Each quarter is decoded once for the four rows. The row totals
// live on the stack at 32(SP), 32 bytes per row, and d at 0(SP).
TEXT ·dotQ6KQ8x4AVX2(SB), NOSPLIT, $160-65
	MOVQ    q+0(FP), SI
	MOVQ    xq+8(FP), DI
	MOVQ    xd+16(FP), R13
	MOVQ    m32+24(FP), R14
	MOVQ    xstride+32(FP), R11
	MOVQ    dstride+40(FP), DX
	MOVQ    nb+48(FP), CX
	LEAQ    (R11)(R11*2), R12
	LEAQ    (DX)(DX*2), R9
	VXORPS  Y0, Y0, Y0
	VMOVUPS Y0, 32(SP)
	VMOVUPS Y0, 64(SP)
	VMOVUPS Y0, 96(SP)
	VMOVUPS Y0, 128(SP)
	BYTES0F(Y15)
	BYTES03(Y14)
	VPSLLW $4, Y14, Y14 // 0x30

x46block:
	PREFETCHT0 3360(SI)
	PREFETCHT0 3424(SI)
	PREFETCHT0 3488(SI)
	PREFETCHT0 3552(SI)
	F16FINITE(208(SI), x46bad)
	F16SCALAR(208(SI), X13)
	VMOVSS X13, 0(SP) // d
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y12, Y12, Y12
	MOVQ   SI, R8       // ql of half 0
	LEAQ   128(SI), AX  // qh of half 0
	LEAQ   192(SI), R10 // scales of half 0
	MOVQ   $2, BX

x46half:
	VPMOVSXBW   (R10), X7
	VINSERTI128 $1, X7, Y7, Y7 // the half's eight scales in both lanes

	VMOVDQU (R8), Y8
	VPAND   Y15, Y8, Y8
	VMOVDQU (AX), Y9
	VPSLLW  $4, Y9, Y9
	VPAND   Y14, Y9, Y9
	VPOR    Y9, Y8, Y8
	VPSHUFB q6scales<>+0(SB), Y7, Y10
	Q6Q8ROWS(0, 0, Y0, Y2, Y4, Y6)

	VMOVDQU 32(R8), Y8
	VPAND   Y15, Y8, Y8
	VMOVDQU (AX), Y9
	VPSLLW  $2, Y9, Y9
	VPAND   Y14, Y9, Y9
	VPOR    Y9, Y8, Y8
	VPSHUFB q6scales<>+32(SB), Y7, Y10
	Q6Q8ROWS(32, 4, Y1, Y3, Y5, Y12)

	VMOVDQU (R8), Y8
	VPSRLW  $4, Y8, Y8
	VPAND   Y15, Y8, Y8
	VMOVDQU (AX), Y9
	VPAND   Y14, Y9, Y9
	VPOR    Y9, Y8, Y8
	VPSHUFB q6scales<>+64(SB), Y7, Y10
	Q6Q8ROWS(64, 8, Y0, Y2, Y4, Y6)

	VMOVDQU 32(R8), Y8
	VPSRLW  $4, Y8, Y8
	VPAND   Y15, Y8, Y8
	VMOVDQU (AX), Y9
	VPSRLW  $2, Y9, Y9
	VPAND   Y14, Y9, Y9
	VPOR    Y9, Y8, Y8
	VPSHUFB q6scales<>+96(SB), Y7, Y10
	Q6Q8ROWS(96, 12, Y1, Y3, Y5, Y12)

	ADDQ $64, R8
	ADDQ $32, AX
	ADDQ $8, R10
	ADDQ $128, DI
	ADDQ $128, R14
	ADDQ $16, R13
	DECQ BX
	JNZ  x46half

	VBROADCASTSS 0(SP), Y9 // d
	Q6Q8END(Y0, Y1, 32)
	Q6Q8END(Y2, Y3, 64)
	Q6Q8END(Y4, Y5, 96)
	Q6Q8END(Y6, Y12, 128)
	ADDQ $210, SI
	DECQ CX
	JNZ  x46block

	MOVQ    out+56(FP), AX
	VMOVUPS 32(SP), Y0
	HSUM1
	MOVSS   X0, 0(AX)
	VMOVUPS 64(SP), Y0
	HSUM1
	MOVSS   X0, 4(AX)
	VMOVUPS 96(SP), Y0
	HSUM1
	MOVSS   X0, 8(AX)
	VMOVUPS 128(SP), Y0
	HSUM1
	MOVSS   X0, 12(AX)
	MOVB    $1, ret+64(FP)
	RET

x46bad:
	VZEROUPPER
	MOVB $0, ret+64(FP)
	RET

// func dotQ8_0Q8AVX2(q *byte, xq *int8, xd *float32, nb int) (float32, bool)
//
// The signed weights are split into magnitude and sign so that
// VPMADDUBSW, which takes one unsigned operand, applies.
TEXT ·dotQ8_0Q8AVX2(SB), NOSPLIT, $0-37
	MOVQ     q+0(FP), SI
	MOVQ     xq+8(FP), DI
	MOVQ     xd+16(FP), R13
	MOVQ     nb+24(FP), CX
	VXORPS   Y0, Y0, Y0
	VPCMPEQW Y14, Y14, Y14
	VPSRLW   $15, Y14, Y14 // int16 ones

q80q8loop:
	F16FINITE(0(SI), q80q8bad)
	VMOVDQU      2(SI), Y8
	VMOVDQU      (DI), Y9
	VPSIGNB      Y8, Y9, Y9 // x with the sign of w
	VPABSB       Y8, Y8     // |w|, at most 128 as unsigned
	VPMADDUBSW   Y9, Y8, Y8
	VPMADDWD     Y14, Y8, Y8
	VCVTDQ2PS    Y8, Y8
	F16SCALAR(0(SI), X10)
	VMULSS       (R13), X10, X10
	VBROADCASTSS X10, Y10
	VFMADD231PS  Y10, Y8, Y0
	ADDQ         $34, SI
	ADDQ         $32, DI
	ADDQ         $4, R13
	DECQ         CX
	JNZ          q80q8loop

	HSUM1
	MOVSS X0, ret+32(FP)
	MOVB  $1, ret1+36(FP)
	RET

q80q8bad:
	VZEROUPPER
	MOVB $0, ret1+36(FP)
	RET

// Kernels for activations in blocks of 256 (Q8Batch rows quantized with
// QuantizeBlocks(x, 256)). A weight block and an activation block then
// cover the same 256 elements, so the integer dot products of the whole
// block, already weighted by the integer sub-block scales, add up exactly
// in int32 and are converted and scaled once per block, by d*dx. The
// activation scale dx is repeated for each run of 32, as for blocks of 32,
// and read at the first. The one-row and four-row kernels do the same
// float operations in the same order, so their results are equal bit for
// bit.

// Q4KSCALES unpacks the 6-bit scales and offsets of the Q4_K block at SI
// as in dotQ4KQ8x4AVX2: the offsets m 0..7 as bytes in AX, the scales in
// R8. It uses BX.
#define Q4KSCALES \
	MOVL 12(SI), AX; \
	SHRL $4, AX; \
	ANDL $0x0f0f0f0f, AX; \
	MOVL 8(SI), BX; \
	SHRL $6, BX; \
	ANDL $0x03030303, BX; \
	SHLL $4, BX; \
	ORL  BX, AX; \
	MOVL 8(SI), BX; \
	ANDL $0x3f3f3f3f, BX; \
	SHLQ $32, AX; \
	ORQ  BX, AX; \
	MOVL 12(SI), R8; \
	ANDL $0x0f0f0f0f, R8; \
	MOVL 4(SI), BX; \
	SHRL $6, BX; \
	ANDL $0x03030303, BX; \
	SHLL $4, BX; \
	ORL  BX, R8; \
	MOVL 4(SI), BX; \
	ANDL $0x3f3f3f3f, BX; \
	SHLQ $32, R8; \
	ORQ  BX, R8

// Q4KPREP leaves dmin*m in Y8 and the scales as int16 pairs in X11 (0..3)
// and X10 (4..7), from AX and R8 as Q4KSCALES sets them and dmin in Y12.
// The callers store the pairs at 0(SP).
#define Q4KPREP \
	VMOVQ      AX, X8; \
	VPMOVZXBD  X8, Y8; \
	VCVTDQ2PS  Y8, Y8; \
	VMULPS     Y12, Y8, Y8; \
	VMOVQ      R8, X10; \
	VPMOVZXBW  X10, X10; \
	VPUNPCKLWD X10, X10, X11; \
	VPUNPCKHWD X10, X10, X10

// Q4KCHUNK splits the nibbles of chunk R8 into Y9 and Y10 and loads their
// scale pairs from R10 into Y11 and Y12.
#define Q4KCHUNK \
	VMOVDQU      (R8), Y8; \
	VPAND        Y15, Y8, Y9; \
	VPSRLW       $4, Y8, Y10; \
	VPAND        Y15, Y10, Y10; \
	VPBROADCASTD (R10), Y11; \
	VPBROADCASTD 4(R10), Y12

// Q4KROW adds the scaled integer dot products of one chunk with one row of
// activations, quants at x0 and x1, to the int32 lanes of acc.
#define Q4KROW(x0, x1, acc) \
	VPMADDUBSW x0, Y9, Y13; \
	VPMADDUBSW x1, Y10, Y14; \
	VPMADDWD   Y11, Y13, Y13; \
	VPMADDWD   Y12, Y14, Y14; \
	VPADDD     Y14, Y13, Y13; \
	VPADDD     Y13, acc, acc

// Q8KEND adds the block's integer sums acc, converted, times d*dx to tot,
// with the weight scale d in the low lane of the X register d and the
// activation scale at dx. It uses Y13.
#define Q8KEND(d, dx, acc, tot) \
	VMULSS       dx, d, X13; \
	VBROADCASTSS X13, Y13; \
	VCVTDQ2PS    acc, acc; \
	VFMADD231PS  Y13, acc, tot

// func dotQ4KQ8KAVX2(q *byte, xq *int8, xd, xdbs *float32, nb int) (float32, bool)
TEXT ·dotQ4KQ8KAVX2(SB), NOSPLIT, $32-45
	MOVQ   q+0(FP), SI
	MOVQ   xq+8(FP), DI
	MOVQ   xd+16(FP), R13
	MOVQ   xdbs+24(FP), R14
	MOVQ   nb+32(FP), CX
	VXORPS Y4, Y4, Y4 // totals
	VXORPS Y5, Y5, Y5 // offsets
	BYTES0F(Y15)

k1block:
	PREFETCHT0 2304(SI)
	PREFETCHT0 2368(SI)
	PREFETCHT0 2432(SI)
	F16FINITE(0(SI), k1bad)
	F16FINITE(2(SI), k1bad)
	F16SCALAR(0(SI), X6) // d
	F16SCALAR(2(SI), X12)
	VBROADCASTSS X12, Y12 // dmin
	Q4KSCALES
	Q4KPREP
	VMOVDQU     X11, 0(SP)
	VMOVDQU     X10, 16(SP)
	VFMADD231PS (R14), Y8, Y5

	VPXOR Y0, Y0, Y0
	LEAQ  16(SI), R8
	LEAQ  0(SP), R10
	MOVQ  $4, BX

k1chunk:
	Q4KCHUNK
	Q4KROW((DI), 32(DI), Y0)
	ADDQ $32, R8
	ADDQ $64, DI
	ADDQ $8, R10
	DECQ BX
	JNZ  k1chunk

	Q8KEND(X6, (R13), Y0, Y4)
	ADDQ $144, SI
	ADDQ $32, R13
	ADDQ $32, R14
	DECQ CX
	JNZ  k1block

	VSUBPS Y5, Y4, Y0
	HSUM1
	MOVSS  X0, ret+40(FP)
	MOVB   $1, ret1+44(FP)
	RET

k1bad:
	VZEROUPPER
	MOVB $0, ret1+44(FP)
	RET

// func dotQ4KQ8Kx4AVX2(q *byte, xq *int8, xd, xdbs *float32, xstride, dstride, nb int, out *[4]float32) bool
//
// dotQ4KQ8KAVX2 for four rows of activations, laid out as for
// dotQ4KQ8x4AVX2. The integer sums are in Y0..Y3 and the totals in
// Y4..Y7; the offsets live on the stack at 64(SP), 32 bytes per row, d at
// 32(SP) and the scale pairs at 0(SP).
TEXT ·dotQ4KQ8Kx4AVX2(SB), NOSPLIT, $192-65
	MOVQ    q+0(FP), SI
	MOVQ    xq+8(FP), DI
	MOVQ    xd+16(FP), R13
	MOVQ    xdbs+24(FP), R14
	MOVQ    xstride+32(FP), R11
	MOVQ    dstride+40(FP), DX
	MOVQ    nb+48(FP), CX
	LEAQ    (R11)(R11*2), R12
	LEAQ    (DX)(DX*2), R9
	VXORPS  Y4, Y4, Y4
	VXORPS  Y5, Y5, Y5
	VXORPS  Y6, Y6, Y6
	VXORPS  Y7, Y7, Y7
	VMOVUPS Y4, 64(SP)
	VMOVUPS Y4, 96(SP)
	VMOVUPS Y4, 128(SP)
	VMOVUPS Y4, 160(SP)
	BYTES0F(Y15)

k4block:
	PREFETCHT0 2304(SI)
	PREFETCHT0 2368(SI)
	PREFETCHT0 2432(SI)
	F16FINITE(0(SI), k4bad)
	F16FINITE(2(SI), k4bad)
	F16SCALAR(0(SI), X13)
	VMOVSS X13, 32(SP) // d
	F16SCALAR(2(SI), X12)
	VBROADCASTSS X12, Y12 // dmin
	Q4KSCALES
	Q4KPREP
	VMOVDQU X11, 0(SP)
	VMOVDQU X10, 16(SP)
	Q4Q8OFF((R14), 64)
	Q4Q8OFF((R14)(DX*1), 96)
	Q4Q8OFF((R14)(DX*2), 128)
	Q4Q8OFF((R14)(R9*1), 160)

	VPXOR Y0, Y0, Y0
	VPXOR Y1, Y1, Y1
	VPXOR Y2, Y2, Y2
	VPXOR Y3, Y3, Y3
	LEAQ  16(SI), R8
	LEAQ  0(SP), R10
	MOVQ  $4, BX

k4chunk:
	Q4KCHUNK
	Q4KROW((DI), 32(DI), Y0)
	Q4KROW((DI)(R11*1), 32(DI)(R11*1), Y1)
	Q4KROW((DI)(R11*2), 32(DI)(R11*2), Y2)
	Q4KROW((DI)(R12*1), 32(DI)(R12*1), Y3)
	ADDQ $32, R8
	ADDQ $64, DI
	ADDQ $8, R10
	DECQ BX
	JNZ  k4chunk

	VMOVSS 32(SP), X8
	Q8KEND(X8, (R13), Y0, Y4)
	Q8KEND(X8, (R13)(DX*1), Y1, Y5)
	Q8KEND(X8, (R13)(DX*2), Y2, Y6)
	Q8KEND(X8, (R13)(R9*1), Y3, Y7)
	ADDQ $144, SI
	ADDQ $32, R13
	ADDQ $32, R14
	DECQ CX
	JNZ  k4block

	// HSUM1 clears the upper lanes, so the rows go to the stack first.
	VSUBPS  64(SP), Y4, Y4
	VSUBPS  96(SP), Y5, Y5
	VSUBPS  128(SP), Y6, Y6
	VSUBPS  160(SP), Y7, Y7
	VMOVUPS Y5, 96(SP)
	VMOVUPS Y6, 128(SP)
	VMOVUPS Y7, 160(SP)
	MOVQ    out+56(FP), AX
	VMOVAPS Y4, Y0
	HSUM1
	MOVSS   X0, 0(AX)
	VMOVUPS 96(SP), Y0
	HSUM1
	MOVSS   X0, 4(AX)
	VMOVUPS 128(SP), Y0
	HSUM1
	MOVSS   X0, 8(AX)
	VMOVUPS 160(SP), Y0
	HSUM1
	MOVSS   X0, 12(AX)
	MOVB    $1, ret+64(FP)
	RET

k4bad:
	VZEROUPPER
	MOVB $0, ret+64(FP)
	RET

// Q6KQ0 to Q6KQ3 decode quarter j of the Q6_K half block at R8 (ql) and
// AX (qh) into 32 unsigned 6-bit quants in Y8, and pick its two scales
// from Y7 into Y10, as dotQ6KQ8x4AVX2 does. They use Y9.
#define Q6KQ0 \
	VMOVDQU (R8), Y8; \
	VPAND   Y15, Y8, Y8; \
	VMOVDQU (AX), Y9; \
	VPSLLW  $4, Y9, Y9; \
	VPAND   Y14, Y9, Y9; \
	VPOR    Y9, Y8, Y8; \
	VPSHUFB q6scales<>+0(SB), Y7, Y10

#define Q6KQ1 \
	VMOVDQU 32(R8), Y8; \
	VPAND   Y15, Y8, Y8; \
	VMOVDQU (AX), Y9; \
	VPSLLW  $2, Y9, Y9; \
	VPAND   Y14, Y9, Y9; \
	VPOR    Y9, Y8, Y8; \
	VPSHUFB q6scales<>+32(SB), Y7, Y10

#define Q6KQ2 \
	VMOVDQU (R8), Y8; \
	VPSRLW  $4, Y8, Y8; \
	VPAND   Y15, Y8, Y8; \
	VMOVDQU (AX), Y9; \
	VPAND   Y14, Y9, Y9; \
	VPOR    Y9, Y8, Y8; \
	VPSHUFB q6scales<>+64(SB), Y7, Y10

#define Q6KQ3 \
	VMOVDQU 32(R8), Y8; \
	VPSRLW  $4, Y8, Y8; \
	VPAND   Y15, Y8, Y8; \
	VMOVDQU (AX), Y9; \
	VPSRLW  $2, Y9, Y9; \
	VPAND   Y14, Y9, Y9; \
	VPOR    Y9, Y8, Y8; \
	VPSHUFB q6scales<>+96(SB), Y7, Y10

// Q6KROW adds the scaled integer dot products of the quarter in Y8, scales
// in Y10, with one row of activations, quants at xq and pair sums at m, to
// the int32 lanes of acc.
#define Q6KROW(xq, m, acc) \
	VPMADDUBSW xq, Y8, Y9; \
	VPSUBW     m, Y9, Y9; \
	VPMADDWD   Y10, Y9, Y9; \
	VPADDD     Y9, acc, acc

// Q6KROWS runs Q6KROW for four rows at byte offset xo.
#define Q6KROWS(xo) \
	Q6KROW(xo(DI), xo(R14), Y0); \
	Q6KROW(xo(DI)(R11*1), xo(R14)(R11*1), Y1); \
	Q6KROW(xo(DI)(R11*2), xo(R14)(R11*2), Y2); \
	Q6KROW(xo(DI)(R12*1), xo(R14)(R12*1), Y3)

// func dotQ6KQ8KAVX2(q *byte, xq *int8, xd *float32, m32 *int16, nb int) (float32, bool)
TEXT ·dotQ6KQ8KAVX2(SB), NOSPLIT, $0-45
	MOVQ   q+0(FP), SI
	MOVQ   xq+8(FP), DI
	MOVQ   xd+16(FP), R13
	MOVQ   m32+24(FP), R14
	MOVQ   nb+32(FP), CX
	VXORPS Y4, Y4, Y4
	BYTES0F(Y15)
	BYTES03(Y14)
	VPSLLW $4, Y14, Y14 // 0x30

k61block:
	PREFETCHT0 3360(SI)
	PREFETCHT0 3424(SI)
	PREFETCHT0 3488(SI)
	PREFETCHT0 3552(SI)
	F16FINITE(208(SI), k61bad)
	F16SCALAR(208(SI), X12) // d
	VPXOR Y0, Y0, Y0
	MOVQ  SI, R8
	LEAQ  128(SI), AX
	LEAQ  192(SI), R10
	MOVQ  $2, BX

k61half:
	VPMOVSXBW   (R10), X7
	VINSERTI128 $1, X7, Y7, Y7
	Q6KQ0
	Q6KROW(0(DI), 0(R14), Y0)
	Q6KQ1
	Q6KROW(32(DI), 32(R14), Y0)
	Q6KQ2
	Q6KROW(64(DI), 64(R14), Y0)
	Q6KQ3
	Q6KROW(96(DI), 96(R14), Y0)
	ADDQ $64, R8
	ADDQ $32, AX
	ADDQ $8, R10
	ADDQ $128, DI
	ADDQ $128, R14
	DECQ BX
	JNZ  k61half

	Q8KEND(X12, (R13), Y0, Y4)
	ADDQ $210, SI
	ADDQ $32, R13
	DECQ CX
	JNZ  k61block

	VMOVAPS Y4, Y0
	HSUM1
	MOVSS   X0, ret+40(FP)
	MOVB    $1, ret1+44(FP)
	RET

k61bad:
	VZEROUPPER
	MOVB $0, ret1+44(FP)
	RET

// func dotQ6KQ8Kx4AVX2(q *byte, xq *int8, xd *float32, m32 *int16, xstride, dstride, nb int, out *[4]float32) bool
//
// dotQ6KQ8KAVX2 for four rows of activations, laid out as for
// dotQ6KQ8x4AVX2. The integer sums are in Y0..Y3 and the totals in Y4,
// Y5, Y6 and Y11.
TEXT ·dotQ6KQ8Kx4AVX2(SB), NOSPLIT, $128-65
	MOVQ   q+0(FP), SI
	MOVQ   xq+8(FP), DI
	MOVQ   xd+16(FP), R13
	MOVQ   m32+24(FP), R14
	MOVQ   xstride+32(FP), R11
	MOVQ   dstride+40(FP), DX
	MOVQ   nb+48(FP), CX
	LEAQ   (R11)(R11*2), R12
	LEAQ   (DX)(DX*2), R9
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y11, Y11, Y11
	BYTES0F(Y15)
	BYTES03(Y14)
	VPSLLW $4, Y14, Y14 // 0x30

k64block:
	PREFETCHT0 3360(SI)
	PREFETCHT0 3424(SI)
	PREFETCHT0 3488(SI)
	PREFETCHT0 3552(SI)
	F16FINITE(208(SI), k64bad)
	F16SCALAR(208(SI), X12) // d
	VPXOR Y0, Y0, Y0
	VPXOR Y1, Y1, Y1
	VPXOR Y2, Y2, Y2
	VPXOR Y3, Y3, Y3
	MOVQ  SI, R8
	LEAQ  128(SI), AX
	LEAQ  192(SI), R10
	MOVQ  $2, BX

k64half:
	VPMOVSXBW   (R10), X7
	VINSERTI128 $1, X7, Y7, Y7
	Q6KQ0
	Q6KROWS(0)
	Q6KQ1
	Q6KROWS(32)
	Q6KQ2
	Q6KROWS(64)
	Q6KQ3
	Q6KROWS(96)
	ADDQ $64, R8
	ADDQ $32, AX
	ADDQ $8, R10
	ADDQ $128, DI
	ADDQ $128, R14
	DECQ BX
	JNZ  k64half

	Q8KEND(X12, (R13), Y0, Y4)
	Q8KEND(X12, (R13)(DX*1), Y1, Y5)
	Q8KEND(X12, (R13)(DX*2), Y2, Y6)
	Q8KEND(X12, (R13)(R9*1), Y3, Y11)
	ADDQ $210, SI
	ADDQ $32, R13
	DECQ CX
	JNZ  k64block

	VMOVUPS Y5, 32(SP)
	VMOVUPS Y6, 64(SP)
	VMOVUPS Y11, 96(SP)
	MOVQ    out+56(FP), AX
	VMOVAPS Y4, Y0
	HSUM1
	MOVSS   X0, 0(AX)
	VMOVUPS 32(SP), Y0
	HSUM1
	MOVSS   X0, 4(AX)
	VMOVUPS 64(SP), Y0
	HSUM1
	MOVSS   X0, 8(AX)
	VMOVUPS 96(SP), Y0
	HSUM1
	MOVSS   X0, 12(AX)
	MOVB    $1, ret+64(FP)
	RET

k64bad:
	VZEROUPPER
	MOVB $0, ret+64(FP)
	RET
