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

//go:build arm64 && !purego

#include "textflag.h"

// NEON kernels for float32 vectors: the dot products and axpy of
// kernels.go, and the attention fast path of attn.go over float32 and f16
// rows. They compute what the AVX2 kernels in kernels_amd64.s and
// attn_amd64.s compute, with the same fused multiply-adds, but on four
// lanes rather than eight, so the sums of products are added in a
// different order; the exponentials and SwiGLU are the same bit for bit.
//
// The Go assembler has no mnemonics for most float vector instructions,
// so those are written as WORDs, each with the instruction it encodes in
// a comment.

// func dotF32NEON(a, b *float32, n int) float32
// n is a positive multiple of 8.
TEXT ·dotF32NEON(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2
	WORD $0x4f00e400 // movi v0.16b, #0
	WORD $0x4f00e401 // movi v1.16b, #0
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0

dot16:
	CMP $16, R2
	BLT dot8
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R1), [V16.S4, V17.S4, V18.S4, V19.S4]
	WORD $0x4e30cc80 // fmla v0.4s, v4.4s, v16.4s
	WORD $0x4e31cca1 // fmla v1.4s, v5.4s, v17.4s
	WORD $0x4e32ccc2 // fmla v2.4s, v6.4s, v18.4s
	WORD $0x4e33cce3 // fmla v3.4s, v7.4s, v19.4s
	SUB $16, R2
	B dot16

dot8:
	CBZ R2, dotdone
	VLD1.P 32(R0), [V4.S4, V5.S4]
	VLD1.P 32(R1), [V16.S4, V17.S4]
	WORD $0x4e30cc80 // fmla v0.4s, v4.4s, v16.4s
	WORD $0x4e31cca1 // fmla v1.4s, v5.4s, v17.4s

dotdone:
	WORD $0x4e22d400 // fadd v0.4s, v0.4s, v2.4s
	WORD $0x4e23d421 // fadd v1.4s, v1.4s, v3.4s
	WORD $0x4e21d400 // fadd v0.4s, v0.4s, v1.4s
	WORD $0x6e20d400 // faddp v0.4s, v0.4s, v0.4s
	WORD $0x7e30d800 // faddp s0, v0.2s
	FMOVS F0, ret+24(FP)
	RET

// func axpyF32NEON(alpha float32, x, y *float32, n int)
// n is a positive multiple of 8.
TEXT ·axpyF32NEON(SB), NOSPLIT, $0-32
	FMOVS alpha+0(FP), F31
	MOVD x+8(FP), R0
	MOVD y+16(FP), R1
	MOVD n+24(FP), R2
	WORD $0x4e0407ff // dup v31.4s, v31.s[0]

axpyloop:
	VLD1.P 32(R0), [V4.S4, V5.S4]
	VLD1 (R1), [V0.S4, V1.S4]
	WORD $0x4e3fcc80 // fmla v0.4s, v4.4s, v31.4s
	WORD $0x4e3fcca1 // fmla v1.4s, v5.4s, v31.4s
	VST1.P [V0.S4, V1.S4], 32(R1)
	SUBS $8, R2
	BNE axpyloop
	RET

// func scoresF32NEON(q, k *float32, d, n int, scale float32, out *float32)
//
// Sets out[j] = scale * (q . k[j]) for the n rows of k, d floats each, d a
// positive multiple of 8. Four rows go together, each with two
// accumulators. Each row's eight partial sums are reduced in the same
// order whether it goes with others or alone: the two accumulators are
// added, then the lanes pairwise, ((s0+s1) + (s2+s3)).
TEXT ·scoresF32NEON(SB), NOSPLIT, $0-48
	MOVD q+0(FP), R0
	MOVD k+8(FP), R1
	MOVD d+16(FP), R2
	MOVD n+24(FP), R3
	FMOVS scale+32(FP), F2
	MOVD out+40(FP), R4
	LSL $2, R2, R5 // bytes per row
	WORD $0x4e040442 // dup v2.4s, v2.s[0]

sc4:
	CMP $4, R3
	BLT sc1
	WORD $0x4f00e410 // movi v16.16b, #0
	WORD $0x4f00e411 // movi v17.16b, #0
	WORD $0x4f00e412 // movi v18.16b, #0
	WORD $0x4f00e413 // movi v19.16b, #0
	WORD $0x4f00e414 // movi v20.16b, #0
	WORD $0x4f00e415 // movi v21.16b, #0
	WORD $0x4f00e416 // movi v22.16b, #0
	WORD $0x4f00e417 // movi v23.16b, #0
	MOVD R0, R6
	MOVD R1, R7
	ADD R5, R7, R8
	ADD R5, R8, R9
	ADD R5, R9, R10
	MOVD R2, R11

sc4d:
	VLD1.P 32(R6), [V0.S4, V1.S4]
	VLD1.P 32(R7), [V24.S4, V25.S4]
	VLD1.P 32(R8), [V26.S4, V27.S4]
	VLD1.P 32(R9), [V28.S4, V29.S4]
	VLD1.P 32(R10), [V30.S4, V31.S4]
	WORD $0x4e38cc10 // fmla v16.4s, v0.4s, v24.4s
	WORD $0x4e39cc31 // fmla v17.4s, v1.4s, v25.4s
	WORD $0x4e3acc12 // fmla v18.4s, v0.4s, v26.4s
	WORD $0x4e3bcc33 // fmla v19.4s, v1.4s, v27.4s
	WORD $0x4e3ccc14 // fmla v20.4s, v0.4s, v28.4s
	WORD $0x4e3dcc35 // fmla v21.4s, v1.4s, v29.4s
	WORD $0x4e3ecc16 // fmla v22.4s, v0.4s, v30.4s
	WORD $0x4e3fcc37 // fmla v23.4s, v1.4s, v31.4s
	SUBS $8, R11
	BNE sc4d

	WORD $0x4e31d610 // fadd v16.4s, v16.4s, v17.4s
	WORD $0x4e33d652 // fadd v18.4s, v18.4s, v19.4s
	WORD $0x4e35d694 // fadd v20.4s, v20.4s, v21.4s
	WORD $0x4e37d6d6 // fadd v22.4s, v22.4s, v23.4s
	WORD $0x6e32d610 // faddp v16.4s, v16.4s, v18.4s
	WORD $0x6e36d694 // faddp v20.4s, v20.4s, v22.4s
	WORD $0x6e34d610 // faddp v16.4s, v16.4s, v20.4s
	WORD $0x6e22de10 // fmul v16.4s, v16.4s, v2.4s
	VST1.P [V16.S4], 16(R4)
	MOVD R10, R1 // the row after the four
	SUB $4, R3
	B sc4

sc1:
	CBZ R3, scdone
	WORD $0x4f00e410 // movi v16.16b, #0
	WORD $0x4f00e411 // movi v17.16b, #0
	MOVD R0, R6
	MOVD R2, R11

sc1d:
	VLD1.P 32(R6), [V0.S4, V1.S4]
	VLD1.P 32(R1), [V24.S4, V25.S4]
	WORD $0x4e38cc10 // fmla v16.4s, v0.4s, v24.4s
	WORD $0x4e39cc31 // fmla v17.4s, v1.4s, v25.4s
	SUBS $8, R11
	BNE sc1d

	WORD $0x4e31d610 // fadd v16.4s, v16.4s, v17.4s
	WORD $0x6e30d610 // faddp v16.4s, v16.4s, v16.4s
	WORD $0x7e30da10 // faddp s16, v16.2s
	WORD $0x1e220a10 // fmul s16, s16, s2
	FMOVS F16, (R4)
	ADD $4, R4
	SUB $1, R3
	B sc1

scdone:
	RET

// func scoresF16NEON(q *float32, k *byte, d, n int, scale float32, out *float32)
//
// scoresF32NEON for keys stored as f16, converted to float32 (exactly) as
// they are loaded; the result is the same bit for bit.
TEXT ·scoresF16NEON(SB), NOSPLIT, $0-48
	MOVD q+0(FP), R0
	MOVD k+8(FP), R1
	MOVD d+16(FP), R2
	MOVD n+24(FP), R3
	FMOVS scale+32(FP), F2
	MOVD out+40(FP), R4
	LSL $1, R2, R5 // bytes per row
	WORD $0x4e040442 // dup v2.4s, v2.s[0]

sh4:
	CMP $4, R3
	BLT sh1
	WORD $0x4f00e410 // movi v16.16b, #0
	WORD $0x4f00e411 // movi v17.16b, #0
	WORD $0x4f00e412 // movi v18.16b, #0
	WORD $0x4f00e413 // movi v19.16b, #0
	WORD $0x4f00e414 // movi v20.16b, #0
	WORD $0x4f00e415 // movi v21.16b, #0
	WORD $0x4f00e416 // movi v22.16b, #0
	WORD $0x4f00e417 // movi v23.16b, #0
	MOVD R0, R6
	MOVD R1, R7
	ADD R5, R7, R8
	ADD R5, R8, R9
	ADD R5, R9, R10
	MOVD R2, R11

sh4d:
	VLD1.P 32(R6), [V0.S4, V1.S4]
	VLD1.P 16(R7), [V4.H8]
	VLD1.P 16(R8), [V5.H8]
	VLD1.P 16(R9), [V6.H8]
	VLD1.P 16(R10), [V7.H8]
	WORD $0x0e217898 // fcvtl v24.4s, v4.4h
	WORD $0x4e217899 // fcvtl2 v25.4s, v4.8h
	WORD $0x0e2178ba // fcvtl v26.4s, v5.4h
	WORD $0x4e2178bb // fcvtl2 v27.4s, v5.8h
	WORD $0x0e2178dc // fcvtl v28.4s, v6.4h
	WORD $0x4e2178dd // fcvtl2 v29.4s, v6.8h
	WORD $0x0e2178fe // fcvtl v30.4s, v7.4h
	WORD $0x4e2178ff // fcvtl2 v31.4s, v7.8h
	WORD $0x4e38cc10 // fmla v16.4s, v0.4s, v24.4s
	WORD $0x4e39cc31 // fmla v17.4s, v1.4s, v25.4s
	WORD $0x4e3acc12 // fmla v18.4s, v0.4s, v26.4s
	WORD $0x4e3bcc33 // fmla v19.4s, v1.4s, v27.4s
	WORD $0x4e3ccc14 // fmla v20.4s, v0.4s, v28.4s
	WORD $0x4e3dcc35 // fmla v21.4s, v1.4s, v29.4s
	WORD $0x4e3ecc16 // fmla v22.4s, v0.4s, v30.4s
	WORD $0x4e3fcc37 // fmla v23.4s, v1.4s, v31.4s
	SUBS $8, R11
	BNE sh4d

	WORD $0x4e31d610 // fadd v16.4s, v16.4s, v17.4s
	WORD $0x4e33d652 // fadd v18.4s, v18.4s, v19.4s
	WORD $0x4e35d694 // fadd v20.4s, v20.4s, v21.4s
	WORD $0x4e37d6d6 // fadd v22.4s, v22.4s, v23.4s
	WORD $0x6e32d610 // faddp v16.4s, v16.4s, v18.4s
	WORD $0x6e36d694 // faddp v20.4s, v20.4s, v22.4s
	WORD $0x6e34d610 // faddp v16.4s, v16.4s, v20.4s
	WORD $0x6e22de10 // fmul v16.4s, v16.4s, v2.4s
	VST1.P [V16.S4], 16(R4)
	MOVD R10, R1
	SUB $4, R3
	B sh4

sh1:
	CBZ R3, shdone
	WORD $0x4f00e410 // movi v16.16b, #0
	WORD $0x4f00e411 // movi v17.16b, #0
	MOVD R0, R6
	MOVD R2, R11

sh1d:
	VLD1.P 32(R6), [V0.S4, V1.S4]
	VLD1.P 16(R1), [V4.H8]
	WORD $0x0e217898 // fcvtl v24.4s, v4.4h
	WORD $0x4e217899 // fcvtl2 v25.4s, v4.8h
	WORD $0x4e38cc10 // fmla v16.4s, v0.4s, v24.4s
	WORD $0x4e39cc31 // fmla v17.4s, v1.4s, v25.4s
	SUBS $8, R11
	BNE sh1d

	WORD $0x4e31d610 // fadd v16.4s, v16.4s, v17.4s
	WORD $0x6e30d610 // faddp v16.4s, v16.4s, v16.4s
	WORD $0x7e30da10 // faddp s16, v16.2s
	WORD $0x1e220a10 // fmul s16, s16, s2
	FMOVS F16, (R4)
	ADD $4, R4
	SUB $1, R3
	B sh1

shdone:
	RET

// func wsumF32NEON(p, v *float32, n, d int, acc *float32)
//
// Adds p[j] * v[j] to acc for the n rows of v, d floats each, d a positive
// multiple of 8, with one fused multiply-add per element and row, in the
// order of the rows, as axpyF32NEON row by row would. Sixteen columns of
// acc stay in registers while the rows go by, then eight.
TEXT ·wsumF32NEON(SB), NOSPLIT, $0-40
	MOVD p+0(FP), R0
	MOVD v+8(FP), R1
	MOVD n+16(FP), R2
	MOVD d+24(FP), R3
	MOVD acc+32(FP), R4
	LSL $2, R3, R12 // bytes per row
	MOVD $0, R5     // the current column

ws16:
	SUB R5, R3, R6
	CMP $16, R6
	BLT ws8
	ADD R5<<2, R4, R7
	VLD1 (R7), [V0.S4, V1.S4, V2.S4, V3.S4]
	MOVD R0, R8
	ADD R5<<2, R1, R9
	MOVD R2, R10

ws16k:
	WORD $0x4ddfc91f // ld1r {v31.4s}, [x8], #4
	VLD1 (R9), [V4.S4, V5.S4, V6.S4, V7.S4]
	ADD R12, R9
	WORD $0x4e3fcc80 // fmla v0.4s, v4.4s, v31.4s
	WORD $0x4e3fcca1 // fmla v1.4s, v5.4s, v31.4s
	WORD $0x4e3fccc2 // fmla v2.4s, v6.4s, v31.4s
	WORD $0x4e3fcce3 // fmla v3.4s, v7.4s, v31.4s
	SUBS $1, R10
	BNE ws16k

	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R7)
	ADD $16, R5
	B ws16

ws8:
	CMP R3, R5
	BGE wsdone
	ADD R5<<2, R4, R7
	VLD1 (R7), [V0.S4, V1.S4]
	MOVD R0, R8
	ADD R5<<2, R1, R9
	MOVD R2, R10

ws8k:
	WORD $0x4ddfc91f // ld1r {v31.4s}, [x8], #4
	VLD1 (R9), [V4.S4, V5.S4]
	ADD R12, R9
	WORD $0x4e3fcc80 // fmla v0.4s, v4.4s, v31.4s
	WORD $0x4e3fcca1 // fmla v1.4s, v5.4s, v31.4s
	SUBS $1, R10
	BNE ws8k

	VST1 [V0.S4, V1.S4], (R7)
	ADD $8, R5
	B ws8

wsdone:
	RET

// func wsumF16NEON(p *float32, v *byte, n, d int, acc *float32)
//
// wsumF32NEON for values stored as f16, with the same result on the
// converted values.
TEXT ·wsumF16NEON(SB), NOSPLIT, $0-40
	MOVD p+0(FP), R0
	MOVD v+8(FP), R1
	MOVD n+16(FP), R2
	MOVD d+24(FP), R3
	MOVD acc+32(FP), R4
	LSL $1, R3, R12 // bytes per row
	MOVD $0, R5     // the current column

wh16:
	SUB R5, R3, R6
	CMP $16, R6
	BLT wh8
	ADD R5<<2, R4, R7
	VLD1 (R7), [V0.S4, V1.S4, V2.S4, V3.S4]
	MOVD R0, R8
	ADD R5<<1, R1, R9
	MOVD R2, R10

wh16k:
	WORD $0x4ddfc91f // ld1r {v31.4s}, [x8], #4
	VLD1 (R9), [V4.H8, V5.H8]
	ADD R12, R9
	WORD $0x0e217898 // fcvtl v24.4s, v4.4h
	WORD $0x4e217899 // fcvtl2 v25.4s, v4.8h
	WORD $0x0e2178ba // fcvtl v26.4s, v5.4h
	WORD $0x4e2178bb // fcvtl2 v27.4s, v5.8h
	WORD $0x4e3fcf00 // fmla v0.4s, v24.4s, v31.4s
	WORD $0x4e3fcf21 // fmla v1.4s, v25.4s, v31.4s
	WORD $0x4e3fcf42 // fmla v2.4s, v26.4s, v31.4s
	WORD $0x4e3fcf63 // fmla v3.4s, v27.4s, v31.4s
	SUBS $1, R10
	BNE wh16k

	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R7)
	ADD $16, R5
	B wh16

wh8:
	CMP R3, R5
	BGE whdone
	ADD R5<<2, R4, R7
	VLD1 (R7), [V0.S4, V1.S4]
	MOVD R0, R8
	ADD R5<<1, R1, R9
	MOVD R2, R10

wh8k:
	WORD $0x4ddfc91f // ld1r {v31.4s}, [x8], #4
	VLD1 (R9), [V4.H8]
	ADD R12, R9
	WORD $0x0e217898 // fcvtl v24.4s, v4.4h
	WORD $0x4e217899 // fcvtl2 v25.4s, v4.8h
	WORD $0x4e3fcf00 // fmla v0.4s, v24.4s, v31.4s
	WORD $0x4e3fcf21 // fmla v1.4s, v25.4s, v31.4s
	SUBS $1, R10
	BNE wh8k

	VST1 [V0.S4, V1.S4], (R7)
	ADD $8, R5
	B wh8

whdone:
	RET

// EXPCONST loads the constants of the exponential of expSumF32NEON and
// siluMulF32NEON, four copies of each, as expc in attn_amd64.s holds
// them: the lower limit of the arguments whose exp is a normal float32,
// log2(e), ln(2) in two parts, 1, the exponent bias, and the polynomial
// of Cephes' expf.
#define EXPCONST \
	MOVW $0xc2aeac4f, R10; WORD $0x4e040d54 /* dup v20.4s, w10 */; \
	MOVW $0x3fb8aa3b, R10; WORD $0x4e040d55 /* dup v21.4s, w10 */; \
	MOVW $0x3f318000, R10; WORD $0x4e040d56 /* dup v22.4s, w10 */; \
	MOVW $0xb95e8083, R10; WORD $0x4e040d57 /* dup v23.4s, w10 */; \
	MOVW $0x3f800000, R10; WORD $0x4e040d58 /* dup v24.4s, w10 */; \
	MOVW $0x0000007f, R10; WORD $0x4e040d59 /* dup v25.4s, w10 */; \
	MOVW $0x39506967, R10; WORD $0x4e040d5a /* dup v26.4s, w10 */; \
	MOVW $0x3ab743ce, R10; WORD $0x4e040d5b /* dup v27.4s, w10 */; \
	MOVW $0x3c088908, R10; WORD $0x4e040d5c /* dup v28.4s, w10 */; \
	MOVW $0x3d2aa9c1, R10; WORD $0x4e040d5d /* dup v29.4s, w10 */; \
	MOVW $0x3e2aaaaa, R10; WORD $0x4e040d5e /* dup v30.4s, w10 */; \
	MOVW $0x3f000000, R10; WORD $0x4e040d5f /* dup v31.4s, w10 */

// EXP sets v0 to exp(v0) as Cephes' expf computes it, with the same fused
// multiply-adds as expSumF32AVX2, for v0 no greater than 0 (the arguments
// there are never positive); arguments below the limit, -inf among them,
// give 0. It uses v1 to v5.
#define EXP \
	WORD $0x6e34e405 /* fcmge v5.4s, v0.4s, v20.4s */; \
	WORD $0x4e34f400 /* fmax v0.4s, v0.4s, v20.4s */; \
	WORD $0x6e35dc01 /* fmul v1.4s, v0.4s, v21.4s */; \
	WORD $0x4e218821 /* frintn v1.4s, v1.4s */; \
	WORD $0x4eb6cc20 /* fmls v0.4s, v1.4s, v22.4s */; \
	WORD $0x4eb7cc20 /* fmls v0.4s, v1.4s, v23.4s */; \
	WORD $0x6e20dc03 /* fmul v3.4s, v0.4s, v0.4s */; \
	WORD $0x4ebb1f62 /* mov v2.16b, v27.16b */; \
	WORD $0x4e20cf42 /* fmla v2.4s, v26.4s, v0.4s */; \
	WORD $0x4ebc1f84 /* mov v4.16b, v28.16b */; \
	WORD $0x4e20cc44 /* fmla v4.4s, v2.4s, v0.4s */; \
	WORD $0x4ebd1fa2 /* mov v2.16b, v29.16b */; \
	WORD $0x4e20cc82 /* fmla v2.4s, v4.4s, v0.4s */; \
	WORD $0x4ebe1fc4 /* mov v4.16b, v30.16b */; \
	WORD $0x4e20cc44 /* fmla v4.4s, v2.4s, v0.4s */; \
	WORD $0x4ebf1fe2 /* mov v2.16b, v31.16b */; \
	WORD $0x4e20cc82 /* fmla v2.4s, v4.4s, v0.4s */; \
	WORD $0x4e23cc40 /* fmla v0.4s, v2.4s, v3.4s */; \
	WORD $0x4e38d400 /* fadd v0.4s, v0.4s, v24.4s */; \
	WORD $0x4ea1b821 /* fcvtzs v1.4s, v1.4s */; \
	WORD $0x4eb98421 /* add v1.4s, v1.4s, v25.4s */; \
	WORD $0x4f375421 /* shl v1.4s, v1.4s, #23 */; \
	WORD $0x6e21dc00 /* fmul v0.4s, v0.4s, v1.4s */; \
	WORD $0x4e251c00 // and v0.16b, v0.16b, v5.16b

// func expSumF32NEON(x *float32, n int, m float32) float32
//
// Sets x[i] = exp(x[i] - m) for n a positive multiple of 4 and returns the
// sum, as expSumF32AVX2 does.
TEXT ·expSumF32NEON(SB), NOSPLIT, $0-28
	MOVD x+0(FP), R0
	MOVD n+8(FP), R1
	FMOVS m+16(FP), F19
	WORD $0x4e040673 // dup v19.4s, v19.s[0]
	EXPCONST
	WORD $0x4f00e412 // movi v18.16b, #0

exploop:
	VLD1 (R0), [V0.S4]
	WORD $0x4eb3d400 // fsub v0.4s, v0.4s, v19.4s
	EXP
	VST1.P [V0.S4], 16(R0)
	WORD $0x4e20d652 // fadd v18.4s, v18.4s, v0.4s
	SUBS $4, R1
	BNE exploop

	WORD $0x6e32d652 // faddp v18.4s, v18.4s, v18.4s
	WORD $0x7e30da52 // faddp s18, v18.2s
	FMOVS F18, ret+24(FP)
	RET

// func siluMulF32NEON(a, b, out *float32, n int)
//
// Sets out[i] to silu(a[i]) * b[i] for n a positive multiple of 4, as
// siluMulF32AVX2 computes it: silu(x) = x / (1+e) for x >= 0 and
// (x*h)*h / (1+e) below, with h = exp(-|x|/2) and e = h*h.
TEXT ·siluMulF32NEON(SB), NOSPLIT, $0-32
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD out+16(FP), R2
	MOVD n+24(FP), R3
	EXPCONST
	WORD $0x4f07f412 // fmov v18.4s, #-0.5

siluloop:
	VLD1.P 16(R0), [V16.S4]
	VLD1.P 16(R1), [V17.S4]
	WORD $0x4ea0fa00 // fabs v0.4s, v16.4s
	WORD $0x6e32dc00 // fmul v0.4s, v0.4s, v18.4s
	EXP
	WORD $0x6e20dc03 // fmul v3.4s, v0.4s, v0.4s
	WORD $0x4e38d463 // fadd v3.4s, v3.4s, v24.4s
	WORD $0x6e20de01 // fmul v1.4s, v16.4s, v0.4s
	WORD $0x6e20dc21 // fmul v1.4s, v1.4s, v0.4s
	WORD $0x6ea0ca05 // fcmge v5.4s, v16.4s, #0.0
	WORD $0x6ea51e01 // bit v1.16b, v16.16b, v5.16b
	WORD $0x6e23fc21 // fdiv v1.4s, v1.4s, v3.4s
	WORD $0x6e31dc21 // fmul v1.4s, v1.4s, v17.4s
	VST1.P [V1.S4], 16(R2)
	SUBS $4, R3
	BNE siluloop
	RET
