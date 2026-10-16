// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mmf, created and owned by Gembit Soultan Shirazi.
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

// 8-bit activation kernels for Q4_K and Q6_K weights with activation
// blocks of 256, on NEON with the dot product instructions (SDOT). They
// compute what the AVX2 kernels in kernels_amd64.s compute, bit for bit:
// each block's integer sums are exact in int32, then converted to float32
// and added with one fused multiply-add per block, in the same order.
//
// The Go assembler has no mnemonics for SDOT and several other vector
// instructions, so those are written as WORDs, each with the instruction
// it encodes in a comment.

// func dotQ4KQ8KNEON(q *byte, xq *int8, xd *float32, xs *int16, nb int) (float32, bool)
//
// For each block, with sub-block scales sc_s and offsets m_s, activation
// quants y and the sums xs_s of y over each sub-block, the integer sums
//
//	S = sum_s sc_s * sum q*y,  M = sum_s m_s * xs_s
//
// are exact in int32. The block adds float32(S) * (d*dx) to the total and
// float32(M) * (dmin*dx) to the offsets, each with one fused multiply-add;
// the result is the total minus the offsets. It is false, and the sum
// meaningless, if d or dmin of a block is not finite.
//
// Each chunk of 32 bytes holds sub-block 2c in its low nibbles and 2c+1 in
// its high ones; SDOT sums their products with the activations four at a
// time, and pairwise adds leave the four sub-block sums of two chunks in
// the lanes of one register, which is multiplied by their scales.
TEXT ·dotQ4KQ8KNEON(SB), NOSPLIT, $0-45
	MOVD q+0(FP), R0
	MOVD xq+8(FP), R1
	MOVD xd+16(FP), R5
	MOVD xs+24(FP), R7
	MOVD nb+32(FP), R9
	WORD $0x4f00e5ff // movi v31.16b, #0x0f
	WORD $0x4f00e41e // movi v30.16b, #0
	WORD $0x4f00e41d // movi v29.16b, #0
k41block:
	MOVHU 0(R0), R13
	AND $0x7C00, R13, R12
	CMP $0x7C00, R12
	BEQ k41bad
	MOVHU 2(R0), R14
	AND $0x7C00, R14, R12
	CMP $0x7C00, R12
	BEQ k41bad
	FMOVS R13, F27
	WORD $0x1ee2437b // fcvt s27, h27
	FMOVS R14, F26
	WORD $0x1ee2435a // fcvt s26, h26
	MOVWU 4(R0), R12
	MOVWU 8(R0), R13
	MOVWU 12(R0), R14
	ANDW $0x3f3f3f3f, R12, R15
	ANDW $0x0f0f0f0f, R14, R16
	LSRW $6, R12, R17
	ANDW $0x03030303, R17, R17
	ORRW R17<<4, R16, R16
	ORR R16<<32, R15, R15
	ANDW $0x3f3f3f3f, R13, R16
	LSRW $4, R14, R17
	ANDW $0x0f0f0f0f, R17, R17
	LSRW $6, R13, R19
	ANDW $0x03030303, R19, R19
	ORRW R19<<4, R17, R17
	ORR R17<<32, R16, R16
	VMOV R15, V22.D[0]
	WORD $0x2f08a6d6 // uxtl v22.8h, v22.8b
	WORD $0x6f10a6d5 // uxtl2 v21.4s, v22.8h
	WORD $0x2f10a6d6 // uxtl v22.4s, v22.4h
	VMOV R16, V20.D[0]
	WORD $0x2f08a694 // uxtl v20.8h, v20.8b
	MOVD R7, R10
	VLD1 (R10), [V0.H8]
	WORD $0x0e60c28c // smull v12.4s, v20.4h, v0.4h
	WORD $0x4e60828c // smlal2 v12.4s, v20.8h, v0.8h
	WORD $0x4eacbd80 // addp v0.4s, v12.4s, v12.4s
	WORD $0x4ea0bc00 // addp v0.4s, v0.4s, v0.4s
	WORD $0x4e21d800 // scvtf v0.4s, v0.4s
	FMOVS (R5), F25
	WORD $0x4f9a9337 // fmul v23.4s, v25.4s, v26.s[0]
	WORD $0x4f9b9338 // fmul v24.4s, v25.4s, v27.s[0]
	WORD $0x4e37cc1d // fmla v29.4s, v0.4s, v23.4s
	WORD $0x4f00e410 // movi v16.16b, #0
	ADD $16, R0, R11
	VLD1.P 32(R11), [V0.B16, V1.B16]
	WORD $0x4e3f1c08 // and v8.16b, v0.16b, v31.16b
	WORD $0x4e3f1c29 // and v9.16b, v1.16b, v31.16b
	WORD $0x6f0c040a // ushr v10.16b, v0.16b, #4
	WORD $0x6f0c042b // ushr v11.16b, v1.16b, #4
	VLD1.P 64(R1), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc4c // addp v12.4s, v2.4s, v3.4s
	VLD1.P 32(R11), [V0.B16, V1.B16]
	WORD $0x4e3f1c08 // and v8.16b, v0.16b, v31.16b
	WORD $0x4e3f1c29 // and v9.16b, v1.16b, v31.16b
	WORD $0x6f0c040a // ushr v10.16b, v0.16b, #4
	WORD $0x6f0c042b // ushr v11.16b, v1.16b, #4
	VLD1.P 64(R1), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea2bd82 // addp v2.4s, v12.4s, v2.4s
	WORD $0x4eb69450 // mla v16.4s, v2.4s, v22.4s
	VLD1.P 32(R11), [V0.B16, V1.B16]
	WORD $0x4e3f1c08 // and v8.16b, v0.16b, v31.16b
	WORD $0x4e3f1c29 // and v9.16b, v1.16b, v31.16b
	WORD $0x6f0c040a // ushr v10.16b, v0.16b, #4
	WORD $0x6f0c042b // ushr v11.16b, v1.16b, #4
	VLD1.P 64(R1), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc4c // addp v12.4s, v2.4s, v3.4s
	VLD1.P 32(R11), [V0.B16, V1.B16]
	WORD $0x4e3f1c08 // and v8.16b, v0.16b, v31.16b
	WORD $0x4e3f1c29 // and v9.16b, v1.16b, v31.16b
	WORD $0x6f0c040a // ushr v10.16b, v0.16b, #4
	WORD $0x6f0c042b // ushr v11.16b, v1.16b, #4
	VLD1.P 64(R1), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea2bd82 // addp v2.4s, v12.4s, v2.4s
	WORD $0x4eb59450 // mla v16.4s, v2.4s, v21.4s
	WORD $0x4eb0be00 // addp v0.4s, v16.4s, v16.4s
	WORD $0x4ea0bc00 // addp v0.4s, v0.4s, v0.4s
	WORD $0x4e21d800 // scvtf v0.4s, v0.4s
	WORD $0x4e38cc1e // fmla v30.4s, v0.4s, v24.4s
	ADD $144, R0
	ADD $32, R5
	ADD $16, R7
	SUBS $1, R9
	BNE k41block
	WORD $0x4ebdd7de // fsub v30.4s, v30.4s, v29.4s
	FMOVS F30, ret+40(FP)
	MOVD $1, R10
	MOVB R10, ret1+44(FP)
	RET
k41bad:
	MOVB ZR, ret1+44(FP)
	RET

// func dotQ4KQ8Kx4NEON(q *byte, xq *int8, xd *float32, xs *int16, xstride, dstride, nb int, out *[4]float32) bool
//
// dotQ4KQ8KNEON for four rows of activations, the quants xstride bytes
// apart, the scales dstride bytes apart and the sums dstride/2. The four
// rows share the decoding of the weights; lane r of V30 and V29 holds the
// total and the offsets of row r, which get the float operations of the
// one-row kernel lane by lane.
TEXT ·dotQ4KQ8Kx4NEON(SB), NOSPLIT, $0-65
	MOVD q+0(FP), R0
	MOVD xq+8(FP), R1
	MOVD xd+16(FP), R5
	MOVD xs+24(FP), R7
	MOVD xstride+32(FP), R10
	MOVD dstride+40(FP), R6
	MOVD nb+48(FP), R9
	LSR $1, R6, R8
	ADD R10, R1, R2
	ADD R10, R2, R3
	ADD R10, R3, R4
	WORD $0x4f00e5ff // movi v31.16b, #0x0f
	WORD $0x4f00e41e // movi v30.16b, #0
	WORD $0x4f00e41d // movi v29.16b, #0
k44block:
	MOVHU 0(R0), R13
	AND $0x7C00, R13, R12
	CMP $0x7C00, R12
	BEQ k44bad
	MOVHU 2(R0), R14
	AND $0x7C00, R14, R12
	CMP $0x7C00, R12
	BEQ k44bad
	FMOVS R13, F27
	WORD $0x1ee2437b // fcvt s27, h27
	FMOVS R14, F26
	WORD $0x1ee2435a // fcvt s26, h26
	MOVWU 4(R0), R12
	MOVWU 8(R0), R13
	MOVWU 12(R0), R14
	ANDW $0x3f3f3f3f, R12, R15
	ANDW $0x0f0f0f0f, R14, R16
	LSRW $6, R12, R17
	ANDW $0x03030303, R17, R17
	ORRW R17<<4, R16, R16
	ORR R16<<32, R15, R15
	ANDW $0x3f3f3f3f, R13, R16
	LSRW $4, R14, R17
	ANDW $0x0f0f0f0f, R17, R17
	LSRW $6, R13, R19
	ANDW $0x03030303, R19, R19
	ORRW R19<<4, R17, R17
	ORR R17<<32, R16, R16
	VMOV R15, V22.D[0]
	WORD $0x2f08a6d6 // uxtl v22.8h, v22.8b
	WORD $0x6f10a6d5 // uxtl2 v21.4s, v22.8h
	WORD $0x2f10a6d6 // uxtl v22.4s, v22.4h
	VMOV R16, V20.D[0]
	WORD $0x2f08a694 // uxtl v20.8h, v20.8b
	MOVD R7, R10
	VLD1 (R10), [V0.H8]
	WORD $0x0e60c28c // smull v12.4s, v20.4h, v0.4h
	WORD $0x4e60828c // smlal2 v12.4s, v20.8h, v0.8h
	ADD R8, R10, R10
	VLD1 (R10), [V0.H8]
	WORD $0x0e60c28d // smull v13.4s, v20.4h, v0.4h
	WORD $0x4e60828d // smlal2 v13.4s, v20.8h, v0.8h
	ADD R8, R10, R10
	VLD1 (R10), [V0.H8]
	WORD $0x0e60c28e // smull v14.4s, v20.4h, v0.4h
	WORD $0x4e60828e // smlal2 v14.4s, v20.8h, v0.8h
	ADD R8, R10, R10
	VLD1 (R10), [V0.H8]
	WORD $0x0e60c28f // smull v15.4s, v20.4h, v0.4h
	WORD $0x4e60828f // smlal2 v15.4s, v20.8h, v0.8h
	WORD $0x4eadbd8c // addp v12.4s, v12.4s, v13.4s
	WORD $0x4eafbdce // addp v14.4s, v14.4s, v15.4s
	WORD $0x4eaebd80 // addp v0.4s, v12.4s, v14.4s
	WORD $0x4e21d800 // scvtf v0.4s, v0.4s
	FMOVS (R5), F25
	MOVD R5, R10
	ADD R6, R10, R10
	MOVWU (R10), R11
	VMOV R11, V25.S[1]
	ADD R6, R10, R10
	MOVWU (R10), R11
	VMOV R11, V25.S[2]
	ADD R6, R10, R10
	MOVWU (R10), R11
	VMOV R11, V25.S[3]
	WORD $0x4f9a9337 // fmul v23.4s, v25.4s, v26.s[0]
	WORD $0x4f9b9338 // fmul v24.4s, v25.4s, v27.s[0]
	WORD $0x4e37cc1d // fmla v29.4s, v0.4s, v23.4s
	WORD $0x4f00e410 // movi v16.16b, #0
	WORD $0x4f00e411 // movi v17.16b, #0
	WORD $0x4f00e412 // movi v18.16b, #0
	WORD $0x4f00e413 // movi v19.16b, #0
	ADD $16, R0, R11
	VLD1.P 32(R11), [V0.B16, V1.B16]
	WORD $0x4e3f1c08 // and v8.16b, v0.16b, v31.16b
	WORD $0x4e3f1c29 // and v9.16b, v1.16b, v31.16b
	WORD $0x6f0c040a // ushr v10.16b, v0.16b, #4
	WORD $0x6f0c042b // ushr v11.16b, v1.16b, #4
	VLD1.P 64(R1), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc4c // addp v12.4s, v2.4s, v3.4s
	VLD1.P 64(R2), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc4d // addp v13.4s, v2.4s, v3.4s
	VLD1.P 64(R3), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc4e // addp v14.4s, v2.4s, v3.4s
	VLD1.P 64(R4), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc4f // addp v15.4s, v2.4s, v3.4s
	VLD1.P 32(R11), [V0.B16, V1.B16]
	WORD $0x4e3f1c08 // and v8.16b, v0.16b, v31.16b
	WORD $0x4e3f1c29 // and v9.16b, v1.16b, v31.16b
	WORD $0x6f0c040a // ushr v10.16b, v0.16b, #4
	WORD $0x6f0c042b // ushr v11.16b, v1.16b, #4
	VLD1.P 64(R1), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea2bd82 // addp v2.4s, v12.4s, v2.4s
	WORD $0x4eb69450 // mla v16.4s, v2.4s, v22.4s
	VLD1.P 64(R2), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea2bda2 // addp v2.4s, v13.4s, v2.4s
	WORD $0x4eb69451 // mla v17.4s, v2.4s, v22.4s
	VLD1.P 64(R3), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea2bdc2 // addp v2.4s, v14.4s, v2.4s
	WORD $0x4eb69452 // mla v18.4s, v2.4s, v22.4s
	VLD1.P 64(R4), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea2bde2 // addp v2.4s, v15.4s, v2.4s
	WORD $0x4eb69453 // mla v19.4s, v2.4s, v22.4s
	VLD1.P 32(R11), [V0.B16, V1.B16]
	WORD $0x4e3f1c08 // and v8.16b, v0.16b, v31.16b
	WORD $0x4e3f1c29 // and v9.16b, v1.16b, v31.16b
	WORD $0x6f0c040a // ushr v10.16b, v0.16b, #4
	WORD $0x6f0c042b // ushr v11.16b, v1.16b, #4
	VLD1.P 64(R1), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc4c // addp v12.4s, v2.4s, v3.4s
	VLD1.P 64(R2), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc4d // addp v13.4s, v2.4s, v3.4s
	VLD1.P 64(R3), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc4e // addp v14.4s, v2.4s, v3.4s
	VLD1.P 64(R4), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc4f // addp v15.4s, v2.4s, v3.4s
	VLD1.P 32(R11), [V0.B16, V1.B16]
	WORD $0x4e3f1c08 // and v8.16b, v0.16b, v31.16b
	WORD $0x4e3f1c29 // and v9.16b, v1.16b, v31.16b
	WORD $0x6f0c040a // ushr v10.16b, v0.16b, #4
	WORD $0x6f0c042b // ushr v11.16b, v1.16b, #4
	VLD1.P 64(R1), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea2bd82 // addp v2.4s, v12.4s, v2.4s
	WORD $0x4eb59450 // mla v16.4s, v2.4s, v21.4s
	VLD1.P 64(R2), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea2bda2 // addp v2.4s, v13.4s, v2.4s
	WORD $0x4eb59451 // mla v17.4s, v2.4s, v21.4s
	VLD1.P 64(R3), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea2bdc2 // addp v2.4s, v14.4s, v2.4s
	WORD $0x4eb59452 // mla v18.4s, v2.4s, v21.4s
	VLD1.P 64(R4), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4e849502 // sdot v2.4s, v8.16b, v4.16b
	WORD $0x4e859522 // sdot v2.4s, v9.16b, v5.16b
	WORD $0x4e869543 // sdot v3.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea2bde2 // addp v2.4s, v15.4s, v2.4s
	WORD $0x4eb59453 // mla v19.4s, v2.4s, v21.4s
	WORD $0x4eb1be10 // addp v16.4s, v16.4s, v17.4s
	WORD $0x4eb3be52 // addp v18.4s, v18.4s, v19.4s
	WORD $0x4eb2be00 // addp v0.4s, v16.4s, v18.4s
	WORD $0x4e21d800 // scvtf v0.4s, v0.4s
	WORD $0x4e38cc1e // fmla v30.4s, v0.4s, v24.4s
	ADD $144, R0
	ADD $32, R5
	ADD $16, R7
	SUBS $1, R9
	BNE k44block
	WORD $0x4ebdd7de // fsub v30.4s, v30.4s, v29.4s
	MOVD out+56(FP), R10
	VST1 [V30.S4], (R10)
	MOVD $1, R10
	MOVB R10, ret+64(FP)
	RET
k44bad:
	MOVB ZR, ret+64(FP)
	RET

// func dotQ6KQ8KNEON(q *byte, xq *int8, xd *float32, nb int) (float32, bool)
//
// For each block, with the sub-block scales sc_s and activation quants y,
// S = sum_s sc_s * sum (q-32)*y is exact in int32, and the block adds
// float32(S) * (d*dx) to the total with one fused multiply-add. It is
// false, and the sum meaningless, if d of a block is not finite. Each
// register of 16 quants is one sub-block.
TEXT ·dotQ6KQ8KNEON(SB), NOSPLIT, $0-37
	MOVD q+0(FP), R0
	MOVD xq+8(FP), R1
	MOVD xd+16(FP), R5
	MOVD nb+24(FP), R9
	WORD $0x4f00e5ff // movi v31.16b, #0x0f
	WORD $0x4f01e61e // movi v30.16b, #0x30
	WORD $0x4f01e41d // movi v29.16b, #0x20
	WORD $0x4f00e41c // movi v28.16b, #0
k61block:
	MOVHU 208(R0), R15
	AND $0x7C00, R15, R12
	CMP $0x7C00, R12
	BEQ k61bad
	ADD $192, R0, R10
	VLD1 (R10), [V0.B16]
	WORD $0x0f08a401 // sxtl v1.8h, v0.8b
	WORD $0x4f08a400 // sxtl2 v0.8h, v0.16b
	WORD $0x0f10a438 // sxtl v24.4s, v1.4h
	WORD $0x4f10a439 // sxtl2 v25.4s, v1.8h
	WORD $0x0f10a41a // sxtl v26.4s, v0.4h
	WORD $0x4f10a41b // sxtl2 v27.4s, v0.8h
	WORD $0x4f00e40e // movi v14.16b, #0
	MOVD R0, R10
	ADD $128, R0, R11
	VLD1.P 64(R10), [V20.B16, V21.B16, V22.B16, V23.B16]
	VLD1.P 32(R11), [V18.B16, V19.B16]
	WORD $0x4f0c5640 // shl v0.16b, v18.16b, #4
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1e8a // and v10.16b, v20.16b, v31.16b
	WORD $0x4ea01d4a // orr v10.16b, v10.16b, v0.16b
	WORD $0x6e3d854a // sub v10.16b, v10.16b, v29.16b
	WORD $0x4f0c5660 // shl v0.16b, v19.16b, #4
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1eab // and v11.16b, v21.16b, v31.16b
	WORD $0x4ea01d6b // orr v11.16b, v11.16b, v0.16b
	WORD $0x6e3d856b // sub v11.16b, v11.16b, v29.16b
	WORD $0x4f0a5640 // shl v0.16b, v18.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1ecc // and v12.16b, v22.16b, v31.16b
	WORD $0x4ea01d8c // orr v12.16b, v12.16b, v0.16b
	WORD $0x6e3d858c // sub v12.16b, v12.16b, v29.16b
	WORD $0x4f0a5660 // shl v0.16b, v19.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1eed // and v13.16b, v23.16b, v31.16b
	WORD $0x4ea01dad // orr v13.16b, v13.16b, v0.16b
	WORD $0x6e3d85ad // sub v13.16b, v13.16b, v29.16b
	VLD1.P 64(R1), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eb8944e // mla v14.4s, v2.4s, v24.4s
	WORD $0x4e3e1e40 // and v0.16b, v18.16b, v30.16b
	WORD $0x6f0c068a // ushr v10.16b, v20.16b, #4
	WORD $0x4ea01d4a // orr v10.16b, v10.16b, v0.16b
	WORD $0x6e3d854a // sub v10.16b, v10.16b, v29.16b
	WORD $0x4e3e1e60 // and v0.16b, v19.16b, v30.16b
	WORD $0x6f0c06ab // ushr v11.16b, v21.16b, #4
	WORD $0x4ea01d6b // orr v11.16b, v11.16b, v0.16b
	WORD $0x6e3d856b // sub v11.16b, v11.16b, v29.16b
	WORD $0x6f0e0640 // ushr v0.16b, v18.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x6f0c06cc // ushr v12.16b, v22.16b, #4
	WORD $0x4ea01d8c // orr v12.16b, v12.16b, v0.16b
	WORD $0x6e3d858c // sub v12.16b, v12.16b, v29.16b
	WORD $0x6f0e0660 // ushr v0.16b, v19.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x6f0c06ed // ushr v13.16b, v23.16b, #4
	WORD $0x4ea01dad // orr v13.16b, v13.16b, v0.16b
	WORD $0x6e3d85ad // sub v13.16b, v13.16b, v29.16b
	VLD1.P 64(R1), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eb9944e // mla v14.4s, v2.4s, v25.4s
	VLD1.P 64(R10), [V20.B16, V21.B16, V22.B16, V23.B16]
	VLD1.P 32(R11), [V18.B16, V19.B16]
	WORD $0x4f0c5640 // shl v0.16b, v18.16b, #4
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1e8a // and v10.16b, v20.16b, v31.16b
	WORD $0x4ea01d4a // orr v10.16b, v10.16b, v0.16b
	WORD $0x6e3d854a // sub v10.16b, v10.16b, v29.16b
	WORD $0x4f0c5660 // shl v0.16b, v19.16b, #4
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1eab // and v11.16b, v21.16b, v31.16b
	WORD $0x4ea01d6b // orr v11.16b, v11.16b, v0.16b
	WORD $0x6e3d856b // sub v11.16b, v11.16b, v29.16b
	WORD $0x4f0a5640 // shl v0.16b, v18.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1ecc // and v12.16b, v22.16b, v31.16b
	WORD $0x4ea01d8c // orr v12.16b, v12.16b, v0.16b
	WORD $0x6e3d858c // sub v12.16b, v12.16b, v29.16b
	WORD $0x4f0a5660 // shl v0.16b, v19.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1eed // and v13.16b, v23.16b, v31.16b
	WORD $0x4ea01dad // orr v13.16b, v13.16b, v0.16b
	WORD $0x6e3d85ad // sub v13.16b, v13.16b, v29.16b
	VLD1.P 64(R1), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eba944e // mla v14.4s, v2.4s, v26.4s
	WORD $0x4e3e1e40 // and v0.16b, v18.16b, v30.16b
	WORD $0x6f0c068a // ushr v10.16b, v20.16b, #4
	WORD $0x4ea01d4a // orr v10.16b, v10.16b, v0.16b
	WORD $0x6e3d854a // sub v10.16b, v10.16b, v29.16b
	WORD $0x4e3e1e60 // and v0.16b, v19.16b, v30.16b
	WORD $0x6f0c06ab // ushr v11.16b, v21.16b, #4
	WORD $0x4ea01d6b // orr v11.16b, v11.16b, v0.16b
	WORD $0x6e3d856b // sub v11.16b, v11.16b, v29.16b
	WORD $0x6f0e0640 // ushr v0.16b, v18.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x6f0c06cc // ushr v12.16b, v22.16b, #4
	WORD $0x4ea01d8c // orr v12.16b, v12.16b, v0.16b
	WORD $0x6e3d858c // sub v12.16b, v12.16b, v29.16b
	WORD $0x6f0e0660 // ushr v0.16b, v19.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x6f0c06ed // ushr v13.16b, v23.16b, #4
	WORD $0x4ea01dad // orr v13.16b, v13.16b, v0.16b
	WORD $0x6e3d85ad // sub v13.16b, v13.16b, v29.16b
	VLD1.P 64(R1), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4ebb944e // mla v14.4s, v2.4s, v27.4s
	WORD $0x4eaebdc2 // addp v2.4s, v14.4s, v14.4s
	WORD $0x4ea2bc42 // addp v2.4s, v2.4s, v2.4s
	WORD $0x4e21d842 // scvtf v2.4s, v2.4s
	FMOVS R15, F0
	WORD $0x1ee24000 // fcvt s0, h0
	FMOVS (R5), F1
	WORD $0x4f809021 // fmul v1.4s, v1.4s, v0.s[0]
	WORD $0x4e21cc5c // fmla v28.4s, v2.4s, v1.4s
	ADD $210, R0
	ADD $32, R5
	SUBS $1, R9
	BNE k61block
	FMOVS F28, ret+32(FP)
	MOVD $1, R10
	MOVB R10, ret1+36(FP)
	RET
k61bad:
	MOVB ZR, ret1+36(FP)
	RET

// func dotQ6KQ8Kx4NEON(q *byte, xq *int8, xd *float32, xstride, dstride, nb int, out *[4]float32) bool
//
// dotQ6KQ8KNEON for four rows of activations, laid out as for
// dotQ4KQ8Kx4NEON. Lane r of V28 holds the total of row r.
TEXT ·dotQ6KQ8Kx4NEON(SB), NOSPLIT, $0-57
	MOVD q+0(FP), R0
	MOVD xq+8(FP), R1
	MOVD xd+16(FP), R5
	MOVD xstride+24(FP), R10
	MOVD dstride+32(FP), R6
	MOVD nb+40(FP), R9
	ADD R10, R1, R2
	ADD R10, R2, R3
	ADD R10, R3, R4
	WORD $0x4f00e5ff // movi v31.16b, #0x0f
	WORD $0x4f01e61e // movi v30.16b, #0x30
	WORD $0x4f01e41d // movi v29.16b, #0x20
	WORD $0x4f00e41c // movi v28.16b, #0
k64block:
	MOVHU 208(R0), R15
	AND $0x7C00, R15, R12
	CMP $0x7C00, R12
	BEQ k64bad
	ADD $192, R0, R10
	VLD1 (R10), [V0.B16]
	WORD $0x0f08a401 // sxtl v1.8h, v0.8b
	WORD $0x4f08a400 // sxtl2 v0.8h, v0.16b
	WORD $0x0f10a438 // sxtl v24.4s, v1.4h
	WORD $0x4f10a439 // sxtl2 v25.4s, v1.8h
	WORD $0x0f10a41a // sxtl v26.4s, v0.4h
	WORD $0x4f10a41b // sxtl2 v27.4s, v0.8h
	WORD $0x4f00e40e // movi v14.16b, #0
	WORD $0x4f00e40f // movi v15.16b, #0
	WORD $0x4f00e410 // movi v16.16b, #0
	WORD $0x4f00e411 // movi v17.16b, #0
	MOVD R0, R10
	ADD $128, R0, R11
	VLD1.P 64(R10), [V20.B16, V21.B16, V22.B16, V23.B16]
	VLD1.P 32(R11), [V18.B16, V19.B16]
	WORD $0x4f0c5640 // shl v0.16b, v18.16b, #4
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1e8a // and v10.16b, v20.16b, v31.16b
	WORD $0x4ea01d4a // orr v10.16b, v10.16b, v0.16b
	WORD $0x6e3d854a // sub v10.16b, v10.16b, v29.16b
	WORD $0x4f0c5660 // shl v0.16b, v19.16b, #4
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1eab // and v11.16b, v21.16b, v31.16b
	WORD $0x4ea01d6b // orr v11.16b, v11.16b, v0.16b
	WORD $0x6e3d856b // sub v11.16b, v11.16b, v29.16b
	WORD $0x4f0a5640 // shl v0.16b, v18.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1ecc // and v12.16b, v22.16b, v31.16b
	WORD $0x4ea01d8c // orr v12.16b, v12.16b, v0.16b
	WORD $0x6e3d858c // sub v12.16b, v12.16b, v29.16b
	WORD $0x4f0a5660 // shl v0.16b, v19.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1eed // and v13.16b, v23.16b, v31.16b
	WORD $0x4ea01dad // orr v13.16b, v13.16b, v0.16b
	WORD $0x6e3d85ad // sub v13.16b, v13.16b, v29.16b
	VLD1.P 64(R1), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eb8944e // mla v14.4s, v2.4s, v24.4s
	VLD1.P 64(R2), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eb8944f // mla v15.4s, v2.4s, v24.4s
	VLD1.P 64(R3), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eb89450 // mla v16.4s, v2.4s, v24.4s
	VLD1.P 64(R4), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eb89451 // mla v17.4s, v2.4s, v24.4s
	WORD $0x4e3e1e40 // and v0.16b, v18.16b, v30.16b
	WORD $0x6f0c068a // ushr v10.16b, v20.16b, #4
	WORD $0x4ea01d4a // orr v10.16b, v10.16b, v0.16b
	WORD $0x6e3d854a // sub v10.16b, v10.16b, v29.16b
	WORD $0x4e3e1e60 // and v0.16b, v19.16b, v30.16b
	WORD $0x6f0c06ab // ushr v11.16b, v21.16b, #4
	WORD $0x4ea01d6b // orr v11.16b, v11.16b, v0.16b
	WORD $0x6e3d856b // sub v11.16b, v11.16b, v29.16b
	WORD $0x6f0e0640 // ushr v0.16b, v18.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x6f0c06cc // ushr v12.16b, v22.16b, #4
	WORD $0x4ea01d8c // orr v12.16b, v12.16b, v0.16b
	WORD $0x6e3d858c // sub v12.16b, v12.16b, v29.16b
	WORD $0x6f0e0660 // ushr v0.16b, v19.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x6f0c06ed // ushr v13.16b, v23.16b, #4
	WORD $0x4ea01dad // orr v13.16b, v13.16b, v0.16b
	WORD $0x6e3d85ad // sub v13.16b, v13.16b, v29.16b
	VLD1.P 64(R1), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eb9944e // mla v14.4s, v2.4s, v25.4s
	VLD1.P 64(R2), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eb9944f // mla v15.4s, v2.4s, v25.4s
	VLD1.P 64(R3), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eb99450 // mla v16.4s, v2.4s, v25.4s
	VLD1.P 64(R4), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eb99451 // mla v17.4s, v2.4s, v25.4s
	VLD1.P 64(R10), [V20.B16, V21.B16, V22.B16, V23.B16]
	VLD1.P 32(R11), [V18.B16, V19.B16]
	WORD $0x4f0c5640 // shl v0.16b, v18.16b, #4
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1e8a // and v10.16b, v20.16b, v31.16b
	WORD $0x4ea01d4a // orr v10.16b, v10.16b, v0.16b
	WORD $0x6e3d854a // sub v10.16b, v10.16b, v29.16b
	WORD $0x4f0c5660 // shl v0.16b, v19.16b, #4
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1eab // and v11.16b, v21.16b, v31.16b
	WORD $0x4ea01d6b // orr v11.16b, v11.16b, v0.16b
	WORD $0x6e3d856b // sub v11.16b, v11.16b, v29.16b
	WORD $0x4f0a5640 // shl v0.16b, v18.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1ecc // and v12.16b, v22.16b, v31.16b
	WORD $0x4ea01d8c // orr v12.16b, v12.16b, v0.16b
	WORD $0x6e3d858c // sub v12.16b, v12.16b, v29.16b
	WORD $0x4f0a5660 // shl v0.16b, v19.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x4e3f1eed // and v13.16b, v23.16b, v31.16b
	WORD $0x4ea01dad // orr v13.16b, v13.16b, v0.16b
	WORD $0x6e3d85ad // sub v13.16b, v13.16b, v29.16b
	VLD1.P 64(R1), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eba944e // mla v14.4s, v2.4s, v26.4s
	VLD1.P 64(R2), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eba944f // mla v15.4s, v2.4s, v26.4s
	VLD1.P 64(R3), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eba9450 // mla v16.4s, v2.4s, v26.4s
	VLD1.P 64(R4), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4eba9451 // mla v17.4s, v2.4s, v26.4s
	WORD $0x4e3e1e40 // and v0.16b, v18.16b, v30.16b
	WORD $0x6f0c068a // ushr v10.16b, v20.16b, #4
	WORD $0x4ea01d4a // orr v10.16b, v10.16b, v0.16b
	WORD $0x6e3d854a // sub v10.16b, v10.16b, v29.16b
	WORD $0x4e3e1e60 // and v0.16b, v19.16b, v30.16b
	WORD $0x6f0c06ab // ushr v11.16b, v21.16b, #4
	WORD $0x4ea01d6b // orr v11.16b, v11.16b, v0.16b
	WORD $0x6e3d856b // sub v11.16b, v11.16b, v29.16b
	WORD $0x6f0e0640 // ushr v0.16b, v18.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x6f0c06cc // ushr v12.16b, v22.16b, #4
	WORD $0x4ea01d8c // orr v12.16b, v12.16b, v0.16b
	WORD $0x6e3d858c // sub v12.16b, v12.16b, v29.16b
	WORD $0x6f0e0660 // ushr v0.16b, v19.16b, #2
	WORD $0x4e3e1c00 // and v0.16b, v0.16b, v30.16b
	WORD $0x6f0c06ed // ushr v13.16b, v23.16b, #4
	WORD $0x4ea01dad // orr v13.16b, v13.16b, v0.16b
	WORD $0x6e3d85ad // sub v13.16b, v13.16b, v29.16b
	VLD1.P 64(R1), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4ebb944e // mla v14.4s, v2.4s, v27.4s
	VLD1.P 64(R2), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4ebb944f // mla v15.4s, v2.4s, v27.4s
	VLD1.P 64(R3), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4ebb9450 // mla v16.4s, v2.4s, v27.4s
	VLD1.P 64(R4), [V6.B16, V7.B16, V8.B16, V9.B16]
	WORD $0x4f00e402 // movi v2.16b, #0
	WORD $0x4f00e403 // movi v3.16b, #0
	WORD $0x4f00e404 // movi v4.16b, #0
	WORD $0x4f00e405 // movi v5.16b, #0
	WORD $0x4e869542 // sdot v2.4s, v10.16b, v6.16b
	WORD $0x4e879563 // sdot v3.4s, v11.16b, v7.16b
	WORD $0x4e889584 // sdot v4.4s, v12.16b, v8.16b
	WORD $0x4e8995a5 // sdot v5.4s, v13.16b, v9.16b
	WORD $0x4ea3bc42 // addp v2.4s, v2.4s, v3.4s
	WORD $0x4ea5bc84 // addp v4.4s, v4.4s, v5.4s
	WORD $0x4ea4bc42 // addp v2.4s, v2.4s, v4.4s
	WORD $0x4ebb9451 // mla v17.4s, v2.4s, v27.4s
	WORD $0x4eafbdce // addp v14.4s, v14.4s, v15.4s
	WORD $0x4eb1be10 // addp v16.4s, v16.4s, v17.4s
	WORD $0x4eb0bdc2 // addp v2.4s, v14.4s, v16.4s
	WORD $0x4e21d842 // scvtf v2.4s, v2.4s
	FMOVS R15, F0
	WORD $0x1ee24000 // fcvt s0, h0
	FMOVS (R5), F1
	MOVD R5, R10
	ADD R6, R10, R10
	MOVWU (R10), R11
	VMOV R11, V1.S[1]
	ADD R6, R10, R10
	MOVWU (R10), R11
	VMOV R11, V1.S[2]
	ADD R6, R10, R10
	MOVWU (R10), R11
	VMOV R11, V1.S[3]
	WORD $0x4f809021 // fmul v1.4s, v1.4s, v0.s[0]
	WORD $0x4e21cc5c // fmla v28.4s, v2.4s, v1.4s
	ADD $210, R0
	ADD $32, R5
	SUBS $1, R9
	BNE k64block
	MOVD out+48(FP), R10
	VST1 [V28.S4], (R10)
	MOVD $1, R10
	MOVB R10, ret+56(FP)
	RET
k64bad:
	MOVB ZR, ret+56(FP)
	RET
