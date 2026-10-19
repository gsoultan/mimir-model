//go:build amd64 && !purego

#include "textflag.h"

// AVX-512 kernel for element-wise float32 operations (elem.go).

// func binF32(op int, y, x, z *float32, n, scalar int, nan float32)
//
// y[j] = x[j] op z[j] (z[0] when scalar != 0) for j < n, op 0 add, 1
// multiply, 2 maximum. An add or multiply whose result is NaN gives nan.
// The maximum is nan when either operand is NaN; when the operands are
// equal (including +0 and -0) it is their bitwise and, which is either
// of them for equal nonzero values, -0 for two -0 and +0 otherwise;
// otherwise the larger.
TEXT ·binF32(SB), NOSPLIT, $0-52
	MOVQ op+0(FP), AX
	MOVQ y+8(FP), DI
	MOVQ x+16(FP), SI
	MOVQ z+24(FP), DX
	MOVQ n+32(FP), CX
	MOVQ scalar+40(FP), BX
	VBROADCASTSS nan+48(FP), Z31
	VBROADCASTSS (DX), Z30
	MOVQ $16, R9
	TESTQ BX, BX
	JZ   stride
	XORQ R9, R9 // z does not advance
stride:
	SHLQ $2, R9
	KXNORW K1, K1, K1 // all 16 lanes

loop:
	CMPQ CX, $16
	JGE  body
	TESTQ CX, CX
	JZ   done
	// The last n < 16 floats: lanes below n.
	MOVL $1, R8
	SHLL CX, R8
	DECL R8
	KMOVW R8, K1
	MOVQ $16, CX

body:
	VMOVUPS.Z (SI), K1, Z0
	VMOVAPS  Z30, Z1
	TESTQ BX, BX
	JNZ  op
	VMOVUPS.Z (DX), K1, Z1

op:
	CMPQ AX, $1
	JEQ  mul
	JGT  max
	VADDPS Z1, Z0, Z2
	JMP  fixnan

mul:
	VMULPS Z1, Z0, Z2

fixnan:
	VCMPPS $3, Z2, Z2, K2 // unordered: the result is NaN
	VMOVAPS Z31, K2, Z2
	JMP  store

max:
	VMAXPS Z1, Z0, Z2
	VCMPPS $0, Z1, Z0, K2 // equal
	VANDPS Z1, Z0, K2, Z2
	VCMPPS $3, Z1, Z0, K2 // either is NaN
	VMOVAPS Z31, K2, Z2

store:
	VMOVUPS Z2, K1, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	ADDQ R9, DX
	SUBQ $16, CX
	JMP  loop

done:
	VZEROUPPER
	RET
