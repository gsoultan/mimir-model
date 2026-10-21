# Generates poolF32: rows combined in groups of 4 and singly, and, for
# rows of at most 8 outputs, two rows to a vector.

class Addr:
    """x row addresses off DX by multiples of R13, R10 as scratch."""
    def __init__(self):
        self.r10 = None  # multiple of R13 that R10 holds beyond DX
    def row(self, m, out):
        if m == 0: return "(DX)"
        if m in (1, 2, 4): return f"(DX)(R13*{m})"
        if m == 8:
            out.append("\tLEAQ (DX)(R13*4), R10\n"); self.r10 = 4
            return "(R10)(R13*4)"
        for base, mul in ((m-1, 1), (m-2, 2), (m-4, 4)):
            if base > 0 and base == self.r10:
                return f"(R10)(R13*{mul})"
        # set R10 = DX + (m-1)*R13
        b = m - 1
        if b in (1, 2, 4):
            out.append(f"\tLEAQ (DX)(R13*{b}), R10\n")
        elif self.r10 is not None and b - self.r10 in (1, 2, 4):
            out.append(f"\tLEAQ (R10)(R13*{b - self.r10}), R10\n")
        else:
            out.append(f"\tLEAQ (DX)(R13*{b-2}), R10\n\tLEAQ (R10)(R13*2), R10\n")
        self.r10 = b
        return "(R10)(R13*1)"

def loads(A, pair, L):
    out = []
    out.append(f"\tCMPQ BX, $1\n\tJNE  {L}even\n")
    ad = Addr()
    for a in range(A):
        if pair:
            # Each row's address before the next one's, which may move R10.
            r0 = ad.row(2*a, out)
            out.append(f"\tVMOVUPS.Z {r0}, K1, Y{4*a+1}\n")
            r1 = ad.row(2*a+1, out)
            out.append(f"\tVMOVUPS.Z {r1}, K1, Y{4*a+2}\n\tVINSERTF32X8 $1, Y{4*a+2}, Z{4*a+1}, Z{4*a+1}\n")
        else:
            out.append(f"\tVMOVUPS.Z {ad.row(a, out)}, K1, Z{4*a+1}\n")
    out.append(f"\tJMP  {L}loaded\n\n{L}even:\n")
    ad = Addr()
    for a in range(A):
        v, b, t = 4*a+1, 4*a+2, 4*a+3
        if pair:
            r0 = ad.row(2*a, out)
            out.append(f"\tVMOVUPS.Z {r0}, K2, Z{v}\n")
            r1 = ad.row(2*a+1, out)
            out.append(f"\tVMOVUPS.Z {r1}, K2, Z{b}\n")
        else:
            r0 = ad.row(a, out)
            out.append(f"\tVMOVUPS.Z {r0}, K2, Z{v}\n\tVMOVUPS.Z 64{r0}, K3, Z{b}\n")
        out.append(f"\tVMOVDQA32 Z30, Z{t}\n\tVPERMI2PS Z{b}, Z{v}, Z{t} // Z{t} = the even lanes of Z{v}:Z{b}\n\tVMOVAPS Z{t}, Z{v}\n")
    out.append(f"\n{L}loaded:\n")
    return "".join(out)

def combine(A, L):
    out = [f"\tCMPQ R15, R14\n\tJNE  {L}acc\n"]
    for a in range(A):
        out.append(f"\tVMOVAPS Z{4*a+1}, Z{4*a}\n")
    out.append(f"\tJMP  {L}next\n\n{L}acc:\n\tCMPQ AX, $1\n\tJEQ  {L}mul\n\tJGT  {L}max\n")
    for a in range(A):
        out.append(f"\tVADDPS Z{4*a+1}, Z{4*a}, Z{4*a}\n")
    out.append(f"\tJMP  {L}next\n\n{L}mul:\n")
    for a in range(A):
        out.append(f"\tVMULPS Z{4*a+1}, Z{4*a}, Z{4*a}\n")
    out.append(f"\tJMP  {L}next\n\n{L}max:\n")
    for a in range(A):
        c, v, t = 4*a, 4*a+1, 4*a+2
        out.append(f"\tVCMPPS $0, Z{v}, Z{c}, K4\n\tVMAXPS Z{c}, Z{v}, Z{t}\n\tVANDPS Z{v}, Z{c}, K4, Z{t}\n\tVCMPPS $3, Z{v}, Z{v}, K4\n\tVMOVAPS Z31, K4, Z{t}\n\tVMOVAPS Z{t}, Z{c}\n")
    out.append(f"\n{L}next:\n")
    return "".join(out)

def group(A, pair, L):
    R = A * (2 if pair else 1)  # rows per pass
    s = f"""{L}chunks:
	MOVQ R11, CX // elements left in the row
	KXNORW K1, K1, K1
	KXNORW K2, K2, K2
	KMOVW full2-32(SP), K3

{L}loop:
	CMPQ CX, $16
	JGE  {L}body
	TESTQ CX, CX
	JZ   {L}rowend
	KMOVW ytail-24(SP), K1
	KMOVW xtail-16(SP), K2
	KMOVW xtail2-8(SP), K3
	MOVQ $16, CX

{L}body:
	MOVQ R9, R8
	MOVQ R14, R15

{L}tap:
	MOVQ (R8), R10
	LEAQ (SI)(R10*4), DX
{loads(A, pair, L)}{combine(A, L)}	ADDQ $8, R8
	DECQ R15
	JNZ  {L}tap
	// With taps combined, any NaN is the canonical one.
	CMPQ R14, $1
	JEQ  {L}store
"""
    for a in range(A):
        s += f"\tVCMPPS $3, Z{4*a}, Z{4*a}, K4\n\tVMOVAPS Z31, K4, Z{4*a}\n"
    s += f"\n{L}store:\n"
    if R > 1:
        s += "\tMOVQ yrs4-40(SP), R10\n\tMOVQ DI, R8\n"
    for a in range(A):
        if pair:
            s += f"\tVMOVUPS Y{4*a}, K1, (R8)\n\tADDQ R10, R8\n\tVEXTRACTF32X8 $1, Z{4*a}, Y{4*a+2}\n\tVMOVUPS Y{4*a+2}, K1, (R8)\n"
        else:
            s += f"\tVMOVUPS Z{4*a}, K1, {'(R8)' if R > 1 else '(DI)'}\n"
        if a < A - 1:
            s += "\tADDQ R10, R8\n"
    s += f"""	ADDQ $64, DI
	ADDQ xchunk-48(SP), SI
	SUBQ $16, CX
	JMP  {L}loop

{L}rowend:
	// y and x moved on by the row's chunks of 16: y back to the row's
	// end and on to the next row, x back to the row's start and on by
	// the rows done.
	MOVQ R11, CX
	ANDQ $15, CX
	JZ   {L}rows
	SUBQ $64, DI
	LEAQ (DI)(CX*4), DI

{L}rows:
	ADDQ ynext-56(SP), DI
	SUBQ xback-64(SP), SI
"""
    for _ in range(R - 1):
        s += "\tADDQ yrs4-40(SP), DI\n\tADDQ R13, SI\n"
    s += f"\tADDQ R13, SI\n\tSUBQ ${R}, R12\n\tJMP  poolrows\n\n"
    return s

head = """// func poolF32(op int, y, x *float32, offs *int, taps, w, rows, stride, xrs, yrs int, nan float32)
//
// rows rows of w outputs, y[r*yrs+j] = the op over the taps t of
// x[offs[t]+r*xrs+j*stride], op as for binF32 and applied in order.
// stride is 1 or 2. Rows go four at a time while there are four, each
// in its own accumulator, so the taps of one row need not wait on each
// other; rows of at most 8 outputs go two to a vector, eight at a time
// while there are eight. Generated by gen/pool.py.
TEXT ·poolF32(SB), NOSPLIT, $64-84
	MOVQ op+0(FP), AX
	MOVQ y+8(FP), DI
	MOVQ x+16(FP), SI
	MOVQ offs+24(FP), R9
	MOVQ taps+32(FP), R14
	MOVQ w+40(FP), R11
	MOVQ rows+48(FP), R12
	MOVQ stride+56(FP), BX
	MOVQ xrs+64(FP), R13
	SHLQ $2, R13
	VBROADCASTSS nan+80(FP), Z31
	VMOVDQU32 evens<>(SB), Z30
	// The lanes of a row's last chunk: of y, and of the two source loads
	// of stride 2, the last even lane and below.
	MOVQ R11, CX
	ANDQ $15, CX
	MOVL $1, R8
	SHLL CX, R8
	DECL R8
	MOVQ R8, ytail-24(SP)
	MOVL $1, R8
	SHLL CX, R8
	SHLL CX, R8
	SHRL $1, R8
	DECL R8
	MOVL R8, R10
	ANDL $0xffff, R10
	MOVQ R10, xtail-16(SP)
	SHRL $16, R8
	MOVQ R8, xtail2-8(SP)
	MOVQ $0x7fff, R8 // a full chunk of stride 2: lanes 0 to 30
	MOVQ R8, full2-32(SP)
	// The row strides of y in bytes; from a row's end in y to the next
	// row's start; a chunk's source, 16 or 32 elements, and a row's.
	MOVQ yrs+72(FP), R8
	SHLQ $2, R8
	MOVQ R8, yrs4-40(SP)
	MOVQ yrs+72(FP), R8
	SUBQ R11, R8
	SHLQ $2, R8
	MOVQ R8, ynext-56(SP)
	MOVQ BX, R8
	SHLQ $6, R8
	MOVQ R8, xchunk-48(SP)
	MOVQ R11, R8
	ADDQ $15, R8
	ANDQ $-16, R8
	IMULQ BX, R8
	SHLQ $2, R8
	MOVQ R8, xback-64(SP)

poolrows:
	TESTQ R12, R12
	JZ   pooldone
	CMPQ R11, $8
	JG   poolwide
	CMPQ R12, $8
	JGE  q4chunks
	CMPQ R12, $2
	JGE  q1chunks
	JMP  p1chunks

poolwide:
	CMPQ R12, $4
	JGE  p4chunks
	JMP  p1chunks

"""
tail = """pooldone:
	VZEROUPPER
	RET

"""
print(head + group(4, False, "p4") + group(1, False, "p1") + group(4, True, "q4") + group(1, True, "q1") + tail, end="")
