#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

namespace zfx::x64 {

/*
 * () ~ * + << <= == & ^ | && || ?: = ,
 */

namespace opcode {
    enum {
        mov = 0x10,
        add = 0x58,
        sub = 0x5c,
        mul = 0x59,
        div = 0x5e,
        min = 0x5d,
        max = 0x5f,
        bit_and = 0x54,
        bit_andn = 0x55,
        bit_or = 0x56,
        bit_xor = 0x57,
        sqrt = 0x51,
        rsqrt = 0x52,
        loadu = 0x10,
        loada = 0x28,
        storeu = 0x11,
        storea = 0x29,
        cmp_eq = 0x00c2,
        cmp_ne = 0x04c2,
        cmp_lt = 0x01c2,
        cmp_le = 0x02c2,
        cmp_gt = 0x0ec2,
        cmp_ge = 0x0dc2,
    };
};

namespace jmpcode {
    enum {
        je = 0x04,
        jne = 0x05,
        jl = 0x0c,
        jle = 0x0e,
        jg = 0x0f,
        jge = 0x0d,
    };
};

namespace opreg {
    enum {
        mm0, mm1, mm2, mm3, mm4, mm5, mm6, mm7,
        mm8, mm9, mm10, mm11, mm12, mm13, mm14, mm15,
    };
    enum {
        rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi,
        r8, r9, r10, r11, r12, r13, r14, r15,
    };

    // Linux: https://stackoverflow.com/questions/18024672/what-registers-are-preserved-through-a-linux-x86-64-function-call
    // Windows: https://docs.microsoft.com/en-us/cpp/build/x64-calling-convention?view=msvc-160#parameter-passing
    // TL;DR: Linux use RDI, RSI, RDX, RCX, R8, R9; Windows use RCX, RDX, R8, R9
#if defined(_WIN32)
    enum {
        a1 = rcx,
        a2 = rdx,
        a3 = r8,
        a4 = r9,
    };
#else
    enum {
        a1 = rdi,
        a2 = rsi,
        a3 = rdx,
        a4 = rcx,
        a5 = r8,
        a6 = r9,
    };
#endif
};

namespace memflag {
    enum {
        reg = 0x00,
        reg_reg = 0x04,
        reg_imm8 = 0x40,
        reg_imm32 = 0x80,
        reg_reg_imm8 = 0x44,
        reg_reg_imm32 = 0x84,
    };
};

namespace simdtype {
    enum {
        xmmps = 0x00,
        xmmpd = 0x01,
        xmmss = 0x02,
        xmmsd = 0x03,
        ymmps = 0x04,
        ymmpd = 0x05,
        ymmss = 0x06,
        ymmsd = 0x07,
    };
};

struct SIMDBuilder {   // requires AVX2
    std::vector<uint8_t> res;

    struct MemoryAddress {
        int adr, mflag, immadr, adr2, adr2shift;

        MemoryAddress
        ( int adr
        , int mflag = memflag::reg
        , int immadr = 0
        , int adr2 = 0
        , int adr2shift = 0
        )
        : adr(adr)
        , mflag(mflag)
        , immadr(immadr)
        , adr2(adr2)
        , adr2shift(adr2shift)
        {}

        void dump(std::vector<uint8_t> &res, int val, int flag = 0) {
            if (mflag & (memflag::reg_imm8 | memflag::reg_imm32)) {
                mflag &= ~(memflag::reg_imm8 | memflag::reg_imm32);
                if (-128 <= immadr && immadr <= 127) {
                    mflag |= memflag::reg_imm8;
                } else {
                    mflag |= memflag::reg_imm32;
                }
            }
            auto adreg = adr & 0x07;
            flag |= mflag | val << 3 & 0x38 | adreg;
            //if (adr == opreg::rsp)
                //flag |= 0x10;
            if (adreg == opreg::rbp)
                flag |= memflag::reg_imm8;
            res.push_back(flag);
            if (adr == opreg::rsp)
                res.push_back(0x24);
            if (mflag & memflag::reg_reg) {
                res.push_back(adr2 | adr2shift << 6);
            }
            if (mflag & memflag::reg_imm8) {
                res.push_back(immadr & 0xff);
            } else if (mflag & memflag::reg_imm32) {
                res.push_back(immadr & 0xff);
                res.push_back(immadr >> 8 & 0xff);
                res.push_back(immadr >> 16 & 0xff);
                res.push_back(immadr >> 24 & 0xff);
            }
        }
    };

    static constexpr size_t scalarSizeOfType(int type) {
        switch (type) {
        case simdtype::xmmps: return sizeof(float);
        case simdtype::xmmpd: return sizeof(double);
        case simdtype::xmmss: return sizeof(float);
        case simdtype::xmmsd: return sizeof(double);
        case simdtype::ymmps: return sizeof(float);
        case simdtype::ymmpd: return sizeof(double);
        default: return 0;
        }
    }

    static constexpr size_t sizeOfType(int type) {
        switch (type) {
        case simdtype::xmmps: return 4 * sizeof(float);
        case simdtype::xmmpd: return 2 * sizeof(double);
        case simdtype::xmmss: return 1 * sizeof(float);
        case simdtype::xmmsd: return 1 * sizeof(double);
        case simdtype::ymmps: return 8 * sizeof(float);
        case simdtype::ymmpd: return 4 * sizeof(double);
        default: return 0;
        }
    }

    void addAvxBroadcastLoadOp(int type, int val, MemoryAddress adr) {
        res.push_back(0xc4);
        res.push_back(0x62 | ~val >> 3 << 7);
        res.push_back(0x79 | type & 0x04);
        res.push_back(0x18 | type & 0x03);
        adr.dump(res, val);
    }

    void addAvxRoundOp(int type, int dst, int src, int opid) {
        res.push_back(0xc4);
        res.push_back(0x43 | ~dst >> 3 << 7 | (~src >> 3 & 1) << 5);
        res.push_back(0x79 | type & 0x04);
        res.push_back(0x09 | type & 0x01);
        res.push_back(0xc0 | dst << 3 & 0x38 | src);
        res.push_back(opid);
    }

    void addAvxMemoryOp(int type, int op, int val, MemoryAddress adr) {
        res.push_back(0xc5);
        res.push_back(type | 0x78 | ~val >> 3 << 7);
        res.push_back(op);
        adr.dump(res, val);
    }

    void addRegularLoadOp(int val, MemoryAddress adr) {
        res.push_back(0x48 | val >> 3);
        res.push_back(0x8b);
        adr.dump(res, val);
    }

    void addRegularStoreOp(int val, MemoryAddress adr) {
        res.push_back(0x48 | val >> 3);
        res.push_back(0x89);
        adr.dump(res, val);
    }

    void addRegularMoveOp(int dst, int src) {
        res.push_back(0x48 | dst >> 3 | src >> 1 & 0x04);
        res.push_back(0x89);
        res.push_back(0xc0 | dst & 0x07 | src << 3 & 0x38);
    }

    void addAdjStackTop(int imm_add) {
        res.push_back(0x48);
        res.push_back(0x83);
        res.push_back(0xc4);
        res.push_back(imm_add);
    }

    void addCallOp(MemoryAddress adr) {
        if (adr.adr & 0x08)
            res.push_back(0x41);
        res.push_back(0xff);
        adr.dump(res, 0, 0x10);
    }

    void addAvxBinaryOp(int type, int op, int dst, int lhs, int rhs) {
        if (rhs >= 8) {
            res.push_back(0xc4);
            res.push_back(0x41 | ~dst >> 3 << 7);
            res.push_back(type | ~lhs << 3 & 0x78);
        } else {
            res.push_back(0xc5);
            res.push_back(type | ~lhs << 3 & 0x78 | ~dst >> 3 << 7);
        }
        res.push_back(op & 0xff);
        res.push_back(0xc0 | dst << 3 & 0x38 | rhs & 0x07);
        if ((op & 0xff) == opcode::cmp_eq) {
           res.push_back(op >> 8);
        }
    }

    void addAvxUnaryOp(int type, int op, int dst, int src) {
        addAvxBinaryOp(type, op, dst, opreg::mm0, src);
    }

    void addAvxBlendvOp(int type, int dst, int lhs, int rhs, int mask) {
        res.push_back(0xc4);
        res.push_back(0x43 | ~dst >> 3 << 7 | (~rhs >> 3 & 1) << 5);
        res.push_back(0x01 | type & 0x04 | ~lhs << 3 & 0x78);
        res.push_back(0x4a | type & 0x01);
        res.push_back(0xc0 | dst << 3 & 0x38 | rhs & 0x07);
        res.push_back(mask << 4);
    }

    void addAvxMoveOp(int type, int dst, int src) {
        addAvxBinaryOp(opcode::loadu, opcode::mov, dst, opreg::mm0, src);
    }

    void addJumpOp(int off) {
        off -= 4;
        if (-128 <= off && off <= 127) {
            res.push_back(0xeb);
            res.push_back(off & 0xff);
        } else {
            off -= 6;
            res.push_back(0xe9);
            res.push_back(off);
        }
    }

    void addPushReg(int reg) {
        if (reg & 0x08)
            res.push_back(0x41);
        res.push_back(0x50 | reg & 0x7);
    }

    void addPopReg(int reg) {
        if (reg & 0x08)
            res.push_back(0x41);
        res.push_back(0x58 | reg & 0x7);
    }

    void addReturn() {
        res.push_back(0xc3);
    }

    void printHexCode() const {
        for (auto const &i: res) {
            printf("%02X", i);
        }
        printf("\n");
    }

    auto const &getResult() const {
        return res;
    }
};

}
