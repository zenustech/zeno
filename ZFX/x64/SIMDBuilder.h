#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <sys/mman.h>
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
        sqrt = 0x51,
        loadu = 0x10,
        loada = 0x28,
        storeu = 0x11,
        storea = 0x29,
    };
};

namespace opreg {
    enum {
        mm0, mm1, mm2, mm3, mm4, mm5, mm6, mm7,
    };
    enum {
        rax, rcx, rdx, rbx,
    };
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

namespace optype {
    enum {
        xmmps = 0xc0,
        xmmpd = 0xc1,
        xmmss = 0xc2,
        xmmsd = 0xc3,
        ymmps = 0xc4,
        ymmpd = 0xc5,
    };
};

class SIMDBuilder {   // requires AVX2
private:
    std::vector<uint8_t> res;

public:
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

        void dump(std::vector<uint8_t> &res, int val) {
            res.push_back(mflag | val << 3 | adr);
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
        case optype::xmmps: return sizeof(float);
        case optype::xmmpd: return sizeof(double);
        case optype::xmmss: return sizeof(float);
        case optype::xmmsd: return sizeof(double);
        case optype::ymmps: return sizeof(float);
        case optype::ymmpd: return sizeof(double);
        default: return 0;
        }
    }

    static constexpr size_t sizeOfType(int type) {
        switch (type) {
        case optype::xmmps: return 4 * sizeof(float);
        case optype::xmmpd: return 2 * sizeof(double);
        case optype::xmmss: return 1 * sizeof(float);
        case optype::xmmsd: return 1 * sizeof(double);
        case optype::ymmps: return 8 * sizeof(float);
        case optype::ymmpd: return 4 * sizeof(double);
        default: return 0;
        }
    }

    void addAvxBroadcastLoadOp(int type, int val, MemoryAddress adr) {
        res.push_back(0xc4);
        res.push_back(0xe2);
        res.push_back(0x79 | type << 2 & 0x04);
        res.push_back(0x18 | type >> 2 & 0x01);
        adr.dump(res, val);
    }

    void addAvxMemoryOp(int type, int op, int val, MemoryAddress adr) {
        res.push_back(0xc5);
        res.push_back(type | 0x38);
        res.push_back(op);
        adr.dump(res, val);
    }

    void addRegularLoadOp(int val, MemoryAddress adr) {
        res.push_back(0x48);
        res.push_back(0x8b);
        adr.dump(res, val);
    }

    void addRegularStoreOp(int val, MemoryAddress adr) {
        res.push_back(0x48);
        res.push_back(0x89);
        adr.dump(res, val);
    }

    void addAvxBinaryOp(int type, int op, int dst, int lhs, int rhs) {
        res.push_back(0xc5);
        res.push_back(type | ~lhs << 3);
        res.push_back(op);
        res.push_back(0xc0 | dst << 3 | rhs);
    }

    void addAvxUnaryOp(int type, int op, int dst, int src) {
        addAvxBinaryOp(type, op, dst, 0, src);
    }

    void addAvxMoveOp(int dst, int src) {
        addAvxBinaryOp(optype::xmmss, opcode::mov, dst, dst, src);
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
