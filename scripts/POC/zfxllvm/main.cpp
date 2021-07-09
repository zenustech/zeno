#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <sys/mman.h>
#include <vector>

/*
 * () ~ * + << <= == & ^ | && || ?: = ,
 */

namespace opcode {
    enum {
        add = 0x58,
        sub = 0x5c,
        mul = 0x59,
        div = 0x5e,
        mov = 0x10,
        sqrt = 0x51,
        loadu = 0x11,
        loada = 0x29,
        storeu = 0x10,
        storea = 0x28,
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

auto make_rw_inst(int op, int val, int adr, int mflag, int type,
    int adr2 = 0, int adr2shift = 0, int immadr = 0) {
    std::vector<uint8_t> res;
    res.push_back(0xc5);
    res.push_back(type | 0x38);
    res.push_back(op);
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
    return res;
}

auto make_inst(int op, int dst, int lhs, int rhs, int type) {
    std::vector<uint8_t> res;
    res.push_back(0xc5);
    res.push_back(type | ~lhs << 3);
    res.push_back(op);
    res.push_back(0xc0 | dst << 3 | lhs);
    return res;
}

auto print_inst(std::vector<uint8_t> const &inst) {
    for (auto const &i: inst) {
        printf("%02X", i);
    }
    printf("\n");
}

int main() {
    auto insts = make_inst(opcode::sqrt, 0, 0, 0, optype::xmmps);
    //auto insts = make_rw_inst(opcode::loadu, opreg::mm0,
    //    opreg::rax, memflag::reg, optype::xmmps);
    print_inst(insts);

    insts.push_back(0xc3);
    size_t memsize = (insts.size() + 4095) / 4096 * 4096;

    uint8_t *mem = (uint8_t *)mmap(NULL, memsize,
        PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS,
        -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap");
        return -1;
    }
    for (size_t i = 0; i < insts.size(); i++) {
        mem[i] = insts[i];
    }
    auto entry = (void (*)())mem;
    entry();

    munmap(mem, memsize);

    return 0;
}
