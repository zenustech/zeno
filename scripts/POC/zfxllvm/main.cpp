#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <sys/mman.h>
#include <vector>
#include <tuple>

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

class ExecutableArena {
protected:
    uint8_t *mem;
    size_t memsize;

public:
    explicit ExecutableArena(size_t memsize_) : memsize(memsize_) {
        mem = (uint8_t *)mmap(NULL, memsize,
            PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS,
            -1, 0);
        if (mem == MAP_FAILED) {
            perror("mmap");
            abort();
        }
    }

    ~ExecutableArena() {
        munmap(mem, memsize);
    }
};

class ExecutableInstance : public ExecutableArena {
public:
    explicit ExecutableInstance(std::vector<uint8_t> const &insts)
        : ExecutableArena((insts.size() + 4095) / 4096 * 4096) {
        for (size_t i = 0; i < insts.size(); i++) {
            mem[i] = insts[i];
        }
    }

    ExecutableInstance(ExecutableInstance const &) = delete;

    inline void operator()() {
        auto entry = (void (*)())this->mem;
        entry();
    }

    inline std::tuple
        < uintptr_t
        , uintptr_t
        , uintptr_t
        > operator()
        ( uintptr_t rax
        , uintptr_t rcx
        , uintptr_t rdx
        ) {
        auto entry = (void (*)())this->mem;
        asm volatile (
            "call *%6"
            : "=a" (rax), "=c" (rcx), "=d" (rdx)
            : "a" (rax), "c" (rcx), "d" (rdx)
            , "" (entry)
            : "cc", "memory"
            );
        return {rax, rcx, rdx};
    }
};

int main() {
    SIMDBuilder builder;
    builder.addRegularLoadOp(opreg::rax, opreg::rdx);
    builder.addAvxMemoryOp(optype::xmmps, opcode::loadu, opreg::mm0, opreg::rax);
    builder.addAvxUnaryOp(optype::xmmps, opcode::sqrt, opreg::mm0, opreg::mm0);
    builder.addAvxBroadcastLoadOp(optype::xmmss, opreg::mm1, opreg::rcx);
    builder.addAvxBinaryOp(optype::xmmps, opcode::mul, opreg::mm0, opreg::mm0, opreg::mm1);
    builder.addAvxMemoryOp(optype::xmmps, opcode::storeu, opreg::mm0,
        {opreg::rax, memflag::reg_imm8, 16});
    builder.printHexCode();
    builder.addReturn();

    std::vector<float> arr(8);
    arr[0] = 1.618f;
    arr[1] = 3.141f;
    arr[2] = 2.718f;
    arr[3] = 2.000f;
    float *ptr = arr.data();
    float scale = 0.500f;

    ExecutableInstance instance(builder.getResult());
    instance(0, (uintptr_t)&scale, (uintptr_t)&ptr);

    printf("%f %f %f %f\n", arr[0], arr[1], arr[2], arr[3]);
    printf("%f %f %f %f\n", arr[4], arr[5], arr[6], arr[7]);

    return 0;
}
