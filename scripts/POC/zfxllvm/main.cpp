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
