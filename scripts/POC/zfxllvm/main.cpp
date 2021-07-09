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
        mov = 0x10,
    };
};

auto make_inst(int op, int dst, int lhs, int rhs, bool is_ss) {
    std::vector<uint8_t> res;
    res.push_back(0xc5);
    res.push_back(0xc0 | ~lhs << 3 | is_ss << 1);
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
    auto insts = make_inst(opcode::sub, 0, 0, 0, false);
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
