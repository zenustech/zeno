#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <sys/mman.h>
#include <vector>

std::vector<uint8_t> insts = {
    0xC5, 0xFA, 0x10, 0xC1, 0xC3,
};

int main() {
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
