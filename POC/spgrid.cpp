#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <tuple>
#include <vector>
#include <cassert>
#include <sys/mman.h>
#include <omp.h>

using std::cout;
using std::endl;
#define show(x) (cout << #x "=" << (x) << endl)


static size_t expandbits3d(size_t v) {
    v = (v * 0x0000000100000001ul) & 0xFFFF00000000FFFFul;
    v = (v * 0x0000000000010001ul) & 0x00FF0000FF0000FFul;
    v = (v * 0x0000000000000101ul) & 0xF00F00F00F00F00Ful;
    v = (v * 0x0000000000000011ul) & 0x30C30C30C30C30C3ul;
    v = (v * 0x0000000000000005ul) & 0x4924924949249249ul;
    return v;
}

static size_t morton3d(size_t x, size_t y, size_t z) {
    x = expandbits3d(x);
    y = expandbits3d(y);
    z = expandbits3d(z);
    return x | (y << 1) | (z << 2);
}

template <size_t NChannels, size_t NElmsize>
struct SPLayout {
};

template <>
struct SPLayout<16, 4> {
    static size_t linearize(size_t c, size_t i, size_t j, size_t k) {
        size_t t = (i & 3) | ((j & 3) << 2) | ((k & 3) << 4) | ((c & 15) << 6);
        size_t m = morton3d(i >> 2, j >> 2, k >> 2);
        return (t << 2) | (m << 12);
    }
};


template <size_t NRes, size_t NChannels, size_t NElmsize>
struct SPGrid {
    void *ptr;
    static constexpr size_t size = NRes * NRes * NRes * NChannels * NElmsize;

    SPGrid() {
        ptr = ::mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        auto s = size;
        show(s);
    }

    void *pointer(size_t c, size_t i, size_t j, size_t k) const {
        size_t offset = SPLayout<NChannels, NElmsize>::linearize(c, i, j, k);
        return static_cast<void *>(static_cast<char *>(ptr) + offset);
    }

    ~SPGrid() {
        ::munmap(ptr, size);
        ptr = nullptr;
    }
};


int main(void)
{
    SPGrid<256, 16, 4> f;

    auto t0 = std::chrono::steady_clock::now();

    *(float *)f.pointer(0, 128, 128, 128) = 3.14;

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    cout << ms << " ms" << endl;
    return 0;
}
