#include <array>
#include <cstring>
#include <sys/mman.h>

namespace spgrid {

static_assert(sizeof(void *) == 8, "SPGrid requies 64-bit architecture");

static void *allocate(size_t size) {
    return ::mmap(0, size, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
}

static void deallocate(void *ptr, size_t size) {
    ::munmap(ptr, size);
}


static constexpr size_t expandbits3d(size_t v) {
    v = (v * 0x0000000100000001ul) & 0xFFFF00000000FFFFul;
    v = (v * 0x0000000000010001ul) & 0x00FF0000FF0000FFul;
    v = (v * 0x0000000000000101ul) & 0xF00F00F00F00F00Ful;
    v = (v * 0x0000000000000011ul) & 0x30C30C30C30C30C3ul;
    v = (v * 0x0000000000000005ul) & 0x4924924949249249ul;
    return v;
}

static constexpr size_t morton3d(size_t x, size_t y, size_t z) {
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

template <>
struct SPLayout<4, 4> {
    static size_t linearize(size_t c, size_t i, size_t j, size_t k) {
        size_t t = (i & 7) | ((j & 7) << 3) | ((k & 3) << 6) | ((c & 3) << 8);
        size_t m = morton3d(i >> 3, j >> 3, k >> 2);
        t = (t << 2) | (m << 12);
        return t;
    }
};

template <>
struct SPLayout<1, 4> {
    static size_t linearize(size_t c, size_t i, size_t j, size_t k) {
        size_t t = (i & 15) | ((j & 7) << 4) | ((k & 7) << 7);
        size_t m = morton3d(i >> 4, j >> 3, k >> 3);
        return (t << 2) | (m << 12);
    }
};

template <>
struct SPLayout<0, 0> {
    static size_t linearize(size_t c, size_t i, size_t j, size_t k) {
        size_t t = ((i >> 3) & 3) | ((j & 31) << 2) | ((k & 31) << 7);
        size_t m = morton3d(i >> 5, j >> 5, k >> 5);
        return t | (m << 12);
    }

    static size_t bit_linearize(size_t c, size_t i, size_t j, size_t k) {
        return i & 7;
    }
};


template <size_t NRes, size_t NChannels, size_t NElmsize>
struct SPGrid {
    using LayoutClass = SPLayout<NChannels, NElmsize>;

    void *ptr;
    static constexpr size_t size =
        NRes * NRes * NRes * NChannels * NElmsize;

    SPGrid() {
        ptr = allocate(size);
    }

    void *pointer(size_t c, size_t i, size_t j, size_t k) const {
        size_t offset = LayoutClass::linearize(c, i, j, k);
        return static_cast<void *>(static_cast<char *>(ptr) + offset);
    }

    ~SPGrid() {
        deallocate(ptr, size);
        ptr = nullptr;
    }
};

template <size_t NRes, size_t NChannels, typename T>
struct SPTypedGrid : SPGrid<NRes, NChannels, sizeof(T)> {
    using ValType = std::array<T, NChannels>;

    T &at(size_t c, size_t i, size_t j, size_t k) const {
        return *(T *)this->pointer(c, i, j, k);
    }

    auto get(size_t i, size_t j, size_t k) const {
        ValType ret;
        for (size_t c = 0; c < NChannels; c++) {
            ret[c] = at(c, i, j, k);
        }
        return ret;
    }

    void set(size_t i, size_t j, size_t k, T const &val) const {
        for (size_t c = 0; c < NChannels; c++) {
            at(c, i, j, k) = val[c];
        }
    }
};

template <size_t NRes, typename T>
struct SPTypedGrid<NRes, 1, T> : SPGrid<NRes, 1, sizeof(T)> {
    using ValType = T;

    T &at(size_t i, size_t j, size_t k) const {
        return *(T *)this->pointer(0, i, j, k);
    }

    auto get(size_t i, size_t j, size_t k) const {
        return at(i, j, k);
    }

    void set(size_t i, size_t j, size_t k, T const &val) const {
        at(i, j, k) = val;
    }
};


template <size_t NRes>
using SPFloatGrid = SPTypedGrid<NRes, 1, float>;
template <size_t NRes>
using SPFloat4Grid = SPTypedGrid<NRes, 4, float>;
template <size_t NRes>
using SPFloat16Grid = SPTypedGrid<NRes, 16, float>;

template <size_t NRes>
struct SPBooleanGrid : SPGrid<NRes, 1, 0> {
    using ValType = bool;

    unsigned char &uchar_at(size_t i, size_t j, size_t k) {
        return *(unsigned char *)this->pointer(0, i, j, k);
    }

    bool get(size_t i, size_t j, size_t k) {
        return uchar_at(i, j, k) & (1 << (i & 7));
    }

    void set_true(size_t i, size_t j, size_t k) {
        uchar_at(i, j, k) |= (1 << (i & 7));
    }

    void set_false(size_t i, size_t j, size_t k) {
        uchar_at(i, j, k) &= ~(1 << (i & 7));
    }

    void set(size_t i, size_t j, size_t k, bool value) {
        if (value)
            set_true(i, j, k);
        else
            set_false(i, j, k);
    }
};

}
