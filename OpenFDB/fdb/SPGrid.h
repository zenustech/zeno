#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#include <tchar.h>
#else
#include <sys/mman.h>
#endif
#include "vec.h"

namespace fdb::spgrid {

static_assert(sizeof(void *) == 8, "SPGrid requies 64-bit architecture");

static void *allocate(size_t size) {
#if defined(_WIN32)
    void *ptr = ::VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE);
    if (!ptr) {
        printf("VirtualAlloc failed!\n");
        exit(-1);
    }
#else
    void *ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        abort();
    }
#endif
    return ptr;
}

static void deallocate(void *ptr, size_t size) {
#if defined(_WIN32)
    ::VirtualFree(ptr, 0, MEM_RELEASE);
#else
    ::munmap(ptr, size);
#endif
}

static void reload_page(void *ptr) {
#if defined(_WIN32)
    ::VirtualAllocEx(GetCurrentProcess(), ptr,
        4096, MEM_COMMIT, PAGE_READWRITE);
#else
    ::mmap(ptr, 4096, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
#endif
}

static void unload_page(void *ptr) {
#if defined(_WIN32)
    ::VirtualFreeEx(GetCurrentProcess(), ptr, 4096, MEM_DECOMMIT);
#else
    ::munmap(ptr, 4096);
#endif
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
    static constexpr size_t Log2ScaleX = 0;
    static constexpr size_t Log2ScaleY = 0;
    static constexpr size_t Log2ScaleZ = 0;

    static constexpr size_t linearize(size_t c, size_t i, size_t j, size_t k) {
        size_t t = (i & 3) | ((j & 3) << 2) | ((k & 3) << 4) | ((c & 15) << 6);
        size_t m = morton3d(i >> 2, j >> 2, k >> 2);
        return (t << 2) | (m << 12);
    }
};

template <>
struct SPLayout<4, 4> {
    static constexpr size_t Log2ScaleX = 1;
    static constexpr size_t Log2ScaleY = 1;
    static constexpr size_t Log2ScaleZ = 0;

    static constexpr size_t linearize(size_t c, size_t i, size_t j, size_t k) {
        size_t t = (i & 7) | ((j & 7) << 3) | ((k & 3) << 6) | ((c & 3) << 8);
        size_t m = morton3d(i >> 3, j >> 3, k >> 2);
        return (t << 2) | (m << 12);
    }
};

template <>
struct SPLayout<1, 4> {
    static constexpr size_t Log2ScaleX = 1;
    static constexpr size_t Log2ScaleY = 0;
    static constexpr size_t Log2ScaleZ = 0;

    static constexpr size_t linearize(size_t c, size_t i, size_t j, size_t k) {
        size_t t = (i & 15) | ((j & 7) << 4) | ((k & 7) << 7);
        size_t m = morton3d(i >> 4, j >> 3, k >> 3);
        return (t << 2) | (m << 12);
    }
};


template <size_t Log2Res, size_t NChannels, size_t NElmsize>
struct SPUntypedGrid {
    using LayoutClass = SPLayout<NChannels, NElmsize>;

    void *m_ptr;
    static constexpr size_t Log2ResX = Log2Res + LayoutClass::Log2ScaleX;
    static constexpr size_t Log2ResY = Log2Res + LayoutClass::Log2ScaleY;
    static constexpr size_t Log2ResZ = Log2Res + LayoutClass::Log2ScaleZ;
    static constexpr size_t NumChannels = NChannels;
    static constexpr size_t ElementSize = NElmsize;
    static constexpr size_t MemorySize =
        (1ul << Log2ResX + Log2ResY + Log2ResZ) * NumChannels * ElementSize;

    SPUntypedGrid() {
        m_ptr = allocate(MemorySize);
    }

    ~SPUntypedGrid() {
        deallocate(m_ptr, MemorySize);
        m_ptr = nullptr;
    }

    SPUntypedGrid(SPGrid const &) = delete;
    SPUntypedGrid(SPGrid &&) = default;
    SPUntypedGrid &operator=(SPGrid const &) = delete;
    SPUntypedGrid &operator=(SPGrid &&) = default;

    void *address(size_t c, vec3i ijk) const {
        size_t i = ijk[0] & ((1l << Log2ResX) - 1);
        size_t j = ijk[1] & ((1l << Log2ResY) - 1);
        size_t k = ijk[2] & ((1l << Log2ResZ) - 1);
        size_t offset = LayoutClass::linearize(c, i, j, k);
        return static_cast<void *>(static_cast<char *>(m_ptr) + offset);
    }
};

template <size_t Log2Res, size_t NElmsize>
struct SPUntypedGrid<Log2Res, 3, NElmsize> : SPGrid<Log2Res, 4, NElmsize> {
};

template <size_t Log2Res, size_t NChannels, typename T>
struct SPGrid : SPUntypedGrid<Log2Res, NChannels, sizeof(T)> {
    using ValueType = vec<T, NChannels>;

    T &at(size_t c, vec3i ijk) const {
        return *(T *)this->address(c, ijk);
    }

    auto get(vec3i ijk) const {
        ValueType ret;
        for (size_t c = 0; c < NChannels; c++) {
            ret[c] = at(c, ijk);
        }
        return ret;
    }

    void set(vec3i ijk, ValueType const &val) const {
        for (size_t c = 0; c < NChannels; c++) {
            at(c, ijk) = val[c];
        }
    }
};

template <size_t Log2Res, typename T>
struct SPGrid<Log2Res, 1, T> : SPUntypedGrid<Log2Res, 1, sizeof(T)> {
    using ValueType = T;

    T &at(vec3i ijk) const {
        return *(T *)this->address(0, ijk);
    }

    T &at(size_t c, vec3i ijk) const {
        return at(ijk);
    }

    auto get(vec3i ijk) const {
        return at(ijk);
    }

    void set(vec3i ijk, T const &val) const {
        at(ijk) = val;
    }
};


template <size_t Log2Res>
using SPFloatGrid = SPGrid<Log2Res, 1, float>;
template <size_t Log2Res>
using SPFloat3Grid = SPGrid<Log2Res, 3, float>;
template <size_t Log2Res>
using SPFloat4Grid = SPGrid<Log2Res, 4, float>;
template <size_t Log2Res>
using SPFloat16Grid = SPGrid<Log2Res, 16, float>;


template <size_t Log2Res, size_t Log2EndRes, size_t NChannels, size_t NElmsize>
struct SPAdaptiveGrid {
    SPUntypedGrid<Log2Res, NChannels, NElmsize> m_head;
    SPAdaptiveGrid<Log2Res - 1, NChannels, NElmsize> m_next;
};

template <size_t Log2Res, size_t NChannels, size_t NElmsize>
struct SPAdaptiveGrid<Log2Res, Log2Res, NChannels, NElmsize> {
    SPUntypedGrid<Log2Res, NChannels, NElmsize> m_head;
};

}
