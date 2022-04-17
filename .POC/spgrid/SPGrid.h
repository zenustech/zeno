#include <array>
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

namespace bate::spgrid {

static_assert(sizeof(void *) == 8, "SPGrid requies 64-bit architecture");

static void *allocate(size_t size) {
    printf("SPGrid allocate size = %zd\n", size);
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

static void reloadPage(void *ptr) {
#if defined(_WIN32)
    ::VirtualAllocEx(GetCurrentProcess(), ptr,
        4096, MEM_COMMIT, PAGE_READWRITE);
#else
    ::mmap(ptr, 4096, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
#endif
}

static void unloadPage(void *ptr) {
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
struct SPLayout<1, 0> {
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

    void *m_ptr;
    static constexpr size_t Resolution = NRes;
    static constexpr size_t NumChannels = NChannels;
    static constexpr size_t ElementSize = NElmsize;
    static constexpr size_t MemorySize =
        NElmsize == 0 ?
        NRes * NRes * NRes * NChannels / 8 :
        NRes * NRes * NRes * NChannels * NElmsize;

    SPGrid() {
        m_ptr = allocate(MemorySize);
    }
    SPGrid(SPGrid const &) = delete;

    void *pointer(size_t c, size_t i, size_t j, size_t k) const {
        //size_t offset = LayoutClass::linearize(c, i, j, k);
        size_t offset = c + NChannels * (i + j * NRes + k * NRes * NRes);
        offset %= NRes * NRes * NRes * NChannels;
        return static_cast<void *>(static_cast<char *>(m_ptr) + offset);
    }

    ~SPGrid() {
        deallocate(m_ptr, MemorySize);
        m_ptr = nullptr;
    }
};

template <size_t NRes, size_t NChannels, typename T>
struct SPTypedGrid : SPGrid<NRes, NChannels, sizeof(T)> {
    using ValueType = std::array<T, NChannels>;

    T &at(size_t c, size_t i, size_t j, size_t k) const {
        return *(T *)this->pointer(c, i, j, k);
    }

    auto get(size_t i, size_t j, size_t k) const {
        ValueType ret;
        for (size_t c = 0; c < NChannels; c++) {
            ret[c] = at(c, i, j, k);
        }
        return ret;
    }

    void set(size_t i, size_t j, size_t k, ValueType const &val) const {
        for (size_t c = 0; c < NChannels; c++) {
            at(c, i, j, k) = val[c];
        }
    }
};

template <size_t NRes, typename T>
struct SPTypedGrid<NRes, 1, T> : SPGrid<NRes, 1, sizeof(T)> {
    using ValueType = T;

    T &at(size_t i, size_t j, size_t k) const {
        return *(T *)this->pointer(0, i, j, k);
    }

    T &at(size_t c, size_t i, size_t j, size_t k) const {
        return at(i, j, k);
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
    using ValueType = bool;

    unsigned char &uchar_at(size_t i, size_t j, size_t k) const {
        return *(unsigned char *)this->pointer(0, i, j, k);
    }

    bool get(size_t i, size_t j, size_t k) const {
        return (uchar_at(i, j, k) & (1 << (i & 7))) != 0;
    }

    void set_true(size_t i, size_t j, size_t k) const {
        uchar_at(i, j, k) |= (1 << (i & 7));
    }

    void set_false(size_t i, size_t j, size_t k) const {
        uchar_at(i, j, k) &= ~(1 << (i & 7));
    }

    void set(size_t i, size_t j, size_t k, bool value) const {
        if (value)
            set_true(i, j, k);
        else
            set_false(i, j, k);
    }
};

template <size_t NRes, size_t NScale = 8>
struct SPActivationMask {
    static constexpr auto Resolution = NRes;
    static constexpr auto MaskScale = NScale;

    SPBooleanGrid<NRes / NScale> m_grid;

    bool is_active(size_t i, size_t j, size_t k) const {
        return m_grid.get(i / NScale, j / NScale, k / NScale);
    }

    void activate(size_t i, size_t j, size_t k) const {
        m_grid.set_true(i / NScale, j / NScale, k / NScale);
    }

    void deactivate(size_t i, size_t j, size_t k) const {
        m_grid.set_false(i / NScale, j / NScale, k / NScale);
    }
};

template <class Grid>
struct SPMasked : Grid {
    static constexpr auto Resolution = Grid::Resolution;
    using typename Grid::ValueType;

    SPActivationMask<Resolution> m_mask;
    static constexpr auto MaskScale = SPActivationMask<Resolution>::MaskScale;

    bool is_active(size_t i, size_t j, size_t k) const {
        return m_mask.is_active(i, j, k);
    }

    void activate(size_t i, size_t j, size_t k) const {
        m_mask.activate(i, j, k);
    }

    void deactivate(size_t i, size_t j, size_t k) const {
        m_mask.deactivate(i, j, k);
    }

    ValueType direct_get(size_t i, size_t j, size_t k) const {
        return Grid::get(i, j, k);
    }

    void direct_set(size_t i, size_t j, size_t k, ValueType value) const {
        return Grid::set(i, j, k, value);
    }

    ValueType get(size_t i, size_t j, size_t k) const {
        if (is_active(i, j, k)) {
            return Grid::get(i, j, k);
        } else {
            return ValueType{};
        }
    }

    void set(size_t i, size_t j, size_t k, ValueType value) const {
        activate(i, j, k);
        return Grid::set(i, j, k, value);
    }
};

/*template <class Grid>
struct SPBitmasked : SPMasked<Grid> {
    static constexpr auto Resolution = Grid::Resolution;
    using typename Grid::ValueType;

    SPBooleanGrid<Resolution> m_bitmask;

    bool is_active(size_t i, size_t j, size_t k) const {
        return SPMasked<Grid>::is_active(i, j, k) && m_bitmask.get(i, j, k);
    }

    void activate(size_t i, size_t j, size_t k) const {
        SPMasked<Grid>::activate(i, j, k);
        m_bitmask.set_true(i, j, k);
    }

    void deactivate(size_t i, size_t j, size_t k) const {
        SPMasked<Grid>::deactivate(i, j, k);
        m_bitmask.set_false(i, j, k);
    }

    ValueType get(size_t i, size_t j, size_t k) const {
        if (is_active(i, j, k)) {
            return Grid::get(i, j, k);
        } else {
            return ValueType{};
        }
    }

    void set(size_t i, size_t j, size_t k, ValueType value) const {
        activate(i, j, k);
        return Grid::set(i, j, k, value);
    }
};*/

}
