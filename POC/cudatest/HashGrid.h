#pragma once

#include "HashMap.h"
#include "vec.h"

namespace fdb {

template <class T>
struct HashGrid {
    struct u64u21x3 {
        uint64_t value;

        FDB_CONSTEXPR u64u21x3() : value((uint64_t)-1L) {}
        FDB_CONSTEXPR u64u21x3(uint64_t value) : value(value) {}
        FDB_CONSTEXPR operator uint64_t() const { return value; }

        FDB_CONSTEXPR u64u21x3(vec3i const &a) {
            uint64_t x = a[0] & 0x1fffffUL;
            uint64_t y = a[1] & 0x1fffffUL;
            uint64_t z = a[2] & 0x1fffffUL;
            value = x | y << 21 | z << 42;
        }

        FDB_CONSTEXPR bool has_value() const {
            return (int64_t)value >= 0l;
        }

        FDB_CONSTEXPR operator vec3i() const {
            size_t x = value & 0x1fffffUL;
            size_t y = (value >> 21) & 0x1fffffUL;
            size_t z = (value >> 42) & 0x1fffffUL;
            return vec3i(x, y, z);
        }
    };

    // https://stackoverflow.com/questions/32370487/64bit-atomicadd-in-cuda
    HashMap<u64u21x3, T, unsigned long long int> m_table;

    inline FDB_CONSTEXPR size_t capacity() const {
        return m_table.capacity();
    }

    inline void reserve(size_t n) {
        return m_table.reserve(n);
    }

    inline void clear() {
        return m_table.clear();
    }

    struct View {
        typename HashMap<u64u21x3, T, unsigned long long int>::View m_view;

        inline View(HashGrid const &parent)
            : m_view(parent.m_table.view())
        {}

        template <class Kernel>
        inline void parallel_foreach(Kernel kernel, ParallelConfig cfg = {512, 2}) const {
            m_view.parallel_foreach([=] FDB_DEVICE (u64u21x3 key, T &value) {
                vec3i coord(key);
                kernel(std::as_const(coord), value);
            }, cfg);
        }

        inline FDB_DEVICE T *emplace(vec3i coord, T value) const {
            u64u21x3 key(coord);
            return m_view.emplace(key, value);
        }

        inline FDB_DEVICE T *touch(vec3i coord) const {
            u64u21x3 key(coord);
            return m_view.touch(key);
        }

        inline FDB_DEVICE T *find(vec3i coord) const {
            u64u21x3 key(coord);
            return m_view.find(key);
        }

        inline FDB_DEVICE T &operator[](vec3i coord) const {
            return *touch(coord);
        }

        inline FDB_DEVICE T &operator()(vec3i coord) const {
            return *find(coord);
        }
    };

    inline View view() const {
        return *this;
    }
};

}
