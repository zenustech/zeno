#pragma once

#include <new>
#include <utility>
#include "Vector.h"
#include "vec.h"

namespace fdb {

// https://github.com/Miruna-Chi/CUDA-Hash-Table/blob/main/gpu_hashtable.cu
template <class K, class T, class UK = K>
struct HashMap {
    static_assert(std::is_trivially_constructible<T>::value);
    static_assert(std::is_trivially_move_constructible<T>::value);
    static_assert(std::is_trivially_move_assignable<T>::value);
    static_assert(std::is_trivially_destructible<T>::value);

    static_assert(std::is_constructible<K>::value);
    static_assert(std::is_trivially_copy_constructible<K>::value);
    static_assert(std::is_trivially_copy_assignable<K>::value);
    static_assert(std::is_trivially_move_constructible<K>::value);
    static_assert(std::is_trivially_move_assignable<K>::value);
    static_assert(std::is_trivially_destructible<K>::value);

    struct KB {
        K key{};
        int initialized{false};
    };

    Vector<KB> m_keys;
    Vector<T> m_values;

    inline FDB_CONSTEXPR size_t capacity() const {
        return m_keys.size();
    }

    inline void reserve(size_t n) {
        if (n > capacity()) {
            m_keys.resize(n);
            m_values.resize(n);
        }
    }

    inline void clear() {
        m_keys.clear();
        m_values.clear();
    }

    struct View {
        KB *m_keys;
        T *m_values;
        size_t m_capacity;

        inline View(HashMap const &parent)
            : m_keys(parent.m_keys.data())
            , m_values(parent.m_values.data())
            , m_capacity(parent.capacity())
        {}

        template <class Kernel>
        inline void parallel_foreach(Kernel kernel, ParallelConfig cfg = {512, 2}) const {
            auto p_keys = m_keys;
            auto p_values = m_values;
            parallel_for(m_capacity, [=] FDB_DEVICE (size_t idx) {
                if (p_keys[idx].key != K()) {
                    kernel(std::as_const(p_keys[idx].key), p_values[idx]);
                }
            }, cfg);
        }

        inline FDB_DEVICE size_t hash_func(K const &key) const {
            return (size_t)(m_capacity * std::fmod((UK)key * 0.6180339887498949, 1.0));
        }

        inline FDB_DEVICE T *touch(K key) const {
            size_t hash = hash_func(key);
            for (size_t cnt = 0; cnt < m_capacity * 32; cnt++) {
                if (atomic_cass((UK *)&m_keys[hash].key, (UK)key, (UK)key)) {
                    return &m_values[hash];
                }
                if (atomic_cass((UK *)&m_keys[hash].key, (UK)K(), (UK)key)) {
                    return &m_values[hash];
                }
                hash++;
                if (hash > m_capacity)
                    hash = 0;
            }
            printf("bad touch\n");
            return nullptr;
        }

        inline FDB_DEVICE T *find(K key) const {
            size_t hash = hash_func(key);
            if (m_keys[hash].key == key) {
                return &m_values[hash];
            }
            for (size_t cnt = 0; cnt < m_capacity * 32; cnt++) {
                hash++;
                if (hash > m_capacity)
                    hash = 0;
                if (m_keys[hash].key == key) {
                    return &m_values[hash];
                }
            }
            printf("bad find\n");
            return nullptr;
        }

        inline FDB_DEVICE T &operator[](K key) const {
            return *touch(key);
        }

        inline FDB_DEVICE T &operator()(K key) const {
            return *find(key);
        }
    };

    inline View view() const {
        return *this;
    }
};

template <class T>
struct HashMap<vec3i, T, vec3i> {
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
            return (int64_t)value >= 0L;
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

        inline View(HashMap const &parent)
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
