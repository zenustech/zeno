#pragma once

#include <new>
#include <utility>
#include "Vector.h"

namespace fdb {

// https://github.com/Miruna-Chi/CUDA-Hash-Table/blob/main/gpu_hashtable.cu
template <class K, class T, class UK = K>
struct HashMap {
    static_assert(std::is_trivially_constructible<T>::value);
    static_assert(std::is_trivially_move_constructible<T>::value);
    static_assert(std::is_trivially_move_assignable<T>::value);
    static_assert(std::is_trivially_destructible<T>::value);

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

}
