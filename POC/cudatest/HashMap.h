#pragma once

#include <utility>
#include "Vector.h"

namespace fdb {

// https://github.com/Miruna-Chi/CUDA-Hash-Table/blob/main/gpu_hashtable.cu
template <class K, class T>
struct HashMap {
    static_assert(std::is_trivially_move_constructible<T>::value);
    static_assert(std::is_trivially_move_assignable<T>::value);
    static_assert(std::is_trivially_destructible<T>::value);

    static_assert(std::is_trivially_copy_constructible<K>::value);
    static_assert(std::is_trivially_copy_assignable<K>::value);
    static_assert(std::is_trivially_move_constructible<K>::value);
    static_assert(std::is_trivially_move_assignable<K>::value);
    static_assert(std::is_trivially_destructible<K>::value);

    Vector<K> m_keys;
    Vector<T> m_values;

    inline FDB_CONSTEXPR size_t capacity() const {
        return m_keys.size();
    }

    inline void reserve(size_t n) {
        if (n > capacity()) {
            m_keys.resize(n);
            m_values.reserve(n);
        }
    }

    struct View {
        K *m_keys;
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
                if (p_keys[idx] != K()) {
                    kernel(std::as_const(p_keys[idx]), p_values[idx]);
                }
            }, cfg);
        }

        inline FDB_DEVICE size_t hash_func(K const &key) const {
            return (size_t)(m_capacity * std::fmod(key * 0.6180339887498949, 1.0));
        }

        inline FDB_DEVICE T *emplace(K key, T val) const {
            size_t hash = hash_func(key);
            for (size_t cnt = 0; cnt < m_capacity; cnt++) {
                if (
                #ifdef FDB_IMPL_CUDA
                atomic_cas(&m_keys[hash], key, key)
                #else
                atomic_load(&m_keys[hash])
                #endif
                == key) {
                    return new (&m_values[hash]) T(val);
                    return &m_values[hash];
                }
                if (atomic_cas(&m_keys[hash], K(), key) == K()) {
                    return new (&m_values[hash]) T(val);
                    return &m_values[hash];
                }
                hash++;
                if (hash > m_capacity)
                    hash = 0;
            }
            return nullptr;
        }

        inline FDB_DEVICE T *find(K key) const {
            size_t hash = hash_func(key);
            if (m_keys[hash] == key) {
                return &m_values[hash];
            }
            for (size_t cnt = 0; cnt < m_capacity - 1; cnt++) {
                hash++;
                if (hash > m_capacity)
                    hash = 0;
                if (m_keys[hash] == key) {
                    return &m_values[hash];
                }
            }
            return nullptr;
        }

        inline FDB_DEVICE T &at(K key) const {
            return *find(key);
        }
    };

    inline View view() const {
        return *this;
    }
};

}
