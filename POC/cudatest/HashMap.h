#pragma once

#include <utility>
#include "Vector.h"

namespace fdb {

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

    struct KTPair {
        K key;
        T value;
    };

    Vector<KTPair> kvs;
    size_t m_size;

    HashMap(size_t n)
    {
        kvs.reserve(n);
    }

    FDB_CONSTEXPR size_t capacity() const {
        return kvs.capacity();
    }

    void reserve(size_t n) const {
        kvs.reserve(n);
    }

    struct View {
        KTPair *m_base;
        size_t m_capacity;

        View(HashMap const &parent)
            : m_base(parent.kvs.data())
            , m_capacity(parent.kvs.capacity())
        {}

        FDB_DEVICE size_t hashFunc(K const &key) const {
            return (size_t)(m_capacity * fmodf(key * 0.618033989f, 1.0f));
        }

        FDB_DEVICE void emplace(K key, T val) const {
            size_t hash = hashFunc(key);
            for (size_t cnt = 0; cnt < m_capacity; cnt++) {
                if (atomicCAS(&m_base[hash].key, key, key) == key) {
                    new (&m_base[hash].value) T(val);
                    return;
                }
                if (atomicCAS(&m_base[hash].key, 0, key) == 0) {
                    new (&m_base[hash].value) T(val);
                    return;
                }
                hash++;
                if (hash > m_capacity)
                    hash = 0;
            }
            //printf("bad HashMap::emplace occurred!\n");
        }

        FDB_DEVICE T &at(K key) const {
            size_t hash = hashFunc(key);
            if (m_base[hash].key == key) {
                return m_base[hash].value;
            }
            for (size_t cnt = 0; cnt < m_capacity - 1; cnt++) {
                hash++;
                if (hash > m_capacity)
                    hash = 0;
                if (m_base[hash].key == key) {
                    return m_base[hash].value;
                }
            }
            //printf("bad HashMap::at occurred!\n");
            return *(T *)(-1);
        }
    };

    View view() const {
        return *this;
    }
};

}
