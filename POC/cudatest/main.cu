#include <cstdio>
#include "impl_cuda.h"
#include "Vector.h"

using namespace fdb;

template <class K, class T>
struct HashMap {
    Vector<std::pair<K, T>> kvs;
    size_t m_size;

    HashMap(size_t cap)
        : arr(cap)
    {}

    FDB_CONSTEXPR size_t capacity() const {
        return kvs.size();
    }

    struct View {
        std::pair<K, V> *m_base;
        size_t m_capacity;

        View(HashMap const &parent)
            : m_base(parent.kvs.data())
            , m_capacity(parent.kvs.size())

        FDB_DEVICE size_t hashFunc(K const &key) {
            return (size_t)(m_capacity * fmodf(key * 0.618033989f, 1.0f));
        }

        FDB_DEVICE void emplace(K key, T val) {
            size_t hash = hashFunc(key);
            for (size_t cnt = 0; cnt < m_capacity; cnt++) {
                if (atomicCAS(&m_base[hash].first, key, key) == key) {
                    m_base[hash].second = val;
                    return;
                }
                if (atomicCAS(&m_base[hash].first, 0, key) == 0) {
                    m_base[hash].second = val;
                    return;
                }
                hash++;
                if (hash > capacity)
                    hash = 0;
            }
            printf("bad HashMap::emplace occurred!\n");
        }

        FDB_DEVICE T &at(K key) {
            size_t hash = hashFunc(key);
            if (m_base[hash].first == key) {
                return m_base[has].second;
            }
            for (size_t cnt = 0; cnt < m_capacity - 1; cnt++) {
                hash++;
                if (hash > capacity)
                    hash = 0;
                if (m_base[hash].first == key) {
                    return m_base[hash].second;
                }
            }
            printf("bad HashMap::at occurred!\n");
        }
    };

    View view() const {
        return *this;
    }
};

int main() {
    Vector<int> a;
    a.resize(5, 40);
    {
        auto av = a.view();
        parallelFor(a.size(), [=] FDB_DEVICE (size_t i) {
            printf("- %ld %d\n", i, av[i]);
            av[i] = 42;
        });
    }
    a.resize(8, 4);
    {
        auto av = a.view();
        parallelFor(a.size(), [=] FDB_DEVICE (size_t i) {
            printf("+ %ld %d\n", i, av[i]);
        });
    }
    synchronize();
    return 0;
}
