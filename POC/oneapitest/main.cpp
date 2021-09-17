//#include "ImplIntel.h"
#include "ImplHost.h"
#include "Vector.h"
#include <cstdio>
#include "vec.h"

using namespace ImplHost;

template <class T>
struct HashFunc {
};

template <>
struct HashFunc<uint32_t> {
    size_t operator()(uint32_t i) const {
        i = (i ^ 61) ^ (i >> 16);
        i = i * 314159265u + 1931127624u;
        return i;
    }
};

template <>
struct HashFunc<vec3I> {
    size_t operator()(vec3I c) const {
        return (73856093u * c[0]) ^ (19349663u * c[1]) ^ (83492791u * c[2]);
    }
};

template <class Key, class T, class Alloc, class KeyHash = HashFunc<Key>>
struct HashMap {
    Vector<std::pair<Key, T>, Alloc> m_table;
    size_t m_size{0};
    Alloc m_alloc;

    explicit HashMap(Alloc alloc)
        : m_table(alloc)
        , m_size(0)
        , m_alloc(alloc)
    {}

    size_t capacity() const {
        return m_table.size();
    }

    void __recapacity(size_t n) {
        m_table.resize(n);
    }

    void reserve(size_t n) {
        if (capacity() < n) {
            __recapacity(n);
        }
    }

    void shrink_to_fit() {
        if (capacity() > m_size) {
            __recapacity(m_size);
        }
    }

    struct View {
        typename Vector<std::pair<Key, T>, Alloc>::View m_table_view;

        View(HashMap const &parent)
            : m_table_view(parent.m_table.view())
        {}

        T &operator[](Key i) const {
            size_t offset = KeyHash{}(i) % m_table_view.size();
            auto &kv = m_table_view[offset];
            return kv.second;
        }

        using iterator = T *;

        iterator begin() const {
            return m_table_view.begin();
        }

        iterator end() const {
            return m_table_view.end();
        }
    };

    View view() const {
        return {*this};
    }
};

int main(void) {
    Queue q;

    HashMap<uint32_t, int, Allocator> v(q.allocator());
    Vector<size_t, Allocator> c(q.allocator(), 1);
    v.reserve(200);

    auto vAxr = v.view();
    auto cAxr = c.view();

    cAxr[0] = 0;
    q.parallel_for(Dim3(100, 1, 1), [=](Dim3 idx) {
        size_t id = make_atomic_ref(cAxr[0])++;
        vAxr[id] = idx.x;
    });

    for (int i = 0; i < 100; i++) {
        printf("%d\n", vAxr[i]);
    }

    return 0;
}
