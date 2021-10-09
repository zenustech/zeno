#include "ImplIntel.h"
//#include "ImplHost.h"
#include "Vector.h"
#include <cstdio>
#include "vec.h"

template <class T>
struct HashFunc {
};

template <>
struct HashFunc<uint32_t> {
    size_t operator()(uint32_t i) const {
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

template <class Alloc>
struct SpinLock {
    int m{0};

    void acquire() {
        static const __INTEL_SYCL_CONSTANT char fmt[] = "spin\n";
        while (!Alloc::make_atomic_ref(m).store_if_equal(0, 1))
            Queue::printf(fmt);
    }

    void release() {
        Alloc::make_atomic_ref(m).store(0);
    }
};

template <class Key, class T, class Alloc, class KeyHash = HashFunc<Key>>
struct HashMap {
    struct HashEntry {
        Key key;
        bool used{false};
        SpinLock<Alloc> spin;
        T value;
    };

    Vector<HashEntry, Alloc> m_table;
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

    struct TouchView {
        typename Vector<HashEntry, Alloc>::View m_view;

        TouchView(HashMap const &parent)
            : m_view(parent.m_table.view())
        {}

        using iterator = HashEntry *;

        iterator find(Key k) const {
            size_t offset = KeyHash{}(k) % m_view.size();
            auto it = m_view.find(offset);
            if (it->used && it->key == k) {
                return it;
            }
            while (1) {
                offset = (offset + 1) % m_view.size();
                it = m_view.find(offset);
                if (!it->used) {
                    it->spin.acquire();
                    if (!it->used) {
                        it->used = true;
                        it->key = k;
                        it->spin.release();
                        return it;
                    } else {
                        it->spin.release();
                    }
                } else if (it->key == k) {
                    return it;
                }
            }
        }

        T &operator[](Key k) const {
            return find(k)->value;
        }
    };

    TouchView touch_view() const {
        return {*this};
    }

    struct View {
        typename Vector<HashEntry, Alloc>::View m_view;

        View(HashMap const &parent)
            : m_view(parent.m_table.view())
        {}

        using iterator = HashEntry *;

        iterator find(Key k) const {
            size_t offset = KeyHash{}(k) % m_view.size();
            auto it = m_view.find(offset);
            if (it->used && it->key == k) {
                return it;
            }
            while (1) {
                offset = (offset + 1) % m_view.size();
                it = m_view.find(offset);
                if (it->used && it->key == k) {
                    return it;
                }
            }
        }

        T &operator[](Key k) const {
            return find(k)->value;
        }
    };

    View view() const {
        return {*this};
    }
};

int main(void) {
    Queue q;

    HashMap<uint32_t, int, Queue> v(q);
    Vector<size_t, Queue> c(q, 1);
    v.reserve(200);

    auto vAxr = v.touch_view();
    auto cAxr = c.view();

    cAxr[0] = 0;
    q.parallel_for(Dim3(1, 1, 1), [=](Dim3 idx) {
        size_t id = Queue::make_atomic_ref(cAxr[0]).fetch_inc();
        vAxr[id] = idx.x;
    });

    for (int i = 0; i < 100; i++) {
        printf("%d\n", vAxr[i]);
    }

    return 0;
}
