#include <cstdio>
#include "impl_cuda.h"
#include <type_traits>

using namespace fdb;

template <class T>
struct Vector {
    T *m_base;
    size_t m_size;
    size_t m_cap;

    static_assert(std::is_trivially_move_constructible<T>::value);
    static_assert(std::is_trivially_move_assignable<T>::value);
    static_assert(std::is_trivially_destructible<T>::value);

    inline Vector()
        : m_base(nullptr)
        , m_size(0)
        , m_cap(0)
    {}

    inline explicit Vector(size_t n)
        : m_base(n ? allocate(n * sizeof(T)) : nullptr)
        , m_size(n)
        , m_cap(n)
    {
        if (m_size) {
            auto p_base = m_base;
            parallelFor(m_size, [=] FDB_DEVICE (size_t i) {
                new (&p_base[i]) T();
            });
        }
    }

    inline explicit Vector(size_t n, T val)
        : m_base(n ? allocate(n * sizeof(T)) : nullptr)
        , m_size(n)
        , m_cap(n)
    {
        if (m_size) {
            auto p_base = m_base;
            parallelFor(m_size, [=] FDB_DEVICE (size_t i) {
                new (&p_base[i]) T(val);
            });
        }
    }

    inline ~Vector() {
        m_size = 0;
        if (m_base) {
            deallocate(m_base);
            m_base = nullptr;
            m_cap = 0;
        }
    }

    Vector(Vector const &) = delete;
    Vector &operator=(Vector const &) = delete;
    Vector(Vector &&) = default;
    Vector &operator=(Vector &&) = default;

    inline Vector clone() const {
        Vector res;
        res.reserve(m_size);
        auto p_res = res.m_base;
        auto p_this = m_base;
        parallelFor(m_size, [=] FDB_DEVICE (size_t i) {
            new (&p_res[i]) T(static_cast<T const &>(p_this[i]));
        });
        return res;
    }

    inline void assign(Vector const &other) const {
        clear();
        resize(other.size());
        auto p_this = m_base;
        auto p_other = other.m_base;
        parallelFor(other.size(), [=] FDB_DEVICE (size_t i) {
            new (&p_this[i]) T(static_cast<T const &>(p_other[i]));
        });
    }

    inline FDB_CONSTEXPR T *data() const {
        return m_base;
    }

    inline FDB_CONSTEXPR size_t size() const {
        return m_size;
    }

    inline FDB_CONSTEXPR T &operator[](size_t i) const {
        return m_base[i];
    }

    inline FDB_CONSTEXPR size_t capacity() const {
        return m_cap;
    }

    inline void __recapacity(size_t n) {
        if (n) {
            if (m_cap) {
                m_base = (T *)reallocate(m_base, m_size * sizeof(T), n * sizeof(T));
            } else {
                m_base = (T *)allocate(n * sizeof(T));
            }
        } else {
            deallocate(m_base);
            m_base = nullptr;
        }
        m_cap = n;
    }

    inline void reserve(size_t n) {
        if (n > m_cap) {
            __recapacity(n);
        }
    }

    inline void shrink_to_fit() {
        if (m_cap > m_size) {
            __recapacity(m_size);
        }
    }

    inline void resize(size_t n) {
        reserve(n);
        if (n > m_size) {
            auto p_base = m_base;
            auto p_size = m_size;
            parallelFor(n - m_size, [=] FDB_DEVICE (size_t i) {
                new (&p_base[p_size + i]) T();
            });
        } else if (n < m_size) {
            auto p_base = m_base;
            auto p_size = m_size;
            parallelFor(m_size - n, [=] FDB_DEVICE (size_t i) {
                new (&p_base[n + i]) T();
            });
        }
    }

    inline void resize(size_t n, T val) {
        reserve(n);
        if (n > m_size) {
            auto p_base = m_base;
            auto p_size = m_size;
            parallelFor(n - m_size, [=] FDB_DEVICE (size_t i) {
                new (&p_base[p_size + i]) T(val);
            });
        } else if (n < m_size) {
            auto p_base = m_base;
            parallelFor(m_size - n, [=] FDB_DEVICE (size_t i) {
                new (&p_base[n + i]) T(val);
            });
        }
    }

    T *view() const {
        return m_base;
    }
};

__global__ void a() { printf("a\n"); }

int main() {
    Vector<int> a;
    a.resize(5, 42);
    auto av = a.view();
    parallelFor(a.size(), [=] FDB_DEVICE (size_t i) {
        printf("%ld %d\n", i, av[i]);
    });
    synchronize();
    return 0;
}
