#include <cstdio>
#include "impl_cuda.h"
#include <type_traits>

using namespace fdb;

template <class T>
struct Vector {
    T *m_base;
    size_t m_size;

    static_assert(std::is_trivially_move_constructible<T>::value);
    static_assert(std::is_trivially_move_assignable<T>::value);
    static_assert(std::is_trivially_destructible<T>::value);

    Vector()
        : m_base(nullptr)
        , m_size(0)
    {}

    template <class Args>
    explicit Vector(size_t n, Args ...args)
        : m_base(n ? allocate(n) : nullptr)
        , m_size(n)
    {
        if (m_size) {
            parallelFor(m_size, [=] FDB_DEVICE (size_t i) {
                new (&m_base[i]) T(args...);
            });
        }
    }

    ~Vector() {
        if (m_size) {
            parallelFor(m_size, [=] FDB_DEVICE (size_t i) {
                m_base[i]->~T();
            });
            m_size = 0;
        }
        if (m_base) {
            deallocate(m_base);
            m_base = nullptr;
        }
    }

    Vector(Vector const &) = delete;
    Vector &operator=(Vector const &) = delete;
    Vector(Vector &&) = default;
    Vector &operator=(Vector &&) = default;

    Vector clone() const {
        Vector res;
        res.reserve(m_size);
        parallelFor(m_size, [p_res = res.m_base, p_this = m_base] FDB_DEVICE (size_t i) {
            new (&p_res[i]) T(static_cast<T const &>(p_this[i]));
        });
        return res;
    }

    void assign(Vector const &other) const {
        clear();
        resize(other.size());
        parallelFor(other.size(), [p_this = m_base, p_other = other.m_base] FDB_DEVICE (size_t i) {
            new (&p_this[i]) T(static_cast<T const &>(p_other[i]));
        });
        return res;
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

    void resize(size_t n) {
        reserve(n);
        if (m_size) {
            if (m_base) {
            }
        }
    }
};

__global__ void a() { printf("a\n"); }

int main() {
    parallelFor(vec3S(2, 2, 4), [=] FDB_DEVICE (vec3S idx) {
        printf("are you ok? %ld %ld %ld\n", idx[0], idx[1], idx[2]);
    });
    synchronize();
    return 0;
}
