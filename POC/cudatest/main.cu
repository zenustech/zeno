#include <cstdio>
#include "impl_cuda.h"
//#include "impl_host.h"
#include "Vector.h"
#include "HashMap.h"

using namespace fdb;

template <class T>
struct HashGrid {
    struct u64_3x21 {
        uint64_t value;

        u64_3x21() : value((uint64_t)-1l) {}

        FDB_CONSTEXPR u64_3x21(vec3S const &a) {
            uint64_t x = a[0] & 0x1ffffful;
            uint64_t y = a[1] & 0x1ffffful;
            uint64_t z = a[2] & 0x1ffffful;
            value = x | y << 21 | z << 42;
        }

        FDB_CONSTEXPR bool has_value() const {
            return (int64_t)value >= 0l;
        }

        FDB_CONSTEXPR operator vec3S() const {
            size_t x = value & 0x1ffffful;
            size_t y = (value >> 21) & 0x1ffffful;
            size_t z = (value >> 42) & 0x1ffffful;
            return vec3S(x, y, z);
        }
    };

    HashMap<u64_3x21, T> m_table;

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
        HashMap<u64_3x21, T> m_view;

        inline View(HashGrid const &parent)
            : m_view(parent.view())
        {}

        template <class Kernel>
        inline void parallel_foreach(Kernel kernel, ParallelConfig cfg = {512, 2}) const {
            m_view.parallel_foreach([=] FDB_DEVICE (u64_3x21 key, T &value) {
                vec3S coord(key);
                kernel(std::as_const(coord), value);
            }, cfg);
        }

        inline FDB_DEVICE T *emplace(vec3S coord, T value) const {
            u64_3x21 key(coord);
            return m_view.emplace(key, value);
        }

        inline FDB_DEVICE T *touch(vec3S coord) const {
            u64_3x21 key(coord);
            return m_view.touch(key);
        }

        inline FDB_DEVICE T *find(vec3S coord) const {
            u64_3x21 key(coord);
            return m_view.find(key);
        }

        inline FDB_DEVICE T &operator[](vec3S coord) const {
            return *touch(key);
        }

        inline FDB_DEVICE T &operator()(vec3S coord) const {
            return *find(key);
        }
    };

    inline View view() const {
        return *this;
    }
};

int main() {
#if 1
    HashGrid<float> a;
    a.reserve(4099);
    {
        auto av = a.view();
        parallel_for(vec3S(4, 4, 4), [=] FDB_DEVICE (vec3S c) {
            av.emplace(c, length(cast<float>(c)));
        });

        av.parallel_foreach([=] FDB_DEVICE (vec3S c, int &v) {
            printf("%d %d %d %f\n", c[0], c[1], c[2], v);
        });
    }

#else
    Vector<int> a;
    a.resize(5, 40);
    {
        auto av = a.view();
        parallel_for(a.size(), [=] FDB_DEVICE (size_t i) {
            printf("- %ld %d\n", i, av[i]);
            av[i] = 42;
        });
    }
    a.resize(8, 4);
    {
        auto av = a.view();
        parallel_for(a.size(), [=] FDB_DEVICE (size_t i) {
            printf("+ %ld %d\n", i, av[i]);
        });
    }

#endif

    synchronize();
    return 0;
}
