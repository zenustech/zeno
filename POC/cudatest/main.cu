#include <cstdio>
//#include "impl_cuda.h"
#include "impl_host.h"
#include "Vector.h"
#include "HashGrid.h"

using namespace fdb;

template <class T>
struct H21D4_Grid {
    struct LeafArray {
        T m_data[16 * 16 * 16]{};

        inline FDB_DEVICE T &operator[](vec3S coord) {
            size_t i = coord[0] | coord[1] << 4 | coord[2] << 8;
            return m_data[i];
        }
    };

    HashGrid<LeafArray> m_grid;

    inline FDB_CONSTEXPR size_t capacity_blocks() const {
        m_grid.capacity();
    }

    inline void reserve_blocks(size_t n) {
        m_grid.reserve(n);
    }

    inline void clear_blocks() {
        m_grid.clear_blocks();
    }

    struct View {
        typename HashGrid<LeafArray>::View m_blk_view;

        View(H21D4_Grid const &parent)
            : m_view(parent.m_grid.view())
        {}

        inline FDB_DEVICE T *touch(vec3S coord) const {
            LeafArray *leaf = m_view.touch(coord >> 4);
            return &leaf[coord & 0xf];
        }

        inline FDB_DEVICE T *find(vec3S coord) const {
            return m_view.find(coord >> 4);
            return &leaf[coord & 0xf];
        }

        inline FDB_DEVICE T &operator[](vec3S coord) const {
            return *touch(coord);
        }

        inline FDB_DEVICE T &operator()(vec3S coord) const {
            return *find(coord);
        }
    };

    View view() const {
        return *this;
    }
};

int main() {
#if 1
    HashGrid<float> a;
    a.reserve(4099);
    {
        auto av = a.view();
        parallel_for(vec3S(16, 16, 16), [=] FDB_DEVICE (vec3S c) {
            av.emplace(c, length(vcast<float>(c)));
        });

        av.parallel_foreach([=] FDB_DEVICE (vec3S c, float &v) {
            printf("%ld %ld %ld %f %f\n", c[0], c[1], c[2], v, length(vcast<float>(c)));
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
