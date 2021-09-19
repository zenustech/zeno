#include <cstdio>
//#include "impl_cuda.h"
#include "impl_host.h"
#include "Vector.h"
#include "HashGrid.h"

using namespace fdb;

template <class T>
struct H21D3_Grid {
    struct Leaf {
        T m_data[8 * 8 * 8]{};

        inline FDB_DEVICE T &at(vec3i coord) {
            size_t i = coord[0] | coord[1] << 3 | coord[2] << 6;
            return m_data[i];
        }
    };

    HashGrid<Leaf> m_grid;

    inline FDB_CONSTEXPR size_t capacity_blocks() const {
        return m_grid.capacity();
    }

    inline void reserve_blocks(size_t n) {
        m_grid.reserve(n);
    }

    inline void clear_blocks() {
        m_grid.clear_blocks();
    }

    struct View {
        typename HashGrid<Leaf>::View m_view;

        View(H21D3_Grid const &parent)
            : m_view(parent.m_grid.view())
        {}

        template <class Kernel>
        inline void parallel_foreach(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            m_view.parallel_foreach([=] FDB_DEVICE (vec3i leaf_coord, Leaf &leaf) {
                leaf_coord <<= 3;
                for (int i = 0; i < 8 * 8 * 8; i++) {
                    int x = i & 0x7;
                    int y = (i >> 3) & 0x7;
                    int z = (i >> 6) & 0x7;
                    vec3i coord = leaf_coord | vec3i(x, y, z);
                    kernel(std::as_const(coord), leaf.m_data[i]);
                }
            }, cfg);
        }

        inline FDB_DEVICE T *touch(vec3i coord) const {
            auto *leaf = m_view.touch(coord >> 3);
            return &leaf->at(coord & 0x7);
        }

        inline FDB_DEVICE T *find(vec3i coord) const {
            auto *leaf = m_view.find(coord >> 3);
            return leaf ? &leaf->at(coord & 0x7) : nullptr;
        }

        inline FDB_DEVICE T &operator[](vec3i coord) const {
            return *touch(coord);
        }

        inline FDB_DEVICE T &operator()(vec3i coord) const {
            return *find(coord);
        }
    };

    View view() const {
        return *this;
    }
};

int main() {
#if 1
    H21D3_Grid<float> a;
    a.reserve_blocks(32);
    {
        auto av = a.view();
        parallel_for(vec3i(16, 16, 16), [=] FDB_DEVICE (vec3i c) {
            av[c] = length(vcast<float>(c));
        });

        av.parallel_foreach([=] FDB_DEVICE (vec3i c, float &v) {
            printf("%d %d %d %f %f\n", c[0], c[1], c[2], v, length(vcast<float>(c)));
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
