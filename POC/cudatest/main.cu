#include <cstdio>
//#include "impl_cuda.h"
#include "impl_host.h"
#include "Vector.h"
#include "H21D3B_Grid.h"

using namespace fdb;

int main() {
#if 1
    H21D3B_Grid<float> a;
    a.reserve_blocks(32);
    {
        auto av = a.view();
        parallel_for(vec3i(16, 16, 14), [=] FDB_DEVICE (vec3i c) {
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
