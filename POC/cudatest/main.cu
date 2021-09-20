#include <cstdio>
#include "impl_cuda.h"
//#include "impl_host.h"
#include "Vector.h"
#include "H21B3_Grid.h"

using namespace fdb;

int main() {
#if 1
    H21B3_Grid<vec3f> vel;
    float dt = 0.01f;

    a.reserve_blocks(16);
    {
        auto _vel = vel.view();
        _vel.parallel_foreach([=] FDB_DEVICE (vec3i c, vec3f &vel) {
            auto btpos = c - vel * dt;
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
i
