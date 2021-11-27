#include <cstdio>
#include "impl_cuda.h"
//#include "impl_host.h"
#include "Vector.h"
#include "H21B3_Grid.h"

using namespace fdb;

template <class Grid>
auto bilerp(Grid &grid, vec3f pos) {
    auto ip = ifloor(pos);
    auto of = pos - ip;
    return (
        of[0] * of[0] * of[0] * grid(pos)
}

template <class Vel, class Qua>
auto advect(Vel vel, Qua qua, Qua new_qua, float dt) {
    auto _vel = vel.view();
    auto _qua = qua.view();
    auto _new_qua = new_qua.view();
    _vel.parallel_foreach([=] FDB_DEVICE (vec3i c, vec3f &vel) {
        auto btpos = c - vel * dt;
        _new_qua[i] = bilerp(_qua, btpos);
    });
}

int main() {
#if 1
    H21B3_Grid<vec3f> vel, new_vel;
    float dt = 0.01f;

    advect(vel, vel, new_vel, dt);
    std::swap(vel, new_vel);

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
