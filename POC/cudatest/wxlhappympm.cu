#include <cstdio>
#include "impl_cuda.h"
//#include "impl_host.h"
#include "Vector.h"
#include "H21B3_Grid.h"

using namespace fdb;

int main() {
#if 1
    H21B3_Grid<vec3f> g_v;
    H21B3_Grid<float> g_m;
    Vector<vec3f> p_x;
    Vector<vec3f> p_v;
    Vector<vec3f> p_C;
    Vector<vec3f> p_J;

    float dx = 0.01f;

    a.reserve_blocks(16);
    {
        auto _g_v = g_v.view();
        auto _g_m = g_m.view();
        auto _p_x = p_x.view();
        auto _p_C = p_C.view();
        auto _p_J = p_J.view();
        parallel_for(p_x.size(), [=] FDB_DEVICE (size_t p) {
            auto Xp = _p_x[p] / dx;
            auto base = vcast<int>(Xp - 0.5f);
            auto fx = Xp - base;
            vec3f w(
                0.5f * pow(1.5f - fx, 2),
                0.75f - pow(xf - 1.0f, 2),
                0.5f * pow(fx - 0.5f, 2));
            auto stress = -4.f * dt * E * pow(dx * 0.5f, 2) * (_p_J[p] - 1.0f) / pow(dx, 2);
        });

        av.parallel_foreach([=] FDB_DEVICE (int i, int &val) {
            printf("%d %d\n", i, val);
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
