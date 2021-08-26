#include <cstdio>
#include <fdb/types.h>
#include <fdb/policy.h>
#include <fdb/SPGrid.h>
#include <fdb/openvdb.h>

using namespace fdb;

int main() {
    spgrid::SPFloatGrid<128> g_pre;
    spgrid::SPFloat3Grid<128> g_vel;
    ndrange_for(policy::Serial{},
    vec<int, 3>(32), vec<int, 3>(64), [&] (auto idx) {
        int i = idx[0], j = idx[1], k = idx[2];
        float c = g_pre.get(i, j, k);
        g_vel.set(i, j, k, vec<float, 3>(
                c - g_pre.get(i+1, j, k),
                c - g_pre.get(i, j+1, k),
                c - g_pre.get(i, j, k+1)));
    });
    return 0;
}
