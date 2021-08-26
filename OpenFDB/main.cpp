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
    vec<int, 3>(0), vec<int, 3>(128), [&] (auto idx) {
        int i = idx[0], j = idx[1], k = idx[2];
        float c = (i / 16 + j / 16 + k / 16) % 2 ? 1.f : 0.f;
        g_pre.set(i, j, k, c);
    });

    ndrange_for(policy::Serial{},
    vec<int, 3>(1), vec<int, 3>(127), [&] (auto idx) {
        int i = idx[0], j = idx[1], k = idx[2];
        float c = g_pre.get(i, j, k);
        g_vel.set(i, j, k, vec<float, 3>(
                c - g_pre.get(i+1, j, k),
                c - g_pre.get(i, j+1, k),
                c - g_pre.get(i, j, k+1)));
    });

    write_dense_vdb("/tmp/a.vdb", [&] (auto coor) {
        return abs(g_vel.get(coor[0], coor[1], coor[2]));
    }, vec<int, 3>(128));

    return 0;
}
